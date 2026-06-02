# Framework

Configuration-driven multi-task computer-vision training on top of PyTorch Lightning, Hydra, Pydantic, timm/smp, albumentations and torchmetrics.

---

## Phase 1 — Setup (runs once before `trainer.fit`)

```mermaid
sequenceDiagram
    participant main
    participant Config as ExperimentConfig<br/>(Pydantic)
    participant Wiring as wiring.py
    participant DM as DataModule
    participant DS as DataSource<br/>(CsvDataSource)
    participant IL as InputLoader<br/>(per input)
    participant Codec as TargetCodec<br/>(per task)
    participant RT as RuntimeContext
    participant TB as TaskBuilder<br/>(TopologyStrategy × ObjectiveStrategy)
    participant CM as CompositeModel<br/>(TimmBackbone + heads)

    main->>Config: load_config(raw_dict)<br/>Hydra DictConfig → validated DTOs
    Config-->>main: ExperimentConfig

    main->>Wiring: build_bindings(config)
    Note over Wiring: infers TargetCodec per task<br/>from preset + objective
    Wiring-->>main: list[TargetBinding(name, column, codec)]

    main->>Wiring: build_data_module(config, target_bindings, runtime)
    Note over Wiring: resolves split/pre-split mode,<br/>builds transforms from configs/transforms/
    Wiring-->>main: DataModule

    main->>DM: setup()

    DM->>DS: read()
    DS-->>DM: DataFrame (full annotation table)

    DM->>DM: _build_input_bindings(inputs_config, frame)
    Note over DM: auto-detects loader per column<br/>from file extensions → image/text
    DM-->>IL: InputBinding(name, column, loader) per input

    loop for each TargetBinding
        DM->>Codec: fit(column_values)
        Note over Codec: LabelIndexCodec → sorts vocab<br/>MultiLabelBinarizeCodec → multi-hot vocab<br/>FloatCodec / MaskCodec → no-op
        Codec-->>RT: num_classes[task_name] = C
    end

    DM->>DM: split_dataframe(ratios, seed, stratify_column?)
    Note over DM: → Dataset per stage<br/>(frame + input_bindings + target_bindings + transform)
    DM-->>RT: dataset_sizes[stage] = N

    main->>Wiring: build_tasks(config, runtime)
    Note over Wiring: runs AFTER setup()<br/>so num_classes is a concrete int

    loop for each task in config.tasks
        Wiring->>TB: build(name, num_classes, objective, ...)
        TB->>TB: validate Topology × Objective
        TB->>TB: out_features / head_spec / codec<br/>criterion / activation / metrics × stages
        TB-->>Wiring: Task { head_spec, codec, criterion,<br/>activation, metrics, weight }
    end
    Wiring-->>main: list[Task]

    main->>CM: build_composite_model(backbone, {name: HeadSpec})
    Note over CM: TimmBackbone + nn.ModuleDict(head per task)<br/>head sized from backbone.feature_dim(feature_key)
    CM-->>main: CompositeModel

    main->>main: LitModule(model, tasks, optimizer_builder)<br/>L.Trainer(…).fit(lit_module, lit_datamodule)
```

---

## Phase 2 — Training step (repeats every batch)

```mermaid
sequenceDiagram
    participant Trainer as L.Trainer
    participant LitM as LitModule
    participant DL as DataLoader
    participant DS as Dataset.__getitem__
    participant CM as CompositeModel
    participant BB as TimmBackbone
    participant Head as LinearHead / ConvHead<br/>(per task)
    participant TC as TaskCodec<br/>(per task)
    participant Crit as Criterion<br/>(per task)
    participant Act as Activation<br/>(per task)
    participant Met as MetricSet<br/>(per task × stage)
    participant Agg as WeightedSumAggregator

    Trainer->>LitM: training_step(batch, batch_idx)

    LitM->>DL: next(dataloader)
    DL->>DS: __getitem__(index)
    loop for each InputBinding
        DS->>DS: InputLoader.load(value) → ndarray / str
    end
    DS->>DS: Transform.apply(sample) → tensors [C,H,W]
    loop for each TargetBinding
        DS->>DS: codec.load(raw) then codec.to_tensor(val) → Tensor
    end
    DS-->>DL: Sample { inputs: {alias: Tensor}, targets: {task: Tensor} }
    DL->>DL: collate_samples(list[Sample])
    DL-->>LitM: Batch { inputs: {…: [B,C,H,W]},<br/>targets: {task_name: Tensor[B,…]} }

    LitM->>CM: forward(batch.inputs)
    CM->>BB: forward({"image": Tensor[B,C,H,W]})
    BB-->>CM: FeatureBundle { "pooled": [B,D] / "decoder": [B,D,H,W] }

    loop for each task head
        CM->>Head: forward(features[task.feature_key])
        Head-->>CM: logits Tensor[B, out_features]
    end
    CM-->>LitM: ModelOutput { task_logits: {name: Tensor} }

    loop for each Task
        LitM->>TC: adapt(batch.targets[task.name])
        Note over TC: MulticlassTaskCodec → [B] long<br/>BinaryTaskCodec    → [B,1] float / [B,1] long<br/>MultilabelTaskCodec→ [B,C] float / [B,C] long<br/>ContinuousTaskCodec→ [B,1] float
        TC-->>LitM: TargetView { loss: Tensor, metric: Tensor }

        LitM->>Crit: forward(logits, target.loss)
        Note over Crit: CrossEntropyCriterion / BCEWithLogitsCriterion<br/>MSECriterion / DiceCriterion / CompositeCriterion
        Crit-->>LitM: LossResult { total, components: {name: Tensor} }

        LitM->>Act: __call__(logits)
        Note over Act: SoftmaxActivation / SigmoidActivation / IdentityActivation
        Act-->>LitM: preds Tensor

        LitM->>Met: update(preds, target.metric)
        Note over Met: TorchMetricsAdapter wraps MetricCollection<br/>accumulates state across batches
    end

    LitM->>Agg: combine(losses, weights)
    Note over Agg: total = Σ weight_i × task_i.total<br/>components namespaced as "task/component"
    Agg-->>LitM: LossResult { total (scalar), components }

    LitM->>LitM: self.log("loss/train/total", …)
    LitM-->>Trainer: combined.total  →  .backward()
```

---

## Epoch end — metrics flush

```mermaid
sequenceDiagram
    participant Trainer as L.Trainer
    participant LitM as LitModule
    participant Met as MetricSet<br/>(per task × stage)
    participant Log as Lightning Logger

    Trainer->>LitM: on_train_epoch_end() / on_validation_epoch_end()

    loop for each Task
        LitM->>Met: compute()
        Met-->>LitM: { "accuracy": 0.83, "macro_f1": 0.79, … }
        LitM->>Log: self.log("{task}/{metric}/{stage}", value)
        LitM->>Met: reset()
    end
```

---

## Key names

| Concept | Class / function |
|---|---|
| Validated config root | `ExperimentConfig` |
| Data annotation source | `CsvDataSource` / `JsonDataSource` |
| Data mode: single file + split | `data.sources: str/list` + `data.split` |
| Data mode: pre-split files | `data.sources: {train: ..., val: ...}` |
| Stratified split | `data.split_stratify: column` (auto-detects categorical/numeric/multilabel) |
| Dataset size cap | `data.max_samples: int / float` |
| Input column(s) config | `data.inputs: str / dict[alias, column] / dict[alias, {column, loader}]` |
| Input column → loader binding | `InputBinding(name, column, loader)` |
| Input loader port | `InputLoader` (ABC) — `ImageLoader` / `TextLoader` (registered in `input_loaders`) |
| Data-layer target decoder | `LabelIndexCodec` / `MultiLabelBinarizeCodec` / `FloatCodec` / `MaskCodec` |
| Target column → codec binding | `TargetBinding(name, column, codec)` |
| Plain data orchestrator | `DataModule` |
| Data wiring helper | `build_data_module(config, target_bindings, runtime)` |
| Lightning data wrapper | `LitDataModule` |
| Per-item assembly | `Dataset.__getitem__` |
| Input transform (YAML-driven) | `AlbumentationsTransform` — built from `configs/transforms/*.yaml` |
| Augmentation pipeline config | `_target_`-keyed YAML; `instantiate(spec)` builds it recursively |
| Batching | `collate_samples` |
| Phase-agnostic batch | `Batch` |
| Backbone output | `FeatureBundle` (named streams: `"pooled"`, `"decoder"`, …) |
| Head build instruction | `HeadSpec(kind, out_features, feature_key)` |
| Backbone + heads | `CompositeModel` |
| Image backbone | `TimmBackbone` |
| Dense (seg) backbone | `SmpBackbone` |
| Heads | `LinearHead` / `ConvHead` |
| Forward output | `ModelOutput` |
| Task bundle | `Task(name, head_spec, codec, criterion, activation, metrics, weight)` |
| Output structure axis | `TopologyStrategy` → `GlobalTopology` / `DenseTopology` |
| Label semantics axis | `ObjectiveStrategy` → `Multiclass` / `Binary` / `Multilabel` / `Continuous` |
| Task assembler | `TaskBuilder` |
| User-facing preset | `classification(…)` / `segmentation(…)` / `regression(…)` |
| Task-layer target shaping | `MulticlassTaskCodec` / `BinaryTaskCodec` / `MultilabelTaskCodec` / `ContinuousTaskCodec` |
| Adapted target | `TargetView(loss, metric)` |
| Loss brick | `CrossEntropyCriterion` / `BCEWithLogitsCriterion` / `MSECriterion` / `DiceCriterion` / `CompositeCriterion` |
| Loss result | `LossResult(total, components)` |
| Post-logit activation | `SoftmaxActivation` / `SigmoidActivation` / `IdentityActivation` |
| Metric collection | `TorchMetricsAdapter` wrapping `torchmetrics.MetricCollection` |
| Loss combiner | `WeightedSumAggregator` |
| Per-head LR | `OptimizerBuilder` |
| Training orchestrator | `LitModule` |
| Runtime inference | `RuntimeContext.num_classes` — populated by `DataModule.setup()` |
| Extension point | `Registry` — `@registry.register("key")` |
| YAML brick spec | `instantiate(spec, registry?)` — recursive; `"key"` / `{name, params}` / `{_target_}` / nested lists |
