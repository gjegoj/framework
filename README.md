# Framework

Configuration-driven multi-task computer-vision training on top of PyTorch Lightning, Hydra, Pydantic, timm, and torchmetrics.

---

## Phase 1 — Setup (runs once before `trainer.fit`)

```mermaid
sequenceDiagram
    participant main
    participant Config as ExperimentConfig<br/>(Pydantic)
    participant Wiring as wiring.py
    participant DM as DataModule
    participant DS as DataSource<br/>(CsvDataSource)
    participant Codec as TargetCodec<br/>(per task)
    participant RT as RuntimeContext
    participant TB as TaskBuilder<br/>(TopologyStrategy × ObjectiveStrategy)
    participant CM as CompositeModel<br/>(TimmBackbone + heads)

    main->>Config: load_config(raw_dict)<br/>Hydra DictConfig → validated DTOs
    Config-->>main: ExperimentConfig

    main->>Wiring: build_bindings(config)
    Note over Wiring: infers codec per task<br/>from preset_meta + objective
    Wiring-->>main: list[TargetBinding(name, column, codec)]

    main->>Wiring: build_transforms(config)
    Wiring-->>main: dict[Stage, BasicTransform]

    main->>DM: DataModule(source, bindings, transforms, runtime)
    main->>DM: setup()

    DM->>DS: read()
    DS-->>DM: DataFrame (full annotation table)

    loop for each TargetBinding
        DM->>Codec: fit(column_values)
        Note over Codec: LabelIndexCodec → sorts vocab<br/>MultiLabelBinarizeCodec → multi-hot vocab<br/>FloatCodec → no-op
        Codec-->>RT: num_classes[task_name] = C
    end

    DM->>DM: split_dataframe(ratios, seed)
    Note over DM: → Dataset per stage<br/>(frame + bindings + transform + loader)
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
    Note over CM: TimmBackbone + nn.ModuleDict(LinearHead per task)<br/>head sized from backbone.feature_dim("pooled")
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
    participant Head as LinearHead<br/>(per task)
    participant TC as TaskCodec<br/>(per task)
    participant Crit as Criterion<br/>(per task)
    participant Act as Activation<br/>(per task)
    participant Met as MetricSet<br/>(per task × stage)
    participant Agg as WeightedSumAggregator

    Trainer->>LitM: training_step(batch, batch_idx)

    LitM->>DL: next(dataloader)
    DL->>DS: __getitem__(index)
    DS->>DS: ImageLoader.load(path) → ndarray
    DS->>DS: Transform.apply(sample) → Tensor [C,H,W]
    DS->>DS: TargetCodec.encode(raw_value) → Tensor
    DS-->>DL: Sample { inputs: {image}, targets: {task: Tensor} }
    DL->>DL: collate_samples(list[Sample])
    DL-->>LitM: Batch { inputs: {image: [B,C,H,W]},<br/>targets: {task_name: Tensor[B,…]} }

    LitM->>CM: forward(batch.inputs)
    CM->>BB: forward({"image": Tensor[B,C,H,W]})
    BB-->>CM: FeatureBundle { "pooled": Tensor[B,D] }

    loop for each task head
        CM->>Head: forward(features["pooled"])
        Head-->>CM: logits Tensor[B, out_features]
    end
    CM-->>LitM: ModelOutput { task_logits: {name: Tensor} }

    loop for each Task
        LitM->>TC: adapt(batch.targets[task.name])
        Note over TC: MulticlassTaskCodec → [B] long<br/>BinaryTaskCodec    → [B,1] float / [B,1] long<br/>MultilabelTaskCodec→ [B,C] float / [B,C] long<br/>ContinuousTaskCodec→ [B,1] float
        TC-->>LitM: TargetView { loss: Tensor, metric: Tensor }

        LitM->>Crit: forward(logits, target.loss)
        Note over Crit: CrossEntropyCriterion / BCEWithLogitsCriterion<br/>MSECriterion / L1Criterion
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
| Data annotation source | `CsvDataSource` |
| Data-layer target decoder | `LabelIndexCodec` / `MultiLabelBinarizeCodec` / `FloatCodec` |
| Task → column → codec binding | `TargetBinding` |
| Plain data orchestrator | `DataModule` |
| Lightning data wrapper | `LitDataModule` |
| Per-item assembly | `Dataset.__getitem__` |
| Input transform | `BasicTransform` / `AlbumentationsTransform` |
| Batching | `collate_samples` |
| Phase-agnostic batch | `Batch` |
| Backbone output | `FeatureBundle` (named streams: `"pooled"`, `"decoder"`, …) |
| Head build instruction | `HeadSpec(kind, out_features, feature_key)` |
| Backbone + heads | `CompositeModel` |
| Feature backbone | `TimmBackbone` |
| Head | `LinearHead` |
| Forward output | `ModelOutput` |
| Task bundle | `Task(name, head_spec, codec, criterion, activation, metrics, weight)` |
| Output structure axis | `TopologyStrategy` → `GlobalTopology` |
| Label semantics axis | `ObjectiveStrategy` → `Multiclass` / `Binary` / `Multilabel` / `Continuous` |
| Task assembler | `TaskBuilder` |
| User-facing preset | `classification(…)` / `regression(…)` |
| Task-layer target shaping | `MulticlassTaskCodec` / `BinaryTaskCodec` / `MultilabelTaskCodec` / `ContinuousTaskCodec` |
| Adapted target | `TargetView(loss, metric)` |
| Loss brick | `CrossEntropyCriterion` / `BCEWithLogitsCriterion` / `MSECriterion` / `L1Criterion` |
| Loss result | `LossResult(total, components)` |
| Post-logit activation | `SoftmaxActivation` / `SigmoidActivation` / `IdentityActivation` |
| Metric collection | `TorchMetricsAdapter` wrapping `torchmetrics.MetricCollection` |
| Loss combiner | `WeightedSumAggregator` |
| Per-head LR | `OptimizerBuilder` |
| Training orchestrator | `LitModule` |
| Runtime inference | `RuntimeContext.num_classes` — populated by `DataModule.setup()` |
| Extension point | `Registry` — `@registry.register("key")` |
| YAML brick spec | `instantiate(spec, registry)` — `"key"` / `{name, params}` / `{_target_}` |
