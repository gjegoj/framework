# Vocabulary

Every name the framework speaks in, one line each. A guide teaches a topic; this
page is looked up. Names are grouped by the package that owns them, and the
groups follow the dependency rule: `core/` first, then the capabilities, then
the composition root.

## Core (`src.core`)

Entities, ports and taxonomy — torch and stdlib only.

| Name | Kind | Meaning |
|---|---|---|
| `Sample`, `Batch` | entity | One example before collation / a collated batch (`inputs`, `targets`, `meta`) |
| `Features` | entity | Named feature streams produced by a backbone |
| `Prediction` | entity | Per-task outputs in the family's shape, optionally with the features and the pre-activation logits they came from |
| `Loss` | entity | Loss value with named parts; `+`, `*`, `.scoped()`, `Loss.sum()` — covers single, weighted and multi-task totals, so there is no separate aggregator |
| `StepResult` | entity | What one model step yields: `loss` + `prediction` + metric-view `targets` |
| `StepPreview` | entity | A step's outputs and targets, detached — enough to draw a page, nothing that holds a graph. Built only when a `StepPreviewConsumer` asked for this batch |
| `Instances` | entity | The objects a batch holds or predicted, concatenated across it: `boxes` (xyxy pixels), `labels`, `sample_index`, and `scores` — `None` for ground truth, which has no confidence. Flat because that is the only shape a ragged quantity has that a tensor can carry |
| `PerClass` | entity | Values a metric produced per class, *carrying which classes they are about*. Position is not the class index: COCO's `map_per_class` covers only the classes that appeared. A dense reading is the case where `classes` is `arrange(n)`, so there is one per-class shape rather than two |
| `TaskOutput` | alias | `Tensor \| Instances` — what one task's prediction or target is. A union, never `Any`: a third shape is one edit and a type error at every consumer that has not considered it |
| `as_tensor` | function | A task's output where the reader serves only tensors (a composed model, a batch transform, an exported graph), refused by name if it is not one |
| `AdaptedTarget` | entity | One raw target shaped into `for_loss` / `for_metrics` views |
| `Task` | entity | One learned objective in family-agnostic terms: `topology` × `objective`, weight, per-task `lr`, metrics. Family-specific bricks live with the family |
| `DataProfile` | entity | Facts inferred from data, written at setup and read at assembly |
| `TargetFacts` | entity | One task's frozen slice of those facts (`num_classes`, `class_names`, `class_values`); every objective brick is built from it, so the *representation* of a target stays the encoder's choice instead of becoming a second axis |
| `ClassDistribution`, `ValueDistribution` | entities | What one target column holds: counted per class, or measured as a five-number summary |
| `DatasetStatistics` | entity | Per-stage row counts plus one distribution per task — what a run prints before its first epoch |
| `Matrix`, `Curve`, `Bars`, `Spread` | entities | Drawable artifacts, each carrying its own axis and series names. A backend draws what it is handed and never asks what the numbers mean |
| `Registry` | mechanism | Keyed component factories: `@registry.register("name")`, `register_instance`, `create(key, **kwargs)`; duplicate protection, missing keys listed in the error |
| `named_by` | function | The offered values a callee names in its signature, and only those — how a derived fact reaches a component without config restating it |
| `one_of` | function | Refuses a value outside a `Literal` alias, naming the alias and listing the options |
| `log_keys` | module | The one owner of the log-key grammar `{stage}/{task}/{leaf}`: tokens (`TOTAL_LOSS`), composition (`join`, `total_loss`) and parsing (`split_for_tracker`) |
| `Topology` | taxonomy | Output structure: `global`, `dense`, `multiview`, `multistream`, `instances` |
| `Objective` | taxonomy | Label semantics: `multiclass`, `binary`, `multilabel`, `continuous`, `metric` |
| `Modality` | taxonomy | Standard input names (open vocabulary): `image`, `embedding`, `text` — the third task axis |
| `Stream` | taxonomy | Standard stream names, each naming a shape class: `features` `[B,D]`, `encoder` `[B,D,H',W']`, `decoder` `[B,D,H,W]`, `logits` (task-shaped), `embeddings` `[B,N,D]` |
| `Stage` | taxonomy | `train` / `val` / `test` |
| `Model` | port | The unit training consumes: `step(batch) → StepResult` (one forward: loss + predictions), `predict(batch)` (no targets). Families: composite, vendor-native, decorators |
| `Backbone` | port | Named inputs → `Features`; `feature_dim(stream)` sizes heads — construction vocabulary of the composite family |
| `Head` | port | One feature stream → raw logits |
| `Criterion` | port | (logits, target) → `Loss`; operates on logits, never activated outputs |
| `DataModule` | port | The data side of an experiment: `setup(profile)` + `dataset(stage)`, plus optional `statistics()` and `collate` — batching belongs to the data, since ragged detection targets cannot be stacked the framework's way |
| `MetricSet` | port | Stateful metrics for one task and stage: update / compute / reset / directions |
| `MetricFamily` | port | A metric whose computed value is several named readings rather than one number; it declares `readings` so a checkpoint monitor learns which keys exist before anything is computed |
| `Activation` | port (alias) | `Callable[[Tensor], Tensor]` — logits → predictions for metrics and inference |
| `TargetAdapter` | port (alias) | `Callable[[Tensor], AdaptedTarget]` |
| `SampleTransform` | port (alias) | `Callable[[Sample], Sample]` — the augmentation seam. Takes a whole sample because geometric augmentation is joint: one crop for the image and its masks alike |
| `BatchTransform` | port (alias) | `Callable[[Batch], Batch]` — the seam for augmentations that mix *different* samples, which a `SampleTransform` cannot reach |
| `MatrixLogger`, `CurveLogger`, `BarsLogger`, `SpreadLogger`, `HtmlLogger`, `SingleValueLogger` | ports | One structural protocol per kind of artifact a tracker may draw. A backend without one keeps its scalars and is told once |
| `StepPreviewConsumer` | port | Something that reads a step's preview and says beforehand whether it wants this batch — so a run with no such consumer builds none |

## Data (`src.data`)

| Name | Kind | Meaning |
|---|---|---|
| `Table` | alias | `pandas.DataFrame` — the annotation-table currency (paths, labels) |
| `TableSource` | port | Reads the annotation table. Built-ins via `FileSource` (kwargs forward to the pandas reader): `CsvSource` (`csv`), `JsonSource` (`json`) in the `table_source_registry`; `InMemorySource` for notebooks and tests; `LimitedSource` wraps any of them to cap the rows read |
| `DataSchema` | entity | Maps table columns to named inputs (`InputColumn`) and per-task targets (`TargetColumn`) |
| `InputLoader` | port (alias) | `Callable[[Any], Any]` — one cell to a model input. Built-in: `ImageLoader` (key `image` in the `input_loader_registry`, the default an input gets when config names none) reads files into HWC RGB uint8 arrays and holds its own `root` |
| `TargetEncoder` | port | sklearn-style `fit` / `encode` into **raw** values (index, float, mask array), so a transform can still touch them; after `fit` exposes facts and a `distribution()`; `spatial` marks targets that follow the image's geometry. Built-ins: `LabelTargetEncoder`, `MultiLabelTargetEncoder`, `ScalarTargetEncoder`, `MaskTargetEncoder`, `GaussianBinsTargetEncoder`, `LinearBinsTargetEncoder` |
| `Splitter` | port (alias) | `Table → {Stage: Table}`; built-ins `random_split(fractions, seed)`, `stratified_split(fractions, by, seed, bins)`, `group_split(fractions, by, seed)` |
| `LoaderCache` | port | Keeps what a loader returned, keyed by the cell value. Built-in: `RamCache` (`ram` in the `cache_registry`) holds decoded arrays up to a byte budget; `cached(loader, cache)` is the one write path |
| `TableDataset` | adapter | torch `Dataset`: one table row → `Sample`; validates the schema up front |
| `collate_samples` | function | `list[Sample] → Batch` |
| `TableDataModule` | adapter | Table-driven `DataModule`: sources → split → fit encoders on train → `DataProfile` → per-stage datasets. Takes one source plus a `Splitter`, several to combine, or one set per stage when the rows arrive already divided |
| `YoloDataModule` | adapter | Vendor-native `DataModule`: ultralytics datasets from a YOLO `data.yaml`, class facts into the profile, its own batching. Detection targets are ragged, so it reports a `collate` of its own — translated into a `Batch` so nothing downstream learns a second shape |
| `SourceWithTransforms` | entity | A source and the transforms its own rows take. A stage becomes a `ConcatDataset` of one dataset per source, each with its own pipeline |
| `counted`, `measured` | functions | The two shapes a target column is described in, for the report drawn before epoch one |

One raw-to-tensor boundary: loaders and encoders produce raw values (arrays,
indices, floats), and tensors are made exactly once afterwards — by the transform
ending in `ToTensorV2`, or by `collate_samples`. That is what lets a mask be
augmented together with its image instead of arriving as a tensor too early.

`setup(profile)` is the hinge of assembly: encoders fit on the train split only,
their facts land in `DataProfile`, and only then are tasks and heads built with
concrete sizes.

See [the data guide](guides/data.md) for the YAML that says each of these things.

## Models (`src.models`)

| Name | Kind | Meaning |
|---|---|---|
| `CompositeModel` | adapter | The constructor-set `Model`: encode once with a shared `Backbone`, serve every task from named streams; heads and criteria are registered submodules |
| `TaskComponents` | entity | How the composite family serves one task: head, criterion, activation, target adapter, stream, weight |
| `YoloModel` | adapter | The first *vendor* `Model`: an ultralytics network held as a submodule, its `box`/`cls`/`dfl` arriving as `Loss` parts scoped by the task, and what survives NMS decoded into `Instances`. `YOLO(model_name)` picks the network class from the name, so one path serves detection, segmentation and pose; the head is rebuilt at the dataset's class count, and `.pt` weights are grafted onto it |
| `DistilledModel` | adapter | Decorator: the student's step plus a soft term against frozen teachers' averaged logits. Off training the teachers do not run |
| `LoraAdapters` | entity | Which projections get a low-rank delta, at what rank; `merge_adapters` folds the delta back before anything reads the weights |
| `model_registry` | registry | Whole model families, separate from `backbone_registry` because the two answer different questions — a backbone is a piece the framework composes with, a model is a family it delegates to. A name found here is what tells assembly which of the two it is looking at |
| `TimmBackbone` | adapter | Any timm model as a pooled-feature backbone (key `timm`); `model_name`, `pretrained`, extras forward to `timm.create_model`; native head: timm's classifier. `checkpoint_path` loads arrived weights with timm's own knobs, stashes the checkpoint's classifier, and the native head transplants it — growing the class space yields `ExpandedHead(base, novel)` with `base` freezable at a module boundary |
| `SmpBackbone` | adapter | smp encoder+decoder (key `smp`): `Stream.ENCODER` + `Stream.DECODER`; `arch`, `encoder_name`; native heads: smp's segmentation head (decoder) and pooling classifier (encoder). DPT with prefix-token ViT encoders (DINO-style) gets the final-LayerNorm patch automatically, logged at INFO |
| `MultiEncoderBackbone` | adapter | One encoder per input (key `multi`): per-encoder projection to a shared space, L2-norm, stacked `[B, N, D]` under `Stream.EMBEDDINGS` — the CLIP/SigLIP construction set |
| `MultiViewBackbone` | adapter | Decorator (key `multiview`): N views of each sample through one shared inner encoder, stacked `[B, N, D]`; optional SimCLR-style projection — the Siamese/triplet construction set |
| `HFTextBackbone` | adapter | transformers text encoder (key `hf_text`): CLS or mask-aware mean pooling to `Stream.FEATURES` |
| `LinearHead`, `ConvHead` | adapters | The default GLOBAL and DENSE projections |
| `CosineHead` | adapter | Learnable class prototypes, cosine logits — the angular-margin classifier |
| `IdentityHead` | adapter | Pass-through, for backbones that already emit task outputs |
| `WrappedHead` | adapter | Any torch module as a `Head` — how backbone-native heads enter the framework |
| `ExpandedHead` | adapter | A carried head beside a fresh one for novel classes, concatenated on the class axis |

Native heads are an explicit choice: a task declaring `native_head: true` asks the
backbone for its own (`Backbone.native_head`) and fails loudly when it has none.

## Tasks and losses (`src.tasks`, `src.losses`)

| Name | Kind | Meaning |
|---|---|---|
| `TaskObjective` | port | Behaviour of one `Objective` member: `out_features`, criterion, activation, target adapter (`None` = structure-supervised) — all built from `TargetFacts`. Built-ins: `MulticlassObjective`, `BinaryObjective`, `MultilabelObjective`, `ContinuousObjective`, `MetricObjective` |
| `TaskTopology` | port | Behaviour of one `Topology` member: head kind, stream, `supports(objective)` pairing check. Built-ins: `GlobalTopology` (linear head, `Stream.FEATURES`), `DenseTopology` (conv head, `Stream.DECODER`), `MultiStreamTopology` and `MultiViewTopology` (identity head over `Stream.EMBEDDINGS`, metric-only) |
| `build_task_components` | function | `Task` + `DataProfile` + `Backbone` → `TaskComponents` — the assembly point of the axes model |
| `resolve_preset` | function | Familiar names as kinds of task (`TaskPreset`), living on the config surface in `config.presets` |
| `expectation_over`, `expectation_of` | functions | The inverse of a binned encoding, used on both sides: a prediction and a target both become the number their distribution stands for, so a binned regression is judged by ordinary regression metrics |
| `WrappedCriterion` | base | Wrap a module, subclass a composer: one tensor-in → tensor-out module logged under `part_name` |
| `CrossEntropyCriterion`, `BinaryCrossEntropyCriterion`, `FocalCriterion`, `MeanSquaredErrorCriterion`, `MeanAbsoluteErrorCriterion`, `HuberCriterion`, `SmoothL1Criterion`, `DiceCriterion`, `IoUCriterion`, `TverskyCriterion` | adapters | Logged as parts `ce` / `bce` / `focal` / `mse` / `mae` / `huber` / `smooth_l1` / `dice` / `iou` / `tversky`; single-channel outputs are squeezed on the channel dim, so every criterion serves GLOBAL and DENSE alike |
| `ExpectationCriterion` | adapter | Regression on `softmax(logits) · class_values` (part `expectation`): the companion of cross-entropy for a binned target |
| `WeightedSumCriterion` | adapter | Several criteria on one output, added with weights, each keeping its own logged name; declared as a list under a task's `loss` |
| `ArcFaceCriterion`, `ProxyAngularCriterion` | adapters | Angular margin over cosine logits (part `arcface`), and the same with the class prototypes held inside the criterion so they never reach the exported model |
| `InfoNceCriterion`, `SigLipCriterion`, `TripletCriterion` | adapters | Over stacked view carriers: pairs `[B, 2, D]` and triplets `[B, 3, D]`, supervised by the in-batch diagonal |
| `MarginRankingCriterion`, `RankNetCriterion` | adapters | Ranking pairs judged by a per-pair number — the hinge (±1) and the logistic form (a probability, ties as 0.5) |
| `KLDivergenceCriterion` | adapter | Distils a student against teacher logits (part `kl`), `T²`-scaled and annealable |

## Transforms (`src.transforms`)

| Name | Kind | Meaning |
|---|---|---|
| `AlbumentationsTransform` | adapter | One joint pipeline over a sample's images and spatial targets — every sampled parameter is shared. Builds its own `Compose`, so registering mask keys never mutates a pipeline reused elsewhere |
| `MultiViewTransform` | adapter | One sample input becomes N independently augmented views — wrap any transform to get per-view sampling |
| `MixUp`, `CutMix` | adapters | Combine two samples and rewrite a global label; `Mosaic` stitches four without resizing, so a segmentation mask composes by the same swap |
| `Rotate90`, `RandomBorderCrop` | adapters | Augmentations that *create* supervision: the same draw that moves the picture rewrites a bound label |

Batch transforms are applied by the `batch_transform` callback, which owns the
schedule and writes the result back — Lightning's hook cannot replace a batch.

## Metrics (`src.metrics`)

| Name | Kind | Meaning |
|---|---|---|
| `WrappedMetricSet` | adapter | Named torchmetrics behind the `MetricSet` port, backed by `MetricCollection` (shared-state metrics update once per group); directions from each metric's `higher_is_better` |
| `MeanAveragePrecisionOverInstances` | adapter | `map`: mean average precision over `Instances`, publishing the readings it was asked for (`map`, `map_50`, `map_75` by default) from one pass — torchmetrics computes all fifteen together. `class_metrics` is derived from the request, `-1` readings are dropped as the "not applicable" sentinel they are, and `faster_coco_eval` is named because the default backend raises where it is not installed |
| `metric_registry` | registry | `accuracy`, `f1`, `precision`, `recall`, `iou`, `mae`, `mse`, `confusion_matrix`, `precision_recall_curve`, `roc`, `map` — torchmetrics under DS names |
| `presentation_of`, `present` | functions | What a computed value *means*, keyed by metric class and looked up along the MRO; scalars and vectors route by geometry, matrices and curves are drawn only as identified artifacts |

Objectives own their metric defaults (`MulticlassObjective` — accuracy,
`ContinuousObjective` — mae, ...): the same place that owns criteria and
activations for that label semantics.

## Training (`src.training`)

| Name | Kind | Meaning |
|---|---|---|
| `TrainingModule` | adapter | The single Lightning module for every `Model` family; registers task metrics for device movement; logs `{stage}/loss`, `{stage}/{task}/{part}`, `{stage}/{task}/{metric}` |
| `TrainingData` | adapter | A set-up `DataModule` as Lightning sees it: per-stage DataLoaders (train shuffled, evaluation ordered) |
| `LoaderSettings` | entity | Batch size, workers, pin_memory, and a swappable `collate` for families with special batching |
| `OptimizerFactory` | port (alias) | `Callable[[groups], Optimizer]`, e.g. `partial(torch.optim.AdamW, lr=1e-3)`. Always named groups — `backbone` plus one per task — because the groups are what `lr_monitor` draws a line each for; a group naming no rate inherits the factory's, and measured, the split changes no update |
| `SchedulerFactory` | port (alias) | `Callable[[Optimizer, FitProfile], LRSchedulerConfig]` — built late, because a scheduler needs the optimizer and the fit-time facts |
| `FitProfile` | entity | Facts that exist only once fitting starts (`total_steps`, `steps_per_epoch`, `epochs`) — the fit-loop counterpart of `DataProfile` |
| `fit_time_kwargs` | function | Fills the canonical fit-time params a scheduler declares and config left unset (`total_steps` wins as the most precise) |
| `per_group_rates` | function | Spreads a schedule's *absolute* rate (`max_lr`, `base_lr`) over the parameter groups, scaled by each group's own — without it a scalar broadcasts and a task's declared rate is silently overwritten |
| `optimizer_registry`, `scheduler_registry`, `profiler_registry` | registries | `adamw` / `adam` / `sgd`; `cosine` / `onecycle` / `plateau` / `step`; `simple` / `advanced` / `pytorch` |

Lightning imports live only inside this package. A training step is the same for
every model family:

```python
result = model.step(batch)     # one forward: loss + predictions
result.loss.total.backward()
```

Inside, `CompositeModel.step` runs `Backbone → per-task Head → Criterion` with
`Loss.sum(weight * task_loss.scoped(name) ...)`; a vendor adapter maps its native
losses into `Loss.parts` instead.

## Callbacks, loggers and export

| Name | Kind | Meaning |
|---|---|---|
| `callback_registry` | registry | `lr_monitor` and `checkpoint` (Lightning's own, registered rather than wrapped), plus `ema`, `freeze`, `anneal`, `batch_transform`, `metric_summary`, `dataset_summary`, `model_summary`, `progress`, `samples` |
| `EmaWeights` | adapter | An exponential moving average of the weights, which validation reports and a checkpoint stores; `EmaModelCheckpoint` writes the averaged copy alone |
| `Freeze` | adapter | Holds named modules still until a share of the run has passed |
| `AnnealCriterion` | adapter | Moves one number of a criterion over the run — focal `gamma`, label smoothing |
| `DatasetSummary`, `MetricSummary`, `TreeModelSummary`, `MetricsProgressBar` | adapters | What a run says about itself: the data before epoch one, the headline numbers after test, the model as a tree, and a live metrics table under the bar |
| `SampleGrid` | adapter | Every N epochs, one batch becomes a self-contained HTML page in the tracker |
| `logger_registry` | registry | `clearml` — experiment trackers behind Lightning's `Logger` |
| `DeployableModel` | adapter | One graph shape for every model family: positional tensors in, task logits out |
| `Exporter` | port | One deployment format. Built-ins in the `exporter_registry`: `torchscript`, `onnx`, `tensorrt` |
| `verify`, `Parity` | function, entity | A written artifact run beside the model it came from, and how far apart they are. An artifact is not an export until it has been read back |
| `Annotator`, `SampleView` | adapter, entity | A task's step tensors become labels and a verdict; one sample projected for display |
| `HtmlRenderer` | adapter | `SampleView`s → one self-contained page: grid, sidebar, lightbox, filters |

## Config and assembly (`src.config`, `src.assembly`)

| Name | Kind | Meaning |
|---|---|---|
| `load_config` | function | The boundary: a raw mapping in, a validated `ExperimentConfig` out — validation happens exactly once |
| `ExperimentConfig` | contract | The single source of truth: `seed`, `data`, `tasks`, `model`, `optimizer`, `scheduler`, `loader`, `trainer`, `callbacks`, `logger`, `transforms`, `adapters`, `distillation`, `export`, `run` |
| `ComponentConfig` | grammar | One way to name any component: `"cross_entropy"` / `{name: ..., knobs}` / `{_target_: ..., knobs}`; every other key becomes a constructor argument. Hydra's other meta-keys are rejected, not ignored |
| `ModelConfig`, `MetricConfig`, `TransformConfig`, `OptimizerConfig` | aliases | The same grammar, named at its points of use |
| `SchedulerConfig`, `LossConfig` | sections | The grammar plus framework-owned fields — a schedule's `interval` / `monitor`, a loss term's `weight` |
| `TaskConfig` | section | `preset` or explicit axes (resolved at load); declares the `target` column once — the data schema derives from tasks. Per-task `weight` scales its loss; per-task `lr` sets the pace of its head and criterion while the backbone keeps the shared rate. `metrics` entries are keyed by the label they log under. `classes` (`{0: cat, 1: dog}`) declares the vocabulary as the source of truth: data is validated against it at fit, the index space survives resampling, and the names label per-class logs and matrices |
| `DataConfig`, `SplitConfig`, `CacheConfig` | sections | Source paths, input columns, per-stage fractions (validated to sum to 1), the RAM budget |
| `LoaderConfig`, `TrainerConfig` | sections | Forward sections: declared fields validated, unknown knobs pass to `DataLoader` / `lightning.Trainer` verbatim. `trainer.profiler` is the one exception, declared because Lightning takes an object there |
| `RunConfig` | section | Which stages run, from which checkpoint, into which directory |
| `TaskPreset`, `task_preset_registry` | entity, registry | Familiar kinds of task on the config surface; nobody writes a preset, only names one |
| `instantiate` / `resolve_target` | functions | The only home of the component grammar. Both forms take one path — resolve the constructor, resolve the params, call — and derived values win over config on conflict. Import paths resolve through `hydra.utils.get_object` |
| `is_vendor_family` / `refuse_what_a_vendor_cannot_serve` | functions | One reading of `config.model.name` decides the model *and* the data pipeline, so the halves cannot disagree; and what such a family cannot serve — `transforms`, `export`, `adapters`, `distillation`, a `batch_transform`, a second task, our bricks on its task — is refused at assembly with the sentence that explains it |
| `build_task_entities` | function | The half of task building a vendor family needs: the `Task` entities, with no bricks behind them. `build_tasks` is that plus the composite family's head, criterion, activation and adapter |
| `shipped_weights` / `load_weights` | functions | A checkpoint carries the weights of the model that ships; one file loads into a distilled model and a plain one alike |
| `assemble` / `run` | functions | `ExperimentConfig` → `Experiment` → fit / test / export. `assemble` names no model family: `build_data_module` returns the `DataModule` port and `build_model(config, profile)` returns `(Model, list[Task])` |
| `Experiment` | entity | `module` + `data` + `trainer` + `tasks`, assembled and ready |

Structural sections are `extra="forbid"` — a typo in a key is a load error naming
it; component and forward sections are `extra="allow"`, so every upstream knob is
reachable from config without a schema change. Pydantic lives only inside
`config/`.

The body of `assemble()` is the build-order contract: `setup(profile)` runs before
the model, which is the only reason head sizes come from data instead of config —
and it holds for every family, so it stays visible there.
