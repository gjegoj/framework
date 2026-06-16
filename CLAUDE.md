# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A configuration-driven framework for training computer-vision models (multi-task,
multi-modal) on top of PyTorch Lightning, Hydra, Pydantic, timm/smp, albumentations
and torchmetrics. It is a clean-architecture rewrite of the prototype in `old/`
(kept only as reference — do not edit or import from it). The full design and the
milestone breakdown live in the approved plan at
`~/.claude/plans/clean-code-refactoring-patterns-softwar-merry-sunbeam.md`.

Status: M1–M7 substantially complete. Tasks: classification + every objective
(multiclass/binary/multilabel/continuous), DENSE segmentation, and **metric learning** —
RANKING (Siamese: N views through one shared backbone) and MULTISTREAM (dual/multi-encoder,
CLIP/SigLIP-style) topologies with triplet / margin-ranking / InfoNCE / SigLIP / ArcFace
losses. Backbones: `TimmBackbone`, multi-stream `SmpBackbone` (`ENCODER_LAST`/`DECODER`,
per-task `feature_key`), precomputed-`EmbeddingBackbone`, and multi-encoder `MultiEncoderBackbone`.
Training: per-head LR via param-groups, LR **schedulers** (`training/scheduler.py`), typed metric
handlers (scalar/vector/matrix/curve), ClearML logger, and M5 callbacks — EMA (thin subclass of
Lightning's `EMAWeightAveraging`), freeze, checkpoint, `MetricsProgressBar`, `SampleLogCallback`,
and batch transforms (MixUp/CutMix/Mosaic). Cross-cutting subsystems: model **export**
(`export/`: ONNX/TorchScript + numerical-parity verification), sample **visualization**
(`visualization/`: interactive HTML GT-vs-pred grid behind `SampleLogCallback`), and an in-RAM
image/mask **cache** (`data/cache.py`). Next: LoRA/PEFT.

## Commands

Dependencies are managed with **uv** and recurring commands live in the **Makefile** —
do not bypass them:

- `uv add <pkg>` / `uv add --dev <pkg>` — add a dependency (never pip; it must land
  in `pyproject.toml`). PyPI is pinned as the default index there.
- `make test` — run the full pytest suite. Single test:
  `uv run pytest tests/test_tasks.py::TestTaskBuilder -v` (or append `::method`).
- `make typecheck` — run mypy over `src` and `tests` (must stay green).
- `make check` — typecheck + tests.
- `make pre-commit` — run pre-commit hooks. `make clean` — remove caches.

mypy is configured strict-ish (untyped defs disallowed). Line length is 120 (ruff).

## Architecture: the Dependency Rule

Source dependencies point inward. `torch` is treated as the "language" (allowed
everywhere); **Lightning, Hydra, data formats, model zoos (timm/smp/HF) and
albumentations are details** kept behind ABC ports. Layers:

- `core/` — framework-agnostic center. Entities (`Sample`, `Batch`, `FeatureBundle`,
  `ModelOutput`, `LossResult`, `TargetView`, `Task`, `HeadSpec`), ABC ports
  (`Backbone`, `Head`, `Criterion`, `Activation`, `MetricSet`, `TaskCodec`,
  `LossAggregator`), the `Registry`, `RuntimeContext`, and canonical string keys
  (`core/keys.py`). Imports only torch + stdlib.
- `config/` — the single Pydantic contract (`ExperimentConfig`). The boundary:
  Hydra → plain dict → `load_config()` → typed DTOs. Nothing downstream re-parses raw
  config. Component sections allow `extra` keys so per-brick overrides / `_target_`
  escape hatch survive validation.
- `data/`, `models/`, `tasks/`, `losses/`, `metrics/`, `training/`, `transforms/` —
  adapters/use cases implementing the core ports. `transforms/input.py` holds the
  per-sample `Transform`/`AlbumentationsTransform`; `transforms/batch/` holds batch
  transforms (MixUp/CutMix/Mosaic). `data/cache.py` is the in-RAM image/mask cache;
  `training/scheduler.py` the LR-scheduler adapters; `models/backbones/` the four
  backbones (timm/smp/embedding/multi); `losses/` the criteria incl. metric-learning
  (`angular.py`=ArcFace, `contrastive.py`=InfoNCE/SigLIP, `ranking.py`=triplet/margin).
- `export/` — deployment-export subsystem behind the `ModelExporter` port
  (`onnx.py`/`torchscript.py`, `pipeline.py`, `verify.py` parity checks, `wrapper.py`
  trace wrappers). `visualization/` — sample-debug subsystem: `Annotator`/`LabelRenderer`
  registries render GT-vs-prediction `SampleView`s into one self-contained interactive
  HTML grid (`renderer.py`), driven by `SampleLogCallback`. Both are details, kept outward.
- composition root: `composition/wiring/` (split by layer: `data`/`model`/`tasks`/
  `training`/`callbacks`/`export`/`checkpointing`/`common`, re-exported from its `__init__`)
  + `main.py`. Wires concrete instances in dependency order; `main.py` is a flat sequence
  of `build_*` calls ending in `run_experiment` (fit → test → export, each gated by a flag).

Concrete adapters wrap third-party libs (e.g. `TimmBackbone`, `TorchMetricsAdapter`,
`AlbumentationsTransform`). Parametric components that live in the autograd graph
(`Backbone`/`Head`/`Criterion`/`MetricSet`) inherit `nn.Module` *and* their ABC port.

## Central idea: a Task is a composition, not an enum type

There is no `TaskType` enum. A `Task` is a bundle of bricks assembled from **three
orthogonal axes**:

- **Modality** (input side): image / text / embedding / multimodal — lives in
  `data` + `backbone`. The task does not know the modality; it consumes a feature
  stream.
- **Topology** (`tasks/strategies/topology.py`, internal): output structure — GLOBAL
  (per-sample), DENSE (per-pixel), RANKING (Siamese: N views stacked through one shared
  backbone → `[B,N,D]`), MULTISTREAM (N separate encoders, e.g. CLIP/SigLIP). Picks the
  head + which `FeatureBundle` stream it consumes (`feature_key`, overridable per task).
- **Objective** (internal): label semantics — multiclass / binary / multilabel /
  continuous / **metric** (metric learning: target implicit, supervision from pair/triplet
  structure or the batch diagonal). Picks the target codec, criterion, activation, metric
  mode, out-features (for `metric`, `num_classes` is reinterpreted as `embedding_dim`).

`TaskBuilder` is the Bridge that combines a `TopologyStrategy` and an
`ObjectiveStrategy`, validating the combination (invalid pairs raise; e.g. `metric` only
pairs with RANKING/MULTISTREAM). Familiar **presets** are thin facades over the builder and
the only user-facing names: `classification`, `segmentation`, `regression`, `triplet`,
`pairwise_ranking`, `contrastive`. The loss *method* that varies within an objective lives
on the preset (`triplet`→`triplet_margin`, `pairwise_ranking`→`margin_ranking`,
`contrastive`→`info_nce`), not the objective. This is why e.g. `segmentation` +
`objective="multilabel"` needs no new type. New topology/objective/modality = one new
class in a registry (OCP).

## Cross-cutting conventions

- **Heads are derived from tasks.** Tasks are declared once (in YAML / via presets);
  `build_composite_model(backbone, {name: HeadSpec})` sizes each head from
  `backbone.feature_dim(feature_key)` and builds it via the head registry. This is
  what removes the prototype's fragile `${ref:}` config graph — nothing is shared by
  reference across model/module/data.
- **Runtime values via `RuntimeContext` + ordering, not string interpolation.**
  `num_classes` is inferred from data by default: `DataModule.setup()` fits target
  codecs and populates `RuntimeContext.num_classes`; tasks (and therefore heads) are
  built *after* setup, so `HeadSpec.out_features` is always a concrete int.
  `RuntimeValue`/`resolve_runtime` remain for other lazy values (e.g. scheduler steps).
- **Split codec.** Heavy/format-specific target decoding (label→index, mask I/O) runs
  in the data layer inside DataLoader workers (`TargetCodec`); light shape/type
  adaptation for loss vs metrics happens task-side (`TaskCodec` → `TargetView`, which
  keeps loss and metric targets separate for future MixUp).
- **Criterion operates on logits; activation is separate** (used only for metrics /
  inference), so losses stay numerically stable.
- **Extension points are registries** (`backbones`, `head_builders`, `criteria`,
  `data_sources`, `target_codecs`, `input_loaders`, `topology_strategies`, `objective_strategies`,
  `task_presets`, `batch_transforms`, `callback_registry`/`callback_builders`, `optimizers`,
  `schedulers`, `metric_factories`, `loggers`, `exporters`, `label_renderers`, `annotators`).
  Register with the `@registry.register("key")` decorator; importing a package's `__init__`
  populates its registries.
- **Two construction families** (see README "How components are built"). *Typed config
  sections* (backbone/optimizer/data/dataloader/logger) have dedicated builders selecting an
  adapter by `kind`/`name`. *Brick-specs* (loss/metrics/codec/head/callbacks/batch transforms)
  all go through `instantiate` — one grammar (`str` / `{name,…}` / `{_target_,…}`, recursive),
  one home for `_target_`. Callbacks needing runtime/config context are built by a
  `callback_builders` Strategy registry (`checkpoint` dirpath, `batch_transform`), keyed by
  registry name — the wiring dispatch loop stays closed for modification (OCP).
- **Batch transforms** (`transforms/batch/`, registry `batch_transforms`) run via
  `BatchTransformCallback` (a thin scheduler). Because the image is *shared* across heads, a
  transform must rewrite *every* task's target: it is injected the tasks' `TargetSpec` list and
  declares `supported_topologies`; the wiring guard rejects incoherent combos (e.g. MixUp + a
  DENSE head) at build time. MixUp/CutMix subclass torchvision `v2` for multi-head label mixing
  (one shared `lam`, per-head one-hot); the multiclass `TaskCodec` is soft-target aware
  (`[B,C]` float → soft loss target + argmax metric target). Mosaic is DENSE-only.
- **Metric direction** is read from `torchmetrics`' declared `higher_is_better`, not guessed:
  `MetricSet.directions()` → `LitModule.metric_directions()` (the `MetricDirectionProvider`
  Protocol) lets consumers (the progress bar) bind direction without parsing metric names.
- Canonical input/feature keys (`IMAGE`, `POOLED`, `DECODER`, `ENCODER_LAST`) live in
  `core/keys.py` — use them instead of literal strings. Other backbone-specific streams
  (e.g. future FPN levels `p3`/`p4`) are documented in each backbone's docstring.
- **DataLoader knobs** live in `config.dataloader` (`DataLoaderConfig`): `num_workers`,
  `pin_memory`, `persistent_workers`, `drop_last`, `prefetch_factor`. `persistent_workers`
  and `prefetch_factor` are auto-disabled when `num_workers=0`. It is a Hydra config group
  (`configs/dataloader/`: `default`/`performance`/`debug`) wired into `config.yaml` defaults —
  override per-run (`dataloader.num_workers=8`) or select a preset (`dataloader=performance`).
  Like the other typed sections, it is `extra="allow"`: unknown keys forward verbatim to
  `torch.utils.data.DataLoader` (`forward_extras` → `DataModule.dataloader_kwargs`), except
  framework-owned keys (`DataLoaderConfig.RESERVED`: dataset/batch_size/shuffle/collate_fn/
  sampler/batch_sampler) which the schema rejects so per-stage conventions hold.
- **The `Trainer` is assembled by `build_trainer`** (one home, composition root), not in
  `main.py`. `TrainerConfig` is a typed section forwarded as kwargs, with two seams: `logger`
  is injected from `build_logger`, and `profiler` is a brick-spec — a `{_target_: ...}` mapping
  (e.g. `SimpleProfiler`/`AdvancedProfiler` with `dirpath`/`filename`) is built via `instantiate`,
  while a string alias (`simple`/`advanced`) or `None` passes straight to Lightning. See
  `configs/trainer/profile.yaml` for the file-output example.
- **LR schedulers** (`training/scheduler.py`, registry `schedulers`: `cosine`/`onecycle`/
  `plateau`/`step`). `SchedulerConfig` mirrors the optimizer (extras forward verbatim);
  `runtime_kwargs` maps a constructor param to a trainer fact (`total_steps`/`steps_per_epoch`/
  `epochs`) resolved at fit time. `None` config → constant LR. It is a Hydra group
  (`configs/scheduler/`); `interval`/`frequency`/`monitor` map to Lightning's `lr_scheduler`.
- **Export** (`run_export`, gated by `run_export`). `ExportConfig` is a per-format Pydantic
  *discriminated union* keyed on `format` (`onnx`/`torchscript`; tensorrt reserved), each
  `extra="forbid"` so a misplaced option fails at `load_config`. Backends implement the
  `ModelExporter` port (`export`/`load`/`validate`) and self-register in `exporters`; the
  generic `verify.py` composes their `load()`/`validate()` into a parity report (`atol`/`rtol`).
  `combined` exports one image→all-logits graph; `split_components` also emits per-part files.
  Output dir defaults to `{save_dir}/export`. Config group `configs/export/` (`onnx`/
  `torchscript`/`all`). `validate_export_preconditions` runs the topology guard *before* training.
- **Sample visualization** (`visualization/`) is driven by `SampleLogCallback` (`sample_log`):
  it annotates a batch's GT vs predictions and renders an interactive self-contained HTML grid.
  Two registries keep it open/closed: `annotators` keyed by `(topology, objective)` write the
  `SampleView` fields; `label_renderers` keyed by `Label` type emit `FieldItem`s (chips for
  labels, full-cell mask overlays for segmentation). The `Renderer` never branches on label type.
- Dataclasses for domain/application objects; Pydantic only at I/O boundaries
  (config). Google-style docstrings with a `Parameters:` block.

## Key data-layer conventions

- `data.inputs` (not `image_column`) drives input loading: `str` = single image shorthand;
  `dict[alias, column]` = multiple inputs, loader auto-detected from file extensions at
  `setup()` time; `dict[alias, {column, loader}]` = explicit loader key.
- `InputBinding(name, column, loader)` is the input-side counterpart of `TargetBinding(name, column, codec)`.
  Both use `target_bindings` / `input_bindings` as parameter names in `DataModule` and `Dataset`.
- `instantiate(spec)` is recursive — handles nested `_target_` graphs (Albumentations pipelines,
  `OneOf`, etc.); `registry` is optional (omit for pure `_target_` mode with no registry lookup).
- `build_data_module(config, target_bindings, runtime)` is the single wiring call that replaces
  manual `DataModule` construction; split/pre-split logic is encapsulated there.

## Environment notes

- The package root is `src/` (import as `from src.core import ...`); run from repo root.
- `albumentationsx` (2.3.1) imports and works. If it crashes, `uv run python -c "import albumentations"` is the canary.
  Augmentation pipelines live in `configs/transforms/` as `_target_`-keyed YAML (`default`/`augmented`);
  transforms always come from config.
- Data sources: subclass `FileDataSource` and implement `_read_file`; `CsvDataSource`/`JsonDataSource`
  registered in `data_sources`. Two data modes: `data.sources: str/list` + `data.split` (random or
  stratified via `data.split_stratify`), or `data.sources: {train: ..., val: ...}` (pre-split files).
  `data.max_samples: int | float` caps dataset size for fast iteration.
- `SmpBackbone` exposes two streams: `ENCODER_LAST [B, D, H, W]` (raw spatial encoder output,
  use with smp's `ClassificationHead` via `prefer_native=True`) and `DECODER [B, D, H, W]`
  (full decoder output for segmentation heads). `POOLED` is not exposed — pooling is the head's job.
  DPT architecture is supported via the `_dpt_style` flag (detected from `name.lower() == "dpt"`).
  ASPP-based architectures (deeplabv3, pan, upernet) require `batch_size ≥ 2` in train mode
  (BatchNorm after global-avg-pool). See `configs/backbone/smp_dpt.yaml` for DPT config example.
- `EmbeddingBackbone` (kind `embedding`) consumes precomputed feature vectors (no image
  encoder) — the modality for embedding/ranking tasks on cached features. `MultiEncoderBackbone`
  (kind `multi`) holds N named sub-encoders producing N `POOLED` streams for MULTISTREAM tasks;
  the encoder name == the `data.inputs` alias == the stream name (wiring derives stream order).
  RANKING instead stacks N input *views* through one shared backbone (`view_keys` from `data.inputs`).
- The reference dataset `old/data/classification.csv` points at remote URLs with empty
  local placeholder images; tests use synthetic images generated in a tmp dir, and the
  offline smoke should use a synthetic local dataset.
