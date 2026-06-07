# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A configuration-driven framework for training computer-vision models (multi-task,
multi-modal) on top of PyTorch Lightning, Hydra, Pydantic, timm/smp, albumentations
and torchmetrics. It is a clean-architecture rewrite of the prototype in `old/`
(kept only as reference — do not edit or import from it). The full design and the
milestone breakdown live in the approved plan at
`~/.claude/plans/clean-code-refactoring-patterns-softwar-merry-sunbeam.md`.

Status: M1–M3 complete (classification, all Objective variants, DENSE segmentation) +
M4-prep complete (multi-stream `SmpBackbone` with `ENCODER_LAST`/`DECODER`, `feature_key`
override in `TaskConfig`, `DataLoaderConfig`, DPT support, multitask smp smoke tests).
Next: M4 (per-head LR, typed metrics, loggers), then M5–M7.

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
- `data/`, `models/`, `tasks/`, `losses/`, `metrics/`, `training/` — adapters/use
  cases implementing the core ports.
- composition root (Hydra entry) wires concrete instances in dependency order.

Concrete adapters wrap third-party libs (e.g. `TimmBackbone`, `TorchMetricsAdapter`,
`AlbumentationsTransform`). Parametric components that live in the autograd graph
(`Backbone`/`Head`/`Criterion`/`MetricSet`) inherit `nn.Module` *and* their ABC port.

## Central idea: a Task is a composition, not an enum type

There is no `TaskType` enum. A `Task` is a bundle of bricks assembled from **three
orthogonal axes**:

- **Modality** (input side): image / text / embedding / multimodal — lives in
  `data` + `backbone`. The task does not know the modality; it consumes a feature
  stream.
- **Topology** (`tasks/strategies/topology.py`, internal): output structure — GLOBAL (per-sample),
  DENSE (per-pixel), SET, SEQUENCE, EMBEDDING. Picks the head + which `FeatureBundle`
  stream it consumes (`feature_key`). `feature_key` can be overridden per task via YAML.
- **Objective** (internal): label semantics — binary / multiclass / multilabel /
  continuous. Picks the target codec, criterion, activation, metric mode, out-features.

`TaskBuilder` is the Bridge that combines a `TopologyStrategy` and an
`ObjectiveStrategy`, validating the combination (invalid pairs raise). Familiar
**presets** (`classification(...)`, `segmentation(...)`) are thin facades over the
builder and are the only user-facing names. This is why e.g. `segmentation` +
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
  `task_presets`). Register with the `@registry.register("key")` decorator; importing a
  package's `__init__` populates its registries.
- Canonical input/feature keys (`IMAGE`, `POOLED`, `DECODER`, `ENCODER_LAST`) live in
  `core/keys.py` — use them instead of literal strings. Other backbone-specific streams
  (e.g. future FPN levels `p3`/`p4`) are documented in each backbone's docstring.
- **DataLoader knobs** live in `config.dataloader` (`DataLoaderConfig`): `num_workers`,
  `pin_memory`, `persistent_workers`, `drop_last`, `prefetch_factor`. `persistent_workers`
  and `prefetch_factor` are auto-disabled when `num_workers=0`.
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
- The reference dataset `old/data/classification.csv` points at remote URLs with empty
  local placeholder images; tests use synthetic images generated in a tmp dir, and the
  offline smoke should use a synthetic local dataset.
