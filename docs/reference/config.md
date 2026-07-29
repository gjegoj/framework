# Config reference

Every YAML key is defined and validated in `src/config/schema.py` — the
single source of truth. Each field there carries its own type, default, constraints, and a
one-line description, so the schema doubles as the authoritative reference; the tables below are
a map of the surface. The **root** model rejects unknown keys (`extra="forbid"`), while the
**component sections** (`backbone` · `optimizer` · `scheduler` · `data` · `dataloader` ·
`logger` · `trainer` · `lora`) allow extras and forward them verbatim to the underlying
constructor. Fixed-schema sections (`distillation`, `export`, `data.cache`) reject typo'd keys
outright.

**Top-level keys** (`ExperimentConfig` root):

| Key | Type | Default | Sets |
|---|---|---|---|
| `project` | `str` | — *required* | Project name for tracking. |
| `run_name` | `str` | `null` | Human-readable run name → logger task. `${now:%Y-%m-%d_%H-%M-%S}` for an auto-timestamp. |
| `save_dir` | `str` | `null` | Root for run outputs; `checkpoint.dirpath` defaults to `{save_dir}/checkpoints`. Use `${hydra:run.dir}` to follow Hydra's run dir. |
| `seed` | `int` | `42` | Global random seed. |
| `epochs` | `int` | — *required* | Number of training epochs. |
| `batch_size` | `int` | — *required* | Batch size. |
| `image_size` | `[int, int]` | — *required* | Image `[height, width]` in pixels. |
| `lr` | `float` | — *required* | Global learning rate; referenced as `${lr}`. Override per task via `tasks.<name>.optimizer.lr`. |
| `mean` / `std` | `list[float]` | ImageNet | Normalization statistics (must be equal length). |
| `run_train` / `run_test` / `run_export` | `bool` | `true` | Gate `fit` / `test` / `export` (at least one must be true). |
| `ckpt_path` | `str` | `null` | Checkpoint for `test` — a `.ckpt` path or alias `best` / `last`. Required for eval-only (`run_train: false`). |
| `init_ckpt_path` | `str` | `null` | Load weights before `fit` (pretrain / fine-tune, **not** resume). Requires `run_train: true`. |

**Sections** (each a nested model — full fields in the schema, deep dives linked):

| Section | Required | Configures | Schema | Details |
|---|---|---|---|---|
| `data` | ✓ | `sources`, `inputs`, `split`, `split_stratify`, `cache`, `max_samples`, `root_path` | `DataConfig` | [Data](../guides/data.md#data) |
| `dataloader` | default | `num_workers`, `pin_memory`, `persistent_workers`, `drop_last`, `prefetch_factor` | `DataLoaderConfig` | [DataLoader & cache](../guides/data.md#dataloader--cache) |
| `model` | ✓ | model selection — assembled backbone or complete model (`kind`, `name`, `pretrained`, …) | `ModelConfig` | [Backbone](../guides/backbones.md) |
| `optimizer` | ✓ | `name`, `lr`, `weight_decay`, … | `OptimizerConfig` | [Optimizer, LR & scheduler](../guides/training.md) |
| `scheduler` | `null` | LR schedule (`name`, `interval`, `monitor`, `runtime_kwargs`, …); `null` = constant LR | `SchedulerConfig` | [Optimizer, LR & scheduler](../guides/training.md) |
| `tasks` | ✓ (≥1) | per task: `preset`, `target`, `objective`, `head`, `feature_key`, `class_mapping`, `loss`, `metrics`, `weight`, per-head `optimizer` | `TaskConfig` | [Tasks & presets](../guides/tasks.md#tasks--presets) |
| `transforms` | `null` | per-stage Albumentations pipelines (`_target_` graphs) | dict | [How components are built](../concepts.md#how-components-are-built) |
| `logger` | default | tracking backend (`none` / `clearml`) | `LoggerConfig` | [Logger](../guides/visualization.md#logger) |
| `callbacks` | `null` | callbacks by registry key / `_target_` | dict | [Callbacks](../guides/callbacks.md) |
| `trainer` | default | Lightning `Trainer` knobs (`accelerator`, `devices`, `precision`, `log_every_n_steps`, `profiler`) | `TrainerConfig` | [How components are built](../concepts.md#how-components-are-built) |
| `export` | default | deployment export (per-format `targets`) | `ExportConfig` (`export.py`) | [Export](../guides/export.md) |
| `distillation` | `null` | online knowledge distillation (`teachers`, `temperature`, `weight`, `loss`, `tasks`) | `DistillationConfig` | [Knowledge distillation](../guides/distillation.md) |
| `lora` | `null` | LoRA fine-tuning (`target_modules`, `rank`, `alpha`, `dropout`, peft extras) | `LoraConfig` | [LoRA fine-tuning](../guides/lora.md) |

> Field-level constraints (`lr > 0`, `cache.ram_fraction ∈ [0, 1]`, split ratios summing to
> 1.0, mutually exclusive split / pre-split modes, …) are enforced in the schema's validators —
> open `src/config/schema.py` for the exact contract behind any key.
