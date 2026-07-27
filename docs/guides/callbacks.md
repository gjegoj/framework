# Callbacks

`callbacks` is a dict of `{registry_key: params}` — the same pattern as `metrics`.
Keys are looked up in `callback_registry`; values are constructor kwargs (`null` = all defaults).
Declaration order in YAML controls registration order, which matters: put `ema` before `checkpoint`.

```yaml
defaults:
  - callbacks: default    # lr_monitor + ema + checkpoint
  # - callbacks: minimal  # checkpoint only
  # - callbacks: none     # no callbacks (smoke tests)
```

| Key | Callback | What it does |
|---|---|---|
| `checkpoint` | `EmaModelCheckpoint` | Saves the best model by a monitored metric; EMA-consistent |
| `ema` | `EmaCallback` | Maintains an EMA shadow; validation and checkpoints use EMA weights |
| `freeze` | `FreezeCallback` | Freezes modules for the first N epochs, then unfreezes |
| `criterion_schedule` | `CriterionScheduleCallback` | Anneals a numeric loss parameter over epochs |
| `batch_transform` | `BatchTransformCallback` | Schedules MixUp / CutMix / Mosaic |
| `lr_monitor` | `LearningRateMonitor` | Logs learning rates to the experiment logger |
| `progress_bar` | `MetricsProgressBar` | Rich progress bar with live metrics & directions |
| `model_summary` | `TreeModelSummary` | Prints the module tree with parameter counts at fit start |
| `sample_log` | `SampleLogCallback` | Renders a GT-vs-prediction HTML grid |
| `metric_summary` | `MetricSummaryCallback` | After test, reports headline metrics to the logger summary |
| `dataset_stats` | `DatasetStatsCallback` | Prints target distributions + logs histograms before training |

**Disable a callback at runtime** — delete its key with the `~` prefix:

```bash
uv run python main.py 'defaults=[{override /callbacks: default}]' '~callbacks.ema'
```

**Custom callback** — register your class (see [Extending the framework](../reference/extending.md))
or point `_target_` straight at it, no registration needed:

```yaml
callbacks:
  my_cb:
    _target_: my_project.callbacks.GradientClipCallback
    max_norm: 1.0
```

## checkpoint

Saves the best model by a monitored metric. `dirpath` defaults to `{save_dir}/checkpoints`;
extra keys forward to Lightning's `ModelCheckpoint`. The `${key:...}` resolver expands the
canonical logged-key tokens:

```yaml
callbacks:
  checkpoint:
    monitor: ${key:LOSS}/val/${key:TOTAL}   # → loss/val/total
    mode: min
    save_top_k: 1
    save_weights_only: true
```

The registry key builds `EmaModelCheckpoint`: with EMA active, even
`save_weights_only: true` checkpoints store the **EMA weights** that produced the monitored
metric (plain Lightning would silently store the live weights there while ranking by the EMA
metric). With LoRA active, checkpoints are pruned to trainable weights only — see
[LoRA fine-tuning](lora.md).

## ema

Exponential moving average of the model weights. EMA weights are swapped in for validation
(where `checkpoint` monitors its metric), swapped back for training, written into
checkpoints, and copied into the model at the end of fit:

```yaml
callbacks:
  ema:
    decay: 0.999            # typical range 0.99–0.9999
    warmup_fraction: 0.1    # fraction of total steps before averaging starts
    use_buffers: true       # also average BatchNorm running statistics
```

During warmup, validation and checkpoints use the **live** weights — a "best" checkpoint can
never hold untrained ones.

## freeze

Freeze named submodules before training, optionally unfreezing later:

```yaml
callbacks:
  freeze:
    targets: [model.backbone]
    unfreeze_at: 0.3    # fraction of max_epochs; int = epoch index; -1 = never
    train_bn: false     # keep BatchNorm stats updating while frozen
```

Note: with [LoRA](lora.md) active, LoRA owns backbone freezing — a `freeze` target inside
`model.backbone` is rejected at startup (it would freeze the adapters too).

## criterion_schedule

Anneal any plain numeric attribute of a task's criterion over epochs (e.g. FocalLoss
`gamma`: start as cross-entropy, ramp into focusing):

```yaml
callbacks:
  criterion_schedule:
    task: mask
    parameter: gamma        # dot-path into weighted_sum terms: focal.gamma
    start: 0.0              # applied from epoch 0 (overrides the constructed value)
    end: 4.0
    schedule: linear        # or cosine
    over: 1.0               # fraction of max_epochs the ramp spans
```

The value is a pure function of the epoch, so the schedule is resume-safe and invisible to
EMA weight swaps and checkpoints. Attributes are resolved and validated at `on_fit_start`
(unknown task / parameter / composite term fails loudly, listing what is available);
learnable `nn.Parameter`s are refused — scheduling them would fight the optimizer. Note:
with a gamma schedule, train and val losses are computed with the epoch's current value, so
prefer monitoring a metric (`mask/iou/val/mean`) over val loss for checkpointing.

## batch_transform

Batch transforms (registry: `mixup` · `cutmix` · `mosaic`) mix samples *after* collation,
rewriting every task's target consistently (one shared `lam`, per-head one-hot; the
multiclass adapter is soft-target aware, so metrics still see hard labels):

```yaml
callbacks:
  batch_transform:
    disable_after_fraction: 0.5   # run MixUp for the first half of training only
    transform:
      name: mixup                  # or cutmix; mosaic is DENSE-only
      alpha: 0.2
```

A wiring guard rejects incoherent combos at build time (e.g. MixUp with a DENSE head, or
Mosaic with a GLOBAL one) — the image is shared across heads, so a transform must be valid
for every task at once.

## lr_monitor

Logs each param-group's learning rate to the experiment logger — useful together with
per-head LR overrides and schedulers:

```yaml
callbacks:
  lr_monitor:
    logging_interval: epoch   # or step
```

## progress_bar

Rich progress bar showing live metric values with direction-aware coloring (improvement vs
regression is read from each metric's own `higher_is_better`, never guessed from its name):

```yaml
callbacks:
  progress_bar: null   # no parameters
```

## model_summary

Prints the assembled module tree with parameter counts at fit start — handy for verifying
what a backbone/head combination actually built and what is frozen:

```yaml
callbacks:
  model_summary:
    max_depth: 3
```

## sample_log

Periodically renders a batch as an interactive self-contained HTML grid — image cells with
toggleable ground-truth / prediction overlays per task. Requires an HTML-capable logger
(ClearML). See [Logging & visualization](visualization.md):

```yaml
callbacks:
  sample_log:
    num_images: 8
    every_n_epochs: 5
    batch_index: 0
    mean: ${mean}       # to de-normalize images for display
    std: ${std}
```

## metric_summary

After `test`, reports the headline metrics (scalars + each vector metric's mean) to the
logger's single-value summary table (ClearML "Single Values"). No-op when the logger does
not support it. See [Logging & visualization](visualization.md#logger):

```yaml
callbacks:
  metric_summary: null   # no parameters
```

## dataset_stats

Before the first stage, reports each task's target distribution per stage — class counts
for classification, numeric stats for regression — as terminal tables plus grouped-bar
histograms to the logger, making class imbalance and train/val/test skew visible at a
glance. See [Data](data.md):

```yaml
callbacks:
  dataset_stats: null   # no parameters
```
