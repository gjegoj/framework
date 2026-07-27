# Data

## Data


**Split mode** — one file, ratios decide the split:

```yaml
data:
  sources: data/annotations.csv
  inputs: image_path          # shorthand: single image column
  split:
    train: 0.8
    val:   0.1
    test:  0.1
```

**Pre-split mode** — separate files per stage:

```yaml
data:
  sources:
    train: data/train.csv
    val:   data/val.csv
  inputs: image_path
```

**Multiple inputs** (multi-view, multimodal):

```yaml
data:
  inputs:
    image:   image_path        # loader auto-detected from extension
    depth:   depth_path        # another image column
    caption: {column: text_col, loader: text}   # explicit loader
```

**Stratified split** — keep class balance across stages:

```yaml
data:
  split: {train: 0.8, val: 0.1, test: 0.1}
  split_stratify: species      # categorical → classification; numeric → quantile-binned
```

**Cap dataset size** for fast iteration:

```yaml
data:
  max_samples: 500       # int → exactly N rows
  max_samples: 0.1       # float → 10% of data
```

**Dataset distribution report.** The `dataset_stats` callback (in the `default` stack) reports,
before the first stage, each task's target distribution per stage — class counts for
classification / multilabel, numeric stats for regression — rendered two ways: a compact table
in the terminal and a grouped-bar histogram per task to the logger (ClearML), with one series
per stage so train/val/test skew is visible at a glance. The data module *computes* the
distributions (`DataModule.statistics()`, each `TargetEncoder` summarizing its own column), so
segmentation (pixel counts) drops in later without touching the report; the callback only
*presents* them. Logging is a no-op without a plot-capable logger.

## DataLoader & cache


`dataloader` is its own config group. Override per-run, swap a preset, or add a block in an
experiment:

```bash
uv run python main.py dataloader.num_workers=8 dataloader.pin_memory=true
uv run python main.py dataloader=performance      # GPU preset: 8 workers, pin_memory, prefetch 4
uv run python main.py dataloader=debug            # num_workers=0 (real tracebacks / breakpoints)
```

| Knob | Meaning |
|---|---|
| `num_workers` | loader subprocesses (`0` = main process, debug-friendly) |
| `pin_memory` | page-locked host memory → faster CPU→GPU copies (CUDA only) |
| `persistent_workers` | keep workers alive between epochs (auto-off at `num_workers=0`) |
| `drop_last` | drop the last incomplete **train** batch (val/test never drop) |
| `prefetch_factor` | batches prefetched per worker (auto-off at `num_workers=0`) |

Extra keys forward verbatim to `torch.utils.data.DataLoader` (e.g. `timeout`,
`multiprocessing_context`); framework-owned keys (`batch_size`/`shuffle`/`collate_fn`/…) are
rejected so per-stage conventions hold.

**In-RAM cache** — decode each image/mask once, warmed in the parent before training and
read-only after (so it stays shared across fork workers). Budget = `min(ram_fraction · free
RAM, max_gb)`:

```yaml
data:
  cache:
    ram_fraction: 0.5     # cap at half of available RAM (0 disables)
    max_gb: 8             # absolute cap in GiB
    workers: 8            # threads used to warm the cache
```

> The cache + multi-worker only share memory under **fork** (Linux). On macOS (spawn),
> pick one: cache with `num_workers=0`, or workers with the cache off.
