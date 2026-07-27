# Logging & visualization

## Logger


```yaml
defaults:
  - logger: none       # no logging (default)
  - logger: clearml    # ClearML experiment tracking
```

**ClearML** config:

```yaml
logger:
  kind: clearml
  project: my-project   # defaults to experiment project
  task: run-001         # optional task name
```

Override at runtime:

```bash
uv run python main.py 'defaults=[{override /logger: clearml}]'
```

**End-of-run summary.** The `metric_summary` callback (in the `default` stack) reports the
headline **test** metrics — scalars plus each vector metric's mean, no per-class noise — to the
logger's single-value summary table (ClearML's "Single Values") via `PlotLogger.log_single_value`,
so the final numbers are visible at a glance. Entries are named as in the live training table
(`species/f1`, `breed/recall`, `mask/iou`, `loss/total` — stage and the `mean` leaf stripped) and
rounded for readability. It is a no-op when the logger does not support it (e.g. `logger: none`);
the detailed per-step scalars and per-class values still log as usual.

## Sample visualization


`SampleLogCallback` periodically takes a batch, runs the model, and renders a
self-contained interactive **HTML grid**: each cell shows the image with toggleable overlays
— chips for classification/regression labels, full-cell colored masks for segmentation —
and a sidebar to switch ground-truth / prediction layers per task and class.

```yaml
callbacks:
  sample_log:
    num_images: 8
    every_n_epochs: 5
    batch_index: 0
    title_prefix: samples
```

It is label-type agnostic: an `annotators` registry keyed by `(topology, objective)` writes
the GT/pred fields, and a `label_renderers` registry keyed by `Label` type emits the cell
overlays — so a new task kind plugs in without touching the renderer.


---
