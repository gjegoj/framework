# Object detection (YOLO)

Detection is a separate **training regime**: the framework's Lightning contour (trainer,
callbacks, logging, checkpoints) drives an [ultralytics](https://github.com/ultralytics/ultralytics)
YOLO model with its native loss and data pipeline. It is *not* a task preset in the
topology × objective sense — a run is either detection or the standard multi-task chain,
and `main.py` branches on the `detection` preset before any of the usual data/model wiring.

> **License note:** ultralytics is **AGPL-3.0**. Whether that is compatible with your use
> is a project-owner licensing decision — the facade (`src/models/yolo.py`) keeps the
> dependency swappable, but the default detection regime depends on it.

## Configuration

The model is declared in the **model section**, like every other model — `kind: yolo`
selects the complete-model family (the run is assembled by the detection assembler
instead of the backbone→heads chain). The task declares *what* is solved and its metrics:

```yaml
run_export: false          # detection export is phase 2 — the guard rejects run_export: true

model: # or select the ready-made brick from the group: model=yolov8n
  kind: yolo
  name: yolov8n.yaml               # ultralytics architecture yaml (offline) or .pt weights path
  # Extra keys forward verbatim as native ultralytics hyperparameters:
  # box: 7.5                       #   loss gains (box / cls / dfl)
  # mosaic: 0.0                    #   and augmentation knobs (mosaic, fliplr, hsv_*, ...)

data:
  sources: data/my_set/data.yaml   # YOLO dataset descriptor (path / train / val / names)

tasks:
  boxes:                           # the task name prefixes every logged key
    preset: detection
    # metrics: {map: {box_format: xyxy}}   # optional (this is the default); any
    #                                        registry metric / kwargs, like every task
```

- `data.sources` is a single YOLO `data.yaml` descriptor; the dataset itself is
  ultralytics-native (images + `labels/*.txt`). `num_classes` comes from the descriptor's
  `names`. If the descriptor has no `test` split, testing reuses the `val` split.
- `name: *.yaml` builds the architecture from scratch (offline, no downloads);
  `name: *.pt` loads pretrained ultralytics weights.
- `image_size`, `batch_size`, `epochs`, `optimizer`, `scheduler`, `dataloader.num_workers`
  and the `trainer`/`logger` sections work exactly as in every other run.

A complete runnable example is `configs/experiment/detect_smoke.yaml`.

## What gets logged

Losses go through the standard grammar — `loss/<stage>/total` plus the components
`loss/<stage>/<task>/box|cls|dfl`. Evaluation runs NMS decoding and accumulates
COCO-style mAP (torchmetrics `MeanAveragePrecision`), logged at epoch end as
`<task>/map50/<stage>` and `<task>/map50_95/<stage>` — the progress bar and ClearML
parse them like any other task metric.

## What composes

EMA, `EmaModelCheckpoint`, freeze, and the ClearML logger work unchanged. For
checkpointing, monitor mAP upward:

```yaml
callbacks:
  ema: {decay: 0.999}
  checkpoint:
    monitor: boxes/map50_95/val
    mode: max
    save_weights_only: true      # stores the EMA weights, as usual
```

## v1 boundaries (phase 2)

`validate_detection_preconditions` fails fast — before anything is built — on anything
outside the v1 scope:

- exactly **one** detection task, no mixing with other tasks (a shared-backbone
  `Topology.DETECTION` is the phase-2 path to mixing);
- **export** (`run_export: true`), **LoRA**, and **distillation** are rejected for
  detection runs;
- also deferred: `sample_log` box annotators and a CSV→YOLO data bridge.
