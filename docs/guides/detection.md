# Object detection

Detection is an ordinary task on the table pipeline: the rows, the encoder, the collate,
the letterbox and the metric all exist and are tested. What does not exist yet is the
criterion that trains a composed detector — stage 3 of the detection roadmap — so today
a detection run builds and is refused before training, naming that stage. This page is
what you can use now.

## The rows

The canonical format is JSON Lines, one row per image, boxes in pixels, classes by name,
an empty list a valid negative:

```json
{"image": "a.jpg", "objects": [{"box": [50, 25, 150, 75], "class": "dog"}]}
{"image": "c.jpg", "objects": []}
```

Two offline converters write it, validating loudly as they go — out-of-bounds boxes are
clipped and counted, degenerate ones refused by name:

```bash
uv run python -m src.data.converters.yolo --data path/data.yaml --into data/pets/
uv run python -m src.data.converters.coco --annotations instances.json --images images/ --into data/pets/
```

A hand-written file is read the same way, and checked the same way. At setup the encoder
refuses a box without area, one off the origin, and a class the task did not declare —
showing the cell, over every split. A box that leaves its image is refused by the
transform seam, naming the row, when the first epoch meets it: only the loaded image
knows its size. Keep hand-written boxes inside the picture; the converters guarantee it.

## The task

```yaml
data:
  source: {train: data/pets/train.jsonl, val: data/pets/val.jsonl}
  inputs: {image: {column: image, loader: {name: image, root: data/pets}}}

tasks:
  boxes:
    kind: detection
    target: objects
    classes: {0: cat, 1: dog}
```

The kind implies the `boxes` encoder, so no `target_encoder` line is written; `classes`
pins the index space, as for every task read against a vocabulary. The encoder parses
list and JSON-string cells alike, so a CSV carrying the same cells is readable too.

## `Instances`

The currency for a per-instance task, in the core beside `Batch` and `Loss`:

```python
Instances(
    boxes=...,  # [N, 4] xyxy, in pixels of the image as the model was fed it
    labels=...,  # [N] class index
    sample_index=...,  # [N] which image in the batch each object belongs to
    scores=...,  # [N] confidence — None for ground truth, which has none
)
```

Flat rather than per-image, because that is the only shape a ragged quantity has that a
tensor can carry; the collate concatenates per-sample `Instances` and rewrites
`sample_index`, keyed on the value's type — no config chooses a collate. `Instances.of(i)`
gives one image's objects back; `.detach()` and `.to(device)` behave as a tensor target does.

A consumer that serves only tensors — a batch transform, an exported graph — refuses an
`Instances` target by name (`require_tensor`), saying which task and which reader.

## Geometry

Boxes follow the picture through the ordinary transforms section: the encoder declares
its geometry as `boxes`, the pipeline binds it, and letterboxing is one declared
operation — it scales, pads and moves the boxes with the image:

```yaml
transforms:
  train:
    _target_: src.transforms.AlbumentationsTransform
    transforms:
      - {_target_: albumentations.LetterBox, size: [640, 640]}
      - {_target_: albumentations.Normalize}
      - {_target_: albumentations.pytorch.ToTensorV2}
```

## The model

```yaml
model: {name: ultralytics, model_name: yolov8n.yaml}
```

Ultralytics' graph behind the framework's ports: everything before `Detect` is the
backbone, exposing the pyramid `p3`/`p4`/`p5` and pooled `features`; the `Detect` head is
rebuilt per task at the class count the data revealed and returns one raw tensor
`[B, 4·reg_max + nc, A]` in every mode. `checkpoint_path: runs/yolov8n.pt` grafts an
ultralytics `.pt` loudly — see [models](models.md#detection-on-a-composed-backbone).
`ultralytics` is imported only inside that family's modules; nothing else names it.
The family serves the one-to-many `Detect` line (v8, 11, 12); `v10Detect`, RT-DETR and
the end-to-end `Detect` of yolo26 are refused by name.

## mAP

The `detection` kind reports `map`, which is one entry publishing a family:

```
val/boxes/map/map       mAP@50-95, the COCO average
val/boxes/map/map_50
val/boxes/map/map_75
```

torchmetrics computes all of its readings in one pass, so asking for three costs what
asking for one would. Widen the request by naming the readings:

```yaml
tasks:
  boxes:
    kind: detection
    metrics:
      map: {name: map, readings: [map, map_50, map_75, map_small, map_medium, map_large]}
```

`map_per_class` and `mar_100_per_class` are the two torchmetrics only computes when told
to, so naming either turns on `class_metrics` — derived from the request, never restated.
They land as per-class leaves named by the classes they are *about*. A reading its data
could not support returns `-1`, which is not a measurement; those are dropped rather than
logged — a scalar reading, and a class inside a per-class one (predicted, never
annotated) alike. The backend is `faster_coco_eval`, named explicitly because torchmetrics reaches
for `pycocotools` by default. A misspelt reading is refused at build time, with the list.

## What is next

Stage 3 of [the roadmap](../superpowers/specs/2026-08-17-detection-roadmap-design.md):
the framework's own criterion over the head's raw tensor (their task-aligned assigner
and box losses, our honest loss parts) and the decoder that turns it into `Instances`.
Until it lands, `configs/experiment/examples/` ships no detection run.
