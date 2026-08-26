# Object detection

Detection is the first **vendor family**: ultralytics owns the head, the loss and
the decoding, and this framework drives the loop around them. One config runs it:

```bash
uv run main.py experiment=examples/detection
```

```yaml
# configs/experiment/examples/detection.yaml
defaults:
  - override /model: yolov8n
  - override /callbacks: default

image_size: [640, 640]

data:
  source: coco128.yaml     # a YOLO descriptor, not an annotation table
  inputs: {}               # the descriptor names its own image directories

tasks:
  boxes:
    preset: detection
```

Everything downstream is unchanged: the same training loop, the same checkpoints,
the same `{stage}/{task}/{leaf}` log keys, the same tracker. It cost one topology
member (`instances`) and one entity in the core (`Instances`) — no second
assembler, no capability flags, and no third place that knows what YOLO is.

## What a vendor family is

Everything else the framework serves is **composed**: it wraps a backbone, builds
a head, declares a criterion. A vendor family arrives whole — its assigner, its
loss and its decoding are one design, and half-replacing it produces a model that
trains against a different objective than it reports.

One name decides which kind of run this is: `model.name` found in the
`vendor_model_registry` rather than the `backbone_registry`. That single reading picks
the model *and* the data pipeline, so the two halves cannot disagree.

## The data

A detection dataset does not arrive as a table. It arrives as a `data.yaml`
descriptor beside `images/` and `labels/` directories:

```yaml
data:
  source: /datasets/defects/data.yaml
  inputs: {}
```

`inputs` is empty and there is no `split`: the descriptor already names one image
directory per stage, and its class list becomes the task's `num_classes` and
`class_names` through the same `DataProfile` every other head is sized from. A
descriptor with no `test:` key simply has no test stage — ordinary YOLO practice.

Ultralytics' own dataset class is used as it stands, which is where the box-aware
augmentation lives — mosaic, HSV, perspective. What the framework adds is one
translation: its batch becomes a `Batch`, with the objects as `Instances` under
the task's own name.

## `Instances`

The currency for a per-instance task, in the core beside `Batch` and `Loss`:

```python
Instances(
    boxes=...,          # [N, 4] xyxy, in pixels of the image as the model was fed it
    labels=...,         # [N] class index
    sample_index=...,   # [N] which image in the batch each object belongs to
    scores=...,         # [N] confidence — None for ground truth, which has none
)
```

Flat rather than per-image, because that is the only shape a ragged quantity has
that a tensor can carry — and it is the shape a detection collate already
produces, so nothing is converted to satisfy the entity. `scores=None` for ground
truth is what lets one entity serve both sides of a comparison.

`Instances.of(i)` gives one image's objects in the same entity; `.detach()` and
`.to(device)` behave the way a tensor target does, so `Batch.to` moves either
alike.

A task predicting objects cannot reach a consumer that serves only tensors — a
composed model's head, a batch transform, an exported graph. `require_tensor` refuses
it by name, saying which task and which reader, instead of failing deeper down on
a missing attribute.

## The model

```yaml
# configs/model/yolov8n.yaml
name: yolo
model_name: yolov8n.yaml
```

`model_name` takes an architecture file or a `.pt` weights path. The network
class follows from the name — measured, `YOLO(...)` picks `DetectionModel`,
`SegmentationModel` or `PoseModel` — so one code path serves all three and this
class never branches on what it is building.

The head is always rebuilt at **this** dataset's class count. With a `.pt` file
the arrived weights are then grafted onto it, which is what ultralytics' own
trainer does when fine-tuning: a checkpoint trained on other classes is the
ordinary case, not an error.

Every other key forwards verbatim to ultralytics' own configuration — the loss
gains and the augmentation knobs alike:

```yaml
model:
  name: yolo
  model_name: yolov8n.pt
  box: 7.5          # loss gains
  cls: 0.5
  dfl: 1.5
  mosaic: 0.0       # augmentation
  hsv_h: 0.015
  degrees: 10.0
```

One namespace, because that is how the vendor keeps them; splitting them across
two of our sections would mean maintaining a table of which key belongs where.

## What it logs

The vendor's three loss components arrive as `Loss` parts, scoped by the task the
way a composed model scopes its criteria:

```
train/loss              the total
train/boxes/box         the box regression term
train/boxes/cls         the classification term
train/boxes/dfl         the distribution-focal term
```

`_loss` is stripped from the vendor's own spelling — `train/boxes/box_loss` would
say "loss" twice, once in the key's grammar and once in ultralytics'.

## mAP

The `detection` preset reports `map`, which is one entry publishing a family:

```
val/boxes/map/map       mAP@50-95, the COCO average
val/boxes/map/map_50
val/boxes/map/map_75
```

torchmetrics computes all fifteen of its readings in one pass, so asking for
three costs exactly what asking for one would. Widen the request by naming the
readings:

```yaml
tasks:
  boxes:
    preset: detection
    metrics:
      map: {name: map, readings: [map, map_50, map_75, map_small, map_medium, map_large]}
```

`map_per_class` and `mar_100_per_class` are the two that torchmetrics only
computes when told to, so naming either turns on `class_metrics` — derived from
the request, never restated in config. They land as per-class leaves named by the
classes they are *about*: COCO reports only the classes that appeared, so the
naming follows the metric's own `classes` tensor rather than position.

A reading its data could not support returns `-1`, which is not a measurement.
Those are dropped rather than logged, so a chart is not dragged and a checkpoint
monitor cannot rank one.

The backend is `faster_coco_eval`, named explicitly because torchmetrics reaches
for `pycocotools` by default and raises for it even where the other is installed.

Available readings: `map`, `map_50`, `map_75`, `map_small`, `map_medium`,
`map_large`, `mar_1`, `mar_10`, `mar_100`, `mar_small`, `mar_medium`,
`mar_large`, `map_per_class`, `mar_100_per_class`. A misspelt one is refused
while the run is being assembled, with the list.

## Decoding

`confidence_threshold` and `iou_threshold` govern non-maximum suppression:

```yaml
model:
  name: yolo
  model_name: yolov8n.pt
  confidence_threshold: 0.25
  iou_threshold: 0.7
```

What survives becomes `Instances`, with empty images keeping their place — an
image that found nothing still exists, and a consumer counting per image would
otherwise read the next image's objects as its own.

**Training steps produce no predictions.** A detection head emits feature maps
while training and only assembles the decodable tensor in eval; ultralytics does
not spend the decode on a step nobody reads, and its own trainer measures on a
separate validation pass for the same reason. So a train-stage mAP is not
reported rather than reported as zero, which would look like a broken model
instead of a measurement nobody took.

## What a vendor family refuses

Most of what an experiment can declare has nothing to attach to here, and each is
refused at assembly with the sentence that explains it — an hour of training is a
long way to carry a section that was never going to apply:

| Section | Why |
|---|---|
| `transforms` | The vendor augments through its own box-aware pipeline; ours is not. Put `mosaic`, `hsv_h`, `degrees` in the model section |
| `export` | The output is not the per-task logits an exported graph is traced from |
| `adapters` | LoRA reparameterizes a backbone this framework composed, and a vendor brings its own |
| `distillation` | The soft term compares per-task logits, which a vendor family does not expose |
| the `batch_transform` callback | It blends targets, and these are objects rather than tensors |
| a second task | The head is built for one; the others would train nothing and report nothing |
| `head` / `loss` / `target_encoder` on the task | The vendor's assigner, loss and decoding are one design |

A section silently ignored is worse than a run that dies: it reports numbers for
a recipe nobody ran, and the difference only shows when somebody tries to
reproduce it.

## Segmentation and pose

`YOLO(model_name)` already picks the right network for a `-seg` or `-pose` file,
and the model class does not branch on which. What is missing is downstream: an
`instances` topology carrying masks or keypoints, a metric that compares them,
and an annotator that draws them. Until those land, a `-seg` file trains but has
nothing to report — see [the backlog](../backlog.md).
