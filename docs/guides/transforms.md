# Transforms

What happens to a sample between loading it and handing it to the model.
Declared per stage at the top level of an experiment — not inside `data`,
because the same pipeline serves any data section.

## The usual pipeline

```yaml
transforms:
  train: &pipeline
    _target_: src.transforms.AlbumentationsTransform
    transforms:
      - {_target_: albumentations.Resize, height: 224, width: 224}
      - {_target_: albumentations.Normalize}
      - {_target_: albumentations.pytorch.ToTensorV2}
  val: *pipeline
  test: *pipeline
```

End with `ToTensorV2`: loaders and encoders produce raw values on purpose — a
mask has to be croppable alongside its image — and the pipeline is where they
become model-ready tensors.

A stage left out of the section keeps its samples exactly as the loaders
produced them, which for images means HWC `uint8` and a conv layer that
refuses them. Declare all three, or none.

## Augmenting train only

```yaml
transforms:
  train:
    _target_: src.transforms.AlbumentationsTransform
    transforms:
      - {_target_: albumentations.RandomResizedCrop, size: [224, 224]}
      - {_target_: albumentations.HorizontalFlip, p: 0.5}
      - {_target_: albumentations.ColorJitter, p: 0.3}
      - {_target_: albumentations.Normalize}
      - {_target_: albumentations.pytorch.ToTensorV2}
  val: &plain
    _target_: src.transforms.AlbumentationsTransform
    transforms:
      - {_target_: albumentations.Resize, height: 224, width: 224}
      - {_target_: albumentations.Normalize}
      - {_target_: albumentations.pytorch.ToTensorV2}
  test: *plain
```

## One sampling for the whole sample

Every image input and every spatial target travels through a **single**
pipeline call, so the crop taken from the image is the crop taken from its mask
and from every other declared image:

```yaml
data:
  inputs:
    image: {column: left_path}
    right_image: {column: right_path}

transforms:
  train:
    _target_: src.transforms.AlbumentationsTransform
    image_inputs: [image, right_image]     # both ride the same geometry
    transforms: [...]
```

`spatial_targets` is never written by hand: it is derived from the encoders —
an encoder that marks itself `spatial` (a mask) is registered automatically, so
a mask cannot silently fall out of step with its image.

Inputs that are not declared — embeddings, captions, class labels — pass
through untouched.

## Any `Compose` knob

Anything `albumentations.Compose` accepts is forwarded verbatim, so a knob
needs no change in the framework to be reachable:

```yaml
transforms:
  train:
    _target_: src.transforms.AlbumentationsTransform
    transforms: [...]
    seed: 42                # a reproducible pipeline
    p: 0.8                  # the chance the whole thing applies at all
    is_check_shapes: false
    mask_interpolation: 0
```

Two arguments are refused, because declaring them would contradict what is
already declared: `additional_targets` comes from `image_inputs` and
`spatial_targets`. `telemetry` is off by default and can
be turned back on.

An argument albumentations does not know fails where it belongs — in
albumentations, naming itself — rather than being swallowed here.

## Boxes and keypoints

`bbox_params` and `keypoint_params` take the plain mapping YAML already writes;
no import path is needed:

```yaml
transforms:
  train:
    _target_: src.transforms.AlbumentationsTransform
    transforms:
      - {_target_: albumentations.HorizontalFlip, p: 0.5}
      - {_target_: albumentations.Normalize}
      - {_target_: albumentations.pytorch.ToTensorV2}
    bbox_params: {coord_format: yolo, label_fields: [classes]}
```

Note `coord_format`, not `format` — albumentationsX renamed it.

Detection runs that use the native YOLO pipeline do **not** need this: the
ultralytics dataset carries its own box-aware augmentation (mosaic, HSV,
perspective), and `transforms` is not consulted for it.

## Several views of one input

Contrastive training needs N independently augmented views of the same image.
That is composition, not a flag — wrap the pipeline that should run per view:

```yaml
transforms:
  train:
    _target_: src.transforms.MultiViewTransform
    views: 2
    base: *pipeline
```

Parameters: `views` (how many), `base` (the transform run afresh for each; omit
it to repeat the input unchanged), `input_name` (which input to multiply,
`image` by default). The result is stacked under the same input name, and the
sample's targets are left alone — augmentation never reaches them here.

## Augmentations that create supervision

Some augmentations do not only perturb the image, they write the label too. A
quarter-turn knows how far it turned, so a folder of upright photographs becomes
a balanced four-class rotation task with no file duplicated and nothing
annotated:

```yaml
tasks:
  angle:
    preset: classification
    target: angle          # a column of zeros in the annotation table

transforms:
  train:
    _target_: src.transforms.AlbumentationsTransform
    label_targets: [angle]
    transforms:
      - {_target_: src.transforms.augmentations.Rotate90, p: 1.0}
      - {_target_: albumentations.Resize, height: 224, width: 224}
      - {_target_: albumentations.Normalize}
      - {_target_: albumentations.pytorch.ToTensorV2}
```

`label_targets` is what binds a column to the rule; the augmentation itself
knows nothing about column names, which is why the same one serves any task.
Every key declared there is rewritten by every augmentation that has a label
rule, so declare the one the pipeline is about.

**Was this image cropped?** — the same idea for a binary signal. The positive
class comes from the crop applying and the negative from it not applying, so the
column starts as the negative class throughout:

```yaml
transforms:
  train:
    _target_: src.transforms.AlbumentationsTransform
    label_targets: [was_cropped]
    transforms:
      - _target_: src.transforms.augmentations.RandomBorderCrop
        crop_left: 0.3
        crop_right: 0.3
        min_crop: 0.15      # a two-pixel trim is not worth labelling as cropped
        p: 0.5              # half the samples stay uncropped — that is the other class
      - {_target_: albumentations.Resize, height: 224, width: 224}
      - {_target_: albumentations.Normalize}
      - {_target_: albumentations.pytorch.ToTensorV2}
```

`applied_label` (default 1) is what the label becomes when the crop applies. It
is a parameter because a label encoder sorts its vocabulary: with classes named
`cropped` and `original`, "cropped" is 0.

## Mixing whole samples

MixUp and CutMix combine *different* samples, so they cannot be sample
transforms: while one sample is being loaded, the ones it would mix with do not
exist yet. They run on the collated batch, and a callback is what applies them:

```yaml
callbacks:
  - name: batch_transform
    transform:
      _target_: src.transforms.MixUp
      alpha: 0.4          # larger mixes more evenly; the default 1.0 is uniform
    until: 0.8            # off for the last fifth, so the run ends on clean data
```

The tasks and their class counts are not written here — assembly offers them to
every callback, and this is one of the few that takes them.

One draw moves the image and every task's label together, so a two-head model
cannot blend its heads one way and its picture another. A class index becomes a
distribution over classes — the mixed sample genuinely belongs to two — which
cross-entropy takes as it is. A continuous target is mixed as the number it
already is.

`CutMix` pastes a rectangle instead of blending, so every pixel comes from
exactly one source and the model sees real texture rather than a ghost of two.
Its labels follow the area actually pasted, which clipping at the frame edge may
have shrunk:

```yaml
callbacks:
  - name: batch_transform
    transform: {_target_: src.transforms.CutMix, alpha: 1.0}
```

Both refuse, while the experiment is being assembled, any task they cannot
rewrite: a blended image has no coherent per-pixel target, and metric learning's
proxy and margin losses break on soft labels. Validation and test are never
touched — the hook they listen on fires in training only.

## Stitching four samples

`Mosaic` splits the batch once into a 2×2 grid: every sample keeps its own
top-left quadrant and takes the other three from batch neighbours at the *same*
spatial positions. Nothing is resized, so every pixel keeps exactly one source:

```yaml
callbacks:
  - name: batch_transform
    transform:
      _target_: src.transforms.Mosaic
      split_range: [0.3, 0.7]   # where the split may fall, as a fraction of the side
    until: 0.8
```

That is what makes it the one mixing transform a **segmentation** task can use:
the mask is stitched by the identical swap and stays a valid index map — no
interpolation, no dtype to juggle. A global label instead takes the four
quadrant areas as its weights, so a mask head and a label head can be served by
one split:

```yaml
tasks:
  mask:  {preset: segmentation, target: mask}
  label: {preset: classification, target: label}
```

Widening `split_range` towards 0 and 1 makes lopsided quadrants more likely. A
batch shorter than four wraps around — less variety, but still correct, because
the label weights follow the same wrap.

## A source with its own pipeline

Declared on the source rather than the stage, for combining datasets that need
different handling. It **replaces** the stage's pipeline for those rows — which
is what lets a source be augmented *less*, not only more — so it has to end the
same way:

```yaml
data:
  source:
    - data/clean.csv
    - path: data/noisy.csv
      transforms:
        train:
          _target_: src.transforms.AlbumentationsTransform
          transforms:
            - {_target_: albumentations.Resize, height: 224, width: 224}
            - {_target_: albumentations.GaussNoise, p: 0.5}
            - {_target_: albumentations.Normalize}
            - {_target_: albumentations.pytorch.ToTensorV2}
```

Stages the source does not mention fall back to the stage's own pipeline, so a
source that differs only in training says only that. See
[data.md](data.md#sources).

## Something entirely your own

A transform is any `Sample -> Sample` callable, so the seam takes anything:

```yaml
transforms:
  train: {_target_: my_pkg.MyTransform, strength: 0.4}
```

It receives the whole sample because geometric augmentation is joint: one crop
for the image and its masks alike.
