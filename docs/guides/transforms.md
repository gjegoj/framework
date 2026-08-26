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

End with `ToTensorV2`: loaders produce raw values on purpose — a mask has to be
croppable alongside its image — and the pipeline is where they become
model-ready tensors.

Every column is **loaded** before the pipeline and every target **encoded**
after it. That order is what lets an augmentation write a target: it hands the
encoder a raw value — a class name, a number — rather than overwriting one the
encoder already made.

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

Every input and every geometric target travels through a **single** pipeline
call, so the crop taken from the image is the crop taken from its mask, its
boxes, and every other declared image:

```yaml
data:
  inputs:
    image: {column: left_path}
    right_image: {column: right_path}      # both follow the same geometry
```

Nothing in the `transforms` section says so. Each value declares its own
**geometry** where it is read — `image` for light, `mask` for per-pixel labels,
`boxes` for rectangles — and assembly derives the mapping from the loaders and
encoders, so a target cannot silently fall out of step with its image.

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
already declared: `additional_targets` comes from the derived geometries, and
`bbox_params` from the boxes target (its knobs are `min_box_visibility` and
`min_box_area`). `telemetry` is off by default and can be turned back on.

An argument albumentations does not know fails where it belongs — in
albumentations, naming itself — rather than being swallowed here.

## Boxes

A detection task's target declares `boxes` geometry, so the pipeline carries it
without a word in this section. Letterboxing is one ordinary operation — it
scales, pads with YOLO's grey, and moves the boxes with the picture:

```yaml
transforms:
  train:
    _target_: src.transforms.AlbumentationsTransform
    transforms:
      - {_target_: albumentations.LetterBox, size: "${image_size}"}
      - {_target_: albumentations.HorizontalFlip, p: 0.5}
      - {_target_: albumentations.Normalize}
      - {_target_: albumentations.pytorch.ToTensorV2}
    min_box_visibility: 0.3      # a crop that leaves less than this drops the box
```

`min_box_visibility` and `min_box_area` are the two real choices here — when a
cropped box stops being a training signal — and each drops a box *with its class
name*, which is why the names travel in their own field inside the call. Everything
else about `bbox_params` follows from the format (xyxy pixels), so declaring it
is refused rather than allowed to contradict the derived value.

One boxes target per pipeline: albumentationsX does not plumb label fields
through additional targets, so two detection tasks over one image are refused at
construction, naming both.

`keypoint_params` still takes the plain mapping YAML writes (note
`coord_format`, not `format` — albumentationsX renamed it), and a pose *target*
geometry will arrive with the encoder that reads one.

Detection runs on the native YOLO pipeline do **not** use this section at all:
the ultralytics dataset carries its own box-aware augmentation, and `transforms`
is not consulted for it.

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
    target: angle          # a stub column of zeros in the annotation table
    classes: {0: "0", 1: "1", 2: "2", 3: "3"}

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

> **An online target declares its vocabulary.** The stub column holds one value,
> so a fitted vocabulary would hold one class — and the augmentation writes three
> more. What cannot be learned from the table must be declared: `classes:` for a
> categorical target, `low`/`high` for a binned one. Without it the run dies on
> its first batch with `IndexError: Target 2 is out of bounds`.

`label_targets` is what binds a column to the rule; the augmentation itself
knows nothing about column names, which is why the same one serves any task.
Every key declared there is rewritten by every augmentation that has a label
rule, so declare the one the pipeline is about.

**Was this image cropped?** — the same idea for a binary signal. The positive
class comes from the crop applying and the negative from it not applying, so the
column starts as the negative class throughout:

```yaml
tasks:
  crop_flag:
    preset: classification
    target: was_cropped     # a stub column of "intact" throughout
    classes: {0: intact, 1: cropped}

transforms:
  train:
    _target_: src.transforms.AlbumentationsTransform
    label_targets: [was_cropped]
    transforms:
      - _target_: src.transforms.augmentations.RandomBorderCrop
        crop_left: 0.3
        crop_right: 0.3
        min_crop: 0.15         # a two-pixel trim is not worth labelling as cropped
        applied_label: cropped # the raw class name; encoding happens after
        p: 0.5                 # half the samples stay uncropped — that is the other class
      - {_target_: albumentations.Resize, height: 224, width: 224}
      - {_target_: albumentations.Normalize}
      - {_target_: albumentations.pytorch.ToTensorV2}
```

`applied_label` is what the column holds when the crop applies. Write it as the
table writes it — encoding runs after the transforms, so the declared vocabulary
turns `cropped` into its index. There is no need to work out which index a
sorted vocabulary will assign.

**A number the augmentation draws** — the same idea for a continuous signal. The
temperature `MaskedPlanckianJitter` applies becomes the target, and a binned
encoder turns it into a distribution the head can learn — *after* the transforms,
on the drawn value. The range cannot be learned from a stub column, so it is
declared:

Its mask arrives as an auxiliary input ([data — columns the model never
sees](data.md#columns-the-model-never-sees)): the augmentation reads it, the
pipeline keeps it aligned with the image, and it never reaches the batch.

```yaml
data:
  inputs: {image: {column: image_path}}
  auxiliary_inputs: {lesion: {column: mask_path}}

tasks:
  warmth:
    preset: regression
    target: warmth          # a stub column; the augmentation writes the real value
    target_encoder: {name: gaussian_bins, bins: 20, low: 3000, high: 4600}

transforms:
  train:
    _target_: src.transforms.AlbumentationsTransform
    label_targets: [warmth]
    transforms:
      - _target_: src.transforms.augmentations.MaskedPlanckianJitter
        mask_key: lesion          # names which auxiliary input bounds the warmth
        temperature_range: [3400, 4200]
        spread: [200, 1400]       # a pair draws a fresh swing per sample; a number is fixed
        roughness: [0.05, 0.6]    # likewise: one sample fades across, the next is mottled
        tint: 0.1                 # an off-locus cast, fading to nothing at 6500 K
        p: 1.0
      - {_target_: albumentations.Resize, height: 224, width: 224}
      - {_target_: albumentations.Normalize}
      - {_target_: albumentations.pytorch.ToTensorV2}
```

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
