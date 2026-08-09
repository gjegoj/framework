# Data

Everything the `data` section can say, with the shortest YAML that says it.
Target encoders are declared on the task that owns the target, so they live at
the end of this guide rather than under `data`.

## The smallest run

```yaml
data:
  source: data/annotations.csv
  inputs:
    image: {column: image_path}
  split: {train: 0.7, val: 0.15, test: 0.15}
```

Three things are implied here and worth knowing:

- the reader is inferred from the extension (`.csv` → `csv`, `.json` → `json`);
- the loader defaults to `image`, this being a vision framework;
- the target encoder follows from each task's preset — see [Targets](#targets).

## Sources

**One file** — `split` divides it into stages:

```yaml
data:
  source: data/annotations.csv
  split: {train: 0.7, val: 0.15, test: 0.15}
```

**One dataset spread over several files** — concatenated, then divided:

```yaml
data:
  source: {path: [data/part1.csv, data/part2.csv]}
  split: {train: 0.7, val: 0.15, test: 0.15}
```

**Several datasets combined** — a list. Each is divided by the same fractions,
so every one is represented in every stage in its own proportion; a small source
cannot land wholly in train by an unlucky draw:

```yaml
data:
  source:
    - data/clean.csv
    - data/synthetic.csv
  split: {train: 0.7, val: 0.15, test: 0.15}
```

**Already divided upstream** — a stage-keyed mapping, and then `split` must be
absent. A competition ships a partition; a temporal or per-patient split is
decided before the data reaches us, and re-cutting it by fractions would undo
the separation it encodes:

```yaml
data:
  source:
    train: data/train.csv
    val: data/val.csv
    test: data/test.csv
```

Declaring only `train` and `val` is allowed, and then the test stage runs on the
validation set — the run finishes and reports rather than dying after the fit. It
says so once, because those `test/*` numbers are computed on the rows the
checkpoint was selected on and are optimistic for that reason:

```
This run declares no test data, so the test stage runs on the validation set —
the same rows the checkpoint was selected on.
```

A `split` says the same thing with a zero share:

```yaml
data:
  split: {train: 0.8, val: 0.2, test: 0}
```

`train` and `val` must stay above zero — encoders are fitted on train, and val is
what the test stage falls back to — but `test: 0` cuts no test rows at all and
takes the same path as an undeclared test source. Note it is a share of *zero*,
not a stage that gets whatever is left: the splitter hands its last stage the
rounding remainder, so nine rows at `0.7 / 0.3 / 0.0` would otherwise go 6/2/1
rather than 6/3, and the "empty" test set would carry one row.

A stage may draw on several sources too:

```yaml
data:
  source:
    train: [data/train_a.csv, data/train_b.csv]
    val: data/val.csv
```

**A reader the extension does not reveal** — declare `format` on the source it
belongs to, so combined datasets may be stored differently from one another:

```yaml
data:
  source:
    - data/clean.csv                        # inferred
    - {path: data/dump.data, format: json}  # declared
```

**Its own augmentation per source** — for combining datasets that need different
handling: a clean set beside a noisy one, a synthetic set that should not be
augmented a second time. A declared pipeline **replaces** the stage's for those
rows (which is what lets a source be augmented *less*, not only more), so it has
to end the way the global one does — in normalisation and a tensor:

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

Stages a source does not mention fall back to the global transform, so a source
that only differs in training says only that.

## A dataset that is not a table

Detection does not arrive as annotation rows. It arrives as a YOLO descriptor beside
`images/` and `labels/` directories, and the descriptor names the class list and one
image directory per stage:

```yaml
data:
  source: data/coco8/data.yaml
  inputs: {}
```

`inputs` is empty because the descriptor names the pictures, and there is no `split`
because it already says which images are which stage. Neither is a special case in the
section: a rule true of a *table* — that it needs at least one input column, and that one
source has to be divided somehow — is stated where the table is built, so a pipeline with
no columns at all is a valid declaration rather than a forbidden one.

Which pipeline reads it follows from `model:`, not from the file's extension — one name
decides the model and the data together, so the two cannot disagree. See
[Vendor families](models.md#vendor-families).

The augmentation is the vendor's too, box-aware and driven from the model section; a
`transforms:` section is refused rather than silently ignored.

→ [Object detection](detection.md) walks one such run from the descriptor to mAP.

## Inputs

The key is the **name the model asks for**; `column` is where the value lives in
the table. They are separate because renaming a column in a CSV should not rename
what a backbone reads:

```yaml
data:
  inputs:
    image: {column: image_path}
```

**Several inputs** — multi-view, stereo, multimodal. Each is loaded on its own
and reaches the batch under its name:

```yaml
data:
  inputs:
    left_image: {column: left_path, loader: {name: image, root: data}}
    right_image: {column: right_path, loader: {name: image, root: data}}
```

**Loader arguments** — the ordinary reason to spell the loader out is a root
path the table's values are relative to:

```yaml
data:
  inputs:
    image: {column: image_path, loader: {name: image, root: data/images}}
```

**A loader that is not ours** — any import path works:

```yaml
data:
  inputs:
    embedding: {column: vector_path, loader: {_target_: my_pkg.read_npy}}
```

Registered loaders: `image`.

## Splitting

```yaml
data:
  split: {train: 0.7, val: 0.15, test: 0.15}
```

Fractions must sum to 1. `seed` (default `42`) is deliberately **separate from
the experiment's seed**: it fixes which samples land in each stage, not what
happens inside a run. Five runs at different experiment seeds must share one test
set, or their metrics are not comparable — so change this one only when the
partition itself should change.

```yaml
data:
  split: {train: 0.7, val: 0.15, test: 0.15, seed: 7}
```

**Keep a distribution steady across stages** — worth setting whenever the target
is imbalanced, since it stops a small validation set from drawing too few rows of
the rare class:

```yaml
data:
  split: {train: 0.7, val: 0.15, test: 0.15, stratify_by: label}
```

How rows are grouped follows from the column's *content*, not its dtype:

| Column | What happens |
|---|---|
| repeating values (`cat`/`dog`, or `0`/`1` as integers) | balanced as classes |
| numeric with more distinct values than `stratify_bins` (default 10) | balanced by quantile |
| cells carrying several labels (`"cat,dog"`, or a list) | balanced one label at a time |

```yaml
data:
  split: {train: 0.7, val: 0.15, test: 0.15, stratify_by: age, stratify_bins: 20}
  # multi-label column separated by something else:
  # split: {..., stratify_by: tags, stratify_separator: "|"}
```

**Keep related rows together** — scans of one patient, frames of one video, crops
of one image. Without it the same patient lands in train and in test, and the
test metric measures memorisation rather than generalisation:

```yaml
data:
  split: {train: 0.7, val: 0.15, test: 0.15, group_by: patient_id}
```

Whole groups move together, so stage sizes approximate the fractions. The
fractions still count **rows**, not groups.

`stratify_by` and `group_by` cannot be combined: balancing classes moves single
rows, keeping groups intact forbids it. Pick the risk that matters more here —
leakage between related rows, or an unbalanced stage.

## Working on a slice

```yaml
data:
  max_samples: 500      # a count
  # max_samples: 0.05   # or a share
```

Rows are drawn at random rather than off the top, since annotation files
routinely arrive grouped by class or ordered by date. The cap applies **per
source**, so combining datasets does not shrink each to a share of it.

## Cache

```yaml
data:
  cache: {name: ram, max_gib: 8, workers: 8}
```

Holds decoded files in RAM so an epoch does not decode what the last one did.
Absent means no cache. It is warmed once in the parent process, before the data
loader forks its workers, and is read-only afterwards — the pixel buffers are
then inherited by every worker instead of decoded in each.

Only arrays are kept, so a loader returning text is simply not cached. When the
budget runs out the remaining files are read from disk each epoch, exactly as
they would be without a cache; an unreadable file is logged and skipped rather
than aborting the run.

Registered caches: `ram`.

## Transforms

Per stage, at the top level of the experiment — they are not part of `data`,
because the same pipeline serves any data section:

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

Every image input and every spatial target of a sample passes through **one**
pipeline call, so a mask is cropped and flipped with the image it belongs to.
`spatial_targets` is never written by hand: it is derived from the encoders.

Anything `albumentations.Compose` accepts is forwarded verbatim — `seed`, `p`,
`bbox_params`, `is_check_shapes`. For augmenting train only, several views of
one input, per-source pipelines and the rest, see
[transforms.md](transforms.md).

## Targets

Declared on the **task**, not under `data` — a target column and its encoder are
declared once, and the data schema is derived from the tasks.

```yaml
tasks:
  label:
    preset: classification
    target: label
```

The encoder follows from the preset, so declaring one is an override:

| Preset | Encoder implied | Column holds |
|---|---|---|
| `classification` | `label` | a class name or index |
| `binary_classification`, `regression` | `scalar` | a number |
| `multilabel_classification` | `multilabel` | `"cat,dog"` or a list |
| `segmentation` | **must be declared** | a mask file path |

A dense target is an image of its own and reading it needs the class count, so
it is the one case config still has to answer for:

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    target_encoder: {name: mask, num_classes: 3, root: data/masks}
```

**Overrides** — a different separator, a pinned class order, bins for a
continuous target learned as a distribution:

```yaml
tasks:
  tags:
    preset: multilabel_classification
    target: tags
    target_encoder: {name: multilabel, separator: "|"}

  quality:
    preset: regression
    target: score
    target_encoder: {name: gaussian_bins, bins: 20}   # or linear_bins
```

Registered encoders: `label`, `multilabel`, `scalar`, `mask`, `gaussian_bins`,
`linear_bins`.

## Everything at once

```yaml
data:
  source:
    - data/clean.csv
    - path: data/noisy.csv
      transforms:
        train: {_target_: src.transforms.AlbumentationsTransform, transforms: [...]}
  inputs:
    image: {column: image_path, loader: {name: image, root: data/images}}
    depth: {column: depth_path, loader: {name: image, root: data/depth}}
  split: {train: 0.7, val: 0.15, test: 0.15, group_by: patient_id, seed: 42}
  cache: {name: ram, max_gib: 8}
  max_samples: 0.5
```

## The dataset report

`callbacks: [{name: dataset_summary}]` prints what a run is about to train on,
before it starts: one table per target, per stage, with the stage's row count as
its `total` row.

Each **encoder describes its own column**, because it owns the vocabulary and the
parsing, and there are two shapes:

- a **class balance** for `label`, `multilabel` and `mask` — counts per class, plus
  each class's share. Counting starts from the declared vocabulary, so a class the
  split never produced appears at zero and is highlighted. That row is usually the
  most useful line in the table.
- a **value spread** for `scalar` and the binned encoders — mean, deviation and the
  five-number summary. `Total` is the stage's row count, and it names the missing
  ones where there are any (`5144 (44 missing)`): a `NaN` is dropped rather than
  propagated, because one missing cell would otherwise turn every statistic into
  `nan`. A second column for the rows that held a number would repeat the first on
  every row of an ordinary column.

An encoder that describes nothing keeps its task in the report, named, with the
reason — rather than disappearing from it.

**A mask is counted in full**, pixel by pixel, because the imbalance a segmentation
loss fights is measured in pixels and no other number stands in for it. Measured:
0.88 ms to decode a mask and bin it, so 3.3 s for a 3680-mask dataset and about
18 s for 20,000 — once, before the first epoch. With `cache` configured these are
the same reads training is about to warm, so the pass costs nothing twice. A pixel
holding a class index the task never declared is refused here, by name, rather than
surfacing as a shape error inside the loss a thousand steps later.

In the tracker the class balance is a grouped bar chart, one series per stage. The
value spread is a box plot — the one artifact the framework draws itself, because
ClearML has no native call for one. Its whiskers are the observed **minimum and
maximum**, not Tukey's 1.5 IQR fences with outlier points: finding outliers needs
the raw values, and holding a whole column in memory for a picture drawn once is
not a trade worth making. A tracker that draws neither simply receives nothing.
