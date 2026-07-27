# Tasks & presets

## Tasks & presets


Tasks are declared as a named dict. The key becomes the task name used in metric logs
(`label/accuracy/val`), loss logs (`loss/val/label`), and per-head LR overrides.

**Classification** (multiclass by default):

```yaml
tasks:
  species:
    preset: classification
    target: species_col
    class_mapping: {0: cat, 1: dog, 2: cow}   # infers num_classes=3
```

**Segmentation**:

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    class_mapping: {0: background, 1: defect, 2: edge}   # infers num_classes=3
```

Or with explicit `num_classes` when class names don't matter:

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    num_classes: 3
```

**Regression**:

```yaml
tasks:
  age:
    preset: regression
    target: age
    dim: 1
```

**Objective override** — same preset, different label semantics:

```yaml
tasks:
  tags:
    preset: classification
    objective: multilabel       # sigmoid + BCE instead of softmax + CE
    target: tags_col
    class_mapping: {0: indoor, 1: outdoor, 2: people}
```

Available objectives: `multiclass` · `multilabel` · `binary` · `continuous` · `metric`
(metric learning — see [Embeddings & metric learning](#embeddings--metric-learning)).

**Custom loss** (registry keys: `cross_entropy` · `bce` · `mse` · `l1` · `dice` · `focal` ·
`weighted_sum` · `kl_divergence` · `arcface` · metric-learning losses `triplet_margin` ·
`margin_ranking` · `ranknet` · `info_nce` · `siglip`):

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    num_classes: 3
    loss:
      name: weighted_sum
      losses: {cross_entropy: 1.0, dice: 2.0}
```

**Custom metrics**:

```yaml
tasks:
  species:
    preset: classification
    target: species_col
    class_mapping: {0: cat, 1: dog, 2: cow}
    metrics:
      accuracy: null
      per_class_f1:
        name: f1
        average: none           # returns [C] vector → logged per class
      confusion_matrix: null
```

**Per-head learning rate** (see [Optimizer, LR & scheduler](training.md)):

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    num_classes: 3
    optimizer:
      lr: 1.0e-4                # this head gets its own param group
```

## Embeddings & metric learning


Metric-learning tasks have no per-sample class label — supervision comes from the
pair/triplet structure or the batch diagonal. The `metric` objective makes the adapter
pass-through and the activation identity; `num_classes` is reinterpreted as the **embedding
dimension** (the projection-head size). The *loss method* is pinned by the preset.

| Preset | Topology | Default loss | Shape of supervision |
|---|---|---|---|
| `triplet` | MULTIVIEW | `triplet_margin` | 3 views: anchor / positive / negative |
| `pairwise_ranking` | MULTIVIEW | `margin_ranking` | 2 views ranked against each other |
| `contrastive` | MULTISTREAM | `info_nce` | N separate encoders aligned (InfoNCE / SigLIP) |

**MULTIVIEW (Siamese)** — N input views go through *one shared backbone* (stacked to
`[B·N, …]`, reshaped to `[B, N, D]`). The view names come from `data.inputs`:

```yaml
data:
  inputs:
    anchor:   anchor_path
    positive: positive_path
    negative: negative_path

tasks:
  embed:
    preset: triplet
    target: anchor_path        # structural; the loss ignores its values
    dim: 128                   # embedding dimension
```

**MULTISTREAM (dual / multi-encoder)** — N *separate* encoders (e.g. image + text), one
named stream each, aligned in a shared space. Use the `multi` backbone whose sub-encoder
names match the `data.inputs` aliases:

```yaml
backbone:
  kind: multi
  encoders:
    image: {kind: timm, name: resnet50}
    text:  {kind: timm, name: ...}      # any registered encoder

tasks:
  align:
    preset: contrastive
    target: image            # structural
    dim: 256
    loss: siglip             # swap info_nce → siglip
```

**Precomputed embeddings** — skip the image encoder entirely with the `embedding`
backbone (the input is a stored feature vector); pair it with `classification` or a metric
preset for ANN/retrieval heads.

> See `configs/experiment/{arcface,contrastive,ranking,embeddings}_smoke.yaml` for runnable
> examples. `arcface` is an angular-margin **loss** you can drop onto a `classification` task.
