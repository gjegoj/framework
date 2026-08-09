# Tasks and presets

A task is what an experiment learns and how it is judged. Everything else in a
config — the model, the loader, the callbacks — serves the tasks.

```yaml
tasks:
  species:
    preset: classification
    target: species
```

The key (`species`) is the task's name. It is the key its targets arrive under,
the key its losses and metrics log under, and the name of its parameter group on
the learning-rate graph — so choose it the way you would choose a column name.

## The axes behind the preset

There is no `TaskType` enum. A task is a point on two axes, plus the input side:

| Axis | Question | Members |
|---|---|---|
| `topology` | What does one prediction look like? | `global`, `dense`, `multiview`, `multistream`, `instances` |
| `objective` | How do labels supervise it? | `multiclass`, `binary`, `multilabel`, `continuous`, `metric` |

A preset is a familiar name for one point, and it is resolved while the config
loads — no preset survives into the built experiment:

| `preset` | topology × objective | Default metrics |
|---|---|---|
| `classification` | `global × multiclass` | f1, precision, recall (per class), confusion matrix |
| `binary_classification` | `global × binary` | the same |
| `multilabel_classification` | `global × multilabel` | the same |
| `regression` | `global × continuous` | mae |
| `metric_learning` | `global × metric` | — |
| `segmentation` | `dense × multiclass` | iou, plus the classification set |
| `binary_segmentation` | `dense × binary` | the same |
| `multilabel_segmentation` | `dense × multilabel` | the same |
| `contrastive` | `multistream × metric` | — |
| `ranking` | `multiview × metric` | — |
| `detection` | `instances × multiclass` | map |

`segmentation` names the *semantic* kind; an instance variant would land under
`instance_segmentation` rather than competing for the name.

A pair with no preset is written out:

```yaml
tasks:
  defect:
    topology: dense
    objective: multilabel      # overlapping classes per pixel
    target: mask_path
```

Declaring both a `preset` and an axis is a config error, not a preference.

Presets whose entry is `—` above are structure-supervised: supervision comes from
the batch's shape (pairs, triplets, the in-batch diagonal) rather than from a
per-sample label, so there is nothing for a per-sample metric to compare.

## What a task declares

| Key | Default | Meaning |
|---|---|---|
| `target` | — | The table column holding this task's ground truth. The data schema derives from the tasks, so a column is named once |
| `classes` | learned | `{0: cat, 1: dog}` — the declared vocabulary, as the source of truth |
| `target_encoder` | from the objective | How a target cell becomes a tensor |
| `loss` | from the objective | One criterion, or a list added with weights |
| `head` | from the topology | Which *kind* of head; sizes stay derived |
| `native_head` | `false` | Keep the pretrained model's own head instead |
| `stream` | from the topology | Which backbone output the head reads |
| `weight` | `1.0` | This task's share of the total loss |
| `lr` | the run's rate | Own rate for this task's head and criterion |
| `metrics` | from the objective | Metrics keyed by the label they log under |

Sizes are never among them. `num_classes` comes from the fitted encoder,
`in_features` from the backbone stream — see [derived values](../concepts.md#sizes-come-from-the-data-never-from-config).

## Declaring the class vocabulary

`classes` turns the class space from something learned into something declared:

```yaml
tasks:
  species:
    preset: classification
    target: species
    classes: {0: cat, 1: dog, 2: rabbit}
```

Three things follow. The data is validated against it at fit, so a typo in a
label is an error rather than a silent extra class. The index space survives
resampling — dropping every `rabbit` row from the train split no longer shifts
`dog` to index 2. And the names label per-class log keys
(`val/species/f1/rabbit`), confusion-matrix axes and the samples grid.

Indices must be exactly `0..n-1` and names must be unique; a continuous objective
refuses `classes` outright, because bins own its value space.

## Several tasks at once

Tasks are a dict, so uniqueness comes free and every task is named:

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    classes: {0: background, 1: defect}
    loss:
      - {name: cross_entropy, weight: 1.0}
      - {name: dice, weight: 1.0}
  label:
    preset: classification
    target: is_defective
    stream: encoder          # read the encoder, not the decoder
    weight: 0.3
    lr: 5.0e-4
```

One backbone encodes the batch once; each task's head reads the stream it names.
`weight` scales a task's contribution to the total loss; `lr` gives its own
bricks a different pace while the backbone keeps the optimizer's — see
[per-task rates](training.md#per-task-learning-rates).

## Overriding the head

The topology picks the head kind, and an override names a kind only:

```yaml
tasks:
  person:
    preset: classification
    target: person_id
    head: {name: cosine}          # learnable prototypes, cosine logits
    loss: {name: arcface, margin: 0.3}
```

`native_head: true` is the other direction: keep the head the pretrained model
ships with, which is what you want when those weights are the point. Declaring
both `head` and `native_head` is refused — they answer the same question.

## Binned regression

A continuous target can be learned as a distribution over bins without leaving
regression semantics. Choosing the encoder is the whole change:

```yaml
tasks:
  score:
    preset: regression
    target: score
    target_encoder: {name: gaussian_bins, bins: 20}
```

The bins then size the head, cross-entropy plus an expectation term replace mean
squared error, and predictions are read back as `softmax(logits) · class_values`
so metrics still compare numbers. See [the data guide](data.md#targets) for what
the encoder learns and why the range is padded.

## What is validated when

At **config load**: the preset resolves, `classes` is checked for completeness
and duplicates, `head` and `native_head` cannot both be set, and an unknown key
in the section is an error naming it.

At **assembly**: the topology validates the objective it was paired with, the
metrics are built with the objective's own arguments, and a task declaring bricks
a vendor family builds itself is refused with the reason.

At **fit**: the data is validated against the declared vocabulary.
