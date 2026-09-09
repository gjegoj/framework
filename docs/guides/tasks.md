# Tasks and kinds

A task is what an experiment learns and how it is evaluated. Everything else in a
config — the model, the loader, the callbacks — serves the tasks.

```yaml
tasks:
  species:
    kind: classification
    target: species
```

The key (`species`) is the task's name. It is the key its targets arrive under,
the key its losses and metrics log under, and the name of its parameter group on
the learning-rate graph — so choose it the way you would choose a column name.

## The kinds

There is no `TaskType` enum and no table of axes. A task is an instance of one
`TaskKind` class in `src/tasks/kinds.py`, and the class states everything the
framework needs to serve it: which encoder its target starts from, which backbone
streams its head reads and what head that is, its loss, how its logits become
predictions, what it is judged by, and how a sample of it is drawn. These are the
kinds the framework ships, under the names config spells them:

| `kind` | One prediction is | Default encoder | Default metrics |
|---|---|---|---|
| `classification` | one class of N | `label` | f1, precision, recall (per class), confusion matrix |
| `binary_classification` | yes or no | `scalar` | the same |
| `multilabel_classification` | any number of classes | `multilabel` | the same |
| `regression` | a number | `scalar` | mae |
| `metric_learning` | an embedding, judged against class proxies | `label` | — |
| `contrastive` | aligned embeddings of two streams (CLIP-style) | — | — |
| `ranking` | embeddings of N views of one item | — | — |
| `segmentation` | one class per pixel | `mask` | iou, plus the classification set |
| `binary_segmentation` | foreground or not, per pixel | `mask` | the same |
| `multilabel_segmentation` | any number of classes per pixel — declared, refused at build until an encoder produces a multi-hot mask | `mask` | the same |
| `detection` | a set of boxes, each one class of N | `boxes` | map |

Kinds whose entry is `—` are structure-supervised: supervision comes from the
batch's shape (pairs, triplets, the in-batch diagonal) rather than from a
per-sample label, so there is nothing for a per-sample metric to compare and
nothing to draw.

Sharing between kinds is inheritance written in that one file, not a legality
matrix: `Segmentation` takes its label semantics from the same piece as
`Classification` and its shape from the same piece as `BinarySegmentation`, so
"one class per pixel" is a subclass that says so, and an impossible pairing is
simply a class nobody wrote. `segmentation` names the *semantic* kind; an instance
variant would land under `instance_segmentation` rather than competing for the name.

A kind of your own is a subclass, reachable with no edit to the framework:

```python
from src.tasks import Classification


class FocalClassification(Classification):
    default_metrics = {"accuracy": {"name": "accuracy"}}

    def loss(self, facts, width):
        return FocalCriterion(gamma=2.0)
```

```yaml
tasks:
  species: {kind: {_target_: my_pkg.FocalClassification}, target: species, classes: {0: cat, 1: dog}}
```

→ [Extending the framework](extending.md#a-kind-of-task)

## What a task declares

| Key | Default | Meaning |
|---|---|---|
| `kind` | **required** | A name from the table above, or `{_target_: ...}` for a kind of your own |
| `target` | — | The table column holding this task's ground truth. The data schema derives from the tasks, so a column is named once |
| `classes` | **required** where the target is read as classes (`label`, `multilabel`, `mask`, `boxes`); refused where the kind's encoder carries no vocabulary | `{0: cat, 1: dog}` — the vocabulary, index to name |
| `target_encoder` | from the kind | How a target cell becomes a tensor |
| `loss` | from the kind | One criterion, or a list added with weights |
| `head` | from the kind | Which *kind* of head; sizes stay derived |
| `streams` | from the kind, or the backbone's pyramid for detection | Which backbone streams the head reads — one name or a list, in reading order |
| `weight` | `1.0` | This task's share of the total loss |
| `lr` | the run's rate | Own rate for this task's head and criterion |
| `metrics` | from the kind | Metrics keyed by the label they log under; a declared mapping replaces the kind's set whole |

Sizes are never among them. `num_classes` is the length of `classes`,
`in_features` comes from the backbone stream — see [derived values](../concepts.md#sizes-come-from-the-data-never-from-config).

## Declaring the class vocabulary

Every target read as classes — `classification`, `multilabel_classification`,
`segmentation`, `detection` and their variants — declares its vocabulary, and
the build refuses a task that reads one without it, before any row is read:

```yaml
tasks:
  species:
    kind: classification
    target: species
    classes: {0: cat, 1: dog, 2: rabbit}
```

The vocabulary is the index space the model's outputs live in, which is why it is
declared rather than learned from the rows: learned, it would shrink when a sample
cap or a resample dropped a rare class from train, reorder when a class was added,
and a checkpoint keyed on it would stop fitting in silence. Declared, the data is
validated against it at fit (a typo is an error, not an extra class), `dog` keeps
its index when every `rabbit` row is dropped, and the names label per-class log
keys (`val/species/f1/rabbit`), confusion-matrix axes and the samples grid.

Indices must be exactly `0..n-1` and names must be unique; a regression task
refuses `classes`, because its encoder carries no vocabulary and bins own its
value space. For a table with many classes, let the script that knows every name
write the block — `scripts/prepare_pet.py` prints them for the pet table, breeds
included.

## Several tasks at once

Tasks are a dict, so uniqueness comes free and every task is named:

```yaml
tasks:
  mask:
    kind: segmentation
    target: mask_path
    classes: {0: background, 1: defect}
    loss:
      - {name: cross_entropy, weight: 1.0}
      - {name: dice, weight: 1.0}
  label:
    kind: classification
    target: is_defective
    streams: encoder         # read the encoder, not the decoder
    weight: 0.3
    lr: 5.0e-4
```

One backbone encodes the batch once; each task's head reads the stream it names.
`weight` scales a task's contribution to the total loss; `lr` gives its own
components a different pace while the backbone keeps the optimizer's — see
[per-task rates](training.md#per-task-learning-rates).

## Overriding the head

The kind picks the head, and an override names a kind of head only:

```yaml
tasks:
  person:
    kind: classification
    target: person_id
    head: {name: cosine}          # learnable prototypes, cosine logits
    loss: {name: arcface, margin: 0.3}
```

A head that reads several layers names them — `streams: [p4, p5]`, or
`[block7, block11]` on a backbone that calls its levels so; a detection task needs
none of this, its backbone declares the pyramid.

`head: native` is the other direction: keep the head the pretrained model ships
with, which is what you want when those weights are the point. `native` is a reserved
name rather than a registered head — the backbone builds it — and it takes no arguments;
the same key answers both questions, so the two cannot disagree.

## Binned regression

A continuous target can be learned as a distribution over bins without leaving
regression semantics. Choosing the encoder is the whole change:

```yaml
tasks:
  score:
    kind: regression
    target: score
    target_encoder: {name: gaussian_bins, bins: 20}
```

The bins then size the head, cross-entropy plus an expectation term replace mean
squared error, and predictions are read back as `softmax(logits) · class_values`
so metrics still compare numbers. See [the data guide](data.md#targets) for what
the encoder learns and why the range is padded.

## What is validated when

At **config load**: `kind` is present and spelled as a name or an import path,
`classes` is checked for completeness and duplicates, `native` refuses arguments,
and an unknown key in the section is an error naming it — the
retired spellings (`preset`, `output_topology`, `input_topology`, `objective`) by
name, pointing at `kind`.

At **build**: an unknown kind is refused listing the known ones, a task whose
encoder reads a vocabulary is refused by name if it declared no `classes`, a
declared `classes` the kind's encoder cannot carry is refused, the metrics are
built with the kind's own arguments (its defaults where none were declared, and
the substitution is logged).

At **fit**: the data is validated against the declared vocabulary.
