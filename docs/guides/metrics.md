# Metrics

How a task is evaluated. The short answer is usually: not at all — every *kind of
task* carries a default set, and `metrics:` is an override.

## Where defaults come from

Two levels, and the upper replaces the lower — declared metrics always win:

1. Declared `metrics:` — always wins, and replaces rather than merges, so a
   set can be narrowed.
2. The *kind of task*: the kind carries the metrics customary for it —
   `classification` brings `f1`/`precision`/`recall` (per class) plus a
   `confusion_matrix`, `regression` brings `mae`, `segmentation` adds `iou`
   to that set; metric-learning kinds bring none.

The kind's word is filled in when the run is built, not when the config
loads: config reads only `core`, and the kinds live in `tasks/`. A loaded
experiment therefore shows `metrics: null` where nothing was declared, and the
run says what it filled in — `Task 'label': metrics from kind 'classification':
f1, precision, recall, confusion_matrix`.

The kinds themselves — what each one is, and what else a task declares — are in
[tasks and kinds](tasks.md).

## Adding your own kind

A kind is a class, and its default metrics are one attribute on it:

```python
from src.tasks import Regression


class Depth(Regression):
    default_metrics = {"mae": {"name": "mae"}, "rmse": {"_target_": "torchmetrics.MeanSquaredError", "squared": False}}
```

Each entry goes through the same metric grammar as a hand-written declaration when
the run is built, so a malformed default fails before any data is read, not an
hour into training. See [a kind of task](extending.md#a-kind-of-task) for the rest
a kind can state.

## One rule

The key is the *label* — where the metric logs (`val/{task}/{label}`). The
value says *what* the metric is, always explicitly: `name` for a registry key,
`_target_` for an import path. The key never stands in for the name, so there
is exactly one way to read every entry:

```yaml
tasks:
  label:
    kind: classification
    target: label
    metrics:
      accuracy: {name: accuracy}
      f1: {name: f1, average: macro}
```

A mapping rather than a list on purpose: the label is an identity the logs
consume, and a mapping makes duplicate labels — two metrics silently
overwriting one log line — unrepresentable. Order carries no meaning here,
unlike `callbacks:`, where it does.

## Per-class and matrix values

A metric does not have to compute one number. With `average: none` a
classification metric returns a value per class, and the run logs the mean at
`val/{task}/{label}/mean` plus one scalar per class under its declared name;
a `confusion_matrix` draws on backends that can (see the
[logging guide](logging.md)). Class names come from the task's `classes:`
declaration or the fitted vocabulary.

A per-class value carries **which classes it is about**, and it is not always all of
them: COCO's `map_per_class` covers only the classes that appeared in the split. Naming
by position would then log one class's number under another's name — silently, with
nothing in the chart to notice. A dense reading is simply the case where the classes are
`0..n-1`, so there is one naming path rather than two that have to agree.

## A metric that computes several numbers at once

Some metrics produce a *family* in one pass — mean average precision computes fifteen
readings together, and computing three of them separately would cost three passes over
the validation set. A family is a namespace rather than a new shape: each reading is
logged under the entry's label and placed by its own geometry, so the scalars share one
graph and a per-class member gets its own.

```yaml
    metrics:
      map: {name: map}          # map, map_50, map_75 — three series on one graph
```

```
val/boxes/map                     val/boxes/map/map_per_class
 ├ map      0.90                   ├ cat    1.00
 ├ map_50   1.00                   ├ truck  0.80
 └ map_75   1.00                   └ mean   0.90
```

Which readings are published is chosen, because fifteen series from the first run is not
a report. `map` publishes `map`, `map_50` and `map_75`; name `readings` to widen it:

```yaml
    metrics:
      map:
        name: map
        readings: [map, map_50, map_75, map_small, map_medium, map_large, map_per_class]
```

Asking for more costs nothing extra — they come from the same pass — with one exception
that pays for itself: a per-class reading turns on torchmetrics' `class_metrics`, which
is derived from the request rather than restated in config, so a run that wants none does
not pay for one. A reading whose data could not support it (`map_large` in a split with
no large objects) is dropped rather than logged: torchmetrics returns `-1` there, and
mean average precision lives in `[0, 1]`, so that is a sentinel and never a measurement.

Every published reading is a key of its own, so a checkpoint may watch one:

```yaml
callbacks:
  - {name: checkpoint, monitor: val/boxes/map/map_50, mode: max, dirpath: "${run.directory}/checkpoints"}
```

## Two flavours of one metric

Because the label is only a key, the same metric may appear under several
labels, each with its own arguments:

```yaml
    metrics:
      top1: {name: accuracy, top_k: 1}
      top5: {name: accuracy, top_k: 5}
      f1_macro: {name: f1, average: macro}
      f1_micro: {name: f1, average: micro}
```

The run then logs `val/label/top1` and `val/label/top5` side by side.

## Sizing comes from the data, and only where it fits

The kind knows its facts — the torchmetrics `task` mode, `num_classes` /
`num_labels` from the fitted vocabulary — and a metric receives the ones it
names in its signature (the one place facts are matched by name, kept for
torchmetrics' constructors). Nothing is forced, so a metric that compares plain
numbers stands beside one that ranks classes without being handed arguments it
would refuse; and none of them is written in config, where a copy could only
disagree — `num_classes: 7` on a metric is refused by name:

```yaml
    metrics:
      accuracy: {name: accuracy}   # receives task="multiclass", num_classes=N — it names them
      mae: {name: mae}             # names neither, receives neither
```

Registry names: `accuracy`, `f1`, `precision`, `recall`, `iou`, `mae`, `mse`,
`confusion_matrix`, `precision_recall_curve`, `roc`.

## Any metric by import path

The registry is a convenience, not a gate. Anything with a torchmetrics-style
`update`/`compute` is reachable, and it too receives the facts it names:

```yaml
    metrics:
      calibration: {_target_: torchmetrics.CalibrationError, n_bins: 10}   # task and num_classes arrive
      rmse: {_target_: torchmetrics.MeanSquaredError, squared: false}
```

## What is validated when

An entry that does not say which metric it is — or says it twice, with both a
`name` and a `_target_` — is refused when the config loads. An unknown
registry name is refused at build time, listing the known ones.
