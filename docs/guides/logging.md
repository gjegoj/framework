# Logging

What a run writes about itself: scalar curves always, per-class breakdowns,
confusion matrices, and PR curves when the backend can draw them — plus, on
request, a report of where its wall clock went.

## The key grammar

Every logged value is keyed `{stage}/{task}/{leaf}` — `val/label/f1`,
`train/loss` — and `core.log_keys` is the grammar's one owner; nobody
re-splits strings by hand. A tracker shows a key as *title* and *series*, and
the stage splits off as the series: `train/loss` and `val/loss` are two lines
on the one `loss` graph, `val/label/f1` sits beside `train/label/f1` — losses
and metrics alike, no special case.

A metric averaged `none` is the exception, because there the interesting
comparison is between *classes* rather than between stages. Its keys keep the
stage in the title and give the leaf to the series, so each stage gets one graph
carrying every class beside the mean:

A key that starts with no stage at all belongs to no one pass, and what it
compares is its own leaves. It splits at the last separator, so the learning
rates of every parameter group — `lr/backbone`, `lr/label` — arrive as lines on
one `lr` graph. A key with no separator (`epoch`) has no leaf to give and stands
alone.

| Key | Graph | Line |
|---|---|---|
| `train/loss`, `val/loss` | `loss` | `train`, `val` |
| `train/label/ce` | `label/ce` | `train` |
| `val/label/accuracy` | `label/accuracy` | `val` |
| `val/label/f1/cat`, `val/label/f1/mean` | `val/label/f1` | `cat`, `mean` |
| `lr/backbone`, `lr/label` | `lr` | `backbone`, `label` |
| `epoch` | `epoch` | `value` |

The rule turns on the key's shape — a fourth segment exists only for a vector
metric — so nothing has to know what a value means to place it, and a backend
never learns the grammar at all.

The trade is deliberate: a vector metric's mean is on its stage's graph rather
than on one of its own, so train and val means sit side by side instead of
overlaid. Scalar metrics and losses keep their stages together as before.

## Where a value lands

A computed metric is routed by what it turned out to be — nobody enumerates
metric configurations, the same `f1` routes differently under
`average: none`:

| Computed value | Where it goes |
|---|---|
| scalar | the scalar log, as always |
| vector `[C]` | mean at `{key}/mean` + one scalar per class at `{key}/{class}` |
| `Matrix` (confusion matrix) | the backend's matrix plot, axes named, class names on the labels |
| `Curve` (PR, ROC) | one figure, one line per class, axes named |
| raw matrix / raw tuple | a loud warning — unidentified artifacts are never drawn by guesswork |
| anything else | a loud warning naming the key and the geometry |

The rule behind the table: *artifacts are drawn only when identified; raw
geometry feeds only scalars.* **A metric identifies its own value** — its
`compute` returns a `Curve`, a `Matrix` or a `PerClass` with the axes
already named (PR and ROC tuples are mirror images, so orientation is
stated, never guessed). Class names come from the task's vocabulary —
declared `classes` or the fitted encoder — filled only where the artifact
left its labels open. A backend that cannot draw an artifact skips it
quietly and keeps the scalars: a CSV run stays useful.

## A metric that draws

Wrap the metric and say what its value means — the same "wrap a module"
shape as a loss:

```python
from src.core import Curve
from src.metrics import WrappedMetric


class DetMetric(WrappedMetric):
    """Detection error tradeoff, drawn with FPR on x."""

    higher_is_better = None

    def __init__(self, task: str, **kwargs):
        super().__init__(BinaryDET(**kwargs))

    def compute(self) -> Curve:
        fpr, fnr, _ = self.inner.compute()
        return Curve(x=(fpr,), y=(fnr,), xaxis="FPR", yaxis="FNR", positive_only=True)
```

Register it under a name like any other metric; there is nothing else to
declare. A subclass inherits the drawing by ordinary inheritance, and
returning `None` means "identified, draws as nothing" — the value is
dropped quietly (the multilabel confusion matrix, `[L, 2, 2]`, is the
built-in example).

> **Name the facts you need.** `task`, `num_classes` and `num_labels` are
> *offered* by the objective and reach whatever **names** them, so a
> constructor of `**kwargs` alone receives none of them and fails inside the
> upstream library about an argument no config mentioned.
> `ClassificationArtifactMetric` writes that signature once for the three
> shipped curves; subclass it and you inherit it.

A metric that computes a **number** needs none of this: it stays
torchmetrics' own class, registered as it comes. Only a value that has to
say what it *means* becomes a class of ours, which is why `iou` — a
`ConfusionMatrix` subclass upstream — draws nothing and logs as the
per-class vector it is.

## ClearML

```bash
uv add clearml && clearml-init
```

The whole validated config reaches the task as hyperparameters — the values that
actually applied, defaults included — so anything about a run is searchable
without being squeezed into a tag first.

```yaml
run:
  project: my-project                      # the run's identity, declared once
  name: ${now:%Y-%m-%d}/${now:%H-%M-%S}

logger:
  name: clearml                            # project_name/task_name arrive from run.* by
  # project_name: other                    # interpolation; override either right here
```

Or from the CLI: `logger=clearml` — the shipped group already reads
`${run.project}`/`${run.name}`, and the Hydra run directory derives from the
same identity (`runs/<project>/<name>`), so one naming feeds the tracker, the
run tree, and the checkpoints root. One ClearML task carries the whole run —
scalars, matrices, curves, and the connected hyperparameters. Only the names
above are declared; every other `Task.init` knob (`output_uri`,
`auto_connect_frameworks`, …) forwards verbatim, so any upstream option is
reachable without adapter changes. No `logger:` section keeps Lightning's
default logging.

A ClearML task also carries the sample grid: the `samples` callback reports its
page through `log_html`, which lands as media with an `html` extension — what
the Debug Samples panel embeds in place. A tracker without `log_html` gets one
warning at setup and the run proceeds. See [visualization.md](visualization.md).

### Tags

Tags are what the experiments list filters by, so the shipped group names the
*shape* of a run rather than its results:

```yaml
logger:
  tags:
    - ${model.name}                          # the family: timm / smp / multi_encoder
    - ${optimizer.name}
    - ${oc.select:scheduler.name,''}         # the section itself may be null
    - ${oc.select:adapters.name,''}          # lora, when a run trains a delta
    - lr=${lr}
    - bs=${batch_size}
    - epochs=${epochs}
    - imgsz=${image_size.0}x${image_size.1}
```

Each is an interpolation, so a group that is off leaves an empty string behind;
empty and repeated tags are dropped rather than shown as blank chips.

**The architecture is not written here**, and cannot be: the key naming one
differs per backbone family — `model_name` for timm and Hugging Face, `arch`
plus `encoder_name` for smp — and a composite backbone has no such key at all.
So the model is asked instead, through `Model.architecture`, and the answer joins
the tags as a derived fact the way `num_classes` reaches a head.

Each backbone answers in the way that is honest for it, which is why this is not
one rule applied centrally:

| Backbone | Answers | Why |
|---|---|---|
| timm | `resnet18` | timm normalises: a weights tag (`resnet18.a1_in1k`) is dropped |
| smp | `unet-resnet34` | smp calls it `u-resnet34`, which is not what anyone filters by |
| Hugging Face | the hub id | that is the name a model is looked up under |
| multi-encoder | `resnet18+resnet34` | it joins what it holds |
| multi-view | the inner encoder's | views are a way of reading, not another model |

A distilled run is filed under its student, and a family with no backbone under
its own class name.

## A backend of your own

Subclass Lightning's `Logger` and register it:

```python
from src.loggers import logger_registry

@logger_registry.register("wandb")
class WandbLogger(Logger): ...
```

Matrices and curves are structural: implement `log_matrix(title, matrix,
iteration)` / `log_curve(title, curve, iteration)` (entities and protocols
alike in `core.reporting`) and the run's artifacts arrive — no inheritance
beyond Lightning's own base, and a backend without them simply keeps its
scalars. The artifact crosses whole, so a new field on `Curve` or `Matrix`
never changes a port signature.

## Where the time went

```
uv run main.py +experiment=examples/classification trainer=profile
```

The preset inherits the default trainer and adds one thing, so a profiling run
measures the settings a real run trains with — `precision` above all. Timing
every hook is not free, so nothing is measured unless it was asked for.

```yaml
trainer:
  profiler: {name: simple}          # or advanced / pytorch, plus that profiler's own arguments
```

| Name | Answers |
|---|---|
| `simple` | wall clock per Lightning hook — data fetch vs `training_step` vs `backward` vs `optimizer_step`. The question worth asking first |
| `advanced` | a cProfile breakdown inside each hook, for when `simple` names the hook but not the line |
| `pytorch` | per-operator detail with a chrome trace, for when the line is a kernel |

Every argument of the chosen profiler is reachable beside its name
(`{name: advanced, line_count_restriction: 20}`), and anything unregistered —
Lightning's `xla`, or your own — by `_target_`.

**The report is written into the run's directory**, beside the job log and the
weights, one file per stage: `fit-profile.txt`, `test-profile.txt`. The preset
says so itself, the same way a saver does:

```yaml
profiler:
  name: simple
  dirpath: ${run.directory}
  filename: profile
```

Both keys, because `filename` is a switch rather than a name — a profiler writes
a file only when it has the two, and with a directory alone the report goes to
the job log instead. Left to Lightning entirely it would land wherever the
*logger* points: nowhere at all for a tracker that uploads, and in a
`lightning_logs/version_0` subdirectory under the default one.

One thing to know before reading the numbers: **a stage's report includes the
stages before it.** Lightning never clears its recorded durations between
stages, so `test-profile.txt` carries the fit's `run_training_epoch` and counts
it in its own total — measured on a one-epoch run, 673 calls over 2.32 s where
the fit alone was 524 over 1.78 s. Read `fit-profile.txt` for training, and
subtract if you want the test stage on its own.
