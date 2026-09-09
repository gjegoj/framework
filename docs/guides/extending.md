# Extending the framework

Every replaceable piece is a registry key. This page is the map of the extension
points; each topic guide carries the depth — how a metric's value is *presented*
is in [logging](logging.md), what an annotator draws is in
[the samples grid](visualization.md).

There are two ways to reach your own code, and the choice is only about whether
the thing is registered:

```yaml
loss: {name: focal_tversky, alpha: 0.7}              # a registry key — short, discoverable
loss: {_target_: my_pkg.losses.FocalTversky, alpha: 0.7}   # an import path — nothing to register
```

Both reach the same constructor by the same path, and every key beyond
`name` / `_target_` becomes a constructor argument. `_target_` needs no
registration at all, which makes it the right answer for a one-off; a registry
key is worth it when several configs will name the thing. What it builds still
has to be the kind of thing the slot takes: a raw torch loss under `loss:` is
refused at build time, because a criterion returns a `Loss` with a named part and
`nn.MSELoss` returns a tensor — wrap it (see [losses](losses.md)).

> A **nested** component builds from `_target_` only. A nested position has no
> registry context, so the short form has nothing to look the name up in.

## The registries

One per capability, in `<package>/registry.py`, named `<singular>_registry`:

| Registry | Module | Holds |
|---|---|---|
| `criterion_registry` | `losses` | Losses, keyed by the part they log under |
| `metric_registry` | `metrics` | Metrics under DS names — torchmetrics' own class where the value is a number, one of ours where it is an artifact |
| `backbone_registry` | `models` | Feature producers the framework composes with |
| `head_registry` | `models` | Kinds of head |
| `adapter_registry` | `models` | Reparameterizations of a built model (LoRA) |
| `task_kind_registry` | `tasks` | The familiar kinds of task, by the name config spells them |
| `table_source_registry`, `input_loader_registry`, `target_encoder_registry`, `cache_registry` | `data` | The data pipeline's replaceable parts |
| `callback_registry` | `callbacks` | What a run does around its steps |
| `logger_registry` | `loggers` | Experiment trackers |
| `optimizer_registry`, `scheduler_registry`, `profiler_registry` | `training` | torch's and Lightning's, by name |
| `exporter_registry` | `export` | Deployment formats |
| `annotation_objective_registry`, `annotation_topology_registry` | `visualization` | How a task's outputs are drawn |
| `label_renderer_registry`, `media_renderer_registry` | `visualization` | How one kind of label or medium becomes HTML — keyed by the entity's *type*, so data chooses, config never does |
| `distribution_reporter_registry` | `callbacks` — declared in `dataset_summary.py` itself, because the package's `registry.py` imports that module and the shared home would be a cycle | How one kind of distribution is reported, keyed by the entity's *type* |

Our own components register by decorator at their definition; third-party classes
are registered explicitly in that `registry.py`, because they are not ours to
decorate. A registry is a convenience, not a gate — anything upstream offers is
reachable by `_target_` without being registered first.

**A registry holds what a declaration names**, which is a smaller set than "every
implementation of the port". Everything a registered class needs comes from its own
declaration — values and nested components filled with `_target_` — plus the facts
the piece of the composition root building it passes explicitly (a head's sizes, a criterion's
`sized` facts). What a declaration only *implies* has no name to be registered under:
`CompositeModel` is what `model:` naming a **backbone** implies, `DistilledModel` is what
`distillation:` being present implies, and `WeightedSumCriterion` is what `loss:` being a
**list** implies. All three are `Model`s or `Criterion`s; none is registered.

The line is not whether the constructor takes a built object — `ExpectationCriterion`
takes a whole `Criterion` in its `distance` slot and *is* registered, because the user
wrote that slot. It is whether a name in the declaration builds it, or the composition root does
from the declaration's shape.

### Registries are not the only way something is chosen

A registry answers one question: *which component does this config name?* — a string
to a factory. Reading the codebase you will meet three other mechanisms, and it is
worth knowing they answer different questions rather than the same one four ways:

| Mechanism | Where | The question it answers |
|---|---|---|
| `Registry` | 25 of them, `<package>/registry.py` | Which component does this **key** mean? Usually a config name; `visualization`'s renderer registries and `dataset_summary`'s reporter registry key by entity *type* — data chooses, config never does |
| `match` over reading kinds | `visualization/annotators.py` | Which labeller of this drawer draws that kind of reading? |
| `match` with shape guards | `loggers/report.py` | What is this value's **geometry** — which is type *and* shape (a 2-D tensor is not a scalar one), and so not expressible as type dispatch |

Extending the framework almost always means adding to a registry. The other three are
internal, and each is where it is because the question it answers is not "which key".

There is deliberately no fourth mechanism for *what a metric's value means*: a metric
says it by returning an artifact from `compute`. A table keyed on the third-party metric
hierarchy used to answer that, and it had to be talked out of claiming values case by
case, because torchmetrics subclasses for state reuse while changing what `compute`
returns.

Config-facing components all share one declaration shape, `ComponentConfig`, and you
will meet it under a dozen aliases — `HeadConfig`, `CallbackConfig`, `LoggerConfig`,
`MetricConfig` and the rest. They are the same class: the alias exists to carry that
position's documentation. One with two or more users lives in `config/components.py`;
one with a single user lives beside that user.

## A loss

Math that ends in one tensor-in → tensor-out module is a `WrappedCriterion`:

```python
# my_pkg/losses.py
from typing import ClassVar

from src.losses import WrappedCriterion
from src.losses.registry import criterion_registry


@criterion_registry.register("focal_tversky")
class FocalTverskyCriterion(WrappedCriterion):
    """Tversky with a focal exponent, for very thin structures."""

    part_name: ClassVar[str] = "focal_tversky"

    def __init__(self, gamma: float = 0.75, **kwargs: Any) -> None:
        super().__init__(FocalTverskyLoss(gamma=gamma, **kwargs))
```

```yaml
tasks:
  mask:
    kind: segmentation
    target: mask_path
    loss: {name: focal_tversky, gamma: 0.5}
```

`part_name` is what the value logs under — `train/mask/focal_tversky` — never an
inline string. A criterion that composes *other criteria* subclasses `Criterion`
directly instead, because its children already return `Loss`.

Declare explicitly only the parameters that need conversion (a YAML list into a
tensor) or a framework default; forward the rest through `**kwargs`, so every
upstream knob stays reachable without touching your wrapper again.

## A metric

Any `torchmetrics.Metric` qualifies:

```python
from src.metrics.registry import metric_registry

metric_registry.register("kappa")(CohenKappa)
```

```yaml
tasks:
  label:
    kind: classification
    target: species
    metrics:
      kappa: {name: kappa, weights: quadratic}
```

The key (`kappa:`) is the label it logs under, and the value names the metric —
so two flavours of one metric can stand side by side:

```yaml
    metrics:
      f1_macro: {name: f1, average: macro}
      f1_per_class: {name: f1, average: none}
```

The kind's own arguments (`task`, `num_classes` / `num_labels`) reach the
metrics that name them in their signature — the one exception kept for
torchmetrics' constructors — so `mae` beside `accuracy` is not handed a `task`
it would refuse, and `num_classes` written on a metric is refused by name.

That one line is the whole of it for a metric computing a **number**. One returning
something else — a curve, a matrix — says what its value *means* by wrapping the metric
in a class of ours; see [a metric that draws](logging.md#a-metric-that-draws).

## A callback

Lightning's `Callback` is the port, so there is nothing to wrap:

```python
import lightning as L

from src.callbacks.registry import callback_registry


@callback_registry.register("gradient_clip")
class GradientClip(L.Callback):
    """Clip the global gradient norm before each optimizer step.

    Parameters:
        max_norm (float): The norm gradients are scaled down to.
    """

    def __init__(self, max_norm: float = 1.0) -> None:
        if max_norm <= 0:
            raise ValueError(f"max_norm must be positive, got {max_norm}.")
        self._max_norm = max_norm

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        torch.nn.utils.clip_grad_norm_(pl_module.parameters(), self._max_norm)
```

```yaml
callbacks:
  - name: gradient_clip
    max_norm: 0.5
```

Callbacks are a **list**, because order is the semantics: one that changes the
weights belongs before one that saves them.

A callback that needs the run's tasks reads them off the module in `setup`, the
way any Lightning callback reads the module — nothing is handed to it by name at
build time:

```python
def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
    self._tasks = tuple(pl_module.tasks)  # TrainingModule publishes them
```

A value config already holds reaches a callback by interpolation on the
declaration — `${lr}`, `${epochs}`, `${mean}`, `${run.directory}` — so one
declaration serves every reader.

## A backbone

```python
from collections.abc import Mapping

from src.core.entities import Features
from src.core.ports import Backbone
from src.models.registry import backbone_registry


@backbone_registry.register("my_encoder")
class MyEncoder(Backbone):
    """..."""

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        return Features(streams={Stream.FEATURES: self._net(inputs[Modality.IMAGE])})

    def feature_dims(self) -> Mapping[str, int]:
        return {Stream.FEATURES: self._width}
```

`feature_dims` declares every stream this backbone offers and how wide each is;
the port turns it into `feature_dim(stream)`, which is what sizes a head — so
nothing about the class count is written in config, and a task asking for a
stream you do not have is refused by name, listing the ones you do. Override
`native_head` to expose the source library's own head, and `architecture` to say
what a run should be filed under in a tracker.

## A model that arrives whole

A model that owns its head, loss and decoding implements the `Model` port and is reached
by `_target_` — no registry to join, no edit to `build.py`:

```python
class DetrModel(Model):
    def step(self, batch: Batch) -> StepResult: ...
    def predict(self, batch: Batch) -> Prediction: ...
```

```yaml
model: {_target_: my_pkg.DetrModel, num_queries: 100}
```

`build_model` takes it as it is, skipping the composite family's head-and-criterion
building; the sections that need a composed model (`adapters`, `distillation`) refuse
it by name. See [models](models.md#a-model-that-arrives-whole).

## A kind of task

A kind is a class, not a table entry. Subclass the shipped kind closest to yours
and override what differs; the pieces kinds are made of — `MulticlassLabels`,
`BinaryLabels`, `MultilabelLabels` for the label semantics, `DenseOutput` for the
per-pixel shape — are importable too:

```python
from src.tasks import DenseOutput, Regression


class Depth(DenseOutput, Regression):
    """A number per pixel: the dense shape over regression semantics."""

    default_encoder = "depth"  # an encoder you register, as below
    default_metrics = {"mae": {"name": "mae"}}
    not_drawn = "a field of numbers has no label kind in the page yet"
```

```yaml
tasks:
  depth: {kind: {_target_: my_pkg.Depth}, target: depth_map}
```

Everything a kind can state is an attribute or a method of `TaskKind`
(`src/tasks/kinds.py`): the encoder its target starts from, the streams its head
reads and the head itself, `loss`, `activation`, `target_adapter`, the facts it
offers metrics (`metric_kwargs`), its `default_metrics`, whether a batch transform
may soften its target (`mixable`), and how a sample of it is read and drawn
(`reader`, `drawer`). A name in the registry is one line more —
`task_kind_registry.register("depth")` — after which `kind: depth` works too, as
long as the module registering it is imported before the experiment is built. `_target_` needs
no such line.

## A data source, loader or encoder

```python
@table_source_registry.register("parquet")
class ParquetSource(FileSource): ...


@target_encoder_registry.register("rle_mask")
class RleMaskEncoder(TargetEncoder): ...
```

An encoder is the one place that knows what its column holds, so it also answers
`facts()` — what fitting revealed — and `distribution()`, which is what the
dataset report draws. Returning `None` from the latter is a valid answer: the
report then names the task rather than dropping it.

Encoders live in `src/data/encoders/`, one module per family (`label`,
`continuous`, `mask`, `boxes`) over `base.py`; a new family is a new module,
imported from the package's `__init__` so its registration runs with the rest.
An encoder whose values are classes subclasses `VocabularyTargetEncoder` and takes
`classes` as a constructor argument — `data/build.py` reads that off the base, hands
it the task's `classes` and refuses a task that declared none, before any row is read.

## An exporter

```python
@exporter_registry.register("coreml")
class CoreMlExporter(Exporter):
    def export(self, model: DeployableModel, example: tuple[Tensor, ...], destination: Path) -> Path: ...
    def load(self, path: Path) -> Runnable: ...
```

Both halves are required, and `load` is why: an artifact is not an export until
it has been read back and compared against the model it came from. Import the
third-party stack *inside* the methods that need it — a format that only exists
on some platforms must not break importing the package.

## Making the module import

A decorator only runs when its module does. Registering from a package the
framework does not import means importing it yourself once — in a notebook, in a
`conftest.py`, or in the script that calls `build`. If that feels like a
detail to remember, use `_target_` instead: an import path resolves itself.

## Where the depth is

| Extension | The full version |
|---|---|
| A criterion, and criteria that hold other criteria | [Losses — writing your own](losses.md#writing-your-own) |
| What a metric's computed value *means* | [Logging — a metric that draws](logging.md#a-metric-that-draws) |
| A tracker backend and the artifact ports | [Logging — a backend of your own](logging.md#a-backend-of-your-own) |
| An augmentation that rewrites a label | [Transforms — something entirely your own](transforms.md#something-entirely-your-own) |
| How a task's outputs are drawn | [The samples grid — adding an annotator](visualization.md#adding-an-annotator) |
| A deployment format | [Export — adding a format](export.md#adding-a-format) |
