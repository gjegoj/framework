# Losses

How a task's criterion is declared. The short answer is usually: not at all —
every objective carries a default (`multiclass` → cross-entropy, `continuous` →
mse, `metric` → InfoNCE), and `loss:` is an override.

## One grammar, with weights

A loss is a component: a registry name or an import path, plus constructor
arguments. One shape whether it stands alone or in a list:

```yaml
tasks:
  label:
    preset: classification
    target: label
    loss: cross_entropy                          # bare name
    # loss: {name: cross_entropy, label_smoothing: 0.1}
    # loss: {name: focal, gamma: 2.0, weight: 0.5}
```

A list declares several criteria on the same output, added with their weights
and logged term by term — `val/label/ce`, `val/label/focal`:

```yaml
    loss:
      - {name: cross_entropy}
      - {name: focal, gamma: 2.0, weight: 0.5}
```

The registry: `cross_entropy` (`ce`), `bce`, `focal`, `mse`, `mae`, `huber`,
`smooth_l1`, `dice`, `iou`, `tversky`, `expectation`, `infonce`, `siglip`,
`triplet`, `margin_ranking`, `ranknet`, `kl_divergence` (`kl`), `arcface`,
`arcface_proxy`. Anything else is an import path:

```yaml
    loss: {_target_: my_pkg.MyCriterion, alpha: 0.3}
```

## What is never written: derived values

Facts only assembly knows are offered to every loss, and a loss receives the
ones it names. Nothing here is repeated in config, so a change of backbone or
dataset cannot desynchronize it:

- `num_classes` — from the fitted label vocabulary,
- `class_values` — the number each bin stands for, from a binned encoder,
- `embedding_dim` — the width of the stream the task reads, from the backbone.

That is why `loss: {name: arcface_proxy}` is a complete declaration.

## Slots: a criterion inside a criterion

Some criteria take another one as a part. The slot is filled with `_target_`;
leaving it out builds the default, and the remaining arguments configure that
default:

```yaml
    # binned regression: cross-entropy shapes the bins, expectation aligns the number
    loss:
      - {name: cross_entropy}
      - name: expectation
        weight: 0.5
        distance: {_target_: src.losses.HuberCriterion, delta: 0.1}
```

Whatever compares the numbers, the term logs as `expectation` — the metric
inside is its own detail.

## Annealing a loss knob

Focal `gamma`, `label_smoothing`, a distillation temperature — plain numbers on
the loss are schedulable by the `anneal` callback, found by name wherever they
live:

```yaml
callbacks:
  - {name: anneal, task: label, parameter: gamma, start: 0.0, end: 2.0, over: 0.5}
```

See [callbacks.md](callbacks.md) for the schedule vocabulary.

## ArcFace: two deliverables, two declarations

The export boundary decides where the class prototypes live.

**An embedder** (faces, retrieval): prototypes are training scaffolding, held
inside the criterion and discarded at export. The head stays an identity over
the embedding stream:

```yaml
tasks:
  person:
    preset: metric_learning
    target: person_id
    target_encoder: {name: label}
    loss: {name: arcface_proxy, margin: 0.3}     # num_classes/embedding_dim are derived
    lr: 1.0e-2                                   # this task's bricks learn faster than the backbone
```

A task's `lr` moves its own bricks — the head and the criterion, prototypes
included — while the backbone keeps the optimizer's rate. Every task becomes a
named optimizer group whether or not it declares one, so `lr_monitor` draws each
by task name beside the backbone's; see
[per-task learning rates](training.md#per-task-learning-rates).

**A classifier** with ArcFace geometry: prototypes are the classifier, so they
live in a `cosine` head and deploy with the model; `arcface` is then only the
training-time margin over its logits:

```yaml
tasks:
  person:
    preset: classification
    target: person_id
    head: {name: cosine}                          # sizes derived; add embedding_dim: 512 to project first
    loss: {name: arcface, margin: 0.3}
```

At inference the margin is gone in both cases — argmax over cosines is the
prediction, the embedding is the embedding.

## Ranking: which of two items comes first

The carrier is a stacked pair — both items through the shared backbone — and
the target is a per-pair *number*, encoded by the `scalar` encoder:

```yaml
tasks:
  prefers:
    preset: ranking
    target: preference
    target_encoder: {name: scalar}
    loss: {name: margin_ranking, margin: 0.5}     # target is +1 / -1
    # loss: {name: ranknet}                       # target is a probability; 0.5 declares a tie
```

`margin_ranking` is the hinge — a hard gap or nothing; `ranknet` is the smooth
logistic form and accepts graded preferences. A single-output head scores each
item directly (sign preserved); a wider embedding scores by its norm.

## Distillation: a teacher's logits as the target

`kl_divergence` compares the student's distribution with a teacher's,
temperature-softened and `T²`-scaled so it adds honestly to a hard loss:

```yaml
    loss:
      - {name: cross_entropy}
      - {name: kl_divergence, temperature: 4.0, weight: 0.7}
```

The target must be the teacher's logits, not a class label — precomputed and
fed as this task's target today; produced online by a teacher-wrapping model
once that decorator family lands.

## Writing your own

Math that ends in one tensor-in → tensor-out module is a `WrappedCriterion`
subclass: build the module, hand it over, name the logged part. Declare only
parameters needing conversion; forward the rest via `**kwargs`:

```python
@criterion_registry.register("poly")
class PolyCriterion(WrappedCriterion):
    part_name: ClassVar[str] = "poly"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(PolyLoss(**kwargs))
```

A criterion that composes *other criteria* (a distance slot, a weighted sum)
subclasses `Criterion` directly — its children already return `Loss`.
