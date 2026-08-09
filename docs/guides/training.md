# Optimizer and scheduler

How fast the weights move, and how that speed changes over the run.

```yaml
lr: 1.0e-3                  # the base rate, at the root, reached by everything as ${lr}
epochs: 10

optimizer: {name: adamw, lr: "${lr}", weight_decay: 1.0e-4}
scheduler: null             # a fixed rate
```

Both are Hydra groups, so the usual way to change either is to swap the whole
file:

```bash
uv run main.py +experiment=examples/classification optimizer=sgd scheduler=onecycle
```

## The optimizer

`adamw`, `adam` and `sgd` are registered under the names a data scientist uses;
anything torch offers is reachable by import path:

```yaml
optimizer: {name: adamw, lr: "${lr}", weight_decay: 1.0e-4, betas: [0.9, 0.95]}
optimizer: {_target_: torch.optim.RAdam, lr: "${lr}"}
```

Every key beyond `name` is a constructor argument, so an upstream knob needs no
change here. Write `lr` as `"${lr}"` rather than a literal — the root value is
what a schedule, a sweep and a log message all read.

## Named parameter groups

The optimizer never receives a flat list of parameters. It receives one group per
task plus one for everything no task claims:

```
backbone   every parameter no task's head or criterion owns
<task>     that task's head and its criterion, one group per task
```

The groups exist whether or not any rate is overridden, because the groups are
what `lr_monitor` draws one line each for. Measured, five AdamW steps through one
group and through three move every parameter to the same value, so the split
costs a run nothing.

A task that owns no parameters at all — a vendor family builds its own head —
gets no group, and a rate declared against it is refused by name rather than
silently ignored.

## Per-task learning rates

A task may set the pace of its own head and criterion; everything else keeps the
optimizer's:

```yaml
lr: 3.0e-4                  # the base rate
tasks:
  label: {preset: classification, target: species, lr: 1.0e-2}
  age:   {preset: regression, target: age}
```

The graph then carries `backbone`, `label` and `age`, with the last two identical
to `backbone` wherever no rate was overridden.

**Schedules that scale** the rate they find — `cosine`, `step`, `plateau` — carry
each group's own rate through untouched.

**Schedules that set** a rate outright — `onecycle`, `cyclic` — take an absolute
number, and a scalar would broadcast over every group and overwrite the declared
rates. So `max_lr` (and `cyclic`'s `base_lr`) is spread across the groups, scaled
by each group's own, and the run says so once at build time:

```
OneCycleLR takes its rate per group: max_lr=[backbone 3.00e-04, label 1.00e-02, age 3.00e-04]
```

Writing `max_lr` as a list instead answers the question yourself, per group, and
is left alone.

## The schedules

| `name` | Clock | What it does |
|---|---|---|
| `cosine` | epoch | One cosine arc over the whole run; `T_max: "${epochs}"` |
| `step` | epoch | Drops the rate by `gamma` every `step_size` epochs |
| `onecycle` | step | Warms up then anneals; `total_steps` is filled from the fit |
| `plateau` | epoch | Reacts to a logged metric instead of counting time |

Beside the constructor's own arguments, a schedule carries four
framework-declared fields that reach Lightning's scheduling policy rather than
the constructor:

| Key | Default | Meaning |
|---|---|---|
| `interval` | `epoch` | Whether the schedule steps per epoch or per batch |
| `frequency` | `1` | Step once every N intervals |
| `monitor` | `null` | The logged metric a reactive schedule watches |
| `strict` | `true` | Fail when the monitored metric is absent, instead of skipping |

`interval: step` is not cosmetic. A step-clocked schedule is *sized* in steps, so
stepping it once per epoch would stretch a run-long schedule over `epochs × ` its
intended length. The shipped `onecycle` group declares it for that reason.

### Fit-time facts

`total_steps`, `steps_per_epoch` and `epochs` only exist once the trainer knows
how long the run is — after gradient accumulation, `drop_last` and any
`limit_*_batches`. A schedule that declares one and leaves it unset is filled
from the fit:

```yaml
scheduler: {name: onecycle, max_lr: ${lr}, interval: step}   # total_steps arrives here
```

`total_steps` wins where a schedule accepts more than one, being the most precise.
A value written in config is left alone.

### Reacting to a metric

```yaml
scheduler:
  name: plateau
  monitor: val/loss
  mode: min
  factor: 0.5
  patience: 3
```

`monitor` is a log key, so it follows the [key grammar](logging.md#the-key-grammar):
`val/loss` is the total validation loss that every run logs, and
`val/species/f1/mean` is one metric of one task. `strict: true` means a misspelt
key fails at fit start rather than quietly never firing.

## Watching the rate

```yaml
callbacks:
  - name: lr_monitor
```

One line per parameter group on a single `lr` graph, named `backbone` and then
one per task. Left without a `logging_interval` it reads on the schedule's own
clock, which is what a step-clocked schedule needs to show its shape at all.

## Trainer knobs

`trainer` is a forward section: declared fields are validated and everything else
reaches `lightning.Trainer` verbatim.

```yaml
trainer:
  max_epochs: ${epochs}
  accelerator: auto
  devices: auto
  default_root_dir: ${run.directory}
  precision: bf16-mixed          # not declared here, forwarded as-is
  accumulate_grad_batches: 4
```

`profiler` is the one exception, declared because Lightning takes an object there
and a forwarded mapping is neither a profiler nor an alias — see
[where the time went](logging.md#where-the-time-went).

## The loader

```yaml
loader:
  batch_size: ${batch_size}
  num_workers: 8
  pin_memory: true
  drop_last: false
```

Also a forward section. `shuffle` and `collate_fn` are refused by name: they are
stage conventions the adapter sets itself — training shuffles, evaluation does
not — and `drop_last` applies to training only, because dropping an incomplete
batch during evaluation would compute metrics on part of the split.
