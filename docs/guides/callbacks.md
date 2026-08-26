# Callbacks

What a run does around its training steps. Declared as a list, and the order is
not cosmetic: a callback that changes the weights belongs before one that saves
them.

```yaml
callbacks:
  - name: lr_monitor
  - name: checkpoint
    monitor: val/label/f1
    mode: max
    dirpath: ${run.directory}/checkpoints
```

`lr_monitor` and `checkpoint` are Lightning's own `LearningRateMonitor` and
`ModelCheckpoint`, registered rather than wrapped — `Callback` is already the
port, so a wrapper would only add a layer to translate across. Every one of
their arguments is reachable by name.

**Write `dirpath` on every saver**, exactly as above. A run has one directory —
the one `export` writes into and the job log sits in — and it is reached the way
`lr` and `epochs` are, by interpolation. Left out, Lightning resolves the
location from the *logger* instead: right for one that writes files, and wrong
for a tracker that uploads, where a run's weights ended up under
`runs/<project>/<name>/<name>/<task id>/checkpoints` — the run's identity spelled
twice plus a hash, three levels below everything else it produced. Every shipped
callback group carries the line, and a test holds them to it.

`lr_monitor` draws one line per parameter group on a single `lr` graph, named:
`backbone` for everything no task claims, then one per task — see
[per-task learning rates](training.md#per-task-learning-rates). Left without a
`logging_interval` it reads on the schedule's own clock, which is what a
step-clocked schedule like `onecycle` needs to show its shape at all.

Anything not in the registry is reachable by import path, the same as a loss or
a transform:

```yaml
callbacks:
  - {_target_: lightning.pytorch.callbacks.EarlyStopping, monitor: val/loss, patience: 5}
```

## Keeping a moving average of the weights

The weights at the last step are not usually the best weights: they carry the
noise of the last few batches. An exponential moving average smooths that out,
and it is what validation reports, what a checkpoint stores, and what training
leaves behind:

```yaml
callbacks:
  - name: ema
    decay: 0.9999    # how much of the average survives each update
    after: 0.1       # train a tenth of the run first, so it starts from something
```

`decay` nearer 1 averages over a longer stretch — 0.9999 suits a long run, 0.99
a short one. `after` is a share of the run rather than a step count, so it
survives a change of epoch count; it is resolved against the run's total steps
when fitting starts.

`device: cpu` keeps the second copy of the weights off the GPU, which matters
for a large model:

```yaml
callbacks:
  - {name: ema, decay: 0.999, device: cpu}
```

### Small checkpoint files

Lightning runs a callback's save hook only for full checkpoints, so with
`checkpoint` and `save_weights_only: true` the file would hold the live weights
while the metric it was chosen by came from the averaged ones. Declaring both
fails when fitting starts. Use `ema_checkpoint` instead — the same
`ModelCheckpoint`, plus lending the model its averaged weights while the file is
written:

```yaml
callbacks:
  - {name: ema, decay: 0.999}
  - name: ema_checkpoint
    monitor: val/label/f1
    mode: max
    save_weights_only: true    # no optimizer state, so the file is far smaller
```

It is the ordinary checkpoint when saving in full, or when no `ema` is declared,
so there is no reason to declare both.

## Annealing a loss parameter

Some loss knobs are meant to move: focal `gamma` easing in, `label_smoothing`
fading out, a distillation temperature cooling. The criterion stays a dumb
component that never sees Lightning — this callback is what knows the epoch:

```yaml
callbacks:
  - name: anneal
    task: label
    parameter: label_smoothing
    start: 0.2
    end: 0.0
    schedule: cosine   # or linear
    over: 0.5          # reach `end` halfway through the run, then hold
```

`start` overrides the constructed value from epoch 0, so the schedule is the
one source of truth while it runs. The value at any epoch is a pure function of
that epoch — a resumed run picks the ramp up exactly where it stood.

The attribute is found by walking the criterion's module tree, so it may live
on the criterion itself or on the torch loss inside it — `label_smoothing` sits
on `nn.CrossEntropyLoss`, and that just works. If several parts of a composite
loss carry the same name, say which one with the part's logging name:
`parameter: ce.label_smoothing` — the same word the run's logs use. A learnable
`nn.Parameter` is refused: scheduling one silently fights the optimizer.

## Holding a backbone still

```yaml
callbacks:
  - name: freeze
    modules: [model.backbone]
    until: 0.3            # held for the first 30% of the run; a whole number is an epoch index
    train_bn: true        # normalisation keeps learning *this* dataset's statistics
```

`modules` are dot-paths from the training module. Built on Lightning's
`BaseFinetuning` rather than on `requires_grad`, because unfreezing has to
return the parameters to the optimizer's groups — the step hand-rolled freezing
usually misses, where the weights thaw but never move. Leave `until` out and
they stay frozen for the whole run — the same word, and the same reading, as a
batch transform's `until`.

## Mixing whole samples

MixUp, CutMix and Mosaic run on the collated batch, applied by a callback that
owns the schedule. See [the transforms guide](transforms.md) for what each one
does to a target:

```yaml
callbacks:
  - name: batch_transform
    transform: {_target_: src.transforms.MixUp, alpha: 0.4}
    until: 0.8      # off for the last fifth, so the run ends on clean data
```

The tasks and their class counts are not written here: assembly offers them to
every callback, and this is one of the few that takes them.

## The final numbers at a glance

After `trainer.test` finishes, `metric_summary` pushes the stage's headline
numbers — the loss, scalar metrics, and each vector metric's `mean`, never
the per-class leaves — to the tracker's summary table (ClearML "Single
Values"):

```yaml
callbacks:
  - name: metric_summary
```

A backend without a summary table is skipped quietly: the numbers are still
in the logs and the plots; this only adds the at-a-glance view where one
exists.

## The model as a tree

`model_summary` replaces Lightning's flat dotted paths with a box-drawing
tree of leaf names — the hierarchy, including the freeze-path modules
(`heads.<task>.base`), reads exactly as a config names it:

```yaml
callbacks:
  - name: model_summary
    max_depth: 3          # Lightning's own knob, forwarded verbatim
```

Only the Name column changes; columns, totals, and the footer are
Lightning's own renderer.

## A live table under the progress bar

`progress` replaces Lightning's bar with one that renders a metrics table
below it: each row a series (`label/f1`, `loss`), columns for the current
Train/Val/Test values and the best observed Train/Val, colour-coded ▲▼
deltas beside them. Improvement direction is each metric's own declared
`higher_is_better` flag, asked from the module — never guessed from a name;
the total loss is the one safe assumption (lower is better).

```yaml
callbacks:
  - name: progress
    # metric_filters: [f1, loss]   # narrow the table; omit to show everything
```

Vector metrics contribute their `mean` row; per-class leaves stay off the
table.

## What the run is about to train on

`dataset_summary` prints one table per target before the first epoch — the class
balance or the value spread, per stage, with the stage's row count as a `total`
row — and sends the same numbers to the tracker as a grouped bar chart or a box
plot.

```yaml
callbacks:
  - {name: dataset_summary}
```

No parameters worth setting: what is counted follows from the targets the run
declares. `title` renames the tracker group if `dataset` collides with something.

Full detail — what each encoder counts, what the mask pass costs, why the box
plot's whiskers are the extremes rather than Tukey fences — is in
[data.md](data.md#the-dataset-report).

## A grid of samples in the tracker

`samples` draws one batch every N epochs as a self-contained HTML page: the
inputs with ground truth and predictions over them, a tree of switches per
field, and the filters that answer *show me the mistakes*. It draws the forward
that already happened — a step returns what it produced and Lightning hands that
to the batch-end hook — so there is no second inference and no way to show
something the run never computed.

Ready-made: `callbacks=samples`. Written out, it is

```yaml
callbacks:
  - name: samples
    every_n_epochs: 5
    stages: [train, val, test]
    mean: "${mean}"        # the run's own normalisation, to undo it
    std: "${std}"
```

Full guide: [visualization.md](visualization.md).
