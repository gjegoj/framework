# Optimizer, LR & scheduler

```yaml
lr: 1.0e-3          # global LR — all param groups start here

optimizer:
  name: adamw       # registry key: adamw · adam · sgd · rmsprop
  lr: ${lr}         # references the top-level lr
  weight_decay: 1.0e-4
```

Available optimizer groups: `adamw.yaml` · `sgd.yaml`.

**Per-head LR override**: add an `optimizer:` block to any task. That head gets its own
param group; the backbone uses the global `lr`.

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    num_classes: 3
    optimizer:
      lr: 5.0e-5    # decoder head trains slower than backbone
```

**Scheduler** is its own config group (`cosine` · `onecycle` · `plateau` · `step`; `none`
= constant LR). `interval`/`frequency`/`monitor` map to Lightning's scheduling; extra keys
forward to the scheduler constructor. `runtime_kwargs` fills a constructor argument from a
trainer fact computed at fit time (`total_steps` / `steps_per_epoch` / `epochs`):

```yaml
defaults:
  - scheduler: onecycle

scheduler:
  name: onecycle
  interval: step
  max_lr: ${lr}
  runtime_kwargs: {total_steps: total_steps}   # filled from the trainer at fit time
```

```yaml
# ReduceLROnPlateau — needs a monitored metric
scheduler:
  name: plateau
  interval: epoch
  monitor: loss/val/total
  factor: 0.5
  patience: 3
```

**Per-head LR + OneCycle/Cyclic.** A scalar `max_lr` (or Cyclic's `base_lr`/`max_lr`) is
expanded per param-group, scaled by each group's lr — so a per-head `optimizer.lr` override
carries into the schedule's peak instead of being overwritten. With `max_lr: ${lr}` and a head
at `lr: 1.0e-4`, the head peaks at `1.0e-4` while the backbone peaks at `${lr}`. (`cosine` /
`step` / `plateau` already scale each group's own lr, so they need nothing special.)

Everything callback-driven — EMA, checkpointing, freezing, batch transforms, loss-parameter
scheduling — lives in the [Callbacks guide](callbacks.md).
