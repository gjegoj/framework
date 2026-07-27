# Knowledge distillation

Turn on with the `distillation` group (`distillation=kl`) or an inline section. Teachers are
**framework models**: each entry is a backbone spec + a `.ckpt` path; their heads are derived
from the *student's* tasks, so logit shapes match by construction. Raw logits are averaged
across teachers; per distilled task the training loss becomes `hard + weight · soft`, where
`soft` is a temperature-scaled KL divergence:

```yaml
distillation:
  teachers:
    - backbone: ${backbone}          # reuse the student backbone spec, or inline a different one
      ckpt_path: runs/.../teacher.ckpt   # EMA checkpoints contribute their EMA weights
  temperature: 2.0                   # softening for the default kl_divergence loss
  weight: 0.7                        # additive soft-loss weight (float, or {task_name: float})
  # loss: {name: kl_divergence, temperature: 4.0}  # explicit brick-spec: used AS IS
  # tasks: [mask]                                   # subset to distill; omit → every task
```

Design guarantees worth knowing:

- **TRAIN only.** Validation/test losses stay pure task losses — checkpoint monitoring is
  directly comparable with non-distilled runs, and teachers never run during evaluation.
- **Additive weighting** (`hard + weight · soft`, not a convex blend): `weight: 0` reproduces
  the baseline experiment bit-for-bit, and hard-loss curves keep their scale across runs.
- **Teachers stay out of the module tree** — invisible to `state_dict`, checkpoints, and EMA;
  they are moved to the training device/dtype automatically at fit start.
- **Everything composes**: the hard loss can be any brick (e.g. `weighted_sum` of focal+dice),
  `criterion_schedule` can anneal it while distilling, and EMA/freeze work unchanged.
- Temperature precedence: the top-level `temperature` feeds the *default* `kl_divergence`
  only; an explicit `loss:` spec wins outright (its own kwargs, criterion defaults for the rest).

The KL component is logged per task as `loss/train/<task>/kl`, so the soft-signal share is
visible next to the hard components.
