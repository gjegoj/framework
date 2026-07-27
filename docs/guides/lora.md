# LoRA fine-tuning

Turn on with the `lora` group (`lora=vit`) or an inline section. Adapters are injected
**in place** into the backbone's matching Linear/Conv2d layers (via peft, no model wrapper —
`state_dict` keys stay natural); the backbone's base weights are frozen, adapters and task
heads train:

```yaml
lora:
  target_modules: [qkv, proj, fc1, fc2]  # module-name suffixes or regexes (ViT attention + MLP)
  rank: 8
  alpha: 16          # effective scale = alpha / rank
  dropout: 0.0
  # extras forward to peft.LoraConfig verbatim:
  # use_dora: true
  # use_rslora: true
```

What changes when `lora:` is set:

- **Checkpoints store trainable weights only** (adapters + heads + criteria — megabytes, not
  gigabytes; buffers like BatchNorm stats are kept). The frozen base is *not* persisted — it
  is rebuilt from the `backbone` config (+ `init_ckpt_path` if one seeded it), so evaluating
  or resuming a LoRA checkpoint requires the same backbone config.
- **Export merges the adapters** into the base weights before tracing — one plain
  ONNX/TorchScript/TensorRT artifact with zero LoRA runtime overhead, verified by the usual
  numerical-parity report.
- **LoRA owns backbone freezing**: a `freeze` callback targeting `model.backbone` is rejected
  at startup (it would recursively freeze the adapters too). Freeze remains available for
  non-LoRA runs or for targets outside the backbone.
- EMA, distillation (LoRA student ← full teacher), batch transforms, and per-head LR
  compose unchanged; a typo'd `target_modules` fails loudly instead of silently training the
  full backbone.
