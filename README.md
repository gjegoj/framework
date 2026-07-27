# Framework

Configuration-driven **multi-task & multimodal** computer-vision training on top of
PyTorch Lightning · Hydra · Pydantic · timm / smp · albumentations · torchmetrics.

> Classification · segmentation · regression · **metric learning** (ranking / dual-encoder) ·
> **knowledge distillation** · **LoRA fine-tuning** — with EMA, MixUp/CutMix/Mosaic, loss-parameter
> scheduling, model **export** (ONNX / TorchScript / TensorRT) and interactive **sample
> visualization** built in.

## Quick start

```bash
uv sync           # install dependencies
make test         # verify everything works
```

The entry point is `main.py`; all configuration lives in `configs/`. Run the built-in
debug experiment (synthetic data, CPU, 2 epochs), then point at your own:

```bash
uv run python main.py
uv run python main.py +experiment=my_exp
```

## Highlights

- **A task is a composition, not a type** — topology × objective × modality; familiar names
  (`classification`, `segmentation`, `triplet`, `contrastive`) are thin presets, so
  `segmentation(objective="multilabel")` needs zero new code.
- **`num_classes` is never hardcoded** — inferred from data at setup and injected into
  heads, losses, and metrics.
- **Hydra groups = swappable blocks** — backbone / optimizer / scheduler / dataloader /
  callbacks / logger / export / distillation / lora, all overridable from the CLI.
- **Training regimes compose** — knowledge distillation (`hard + weight·KL` from frozen
  teachers) and LoRA (frozen base + adapters, merged back at export) are one YAML section
  each, and work together with EMA, freeze, and batch transforms.
- **Train → test → export, one pipeline** — with numerical-parity verification of every
  exported artifact and interactive HTML grids of predictions along the way.

## Documentation

Full documentation lives in the [`docs/`](docs/README.md) folder:

| | |
|---|---|
| [Core concepts](docs/concepts.md) | The three-axis task model, runtime values, construction families |
| [Data](docs/guides/data.md) · [Tasks](docs/guides/tasks.md) · [Backbones](docs/guides/backbones.md) · [Optimizer & scheduler](docs/guides/training.md) | Everyday configuration guides |
| [Callbacks](docs/guides/callbacks.md) | EMA, checkpointing, freeze, batch transforms, loss scheduling — every callback with usage |
| [Distillation](docs/guides/distillation.md) · [LoRA](docs/guides/lora.md) · [Export](docs/guides/export.md) · [Visualization](docs/guides/visualization.md) | Feature guides |
| [Recipes](docs/recipes.md) | Complete copy-paste experiment configs |
| [Config reference](docs/reference/config.md) · [CLI](docs/reference/cli.md) · [Extending](docs/reference/extending.md) | Reference |
| [Internals](docs/internals.md) | Assembly pipeline with diagrams |

## Development

```bash
make test         # full pytest suite (unit + e2e)
make test-unit    # unit tests only (the pre-commit gate)
make typecheck    # mypy over src and tests
make check        # typecheck + full tests
```
