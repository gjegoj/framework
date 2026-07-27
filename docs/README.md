# Framework

Configuration-driven **multi-task & multimodal** computer-vision training on top of
PyTorch Lightning · Hydra · Pydantic · timm / smp · albumentations · torchmetrics.

Classification · segmentation · regression · **metric learning** (ranking / dual-encoder) ·
**object detection** (YOLO) · **knowledge distillation** · **LoRA fine-tuning** — with EMA,
MixUp/CutMix/Mosaic, loss-parameter scheduling, model **export** (ONNX / TorchScript / TensorRT)
and interactive **sample visualization** built in.

## Quick start

```bash
uv sync           # install dependencies
make test         # verify everything works
```

The entry point is `main.py`. All configuration lives in `configs/`.
Run with the built-in debug experiment (synthetic data, CPU, 2 epochs):

```bash
uv run python main.py
```

To point at your own data, override the experiment:

```bash
uv run python main.py +experiment=my_exp
```

## Where to go next

| You want to… | Read |
|---|---|
| Understand the design in five minutes | [Core concepts](concepts.md) |
| Point the framework at your data | [Data](guides/data.md) |
| Declare classification / segmentation / regression tasks | [Tasks & presets](guides/tasks.md) |
| Pick an encoder (timm / smp / DINOv3 / embeddings) | [Backbones](guides/backbones.md) |
| Tune the optimizer and LR schedule | [Optimizer, LR & scheduler](guides/training.md) |
| Configure EMA, checkpoints, freeze, MixUp/Mosaic, loss scheduling | [Callbacks](guides/callbacks.md) |
| Distill a big model into a small one | [Knowledge distillation](guides/distillation.md) |
| Fine-tune a large backbone cheaply | [LoRA fine-tuning](guides/lora.md) |
| Train a YOLO object detector | [Object detection](guides/detection.md) |
| Ship a model (ONNX / TorchScript / TensorRT) | [Export](guides/export.md) |
| Copy a complete working config | [Recipes](recipes.md) |
| Look up any YAML key | [Config reference](reference/config.md) |
| Add your own loss / metric / callback / backbone | [Extending the framework](reference/extending.md) |
| See how it all fits together | [Internals](internals.md) |
