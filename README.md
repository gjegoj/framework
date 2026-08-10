# ml-framework

Config-driven multi-task computer-vision training on PyTorch Lightning · Hydra ·
Pydantic · timm / smp / ultralytics · albumentations · torchmetrics.

Built around a strict dependency discipline — a thin core, capability packages
around it, one composition root — and a vocabulary any data scientist can read
without a glossary.

> Classification · segmentation · regression · **metric learning** (ranking,
> dual-encoder) · **object detection** (YOLO) · **knowledge distillation** ·
> **LoRA fine-tuning** — with EMA, MixUp/CutMix/Mosaic, loss-parameter annealing,
> per-task learning rates, model **export** (TorchScript / ONNX / TensorRT) with
> numerical-parity verification, and interactive HTML grids of predictions.

## Quick start

```bash
make install
make test-run                    # fetches a real dataset and trains three tasks on one backbone
```

`make test-run` is the whole framework in one command: it downloads Oxford-IIIT
Pet, writes the table, and runs the multitask example — classification,
regression and segmentation together. From there, every other shipped run is one
line, and [`configs/experiment/examples/`](configs/experiment/examples/) is where
they live:

```bash
uv run main.py +experiment=examples/classification
uv run main.py +experiment=examples/segmentation
uv run main.py +experiment=examples/detection      # COCO128, which downloads itself
uv run main.py +experiment=examples/classification lr=3e-4 trainer.max_epochs=50 loader=performance scheduler=onecycle
```

Configs live in [`configs/`](configs/). `config.yaml` holds the shared knobs
(`lr`, `epochs`, `batch_size`, `image_size`, `mean`, `std`) and the group files
interpolate from them, so one edit in an experiment file reaches every consumer.
Override a knob (`lr=3e-4`), not its mirror (`optimizer.lr=3e-4`). Adding a key a
group file does not declare needs Hydra's `+` (`+trainer.precision=bf16-mixed`).

## Architecture

```
cli.py + assembly/   composition root: Hydra composes, one grammar builds
      │ creates and wires
capability packages  data · models · tasks · losses · metrics · transforms ·
      │              training · callbacks · loggers · export · visualization
      │ implement and consume
core/                entities · ports · taxonomy · the log-key grammar — torch and stdlib only
```

Arrows point down only. The core never imports a capability; a capability never
imports `config/`; only the composition root reads config. That is what keeps the
third-party stacks contained — Lightning in `training/`, pydantic in `config/`,
Hydra in `cli.py`.

Three ideas carry most of the design:

- **A task is a composition, not a type.** `topology × objective × modality`;
  `classification` and `segmentation` are thin presets over that, so
  `dense × multilabel` is a config change rather than new code.
- **Sizes come from the data.** Encoders fit on the train split, their facts land
  in a `DataProfile`, and only then are heads built — `num_classes` is never
  written in a config file.
- **One grammar for every component.** `name` (a registry key) or `_target_` (an
  import path); every other key is a constructor argument, so an upstream knob is
  reachable without a schema change.

Third-party models plug in through adapters to narrow ports; the ports never bend
toward a vendor's signatures. What the model brings decides where it lands — and
every row below is one new class, with no edit to existing code:

| The model provides | You write |
|---|---|
| Features only (timm, DINO, an smp encoder) | a `Backbone` adapter |
| Features behind a removable head (torchvision) | a `Backbone` adapter that strips it |
| Logits but no loss (HF `*ForClassification`) | a `Backbone` exposing a `logits` stream + `IdentityHead` |
| Everything: preprocessing, head, loss, decoding (YOLO, DETR) | a `Model` adapter in `model_registry` |
| Just weights for our own topology | nothing — `checkpoint_path` loads them |

## Documentation

Full documentation is in [`docs/`](docs/README.md).

| | |
|---|---|
| [Core concepts](docs/concepts.md) | The design in five minutes, with the assembly order |
| [Data](docs/guides/data.md) · [Tasks](docs/guides/tasks.md) · [Models](docs/guides/models.md) · [Optimizer & scheduler](docs/guides/training.md) | Everyday configuration |
| [Losses](docs/guides/losses.md) · [Metrics](docs/guides/metrics.md) · [Transforms](docs/guides/transforms.md) · [Callbacks](docs/guides/callbacks.md) | The pieces of a run |
| [Detection](docs/guides/detection.md) · [Export](docs/guides/export.md) · [Samples grid](docs/guides/visualization.md) · [Logging](docs/guides/logging.md) | Feature guides |
| [Extending](docs/guides/extending.md) | Your own loss, metric, callback, backbone or task kind |
| [Backlog](docs/backlog.md) | Known defects and deferred decisions, with the reasoning kept |

## Development

```bash
make install     # uv sync
make test        # full pytest suite
make test-unit   # unit tests only (the pre-commit gate)
make typecheck   # mypy --strict over src and tests
make check       # typecheck + full tests — the gate
make pre-commit  # every hook: typos, isort, black, ruff, mypy, unit tests
make clean       # caches and temporary files
```

The framework runs end to end from YAML: `main.py` composes a config, assembly
builds the experiment, and it trains, tests and exports — covered by acceptance
tests that go from a file on disk to logged metrics.
