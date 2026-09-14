# ml-framework

Config-driven multi-task computer-vision training on PyTorch Lightning · Hydra ·
Pydantic · timm / smp · albumentations · torchmetrics.

Built around a strict dependency discipline — a thin core, capability packages
around it, one composition root — and a vocabulary any data scientist can read
without a glossary.

Classification, segmentation, regression and metric learning, several of them on
one backbone, with EMA, freezing, MixUp/CutMix, loss-parameter annealing and
per-task learning rates, a grid of samples and a summary of the data a run is
about to read. What a run ends with is deployable: ONNX, PT2, TorchScript, ncnn
or a TensorRT engine, each proven against the model it was written from and
described by a record a deployment reads. Metric learning learns its identities
from the training split alone, so a run can hold whole identities out and be
judged on ones it never saw — by retrieval and by verification, since the
objective over the learned identities says nothing about the others. Detection is
not here yet. What is written below is what runs.

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
uv run main.py experiment=examples/classification
uv run main.py experiment=examples/segmentation
uv run main.py experiment=examples/finetuning
uv run main.py experiment=examples/metric_learning
uv run main.py experiment=examples/classification lr=3e-4 epochs=50 loader=performance scheduler=onecycle
```

Configs live in [`configs/`](configs/). `config.yaml` holds the knobs a run is
usually steered by — `seed`, `lr`, `epochs`, `batch_size` — and the group files
interpolate from them, so one edit reaches every consumer. The picture's size and
statistics are declared once in `configs/preprocessing/image.yaml`, and the pixel
chain reads them from there. Override a knob (`lr=3e-4`), not its mirror: the
optimizer group declares no `lr`, so `optimizer.lr=3e-4` is refused by Hydra,
which offers you `+optimizer.lr=3e-4` — take that offer and the run is refused by
name, because `lr` has one home. Adding a key a group file genuinely does not
declare is what the `+` is for (`+trainer.precision=bf16-mixed`).

## Architecture

```
cli.py + build.py    composition root: Hydra composes, one grammar builds
      │ creates and wires
capability packages  data · transforms · models · tasks · losses · metrics ·
      │              training · tracking · callbacks · export · integrations
      │              (visualization is a library of its own beside them)
      │ implement and consume
core/                entities · taxonomy · the registry — torch and stdlib only
```

Arrows point down only. The core never imports a capability; a capability never
imports `config/`; only the composition root and a package's own `build.py` read
declarations. That is what keeps the third-party stacks contained — Lightning in
`training/` and `callbacks/`, albumentations in `transforms/`, pandas and OpenCV
in `data/`, pydantic in `config/`, ONNX and TensorRT in `export/backends/`,
Hydra in `cli.py` and the one resolver it needs in `config/instantiate.py`. The
rules are not a
convention: [`tests/test_layering.py`](tests/test_layering.py) walks every import
in the tree and fails on one that is not declared.

Three ideas carry most of the design:

- **A task is a kind.** `classification`, `segmentation`, `regression` are
  classes that each state what the task needs — encoder, head, loss, metrics — in
  one place; a kind of your own is a subclass reachable by `_target_`, not a new
  subsystem.
- **Sizes come from the data.** Encoders fit on the train split, their facts land
  on the prepared module's `info`, and only then are heads built — `num_classes`
  is never written in a config file.
- **One grammar for every component.** `name` (a registry key) or `_target_` (an
  import path); every other key is a constructor argument, so an upstream knob is
  reachable without a schema change.

Third-party models plug in through adapters to narrow ports; the ports never bend
toward a library's signatures. What the model brings decides where it lands — and
every row below is one new class, with no edit to existing code:

| The model provides | You write |
|---|---|
| Features only (timm, DINO, an smp encoder) | a `Backbone` adapter |
| Features behind a removable head (torchvision) | a `Backbone` adapter that strips it |
| Logits but no loss (HF `*ForClassification`) | a `Backbone` exposing a `logits` stream + `head: {_target_: torch.nn.Identity}` |
| Everything: head, loss, decoding (a DETR of your own) | a `Model`, reached by `_target_` and built as it is |
| Just weights for our own topology | nothing — `run.checkpoint_path` loads them |

The same holds off the model path: a loss, a metric, a callback, an encoder, a
pixel augmentation and a rule for dividing a table all arrive the same way.
[`tests/e2e/test_custom_extension.py`](tests/e2e/test_custom_extension.py) is the
promise written down — a task kind, a network and a splitting rule of one's own,
run end to end with nothing under `src/` knowing about them.

## Documentation

There is none yet, deliberately: the guides are written once the shape stops
moving. Until then the configs under [`configs/`](configs/) carry the reasoning
in comments, and every contract is stated in the docstring of the `base.py` that
declares it.

## Development

```bash
make install     # uv sync
make test        # full pytest suite
make test-unit   # the package tests only
make test-e2e    # the runs that go from a file on disk to logged metrics
make test-gate   # the whole suite minus the tests that need a model hub (the pre-commit gate)
make typecheck   # mypy over src and tests
make check       # typecheck + full tests — the gate
make pre-commit  # every hook: file hygiene, typos, ruff check, ruff format, mypy, the test gate
make clean       # caches and temporary files
```
