# Documentation

Start with [core concepts](concepts.md) if you have five minutes; otherwise find
what you came for below. The project overview and quick start live in the
[root README](../README.md).

## Guides

| You want to… | Read |
|---|---|
| Understand the design in five minutes | [Core concepts](concepts.md) |
| Point the framework at your data | [Data](guides/data.md) |
| Declare classification / segmentation / regression / metric-learning tasks | [Tasks and presets](guides/tasks.md) |
| Pick an encoder — timm, smp, HF text, multi-encoder, multi-view | [Models](guides/models.md) |
| Tune the optimizer, the LR schedule, per-task rates | [Optimizer and scheduler](guides/training.md) |
| Choose or compose a loss | [Losses](guides/losses.md) |
| Decide what a run is judged by | [Metrics](guides/metrics.md) |
| Augment samples, or mix whole batches | [Transforms](guides/transforms.md) |
| Configure EMA, checkpoints, freezing, annealing, the dataset report | [Callbacks](guides/callbacks.md) |
| Train a YOLO object detector | [Detection](guides/detection.md) |
| See where the model gets it wrong, sample by sample | [The samples grid](guides/visualization.md) |
| Read the log keys, or add a tracker | [Logging](guides/logging.md) |
| Ship a model — TorchScript, ONNX, TensorRT | [Export](guides/export.md) |
| Add your own loss, metric, callback, backbone or task kind | [Extending](guides/extending.md) |

## Reference

| | |
|---|---|
| [Backlog](backlog.md) | Known defects and deferred decisions, with the reasoning kept |
| [Specs and plans](superpowers/) | The design record: why each feature has the shape it has |

## Runnable configs

[`configs/experiment/examples/`](../configs/experiment/examples/) holds working
runs rather than copy-paste snippets, and a test composes every one of them, so
they cannot drift from the code. Five read the same table — the Oxford-IIIT Pet
dataset, which [`scripts/prepare_pet.py`](../scripts/prepare_pet.py) writes and
`make test-run` fetches for you:

| File | What it runs |
|---|---|
| [`classification.yaml`](../configs/experiment/examples/classification.yaml) | Cat or dog, from a pretrained ResNet |
| [`regression.yaml`](../configs/experiment/examples/regression.yaml) | A continuous target, on a column that is deliberately noise |
| [`segmentation.yaml`](../configs/experiment/examples/segmentation.yaml) | Pet / background / boundary, per pixel |
| [`multitask.yaml`](../configs/experiment/examples/multitask.yaml) | All three at once, on one backbone |
| [`all_callbacks.yaml`](../configs/experiment/examples/all_callbacks.yaml) | Every shipped callback in one run |
| [`detection.yaml`](../configs/experiment/examples/detection.yaml) | A YOLO detector on COCO128, which downloads itself |
| [`pet.yaml`](../configs/experiment/examples/pet.yaml) | What the first five share; inherited, never run |

```bash
make test-run                                # the dataset, then the multitask run
uv run main.py +experiment=examples/classification # or any of the others
```
