# Extending the framework

Every component is a registry key. Register your own with the `@registry.register` decorator — importing the module is enough to make it available.

**Custom loss**:

```python
# src/losses/my_loss.py
from src.losses.registry import criteria

@criteria.register("focal_tversky")
class FocalTverskyLoss(nn.Module, Criterion):
    ...
```

```yaml
tasks:
  mask:
    loss: {name: focal_tversky, alpha: 0.7, beta: 0.3}
```

**Custom metric**:

```python
from src.metrics.registry import metric_factories
metric_factories.register("my_metric")(MyTorchMetric)
```

```yaml
tasks:
  label:
    metrics:
      my_score:
        name: my_metric
        some_param: 42
```

**Custom callback**:

```python
# src/callbacks/my_callback.py
import lightning as L
from src.callbacks.registry import callback_registry

@callback_registry.register("gradient_clip")
class GradientClipCallback(L.Callback):
    def __init__(self, max_norm: float = 1.0) -> None:
        if max_norm <= 0:
            raise ValueError(f"max_norm must be positive, got {max_norm}.")
        self._max_norm = max_norm

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        import torch
        torch.nn.utils.clip_grad_norm_(pl_module.parameters(), self._max_norm)
```

Import the module once (e.g. in `main.py`) so the decorator runs, then use it by key:

```yaml
callbacks:
  gradient_clip:
    max_norm: 0.5
```

Or use `_target_` to skip registration entirely:

```yaml
callbacks:
  my_clip:
    _target_: src.callbacks.my_callback.GradientClipCallback
    max_norm: 0.5
```

**Custom data source** (e.g. Parquet):

```python
from src.data.sources import data_sources, FileDataSource

@data_sources.register("parquet")
class ParquetDataSource(FileDataSource):
    def _read_file(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path)
```

```yaml
data:
  sources: data/annotations.parquet
  source_type: parquet
```

**Complete-model family** (a third-party model that owns its head and loss, like YOLO —
the detection files are the worked example). Four pieces, each in its layer:

```python
# 1. models layer — the fused model behind the CompleteModel port (src/models/yolo.py)
from src.models.complete import CompleteModel
from src.models.registry import register_complete_model

@register_complete_model("my_kind")            # 'kind' shares one namespace with backbones
class MyModel(CompleteModel[MyPredictions, MyTargets]):
    family = "my_family"                       # == the task preset of such runs
    # prepare_batch / forward / training_loss / evaluation_loss / predictions / targets

# 2. data layer (optional) — a native LightningDataModule that writes facts
#    (num_classes, ...) into RuntimeContext at setup (src/data/detection.py);
#    families without a native format reuse the standard bindings contour instead.

# 3. metrics layer — a default metrics spec for build_metric_bundle (src/metrics/bundle.py)
MY_FAMILY_DEFAULT_METRICS = {"my_metric": None}

# 4. composition — the assembler gluing them, registered as a singleton
#    (src/composition/wiring/detection.py)
from src.composition.wiring.experiment import Capabilities, ExperimentAssembler, experiment_assemblers

class MyFamilyAssembler(ExperimentAssembler):
    name = "my_family"
    capabilities = Capabilities(export=False, lora=False, distillation=False,
                                batch_transforms=False, task_mixing=False)
    def build(self, config, runtime): ...      # datamodule + MyModel + CompleteModelLitModule

experiment_assemblers.register_instance(MyFamilyAssembler.name, MyFamilyAssembler())
```

```yaml
model: {kind: my_kind, name: ...}      # extras forward as the model's native knobs
tasks: {my_task: {preset: my_family}}
```

`main.py` never changes: `resolve_experiment_assembler` dispatches by the model
section's `kind` and validates the kind⇔preset pairing in both directions.

**Other extension points** follow the same pattern — `backbones`, `head_builders`,
`target_encoders`, `input_loaders`, `topology_strategies`, `objective_strategies`,
`task_presets`, `batch_transforms`, `schedulers`, `exporters`, `label_renderers`,
`annotators`.


---
