"""Registry of optimizer classes selectable from YAML by ``name``.

These are third-party ``torch.optim`` classes (no local abstraction to co-locate
with), so the registry lives in its own module — mirroring ``metrics/registry.py``.
Users register their own optimizer the same way, or bypass the registry with a
``_target_`` spec.
"""

from __future__ import annotations

import torch

from src.core.registry import Registry

optimizers: Registry[torch.optim.Optimizer] = Registry("optimizer")

optimizers.register("adamw")(torch.optim.AdamW)
optimizers.register("adam")(torch.optim.Adam)
optimizers.register("sgd")(torch.optim.SGD)
optimizers.register("rmsprop")(torch.optim.RMSprop)
