"""What runs alongside the loop: Lightning's own hooks, under the names a declaration writes.

Importing this package is what makes those names resolvable: `callbacks: [{name: checkpoint}]` finds
one here. No second callback framework is introduced — a callback is a Lightning callback.
"""

from __future__ import annotations

from src.callbacks.registry import callback_registry

__all__ = ["callback_registry"]
