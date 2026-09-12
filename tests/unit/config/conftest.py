"""The smallest experiment that validates, and a registry to resolve names against."""

from __future__ import annotations

from typing import Any

import pytest

from src.core import Registry


class Widget:
    def __init__(self, size: int = 1, *, tag: str = "", inner: object = None) -> None:
        self.size, self.tag, self.inner = size, tag, inner


@pytest.fixture
def registry() -> Registry[Widget]:
    widgets = Registry[Widget]("widget")
    widgets.register("widget")(Widget)
    return widgets


@pytest.fixture
def minimal() -> dict[str, Any]:
    return {
        "data": {"name": "table", "source": "data.csv"},
        "preprocessing": {"name": "standard", "inputs": {"image": {"name": "image", "image_size": [224, 224]}}},
        "model": {"name": "timm", "model_name": "resnet18"},
        "tasks": {"species": {"kind": "classification", "target_column": "species", "classes": {0: "cat", 1: "dog"}}},
    }
