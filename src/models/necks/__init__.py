"""Necks, one module per kind; importing this registers every one of them.

A neck reads what a backbone published and publishes its own, which is why these are a package beside
`backbones/` rather than a module inside it: what makes a backbone one is that it reads a sample.
"""

from __future__ import annotations

from src.models.necks.projector import Projector

__all__ = ["Projector"]
