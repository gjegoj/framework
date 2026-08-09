"""The model summary's Name column as a module tree."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, override

from lightning.pytorch.callbacks import RichModelSummary


def tree_names(names: list[str]) -> list[str]:
    """Ordered dotted module paths as tree-connector-prefixed leaf labels.

    ``["model", "model.backbone", "model.backbone.encoder", "model.heads"]``
    becomes ``["model", "├─ backbone", "│  └─ encoder", "└─ heads"]``, input
    order preserved. A pure function, so the tree logic tests without
    Lightning.
    """
    children: dict[str | None, list[str]] = defaultdict(list)
    for name in names:
        parent = name.rsplit(".", 1)[0] if "." in name else None
        children[parent].append(name)

    label: dict[str, str] = {}

    def render(parent: str | None, prefix: str) -> None:
        siblings = children.get(parent, [])
        for index, name in enumerate(siblings):
            is_last = index == len(siblings) - 1
            leaf = name.rsplit(".", 1)[-1]
            if parent is None:
                label[name] = leaf
                render(name, "")
            else:
                label[name] = f"{prefix}{'└─ ' if is_last else '├─ '}{leaf}"
                render(name, prefix + ("   " if is_last else "│  "))

    render(None, "")
    return [label[name] for name in names]


class TreeModelSummary(RichModelSummary):
    """``RichModelSummary`` with the ``Name`` column rendered as a module tree.

    Lightning prints flat dotted paths; a tree of leaf names (``├─ backbone``) reads the
    hierarchy at a glance — including the freeze-path modules (``heads.<task>.base``)
    exactly as a config names them.

    Only the Name column changes. The transformed rows delegate to Lightning's own
    renderer, so columns, footer, totals and rank-zero gating stay upstream's, and an
    upstream layout change arrives for free.
    """

    @staticmethod
    @override
    def summarize(
        summary_data: list[tuple[str, list[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        total_flops: int,
        **summarize_kwargs: Any,
    ) -> None:
        treed = [(header, tree_names(values) if header == "Name" else values) for header, values in summary_data]
        RichModelSummary.summarize(
            treed,
            total_parameters,
            trainable_parameters,
            model_size,
            total_training_modes,
            total_flops,
            **summarize_kwargs,
        )
