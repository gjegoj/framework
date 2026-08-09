"""``TreeModelSummary``: the summary's Name column reads as a module tree."""

from __future__ import annotations

import pytest

from src.callbacks.model_summary import TreeModelSummary, tree_names


def test_dotted_paths_become_tree_connected_leaf_labels() -> None:
    """The docstring example is the contract: order preserved, connectors carry structure."""
    names = ["model", "model.backbone", "model.backbone.encoder", "model.heads"]

    assert tree_names(names) == ["model", "├─ backbone", "│  └─ encoder", "└─ heads"]


def test_the_last_sibling_closes_its_branch() -> None:
    names = ["heads", "heads.tags", "heads.tags.base", "heads.tags.novel"]

    assert tree_names(names) == ["heads", "└─ tags", "   ├─ base", "   └─ novel"]


def test_the_rendered_summary_speaks_connectors_not_dotted_paths(capsys: pytest.CaptureFixture[str]) -> None:
    summary_data = [
        (" ", ["0", "1"]),
        ("Name", ["model", "model.backbone"]),
        ("Type", ["CompositeModel", "TimmBackbone"]),
        ("Params", ["1.0 K", "1.0 K"]),
        ("Mode", ["train", "train"]),
        ("FLOPs", ["0", "0"]),
    ]

    TreeModelSummary.summarize(
        summary_data,
        total_parameters=1000,
        trainable_parameters=1000,
        model_size=0.1,
        total_training_modes={"train": 2, "eval": 0},
        total_flops=0,
    )

    rendered = capsys.readouterr().out
    assert "└─ backbone" in rendered
    assert "model.backbone" not in rendered
    assert "Trainable params" in rendered  # the parent's footer, not a re-implementation
