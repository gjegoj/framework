"""``ExpandedHead``: a grown classifier whose freeze boundary is a module boundary."""

from __future__ import annotations

import torch
from torch import nn

from src.models import ExpandedHead


def test_forward_is_the_concat_of_base_and_novel() -> None:
    head = ExpandedHead(base=nn.Linear(4, 3), novel=nn.Linear(4, 1))
    features = torch.randn(2, 4)

    combined = head(features)

    assert combined.shape == (2, 4)
    assert torch.equal(combined, torch.cat((head.base(features), head.novel(features)), dim=-1))


def test_submodule_names_are_the_freeze_path_contract() -> None:
    """Freeze configs name ``...head.base``; renaming the attribute is a breaking change."""
    head = ExpandedHead(base=nn.Linear(4, 3), novel=nn.Linear(4, 1))

    assert dict(head.named_children()).keys() == {"base", "novel"}


def test_a_frozen_base_survives_an_adamw_step_with_weight_decay_bit_for_bit() -> None:
    """The test a gradient mask could not pass: decoupled decay moves grad-masked rows,
    while a requires_grad=False module is skipped entirely."""
    base, novel = nn.Linear(4, 3), nn.Linear(4, 1)
    head = ExpandedHead(base=base, novel=novel)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    frozen_before = {name: tensor.clone() for name, tensor in base.state_dict().items()}
    novel_before = novel.weight.clone()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in head.parameters() if parameter.requires_grad), lr=0.1, weight_decay=0.1
    )

    head(torch.randn(8, 4)).sum().backward()
    optimizer.step()

    assert all(torch.equal(frozen_before[name], tensor) for name, tensor in base.state_dict().items())
    assert not torch.equal(novel.weight, novel_before)


def test_dense_logits_grow_along_the_class_axis() -> None:
    """Classes live at dim 1 for batch-first logits — [B, C] and [B, C, H, W] alike;
    concatenating along the last dim would splice segmentation maps side by side."""
    head = ExpandedHead(base=nn.Conv2d(4, 3, kernel_size=1), novel=nn.Conv2d(4, 1, kernel_size=1))

    combined = head(torch.randn(2, 4, 8, 8))

    assert combined.shape == (2, 4, 8, 8)
