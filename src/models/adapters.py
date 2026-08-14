"""Training a low-rank delta instead of the weights it stands in for."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.models.registry import adapter_registry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor, nn

log = logging.getLogger(__name__)

type Adapters = Callable[[nn.Module], None]
"""Injects trainable adapters into a model in place and freezes what they stand in for.

Returns nothing on purpose: the model is rewritten, and a returned module would
suggest the original survived it. Not to be read as ``core.ports.TargetAdapter``,
which shapes one batch's target — this adapts a model's parameters.
"""


@adapter_registry.register("lora")
class LoraAdapters:
    """Low-rank adapters on the named projections, everything else frozen.

    A backbone arrives pretrained and enormous; this leaves it frozen and learns a small
    ``lora_A``/``lora_B`` pair beside each targeted projection — measured on a timm
    ViT-tiny, 118K trainable parameters out of 5.6M. Before anything reads the weights
    the delta folds back, and what is left is the architecture that arrived.

    Applied as a *transformation* of a built model rather than as a component it
    contains: peft rewrites the targeted layers in place, so wrapping would only add a
    name level over keys peft already renamed. peft itself is imported inside the calls
    that need it — measured at 5.5 s, it is the heaviest dependency here, and
    ``src.models`` is on the import path of every run.

    Parameters:
        target_modules (Sequence[str]): Module-name suffixes or regexes to adapt,
            e.g. ``[qkv, proj, fc1, fc2]`` for a timm ViT. No default: it is the
            one setting that cannot be guessed from the architecture, and a value
            that matches nothing is refused rather than left to train everything.
        rank (int): Width of the delta — peft's ``r``, under the name the paper
            uses.
        alpha (float): Scaling numerator; the effective scale is ``alpha / rank``.
        dropout (float): Dropout on the adapter's input.
        **kwargs: Forwarded verbatim to ``peft.LoraConfig`` — ``use_dora``,
            ``use_rslora``, ``bias``, ``exclude_modules`` and the rest stay
            reachable without this class declaring them.
    """

    def __init__(
        self,
        target_modules: Sequence[str],
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if not target_modules:
            raise ValueError("LoraAdapters needs at least one target module, e.g. target_modules: [qkv, proj].")
        if rank <= 0:
            raise ValueError(f"LoRA rank is the width of the delta and must be positive, got {rank}.")
        self.target_modules = list(target_modules)
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self._options = kwargs

    def __call__(self, model: nn.Module) -> None:
        from peft import LoraConfig, inject_adapter_in_model

        inject_adapter_in_model(
            LoraConfig(
                r=self.rank,
                # peft annotates lora_alpha as int but only ever divides by r — keep the float.
                lora_alpha=self.alpha,  # type: ignore[arg-type]
                lora_dropout=self.dropout,
                target_modules=self.target_modules,
                **self._options,
            ),
            model,
        )
        adapted = _adapted_layers(model)
        if not adapted:
            raise ValueError(
                f"LoRA matched no module against target_modules={self.target_modules}. Check the names "
                "against the backbone's named_modules() — an unmatched target would train every weight."
            )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "lora_" in name
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        log.info("LoRA adapted %d layers; %d of %d parameters train.", len(adapted), trainable, total)


def graft_base_weights(model: nn.Module, weights: Mapping[str, Tensor]) -> bool:
    """Load a plain checkpoint's weights beneath the model's adapters, exactly.

    Warm-starting adapters from an earlier run's weights: peft renamed every
    targeted layer (``net.weight`` became ``net.base_layer.weight``), so the plain
    keys are rewritten to the adapted names — a mechanical rename over
    ``_adapted_layers``, no guessing — and the deltas keep their fresh start.
    ``lora_B`` initialises at zero, so the grafted model computes exactly what the
    checkpoint's weights say until training moves the delta (asserted in
    ``test_checkpoints.py`` against a hand-computed forward).

    Returns ``False`` when there is nothing to graft: the model has no adapters, or
    the weights already carry adapter keys — an adapted run's own checkpoint loads
    strictly and needs no rename. Raises when the renamed weights still do not fit:
    the graft fixes exactly one mismatch, the adapters' rename, and nothing else.
    """
    adapted = {name for name, _ in _adapted_layers(model)}
    if not adapted or any(".base_layer." in key or "lora_" in key for key in weights):
        return False
    renamed = {_beneath(key, adapted): value for key, value in weights.items()}
    try:
        report = model.load_state_dict(renamed, strict=False)
    except RuntimeError as error:
        # ``strict=False`` forgives absent keys, not mismatched shapes — a head sized
        # for different classes still raises, and it deserves the named refusal too.
        raise ValueError(
            f"The checkpoint does not fit {type(model).__name__} even beneath the adapters — "
            "the mismatched shapes are above. The graft renames the adapted layers and nothing "
            "else; a layer of a different width is the architecture disagreeing, not the naming."
        ) from error
    strays = [key for key in report.missing_keys if "lora_" not in key] + list(report.unexpected_keys)
    if strays:
        raise ValueError(
            f"The checkpoint does not fit {type(model).__name__} even beneath the adapters: "
            f"{', '.join(strays)}. The graft renames the adapted layers and nothing else — "
            "this checkpoint disagrees with the model's architecture itself."
        )
    log.info(
        "Grafted the checkpoint's weights beneath %d adapted layers; the deltas keep their fresh start.", len(adapted)
    )
    return True


def _beneath(key: str, adapted: set[str]) -> str:
    """The adapted spelling of one plain parameter key; untouched where no adapter sits."""
    owner, _, parameter = key.rpartition(".")
    return f"{owner}.base_layer.{parameter}" if owner in adapted else key


def merge_adapters(model: nn.Module) -> int:
    """Fold every adapter into the layer it stands in for, and return how many were folded.

    Exact, measured: outputs are unchanged and the keys come back identical to a
    model that never saw peft — which is what makes a run's checkpoint and its
    exported artifact indistinguishable from a plain run's. Destructive by
    nature, so it belongs after the run's final weights are in place, never
    before restoring them: a checkpoint is keyed with ``lora_`` and
    ``base_layer`` names that a folded model no longer has.

    A no-op on a model that was never adapted, so the caller needs no branch.
    """
    adapted = _adapted_layers(model)
    for name, module in adapted:
        module.merge(safe_merge=True)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, module.get_base_layer())
    if adapted:
        log.info("Folded %d LoRA layers back into their weights.", len(adapted))
    return len(adapted)


def _adapted_layers(model: nn.Module) -> list[tuple[str, Any]]:
    """Every peft LoRA layer in the tree, named — empty when peft never ran."""
    from peft.tuners.lora import LoraLayer

    return [(name, module) for name, module in model.named_modules() if isinstance(module, LoraLayer)]
