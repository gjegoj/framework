"""Parameters a run adds to a network it did not build, and folds back into it before the run ends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, override

from torch import nn

from src.core import submodule_at
from src.models.registry import adapter_registry

if TYPE_CHECKING:
    from peft import LoraModel

DECLARATION = "adapter"
"""The section an adapter is declared in, so a refusal names the key a reader would go and edit."""

DELTA_NAME = "default"
"""What this run's added parameters are called inside peft, which keeps several sets under names.

A run declares one ``adapter`` section, so there is one, and nothing this framework writes ever carries
the name: the delta is folded back in before a checkpoint of it is ever read.
"""


class Adapter(ABC):
    """A run's own parameters over weights it did not learn, kept apart until the run is over.

    Two moments, and the order between them is the design. A delta is attached after the network is
    built and before it is trained, so that the epochs are spent on it; it is folded back in after the
    epoch the run kept has been restored and before anything is evaluated or shipped, so that what the
    run leaves behind is named exactly as a run that adapted nothing. Measured on peft 0.20.0: folding
    reproduces the adapted answer to 2.4e-08 and restores every ``state_dict`` key.

    A *part* of the model rather than the whole of it, and that is the difference this class exists to
    make. Wrapping the model moves every path under it — ``backbone`` becomes ``base_model.model.backbone``
    (measured) — and those paths are a contract: a freeze callback writes them, a checkpoint carries
    them, a parameter group is split by them. Adapting a named part leaves all of them naming what they
    named, and only the adapted leaves gain a level.

    Attached in the constructor rather than by a call of its own: between an adapter that exists and one
    that has been attached there is no state worth representing, and a second step would leave ``merge``
    with a question to answer about a run that never took the first.

    Parameters:
        module: Dot-path of what to adapt, relative to the model — ``backbone``, ``backbone.blocks``.
        model: The network this run will train; it is adapted in place.
    """

    def __init__(self, module: str, model: nn.Module) -> None:
        self.module = module
        self.target = submodule_at(model, module, reader=f"`{DECLARATION}.module`")

    @abstractmethod
    def merge(self) -> None:
        """Fold what was learned into the weights it was added to, leaving the network named as it was."""
        raise NotImplementedError


@adapter_registry.register("lora")
class LoraAdapter(Adapter):
    """A low-rank delta beside chosen weights: the network is held still and two small matrices learn.

    Which weights is the one thing with no safe default. peft matches a name against the end of a
    module's path, so ``qkv`` reaches every attention projection of a transformer and ``conv1`` every
    first convolution of a residual block — and a name reaching none of them is refused here with the
    parts that are actually there, which peft's own message cannot know to offer.

    Parameters:
        module: Dot-path of what to adapt, relative to the model.
        model: The network this run will train; it is adapted in place.
        target_modules: Which parts under ``module`` the delta is added beside, by the end of their path.
        r: The rank of the delta — how many columns the pair of matrices shares, and the whole of what
            this costs in parameters.
        alpha: What the delta is scaled by, as ``alpha / r``; with the default pair it is doubled.
        dropout: Dropout over the input of the delta during training, in the usual place.
    """

    def __init__(
        self,
        module: str,
        model: nn.Module,
        *,
        target_modules: Sequence[str],
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(module, model)
        from peft import LoraConfig, LoraModel

        declared = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(target_modules))
        try:
            self._delta: LoraModel = LoraModel(self.target, declared, DELTA_NAME)
        except ValueError as error:
            raise ValueError(self._nothing_it_could_adapt(target_modules)) from error

    @override
    def merge(self) -> None:
        """Fold the delta into the weights it was added beside, and take its layers back out.

        The module it answers with is the one it was handed, adapted in place, which is why nothing here
        puts anything back: the network already holds what was merged, under the names it started with.
        """
        self._delta.merge_and_unload()

    def _nothing_it_could_adapt(self, target_modules: Sequence[str]) -> str:
        """What to write instead, which is the half of this refusal peft has no way to answer.

        Shaped like ``load_weights``: the library's own sentence stands above, because it says precisely
        what it could not do, and this adds what a reader needs — the declaration to edit, the module it
        was searched under, and the names that are there.
        """
        named = ", ".join(sorted(_where_a_delta_fits(self.target))) or "no part at all"
        return (
            f"`{DECLARATION}.target_modules` names {', '.join(target_modules)}, and peft could not adapt "
            f"{type(self.target).__name__} at `{DECLARATION}.module: {self.module}` with it — peft's own "
            f"words are above. A name is matched against the end of a module's path, and a low-rank delta "
            f"goes beside a weight that is a matrix; under this module that is {named}."
        )


def _where_a_delta_fits(module: nn.Module) -> set[str]:
    """The end of the path of every part a low-rank delta could go beside, which is every weight matrix.

    A rule about the tensor rather than a copy of the list of classes peft handles: that list is peft's
    to grow, and a second copy of it here would drift into offering parts that cannot be adapted. What
    the rule answers was measured — ``qkv, proj, fc1, fc2`` over a ViT, ``conv1, conv2`` over a resnet —
    which is the whole of what a reader wants to see and four names rather than the forty-one a module
    tree holds.
    """
    return {
        name.rsplit(".", 1)[-1]
        for name, part in module.named_modules()
        if name and isinstance(getattr(part, "weight", None), nn.Parameter) and part.weight.dim() >= 2
    }
