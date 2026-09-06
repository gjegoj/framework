"""The ultralytics detection graph behind our ports — parsed, not ported."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import torch
from torch import Tensor, nn

from src.core.entities import Features
from src.core.ports import Backbone
from src.core.taxonomy import Modality, Stream
from src.models.heads import DetectHead
from src.models.registry import backbone_registry

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)


@backbone_registry.register("ultralytics")
class UltralyticsBackbone(Backbone):
    """Everything before ``Detect`` in an ultralytics yaml, as a pyramid backbone.

    The head leaves the forward path and is rebuilt per task through ``native_head``, sized
    from the profile like every head. The pyramid — how many levels, at which strides — is
    read off the graph and named by stride (``p2`` … ``p6``), so a ``-p2`` yaml declares four
    levels with no config. ``ultralytics`` is imported inside the methods that need it.
    Serves the ``Detect`` line (v8/11/12); other heads are refused by name.

    Parameters:
        model_name (str): An architecture yaml ultralytics ships (``yolov8n.yaml`` …
            ``yolo12x.yaml``). Weights never come with the name — declare ``checkpoint_path``.
        input_name (str): Which batch input to encode.
        checkpoint_path (str | Path | None): An ultralytics ``.pt``. It is a pickled module, so it
            is read with ``weights_only=False`` — only a path written here is ever unpickled.
            The body loads strictly; the head is stashed and transplanted by ``native_head``.
        use_ema (bool): Prefer the checkpoint's EMA weights when it carries them.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.yaml",
        input_name: str = Modality.IMAGE,
        checkpoint_path: str | Path | None = None,
        use_ema: bool = True,
    ) -> None:
        super().__init__()
        if not model_name.endswith(".yaml"):
            raise ValueError(
                f"'{model_name}' is not an architecture yaml. Weights arrive through 'checkpoint_path', "
                f"never through the model name."
            )
        from ultralytics.nn.modules import Detect
        from ultralytics.nn.tasks import DetectionModel

        # nc is a placeholder: the head is rebuilt per task. Any, as every vendor object here:
        # nn.Module types each attribute as Tensor | Module, and this graph is walked by attribute.
        graph: Any = DetectionModel(model_name, nc=1, verbose=False)
        if type(graph.model[-1]) is not Detect:
            found = type(graph.model[-1]).__name__
            raise ValueError(
                f"'{model_name}' ends in {found}; this family serves the Detect line (v8/11/12), "
                f"and {found} needs a head adapter of its own."
            )
        head: Any = graph.model[-1]
        self._architecture = Path(model_name).stem
        self._input_name = input_name
        # Named `model` so the state-dict keys are the checkpoint's own: model.0.* … model.{n-1}.*.
        self.model: nn.Sequential = graph.model[:-1]
        self._kept = set(graph.save)
        self._level_indices = tuple(int(index) for index in head.f)
        self._strides = tuple(int(stride) for stride in head.stride.tolist())
        self._levels = tuple(f"p{stride.bit_length() - 1}" for stride in self._strides)  # 8 → p3, 4 → p2
        self._widths = tuple(int(branch[0].conv.in_channels) for branch in head.cv2)
        self._reg_max = int(head.reg_max)
        # The template never runs in forward; the 1-tuple keeps it out of the module tree.
        self._template: tuple[Any] = (head,)
        self._carried_head: dict[str, Tensor] | None = None
        if checkpoint_path is not None:
            self._carried_head = _load_body(
                self, model_name, checkpoint_path, head_index=len(graph.model) - 1, use_ema=use_ema
            )

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        x: Any = inputs[self._input_name]
        kept: dict[int, Tensor] = {}
        layer: Any
        for index, layer in enumerate(self.model):
            if layer.f != -1:  # ultralytics' own walk: -1 is the previous output, else saved layers
                x = kept[layer.f] if isinstance(layer.f, int) else [x if j == -1 else kept[j] for j in layer.f]
            x = layer(x)
            if index in self._kept:
                kept[index] = x
        levels = {name: kept[index] for name, index in zip(self._levels, self._level_indices, strict=True)}
        return Features(streams={**levels, Stream.FEATURES: levels[self._levels[-1]].mean(dim=(2, 3))})

    def feature_dims(self) -> Mapping[str, int]:
        return {**dict(zip(self._levels, self._widths, strict=True)), Stream.FEATURES: self._widths[-1]}

    @override
    def pyramid(self) -> tuple[str, ...]:
        return self._levels

    @property
    def strides(self) -> tuple[int, ...]:
        """The pyramid's strides, level by level — what a criterion and a decoder are built from."""
        return self._strides

    @property
    @override
    def architecture(self) -> str:
        return self._architecture

    @override
    def native_head(
        self, streams: tuple[str, ...], in_features: int | tuple[int, ...], out_features: int
    ) -> nn.Module | None:
        if streams != self._levels or isinstance(in_features, int):
            return None
        from ultralytics.nn.modules import Detect

        detect = Detect(nc=out_features, ch=in_features)
        detect.stride = self._template[0].stride
        detect.bias_init()  # ultralytics' own prior, which needs the strides
        if self._carried_head is not None:
            _transplant(detect, self._carried_head, self._architecture)
        return DetectHead(detect, streams=self._levels, strides=self._strides, reg_max=self._reg_max)


def _load_body(
    backbone: nn.Module, model_name: str, path: str | Path, *, head_index: int, use_ema: bool
) -> dict[str, Tensor]:
    """Load the checkpoint's body into ``backbone`` strictly; return the head's tensors, unprefixed."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)  # a pickled module, by their format
    source = (checkpoint.get("ema") if use_ema else None) or checkpoint["model"]
    state = {key: value.float() for key, value in source.state_dict().items()}
    prefix = f"model.{head_index}."
    body = {key: value for key, value in state.items() if not key.startswith(prefix)}
    head = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    try:
        backbone.load_state_dict(body, strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"'{path}' was trained for another architecture than '{model_name}' — its body does not "
            f"fit key for key: {str(error).splitlines()[0]}"
        ) from error
    log.info("Loaded %d body tensors from %s into %s; %d head tensors stashed.", len(body), path, model_name, len(head))
    return head


def _transplant(detect: nn.Module, stash: Mapping[str, Tensor], architecture: str) -> None:
    """Move every head tensor whose shape still fits; the class branch fits only on an equal class count."""
    fresh = detect.state_dict()
    fitting = {key: value for key, value in stash.items() if key in fresh and fresh[key].shape == value.shape}
    detect.load_state_dict(fitting, strict=False)
    left = sorted(key for key in stash if key not in fitting)
    log.info(
        "Transplanted %d of %d head tensors into %s's Detect (%d classes); left fresh: %s.",
        len(fitting),
        len(stash),
        architecture,
        detect.nc,
        ", ".join(left) or "none",
    )
