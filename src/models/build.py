"""The model section as a graph: a backbone from its declaration, one head per task, sized from both."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from src.config import ComponentConfig, HeadConfig, ModelConfig
from src.config.instantiate import instantiate
from src.core import Axis, TensorShape
from src.models.base import Backbone, HeadConnection, Model, ShapeAware
from src.models.registry import backbone_registry, head_registry, model_registry

NATIVE = "native"
"""The reserved head name: the head the backbone itself brings, built by the backbone."""


def build_model(declared: ModelConfig, heads: Mapping[str, HeadConfig], outputs: Mapping[str, TensorShape]) -> Model:
    """A network reached by import path arrives whole; a named family composes heads onto a backbone.

    Per task, the model is told two things and no more: the head its run declared (already merged with
    the task's own default) and the shape that task outputs. That is the whole of what ``models`` needs
    to know about ``tasks``, which is why it needs to know nothing about ``Task``.
    """
    if declared.import_path is not None:
        if declared.backbone is not None:
            raise ValueError(
                f"{declared.spelled!r} is a whole model and brings its own backbone; drop 'model.backbone'."
            )
        whole = instantiate(declared)
        if not isinstance(whole, Model):
            raise TypeError(
                f"{declared.spelled!r} built {type(whole).__name__}, which is not a Model: a run asks a model "
                "for one thing, that inputs become outputs, and this cannot answer."
            )
        return whole
    if declared.backbone is None:
        raise ValueError(f"{declared.spelled!r} composes heads onto a backbone; declare 'model.backbone'.")
    if set(heads) != set(outputs):
        raise ValueError(f"Heads are declared for {sorted(heads)}, but outputs are known for {sorted(outputs)}.")
    backbone = build_backbone(declared.backbone)
    connections = {name: build_head(name, head, outputs[name], backbone) for name, head in heads.items()}
    composed: Model = instantiate(declared, model_registry, backbone=backbone, heads=connections)
    return composed


def build_backbone(declared: ComponentConfig) -> Backbone:
    built = instantiate(declared, backbone_registry)
    if not isinstance(built, Backbone):
        raise TypeError(
            f"'model.backbone' built {type(built).__name__}, which is not a Backbone: it publishes no feature "
            "streams to size a head from. A network that arrives whole is declared as the model itself."
        )
    return built


def build_head(task: str, declared: HeadConfig, output_shape: TensorShape, backbone: Backbone) -> HeadConnection:
    """The declared head at the widths nobody has to write down: the stream's, and the task's output."""
    if declared.stream is None:
        raise ValueError(f"Task {task!r}: head {declared.spelled!r} names no feature stream to read.")
    stream, out_features = declared.stream, _out_features(task, output_shape)
    if declared.name == NATIVE:
        return HeadConnection(_native_head(task, declared, stream, out_features, backbone), stream=stream)
    published = _published(task, stream, backbone)
    head: nn.Module = instantiate(
        declared, head_registry, in_features=_width(published, stream, backbone), out_features=out_features
    )
    _refuse_a_head_that_cannot_read(task, declared.spelled, head, published, stream)
    return HeadConnection(head, stream=stream)


def _native_head(task: str, declared: HeadConfig, stream: str, out_features: int, backbone: Backbone) -> nn.Module:
    if params := sorted(declared.params):
        raise ValueError(
            f"Task {task!r}: {NATIVE!r} names the head the backbone brings, built by the backbone; "
            f"drop {', '.join(params)}."
        )
    native = backbone.native_head(stream, out_features)
    if native is None:
        raise LookupError(
            f"Task {task!r} asks for a native head over {stream!r}, but {type(backbone).__name__} offers no "
            "native head for that stream. Declare a head of your own instead."
        )
    return native


def _published(task: str, stream: str, backbone: Backbone) -> TensorShape:
    shapes = backbone.feature_shapes
    if stream not in shapes:
        raise ValueError(
            f"Task {task!r} reads {stream!r}, but {type(backbone).__name__} publishes {', '.join(shapes)}."
        )
    return shapes[stream]


def _width(published: TensorShape, stream: str, backbone: Backbone) -> int:
    width = published.size(Axis.CHANNELS)
    if width is None:
        raise ValueError(f"{type(backbone).__name__} declares no width for {stream!r}; a head cannot be sized.")
    return width


def _refuse_a_head_that_cannot_read(
    task: str, spelled: str, head: nn.Module, published: TensorShape, stream: str
) -> None:
    """A head that declares its axes meets a stream that declares its own; a mismatch dies here, not in a matmul."""
    if isinstance(head, ShapeAware) and tuple(head.reads) != published.axes:
        raise ValueError(
            f"Task {task!r}: head {spelled!r} reads a [{', '.join(head.reads)}] feature, but {stream!r} is "
            f"[{', '.join(published.axes)}]. Read a stream of that shape, or a head that takes this one "
            "(a feature map needs 'conv', or the backbone's own head through 'native')."
        )


def _out_features(task: str, shape: TensorShape) -> int:
    """How many values a head produces per position: one per class, or one when a task has no classes."""
    if Axis.CLASSES not in shape.axes:
        return 1
    classes = shape.size(Axis.CLASSES)
    if classes is None:
        raise ValueError(f"Task {task!r} outputs classes without a count; a head cannot be sized for it.")
    return classes
