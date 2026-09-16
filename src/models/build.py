"""The model section as a graph: a backbone from its declaration, one head per task, sized from both."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from torch import Tensor, nn

from src.config import ComponentConfig, HeadConfig, ModelConfig
from src.config.instantiate import instantiate
from src.core import SPATIAL, Axis, TensorShape, naming
from src.models.adapters import Adapter
from src.models.base import Backbone, HeadConnection, Model, ShapeAware
from src.models.heads import ExpandedHead, StackedHeads
from src.models.registry import adapter_registry, backbone_registry, head_registry, model_registry

log = logging.getLogger(__name__)

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
    _refuse_a_carried_classifier_with_two_claimants(backbone, heads)
    connections = {name: build_head(name, head, outputs[name], backbone) for name, head in heads.items()}
    composed: Model = instantiate(declared, model_registry, backbone=backbone, heads=connections)
    return composed


def build_adapter(declared: ComponentConfig | None, model: Model) -> Adapter | None:
    """The parameters this run adds to the network, attached to it, or None where it declared none.

    The model is imposed rather than offered: an adapter is a change to a network, so there is no such
    thing as one that does not take it. Nothing else about the run reaches here — how many epochs, which
    optimizer, what is frozen — because none of it changes what a delta is.
    """
    if declared is None:
        return None
    built = instantiate(declared, adapter_registry, model=model)
    if not isinstance(built, Adapter):
        raise TypeError(
            f"{declared.spelled!r} built {type(built).__name__}, which is not an Adapter: a run asks one for "
            "parameters to add before it trains and a way to fold them back after, and this answers neither."
        )
    return built


def build_backbone(declared: ComponentConfig) -> Backbone:
    built = instantiate(declared, backbone_registry)
    if not isinstance(built, Backbone):
        raise TypeError(
            f"'model.backbone' built {type(built).__name__}, which is not a Backbone: it publishes no feature "
            "streams to size a head from. A network that arrives whole is declared as the model itself."
        )
    return built


def build_head(task: str, declared: HeadConfig, output_shape: TensorShape, backbone: Backbone) -> HeadConnection:
    """The declared head at the widths nobody has to write down: each stream's, and the task's output."""
    streams = declared.streams
    if not streams:
        raise ValueError(f"Task {task!r}: head {declared.spelled!r} names no feature stream to read.")
    out_features = _out_features(task, output_shape)
    at = _sized(task, declared, streams, backbone)
    if not backbone.carried_head:
        return HeadConnection(at(out_features), streams=streams)
    return HeadConnection(_started_from(task, at, backbone.carried_head, out_features), streams=streams)


def _sized(task: str, declared: HeadConfig, streams: tuple[str, ...], backbone: Backbone) -> Callable[[int], nn.Module]:
    """The declared head at any number of outputs — one of it per stream, stacked where it reads several.

    A factory rather than one head, because growing means building the same declaration at two counts —
    the rows a file carried and the classes added since — and a second spelling of "what this task's
    head is" would be free to disagree with the first.

    A pairing is that same declaration built once per stream rather than a head of its own: the towers
    differ in width and in nothing else a head can see, so ``linear`` over a pair is two of it, and a
    head added to the registry reads a pairing the day it arrives without knowing that pairings exist.
    """
    over = {stream: _over(task, declared, stream, backbone) for stream in streams}
    if len(over) == 1:
        return next(iter(over.values()))
    return lambda count: StackedHeads({stream: build(count) for stream, build in over.items()})


def _over(task: str, declared: HeadConfig, stream: str, backbone: Backbone) -> Callable[[int], nn.Module]:
    """The declared head over one stream, at the width that stream publishes and any number of outputs."""
    if declared.name == NATIVE:
        return lambda count: _native_head(task, declared, stream, count, backbone)
    published = _published(task, stream, backbone)
    width = _width(published, stream, backbone)

    def built(count: int) -> nn.Module:
        with naming(f"tasks.{task}.head"):
            head: nn.Module = instantiate(declared, head_registry, in_features=width, out_features=count)
        _refuse_a_head_that_cannot_read(task, declared.spelled, head, published, stream)
        return head

    return built


def _started_from(task: str, at: Callable[[int], nn.Module], carried: Mapping[str, Tensor], declared: int) -> nn.Module:
    """The declared head holding the rows a weight file carried, grown where the task declares more classes.

    By index, because the vocabulary is declared: `classes` pins a name to a position, so a run that
    added names to the end reads the carried rows as the run that wrote them did. Narrowing is refused
    instead — dropping rows is a mapping from old positions to new, and a transplant that guessed one
    would report every class under another's name.
    """
    rows = _carried_classes(task, carried)
    if rows > declared:
        raise ValueError(
            f"Task {task!r}: the weights this backbone started from carry {rows} classes and the task "
            f"declares {declared}. Which of the {rows} stay, and at which positions, is a mapping nobody "
            f"has written; declare the {rows} classes the file was trained on, in their order, and add to "
            f"the end of them."
        )
    base = at(rows)
    _fill(task, base, carried, rows)
    if rows == declared:
        log.info("Task %r starts every one of its %d classes from the weights the backbone carried.", task, rows)
        return base
    log.info("Task %r starts %d of its %d classes from the weights carried; the rest are fresh.", task, rows, declared)
    return ExpandedHead(base=base, novel=at(declared - rows))


def _carried_classes(task: str, carried: Mapping[str, Tensor]) -> int:
    """How many classes the carried classifier answers for, which every one of its tensors agrees on.

    Read off the tensors rather than told: a classifier is weights and biases indexed by class, whatever
    the family shaped them like — timm's ``[classes, width]`` and smp's ``[classes, width, k, k]`` alike.
    """
    counts = {int(value.shape[0]) for value in carried.values() if value.ndim}
    if len(counts) != 1:
        raise ValueError(
            f"Task {task!r}: the weights this backbone started from carry more than one head — "
            f"{', '.join(sorted(carried))} answer for {sorted(counts)} classes between them — and which of "
            f"them this task continues is not written anywhere. Point the run at a file with one head."
        )
    return counts.pop()


def _fill(task: str, head: nn.Module, carried: Mapping[str, Tensor], rows: int) -> None:
    """Put the carried tensors into the head built for them, matched by shape, or refuse naming both sides.

    By shape, because the two names come from different vocabularies — the file writes the library's
    (``fc.weight``, ``segmentation_head.0.weight``) and the head writes this framework's — while the
    shapes are the same fact on both sides. A carried tensor matching nothing is a classifier read off
    another feature space, and one matching two is a head this transplant cannot tell apart.
    """
    held = head.state_dict()
    wanted = {name: tensor.shape for name, tensor in held.items() if tuple(tensor.shape[:1]) == (rows,)}
    landed: dict[str, Tensor] = {}
    for name, value in carried.items():
        matched = [key for key, shape in wanted.items() if shape == value.shape and key not in landed]
        if len(matched) != 1:
            raise ValueError(
                f"Task {task!r}: the carried {name!r} is {tuple(value.shape)}, and this head has "
                f"{'no part' if not matched else 'more than one part'} of that shape "
                f"({', '.join(f'{key} {tuple(shape)}' for key, shape in wanted.items()) or 'none'}). "
                f"A classifier over another feature space is not this one's to start from."
            )
        landed[matched[0]] = value
    if set(landed) != set(wanted):
        raise ValueError(
            f"Task {task!r}: the weights carried fill {', '.join(sorted(landed)) or 'nothing'} and leave "
            f"{', '.join(sorted(set(wanted) - set(landed)))} at whatever `seed` produced. A head half from "
            f"a file is not that file's head."
        )
    head.load_state_dict({**held, **landed})


def _refuse_a_carried_classifier_with_two_claimants(backbone: Backbone, heads: Mapping[str, HeadConfig]) -> None:
    """One file carries one classifier, and a run of several tasks gives it more than one place to land.

    Refused rather than shared or guessed: every one of them would be starting from rows trained to
    answer another question, and the run would report it as a warm start. Named here because this is
    where all of a run's tasks are visible at once.
    """
    if backbone.carried_head and len(heads) > 1:
        raise ValueError(
            f"The weights this backbone started from carry one classifier, and this run declares "
            f"{', '.join(sorted(heads))}: which of them it was trained to answer is not written anywhere. "
            f"Point a single-task run at that file, or drop its `checkpoint_path` and let the heads "
            f"start fresh."
        )


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
    if isinstance(head, ShapeAware) and tuple(head.reads_axes) != published.axes:
        raise ValueError(
            f"Task {task!r}: head {spelled!r} reads a [{', '.join(head.reads_axes)}] feature, but {stream!r} is "
            f"[{', '.join(published.axes)}]. Read a stream of that shape, or a head that takes this one "
            "(a feature map needs 'conv', or the backbone's own head through 'native')."
        )


def _out_features(task: str, shape: TensorShape) -> int:
    """How many values a head produces per position: the one axis of the output that is not spatial.

    A rule rather than a list of axis names, because each kind of output names its width with an axis of
    its own. A shape naming no width, or two, is refused with the axes it was handed rather than guessed
    at — a silently assumed 1 is a head built for the wrong task.
    """
    widths = [axis for axis in shape.axes if axis not in SPATIAL]
    if len(widths) != 1:
        named = ", ".join(shape.axes) or "nothing"
        raise ValueError(
            f"Task {task!r} outputs [{named}]; a head is sized by how many values it makes per position, "
            f"which is the one axis of an output that is neither height nor width, and this names {len(widths)}."
        )
    size = shape.size(widths[0])
    if size is None:
        raise ValueError(f"Task {task!r} outputs {widths[0]!r} without a count; a head cannot be sized for it.")
    return size
