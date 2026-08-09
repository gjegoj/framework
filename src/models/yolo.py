"""Ultralytics as a model family: the head, the loss and the decoding are the vendor's."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import torch

from src.core.entities import Instances, Loss, Prediction, StepResult
from src.core.ports import Model
from src.core.taxonomy import Modality
from src.models.registry import model_registry

if TYPE_CHECKING:
    from torch import Tensor, nn

    from src.core.entities import Batch

_LOSS_SUFFIX = "_loss"
"""What ultralytics appends to each component's name; a part is logged without it.

``train/boxes/box_loss`` would say "loss" twice — once in the key's own grammar and once
in the vendor's spelling — while every other criterion here contributes ``ce`` or ``mse``.
"""

_WEIGHTS_SUFFIX = ".pt"
"""What tells trained weights from a bare architecture — the one branch this class has.

The branch is about where the weights come from, never about what kind of task it is:
detection, segmentation and pose all take the same path, and only the file's extension
decides whether anything is grafted.
"""

_DETECTION_COLUMNS = 6
"""Columns of one suppressed detection: ``x1 y1 x2 y2 conf cls``."""


@model_registry.register("yolo")
class YoloModel(Model):
    """A YOLO network trained through this framework's loop rather than ultralytics'.

    The network is held as a submodule, so device moves, EMA and checkpoints see its
    parameters like any other model's.

    What this class owns is the translation between the framework's currency and the
    vendor's dialect, in both directions and nowhere else: a ``Batch`` becomes the three
    keys the detection criterion reads, and what survives suppression becomes
    ``Instances``. Everything downstream — the training loop, callbacks, checkpointing,
    metrics — therefore never learns a vendor's shape. Both directions rebuild dicts of
    references and copy nothing.

    ``ultralytics`` is imported inside the methods that need it, so a run that never
    touches detection does not pay for the import.

    Parameters:
        model_name (str): An ultralytics architecture file (``yolov8n.yaml``) or a
            ``.pt`` weights path. One path serves every kind: measured, ``YOLO`` picks
            ``DetectionModel``, ``SegmentationModel`` or ``PoseModel`` from the name, so
            this class does not branch on what it is building.
        num_classes (int): Offered by assembly from the dataset descriptor, never written
            in config — the same derived fact every head is sized from.
        confidence_threshold (float): Minimum confidence a detection is kept at.
        iou_threshold (float): Overlap above which suppression drops the weaker box.
        **hyperparameters (Any): Forwarded verbatim to ultralytics' own configuration —
            the loss gains (``box``/``cls``/``dfl``) and the augmentation knobs alike,
            because the vendor keeps them in one namespace and splitting them across two
            of our sections would mean maintaining a table of which key belongs where.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        **hyperparameters: Any,
    ) -> None:
        super().__init__()
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg

        # `YOLO` reads the name and hands back the right kind of network — measured, a
        # `-seg` file gives a SegmentationModel and a `-pose` file a PoseModel — so this
        # class never branches on what it is building.
        declared = cast("Any", YOLO(model_name).model)
        # Rebuilt at *this* dataset's class count, from the architecture the file
        # describes, because the head's width is a fact of the data and never of config.
        network = cast("nn.Module", type(declared)(cfg=declared.yaml, nc=num_classes, verbose=False))
        if model_name.endswith(_WEIGHTS_SUFFIX):
            # What ultralytics' own trainer does with pretrained weights: graft the layers
            # that fit and leave the head this run's size. A checkpoint trained on other
            # classes is the ordinary case for fine-tuning, not an error.
            cast("Any", network).load(declared)
        # Weights arrive inference-frozen: ultralytics re-enables gradients in its own
        # trainer's setup, which this contour replaces.
        network.requires_grad_(True)
        # `args` is the vendor's dynamic configuration namespace, invisible to nn.Module
        # typing; its criterion reads the loss gains from there.
        cast("Any", network).args = get_cfg(overrides=dict(hyperparameters))
        self.detector = network
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold

    @override
    def step(self, batch: Batch) -> StepResult:
        """One walk of the network serving both what backward needs and what metrics compare.

        The output is handed to both readers rather than fetched twice: the vendor's own
        criterion takes it as ``preds``, and suppression decodes the same tensor. Asked
        separately — which is what ``loss()`` does when it is given nothing, since it then
        forwards internally — every eval batch walks the whole network twice for numbers
        that are identical either way (measured: same loss to six decimals, same boxes).
        """
        task_name, objects = _objects_of(batch)
        output = self.detector(batch.inputs[Modality.IMAGE])
        components, named = cast("Any", self.detector).loss(_vendor_batch(batch, objects), preds=output)
        # Scoped by the task, exactly as a composed model scopes its criteria's parts, so
        # `train/boxes/box` reads like `train/label/ce` and the log grammar has no family
        # to learn. Unscoped, the vendor's spelling would sit one level too high and two
        # detection tasks could not be told apart.
        loss = Loss(
            total=components.sum(),
            parts={name.removesuffix(_LOSS_SUFFIX): value for name, value in named.items()},
        ).scoped(task_name)
        return StepResult(loss=loss, prediction=self._decoded(output, batch), targets=dict(batch.targets))

    @override
    def predict(self, batch: Batch) -> Prediction:
        """Inference: walk the network, and decode what survives suppression."""
        return self._decoded(self.detector(batch.inputs[Modality.IMAGE]), batch)

    def _decoded(self, output: Any, batch: Batch) -> Prediction:
        """One walk's output as the framework's own ragged shape, per image.

        Training is the one mode with nothing to hand back. A detection head emits its
        feature maps while training and only assembles the decodable tensor in eval —
        ultralytics does not spend the decode on a step whose output nobody reads, and
        its own trainer measures on a separate validation pass for the same reason.
        Returning empty objects instead would be a fabricated answer that a train-stage
        mAP would then report as zero, which looks like a broken model rather than a
        measurement nobody took.

        The mode is read here rather than at each caller because it is a fact about the
        *output* — whether the head assembled something decodable — not about who asked.
        """
        from ultralytics.utils.nms import non_max_suppression

        if self.training:
            return Prediction(outputs={})
        raw = output[0] if isinstance(output, (list, tuple)) else output
        kept = non_max_suppression(raw, conf_thres=self._confidence_threshold, iou_thres=self._iou_threshold)
        return Prediction(outputs={name: _found(kept) for name in batch.targets})


def _objects_of(batch: Batch) -> tuple[str, Instances]:
    """The one per-instance target a vendor family is given: its task's name, and its objects.

    Both halves are wanted at once and by the same caller — the loss scopes its parts by
    the name, the vendor's dialect is built from the objects — so they are found together.
    Searched twice, the refusal below was written twice too, and two copies of a sentence
    are two things to keep in step.
    """
    for name, value in batch.targets.items():
        if isinstance(value, Instances):
            return name, value
    raise ValueError(
        "A detection step needs the batch's objects, and this batch carries none. "
        "A per-instance task's targets arrive as 'Instances' from the pipeline that read them."
    )


def _vendor_batch(batch: Batch, objects: Instances) -> dict[str, Tensor]:
    """The three keys the detection criterion reads, plus the image it reads them against.

    Measured: it takes ``batch_idx``, ``cls`` and ``bboxes`` and nothing else, and scales
    the normalised boxes by the image size itself — so the framework's xyxy pixels are
    converted back here, using the shape the image tensor already carries. Nobody has to
    be told the size, so there is no second place that could hold a stale one.

    The objects arrive rather than being looked for: this is the translation, and finding
    what to translate belongs to whoever also needed the name (:func:`_objects_of`).
    """
    from ultralytics.utils.ops import xyxy2xywhn

    image = batch.inputs[Modality.IMAGE]
    height, width = image.shape[-2:]
    return {
        "img": image,
        "cls": objects.labels.reshape(-1, 1).float(),
        "bboxes": xyxy2xywhn(objects.boxes, w=width, h=height),
        "batch_idx": objects.sample_index.float(),
    }


def _found(per_image: list[Tensor]) -> Instances:
    """Suppression's per-image lists folded into the one flat entity.

    Empty images keep their place: an image that found nothing still exists, and a
    consumer counting per image would otherwise read the next image's objects as its own.
    """
    if not per_image:
        return Instances(
            boxes=torch.zeros(0, 4),
            labels=torch.zeros(0, dtype=torch.int64),
            sample_index=torch.zeros(0, dtype=torch.int64),
            scores=torch.zeros(0),
        )
    stacked = torch.cat([one if len(one) else one.new_zeros(0, _DETECTION_COLUMNS) for one in per_image])
    index = torch.cat([torch.full((len(one),), position, dtype=torch.int64) for position, one in enumerate(per_image)])
    return Instances(
        boxes=stacked[:, :4],
        labels=stacked[:, 5].to(torch.int64),
        sample_index=index,
        scores=stacked[:, 4],
    )
