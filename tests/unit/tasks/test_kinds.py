"""``TaskKind``: what one kind of task needs, stated in one class, and the components it builds."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import Tensor, nn

from src.core import Backbone, Batch, Features, Stream, TaskFacts
from src.losses import InfoNceCriterion, ProxyAngularCriterion
from src.models import CompositeModel, ConvHead, LinearHead, TaskComponents
from src.tasks import (
    BinaryClassification,
    BinarySegmentation,
    Classification,
    Contrastive,
    Detection,
    MetricLearning,
    MultilabelClassification,
    MultilabelSegmentation,
    Overrides,
    Ranking,
    Regression,
    Segmentation,
    Task,
    TaskKind,
)
from src.tasks.registry import task_kind_registry
from tests.support.entities import a_task
from tests.support.fakes import FakeEncoder, FlattenBackbone
from tests.support.narrowing import tensor

KINDS: dict[str, TaskKind] = {
    "classification": Classification(),
    "binary_classification": BinaryClassification(),
    "multilabel_classification": MultilabelClassification(),
    "regression": Regression(),
    "metric_learning": MetricLearning(),
    "segmentation": Segmentation(),
    "binary_segmentation": BinarySegmentation(),
    "multilabel_segmentation": MultilabelSegmentation(),
    "contrastive": Contrastive(),
    "detection": Detection(),
    "ranking": Ranking(),
}
"""The eleven familiar names, spelled as the config spells them."""

FACTS = TaskFacts(num_classes=3)
WIDTH = 8


def classified(kind: TaskKind | None = None, classes: int = 3) -> Task:
    return a_task(kind=kind, facts=TaskFacts(num_classes=classes))


@pytest.mark.parametrize(("name", "kind"), list(KINDS.items()))
def test_every_kind_is_reachable_from_config_by_name(name: str, kind: TaskKind) -> None:
    assert isinstance(task_kind_registry.create(name), type(kind))


def test_a_kind_declared_by_target_builds_like_a_shipped_one() -> None:
    """The extension path: one class, reachable with no edit to the framework."""
    from src.build import kind_of
    from src.config import TaskConfig

    declared = TaskConfig.model_validate({"kind": {"_target_": "tests.support.kinds.FocalClassification"}})

    assert isinstance(kind_of(declared), Classification)


def test_an_unknown_kind_is_refused_listing_the_known_ones() -> None:
    from src.build import kind_of
    from src.config import TaskConfig

    with pytest.raises(LookupError, match="classification"):
        kind_of(TaskConfig.model_validate({"kind": "object_finding"}))


# --- what each kind states -------------------------------------------------------------------


def test_out_features_follow_the_kind() -> None:
    assert Classification().out_features(TaskFacts(num_classes=10)) == 10
    assert MultilabelClassification().out_features(TaskFacts(num_classes=5)) == 5
    assert BinaryClassification().out_features(TaskFacts()) == 1
    assert Regression().out_features(TaskFacts()) == 1
    assert Regression().out_features(TaskFacts(class_values=(1.0, 2.0, 3.0))) == 3
    assert MetricLearning().out_features(TaskFacts()) is None


def test_a_kind_that_projects_onto_classes_refuses_to_build_without_their_count() -> None:
    with pytest.raises(LookupError, match="num_classes"):
        Classification().out_features(TaskFacts())


def test_only_metric_learning_has_no_target_adapter() -> None:
    """No target column means there is nothing to adapt; a declared vocabulary brings one back."""
    assert MetricLearning().target_adapter(TaskFacts()) is None
    assert MetricLearning().target_adapter(TaskFacts(num_classes=3)) is not None
    assert Classification().target_adapter(TaskFacts()) is not None
    assert Regression().target_adapter(TaskFacts()) is not None


def test_metric_kwargs_carry_the_label_semantics() -> None:
    assert Classification().metric_kwargs(TaskFacts(num_classes=3)) == {"task": "multiclass", "num_classes": 3}
    assert BinaryClassification().metric_kwargs(TaskFacts()) == {"task": "binary"}
    assert MultilabelClassification().metric_kwargs(TaskFacts(num_classes=5)) == {"task": "multilabel", "num_labels": 5}
    assert Regression().metric_kwargs(TaskFacts()) == {}


def test_metric_learning_defaults_to_proxies_over_labels_and_contrastive_to_infonce_over_views() -> None:
    """Each kind's default steps on what its own encoder and head produce: labelled embeddings against
    proxies sized by the facts, or stacked views against each other with no target at all."""
    kind = MetricLearning()
    carrier = torch.randn(4, 2, 8)

    assert isinstance(kind.loss(FACTS, WIDTH), ProxyAngularCriterion)
    assert isinstance(KINDS["contrastive"].loss(TaskFacts(), WIDTH), InfoNceCriterion)
    assert torch.equal(kind.activation(TaskFacts())(carrier), carrier)


def test_classification_bricks_work_together() -> None:
    kind = Classification()
    logits = torch.randn(4, 3)

    adapted = kind.target_adapter(FACTS)(torch.tensor([0, 1, 2, 0]))  # type: ignore[misc]
    loss = kind.loss(FACTS, WIDTH)(logits, adapted.for_loss)
    probabilities = kind.activation(FACTS)(logits)

    assert set(loss.parts) == {"ce"}
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4))


def test_binary_adapter_floats_the_loss_view_only() -> None:
    adapted = BinaryClassification().target_adapter(TaskFacts())(torch.tensor([0, 1, 1]))  # type: ignore[misc]

    assert adapted.for_loss.dtype == torch.float32
    assert adapted.for_metrics.dtype == torch.long


def test_binary_activation_squeezes_the_single_logit_flat_and_dense() -> None:
    activation = BinaryClassification().activation(TaskFacts())

    assert activation(torch.zeros(4, 1)).shape == (4,)
    assert torch.allclose(activation(torch.zeros(4, 1)), torch.full((4,), 0.5))
    assert activation(torch.zeros(2, 1, 8, 8)).shape == (2, 8, 8)


def test_regression_activation_squeezes_single_output_heads() -> None:
    predictions = Regression().activation(TaskFacts())(torch.tensor([[1.0], [2.0]]))

    assert torch.equal(predictions, torch.tensor([1.0, 2.0]))


def test_a_binned_regression_is_learned_as_a_distribution_and_read_back_as_a_number() -> None:
    facts = TaskFacts(class_values=(0.0, 1.0, 2.0))
    kind = Regression()

    predicted = kind.activation(facts)(torch.tensor([[0.0, 0.0, 10.0]]))
    loss = kind.loss(facts, WIDTH)(torch.zeros(1, 3), torch.tensor([[0.0, 0.0, 1.0]]))

    assert predicted.item() == pytest.approx(2.0, abs=1e-3)
    assert {"ce", "expectation"} <= set(loss.parts)


def test_classification_bricks_work_on_dense_logits() -> None:
    kind = Segmentation()
    logits = torch.randn(2, 3, 8, 8)

    loss = kind.loss(FACTS, WIDTH)(logits, torch.randint(0, 3, (2, 8, 8)))
    probabilities = kind.activation(FACTS)(logits)

    assert loss.total.shape == ()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2, 8, 8))


def test_the_default_encoder_is_the_kinds_own_fact() -> None:
    """What ``build_target_encoder`` builds when the task declares no ``target_encoder``."""
    assert {name: kind.default_encoder for name, kind in KINDS.items()} == {
        "classification": "label",
        "binary_classification": "scalar",
        "multilabel_classification": "multilabel",
        "regression": "scalar",
        "metric_learning": "label",
        "segmentation": "mask",
        "binary_segmentation": "mask",
        "multilabel_segmentation": "mask",
        "contrastive": None,
        "ranking": None,
        "detection": "boxes",
    }


SEGMENTATION_JUDGMENT = {"iou", "f1", "precision", "recall", "confusion_matrix"}


def test_every_kinds_default_judgment_reads_off_its_class() -> None:
    assert set(Classification().default_metrics) == {"f1", "precision", "recall", "confusion_matrix"}
    assert set(Segmentation().default_metrics) == SEGMENTATION_JUDGMENT
    assert set(BinarySegmentation().default_metrics) == SEGMENTATION_JUDGMENT
    assert set(Regression().default_metrics) == {"mae"}
    assert set(Detection().default_metrics) == {"map"}
    assert MetricLearning().default_metrics == {} and Contrastive().default_metrics == {}


def test_a_kind_declares_which_streams_its_head_reads() -> None:
    """A fact each kind states literally, in its own class."""
    assert Classification().streams == (Stream.FEATURES,)
    assert MetricLearning().streams == (Stream.FEATURES,)
    assert Contrastive().streams == (Stream.EMBEDDINGS,)
    assert Ranking().streams == (Stream.EMBEDDINGS,)
    assert Segmentation().streams == (Stream.DECODER,)
    assert Detection().streams is None  # the backbone's pyramid: its levels, its count, its order


def test_a_kind_says_whether_a_batch_transform_may_soften_its_target() -> None:
    """Soft labels break metric learning, and objects have no weighted sum."""
    assert Classification().mixable and BinaryClassification().mixable and Regression().mixable
    assert Segmentation().mixable
    assert not MetricLearning().mixable and not Contrastive().mixable and not Ranking().mixable
    assert not Detection().mixable


def test_a_kind_softens_its_target_into_what_a_weighted_sum_needs() -> None:
    """Class indices widen to one-hot at the class count; anything else is already a number."""
    widened = Classification().soften(torch.tensor([0, 2]), FACTS)
    assert torch.equal(widened, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    assert BinaryClassification().soften(torch.tensor([0, 1]), TaskFacts()).dtype == torch.float32
    assert torch.equal(Regression().soften(torch.tensor([1.5, 2.0]), TaskFacts()), torch.tensor([1.5, 2.0]))


def test_two_kinds_are_declared_but_not_yet_trainable_and_each_says_why() -> None:
    """Detection waits for its criterion (roadmap stages 3–4); multilabel segmentation for an encoder that
    produces a multi-hot mask. Both build a model and refuse to train it, by name, at build."""
    assert (reason := Detection().unavailable) is not None and "roadmap" in reason
    assert (reason := KINDS["multilabel_segmentation"].unavailable) is not None and "multi-hot" in reason
    assert all(
        kind.unavailable is None for name, kind in KINDS.items() if name not in {"detection", "multilabel_segmentation"}
    )


def test_the_kinds_with_nothing_to_show_per_sample_say_so() -> None:
    assert Classification().not_drawn is None and Segmentation().not_drawn is None
    for kind in (MetricLearning(), Contrastive(), Ranking(), Detection()):
        assert kind.not_drawn, type(kind).__name__


# --- heads ------------------------------------------------------------------------------------


def test_a_global_kind_builds_a_linear_head_of_the_requested_size() -> None:
    head = Classification().head(in_features=8, out_features=3)

    assert isinstance(head, LinearHead)
    assert head(torch.zeros(2, 8)).shape == (2, 3)


def test_metric_learning_serves_an_identity_head() -> None:
    """No width asked for is the metric contract — the embedding is the output."""
    assert isinstance(MetricLearning().head(in_features=16, out_features=None), nn.Identity)


def test_a_dense_kind_builds_a_conv_head_preserving_spatial_dims() -> None:
    head = Segmentation().head(in_features=16, out_features=3)

    assert isinstance(head, ConvHead)
    assert head(torch.zeros(2, 16, 8, 8)).shape == (2, 3, 8, 8)


@pytest.mark.parametrize("kind", [Classification(), Segmentation()])
def test_a_single_stream_kind_refuses_pyramid_widths_by_name(kind: TaskKind) -> None:
    """Sized from three widths, a linear or conv head would silently pick one of them."""
    with pytest.raises(ValueError, match=f"{type(kind).__name__} reads one stream"):
        kind.head(in_features=(64, 128, 256), out_features=3)


# --- components -------------------------------------------------------------------------------


def test_a_kind_builds_the_components_that_serve_its_task() -> None:
    components = Classification().components(classified(), FlattenBackbone(dim=12))

    assert isinstance(components, TaskComponents)
    assert components.head(torch.zeros(2, 12)).shape == (2, 3)
    assert components.weight == 1.0


def test_the_tasks_weight_flows_into_its_components() -> None:
    task = a_task(facts=FACTS, weight=0.5)

    assert Classification().components(task, FlattenBackbone(dim=12)).weight == 0.5


def test_missing_num_classes_names_the_task_and_hints_setup() -> None:
    with pytest.raises(LookupError, match="label"):
        Classification().components(a_task(), FlattenBackbone(dim=12))


class TwoStreamBackbone(Backbone):
    """Exposes a second stream and a native head for it."""

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        flat = inputs["image"].flatten(start_dim=1)
        return Features(streams={"features": flat, "extra": flat * 2})

    def feature_dims(self) -> Mapping[str, int]:
        return {"features": 12, "extra": 12}

    def native_head(
        self, streams: tuple[str, ...], in_features: int | tuple[int, ...], out_features: int
    ) -> nn.Module | None:
        if streams == ("extra",):
            assert isinstance(in_features, int)
            return nn.Linear(in_features, out_features)
        return None


def test_declared_streams_read_another_stream() -> None:
    components = Classification().components(classified(), TwoStreamBackbone(), Overrides(streams=("extra",)))

    assert components.streams == ("extra",)


def test_a_declared_native_head_uses_the_backbones_head() -> None:
    components = Classification().components(
        classified(), TwoStreamBackbone(), Overrides(streams=("extra",), head="native")
    )

    assert components.head(torch.zeros(2, 12)).shape == (2, 3)


def test_a_declared_head_on_a_task_that_projects_nothing_is_refused() -> None:
    """Metric learning's embedding *is* the output, so a named head has no width to build at."""
    with pytest.raises(ValueError, match="nothing to project onto"):
        MetricLearning().components(
            a_task(kind=MetricLearning()),
            FlattenBackbone(dim=12),
            Overrides(head=lambda inputs, outputs: LinearHead(12, 3)),
        )


def test_a_native_head_the_backbone_does_not_offer_fails_loud() -> None:
    with pytest.raises(LookupError, match="native"):
        Classification().components(classified(), FlattenBackbone(dim=12), Overrides(head="native"))


def test_a_native_head_is_used_as_it_is() -> None:
    """No wrapper around it: a freeze config names ``heads.<task>.base`` and a checkpoint holds
    ``heads.<task>.weight``, and either would be buried under a private attribute."""
    components = Classification().components(
        classified(), TwoStreamBackbone(), Overrides(streams=("extra",), head="native")
    )
    model = CompositeModel(backbone=TwoStreamBackbone(), components={"label": components})

    assert isinstance(components.head, nn.Linear)
    assert {key for key in model.state_dict() if key.startswith("heads.")} == {"heads.label.weight", "heads.label.bias"}


def test_a_declared_loss_replaces_the_kinds_default() -> None:
    from src.losses import FocalCriterion

    components = Classification().components(
        classified(), FlattenBackbone(dim=12), Overrides(loss=lambda facts, width: FocalCriterion(gamma=2.0))
    )

    assert isinstance(components.criterion, FocalCriterion)


class DecoderBackbone(Backbone):
    """A dense-capable fake: exposes a decoder stream with spatial dims."""

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        return Features(streams={"decoder": inputs["image"].repeat(1, 4, 1, 1)})

    def feature_dims(self) -> Mapping[str, int]:
        return {"decoder": 12}


def test_a_contrastive_task_builds_and_steps_without_targets() -> None:
    """Metric learning closes: two encoders, stacked embeddings, InfoNCE, no target column."""
    from src.models import MultiEncoderBackbone

    torch.manual_seed(0)
    backbone = MultiEncoderBackbone(
        encoders={"image": FakeEncoder("image", 4), "text": FakeEncoder("text", 6)}, embedding_dim=8
    )
    task = a_task(name="pair", kind=Contrastive())

    model = CompositeModel(backbone=backbone, components={"pair": task.kind.components(task, backbone)})
    loss, prediction, targets = model.step(
        Batch(inputs={"image": torch.randn(4, 5), "text": torch.randn(4, 7)}, targets={})
    )

    assert set(loss.parts) == {"pair/infonce"}
    assert loss.total.requires_grad
    assert tensor(prediction.outputs["pair"]).shape == (4, 2, 8)
    assert tensor(targets["pair"]).numel() == 0


def test_a_segmentation_task_builds_and_steps_end_to_end() -> None:
    torch.manual_seed(0)
    task = classified(Segmentation())
    backbone = DecoderBackbone()

    model = CompositeModel(backbone=backbone, components={"label": Segmentation().components(task, backbone)})
    loss, prediction, _ = model.step(
        Batch(inputs={"image": torch.randn(2, 3, 8, 8)}, targets={"label": torch.randint(0, 3, (2, 8, 8))})
    )

    assert set(loss.parts) == {"label/ce"}
    assert loss.total.requires_grad
    assert tensor(prediction.outputs["label"]).shape == (2, 3, 8, 8)


def test_built_components_run_inside_a_composite_model() -> None:
    torch.manual_seed(0)
    backbone = FlattenBackbone(dim=12)
    tasks = [classified(), a_task(name="score", kind=Regression(), weight=0.5)]

    model = CompositeModel(
        backbone=backbone, components={task.name: task.kind.components(task, backbone) for task in tasks}
    )
    loss, prediction, _ = model.step(
        Batch(
            inputs={"image": torch.randn(4, 3, 2, 2)},
            targets={"label": torch.tensor([0, 1, 2, 0]), "score": torch.rand(4)},
        )
    )

    assert set(loss.parts) == {"label/ce", "score/mse"}
    assert tensor(prediction.outputs["label"]).shape == (4, 3)
    assert tensor(prediction.outputs["score"]).shape == (4,)


class PyramidBackbone(Backbone):
    """Four levels under names of its own, a pooled vector, and a native head for the pyramid."""

    LEVELS = ("l4", "l7", "l9", "l11")

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        image = inputs["image"]
        levels = {name: image[:, :, ::step, ::step] for name, step in zip(self.LEVELS, (1, 2, 4, 8), strict=True)}
        return Features(streams={**levels, Stream.FEATURES: image.mean(dim=(2, 3))})

    def feature_dims(self) -> Mapping[str, int]:
        return {**dict.fromkeys(self.LEVELS, 3), Stream.FEATURES: 3}

    def pyramid(self) -> tuple[str, ...]:
        return self.LEVELS

    def native_head(
        self, streams: tuple[str, ...], in_features: int | tuple[int, ...], out_features: int
    ) -> nn.Module | None:
        return SumHead(streams, out_features) if streams == self.LEVELS else None


class SumHead(nn.Module):
    """A stand-in detection head: reads the mapping in the order it was built for, returns one tensor."""

    def __init__(self, streams: tuple[str, ...], out_features: int) -> None:
        super().__init__()
        self.streams = streams
        self._out = out_features

    def forward(self, features: Tensor | Mapping[str, Tensor]) -> Tensor:
        assert not isinstance(features, Tensor)
        pooled = [features[name].mean(dim=(2, 3)) for name in self.streams]
        return torch.stack(pooled, dim=1)[:, :, : self._out]


def test_detection_reads_the_backbones_pyramid_through_its_native_head() -> None:
    """The framework composes no detection head, so 'native' is the default — and the
    backbone, not the kind, says which levels there are, how many, and in what order."""
    components = Detection().components(classified(Detection()), PyramidBackbone())

    assert isinstance(components.head, SumHead)
    assert components.streams == ("l4", "l7", "l9", "l11")


def test_a_backbone_without_a_pyramid_is_refused_naming_it_and_the_task() -> None:
    with pytest.raises(LookupError, match="TwoStreamBackbone declares no pyramid, and task 'label' reads one"):
        Detection().components(classified(Detection()), TwoStreamBackbone())


def test_declared_streams_reach_a_custom_head_in_that_order_whatever_the_pyramid() -> None:
    components = Detection().components(
        classified(Detection()),
        PyramidBackbone(),
        Overrides(streams=("l11", "l7"), head=lambda widths, out: SumHead(("l11", "l7"), out)),
    )

    assert components.streams == ("l11", "l7")
    assert isinstance(components.head, SumHead)


def test_multilabel_segmentation_inherits_the_dense_head_and_changes_only_the_semantics() -> None:
    """Sharing is inheritance the author wrote: no matrix decides whether the pairing is legal."""
    kind = MultilabelSegmentation()

    assert kind.shape is Segmentation().shape
    assert isinstance(kind.head(16, 3), ConvHead)
    assert kind.metric_kwargs(FACTS) == {"task": "multilabel", "num_labels": 3}
    assert kind.activation(FACTS)(torch.zeros(2, 3, 4, 4)).shape == (2, 3, 4, 4)
