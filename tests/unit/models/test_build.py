"""The model section becomes a graph: a backbone from its declaration, one head per task, sized from both."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from torch import Tensor, nn

from src.config import ComponentConfig, HeadConfig, ModelConfig
from src.core import Axis, ModelOutput, Stream, TensorShape, TensorTree
from src.models import CompositeModel, Model
from src.models.build import build_model
from src.models.heads import ConvHead, LinearHead
from tests.unit.models.conftest import MAP_WIDTH, POOLED_WIDTH

CLASSES = TensorShape(axes=(Axis.CLASSES,), sizes=(3,))
DENSE = TensorShape(axes=(Axis.CLASSES, Axis.HEIGHT, Axis.WIDTH), sizes=(3, None, None))
SCALAR = TensorShape(axes=(), sizes=())
ENCODER = ComponentConfig(_target_="tests.unit.models.conftest.Encoder")


def composite(**overrides: Any) -> ModelConfig:
    return ModelConfig(name="composite", backbone=ENCODER, **overrides)


def head(name: str = "linear", stream: str = Stream.POOLED, **params: Any) -> HeadConfig:
    return HeadConfig(name=name, stream=stream, **params)


def projection(head: nn.Module) -> nn.Linear | nn.Conv2d:
    return next(module for module in head.modules() if isinstance(module, nn.Linear | nn.Conv2d))


class TestSizing:
    @pytest.mark.parametrize(
        ("name", "stream", "shape", "kind", "width"),
        [
            pytest.param("linear", Stream.POOLED, CLASSES, LinearHead, POOLED_WIDTH, id="a vector onto classes"),
            pytest.param("conv", Stream.DECODER, DENSE, ConvHead, MAP_WIDTH, id="a map onto dense classes"),
        ],
    )
    def test_a_head_is_sized_from_the_stream_it_reads_and_the_output_its_task_needs(
        self, name: str, stream: str, shape: TensorShape, kind: type[nn.Module], width: int
    ) -> None:
        """Neither width is ever written in config: one comes from the backbone, the other from the data."""
        model = build_model(composite(), {"t": head(name, stream)}, {"t": shape})

        assert isinstance(model, CompositeModel)
        built = model.heads["t"]
        assert isinstance(built, kind)
        sizes = projection(built)
        assert (sizes.in_channels if isinstance(sizes, nn.Conv2d) else sizes.in_features) == width
        assert (sizes.out_channels if isinstance(sizes, nn.Conv2d) else sizes.out_features) == 3

    def test_a_task_with_no_classes_gets_one_output(self) -> None:
        model = build_model(composite(), {"age": head()}, {"age": SCALAR})

        assert isinstance(model, CompositeModel) and projection(model.heads["age"]).out_features == 1

    def test_the_built_graph_runs_every_head_it_was_given(self, images: dict[str, Tensor]) -> None:
        model = build_model(
            composite(), {"label": head(), "mask": head("conv", Stream.DECODER)}, {"label": CLASSES, "mask": DENSE}
        )

        output = model(images)

        assert isinstance(output, ModelOutput) and set(output.outputs) == {"label", "mask"}

    @pytest.mark.parametrize("restated", ["in_features", "out_features"])
    def test_a_size_the_framework_derives_is_refused_where_it_was_restated(self, restated: str) -> None:
        declared = HeadConfig.model_validate({"name": "linear", "stream": Stream.POOLED, restated: 3})

        with pytest.raises(ValueError, match=restated):
            build_model(composite(), {"t": declared}, {"t": CLASSES})

    @pytest.mark.parametrize(
        ("name", "stream", "shape"),
        [
            pytest.param("linear", Stream.DECODER, DENSE, id="a pooled head on a feature map"),
            pytest.param("conv", Stream.POOLED, CLASSES, id="a dense head on a pooled vector"),
        ],
    )
    def test_a_head_that_cannot_read_its_stream_is_refused_before_the_first_batch(
        self, name: str, stream: str, shape: TensorShape
    ) -> None:
        """Both shapes are declared, so this is a build-time contradiction, not a matmul error mid-run."""
        with pytest.raises(ValueError, match="channels"):
            build_model(composite(), {"t": head(name, stream)}, {"t": shape})

    def test_a_head_reading_a_stream_the_backbone_does_not_publish_is_told_what_there_is(self) -> None:
        with pytest.raises(ValueError, match="pooled, decoder"):
            build_model(composite(), {"t": head(stream="logits")}, {"t": CLASSES})

    def test_a_head_without_a_stream_to_read_is_refused_by_task(self) -> None:
        with pytest.raises(ValueError, match="feature stream"):
            build_model(composite(), {"t": HeadConfig(name="linear")}, {"t": CLASSES})

    def test_a_task_with_a_head_but_no_output_is_refused_before_anything_is_built(self) -> None:
        with pytest.raises(ValueError, match="outputs are known"):
            build_model(composite(), {"t": head(), "other": head()}, {"t": CLASSES})


class TestNativeHead:
    def test_the_backbone_builds_its_own_head_when_a_task_asks_for_it(self) -> None:
        model = build_model(composite(), {"t": head("native")}, {"t": CLASSES})

        assert isinstance(model, CompositeModel)
        assert isinstance(model.heads["t"], nn.Linear) and model.heads["t"].out_features == 3

    def test_a_backbone_offering_none_says_so_instead_of_quietly_using_another_head(self) -> None:
        with pytest.raises(LookupError, match="native head"):
            build_model(composite(), {"t": head("native", Stream.DECODER)}, {"t": DENSE})

    def test_native_takes_no_arguments_because_the_backbone_builds_it(self) -> None:
        with pytest.raises(ValueError, match="native"):
            build_model(composite(), {"t": head("native", kernel_size=3)}, {"t": CLASSES})


class TestFamilies:
    def test_a_model_that_arrives_whole_brings_its_own_heads(self) -> None:
        model = build_model(ModelConfig(_target_="tests.unit.models.test_build.Whole"), {"t": head()}, {"t": CLASSES})

        assert isinstance(model, Whole)

    @pytest.mark.parametrize(
        ("declared", "reason"),
        [
            pytest.param(ModelConfig(name="composite"), "backbone", id="a family without a backbone"),
            pytest.param(
                ModelConfig(_target_="tests.unit.models.test_build.NotAModel"),
                "not a Model",
                id="an import path to something else entirely",
            ),
            pytest.param(
                ModelConfig(_target_="tests.unit.models.test_build.Whole", backbone=ENCODER),
                "brings its own",
                id="a whole model with a backbone",
            ),
            pytest.param(
                ModelConfig(name="composite", backbone=ComponentConfig(_target_="tests.unit.models.test_build.Whole")),
                "not a Backbone",
                id="a backbone position holding a network",
            ),
        ],
    )
    def test_refuses_a_model_section_that_contradicts_itself(self, declared: ModelConfig, reason: str) -> None:
        with pytest.raises((ValueError, TypeError), match=reason):
            build_model(declared, {"t": head()}, {"t": CLASSES})


class Whole(Model):
    """A network reached by import path: it owns its heads, so no task's head is built for it."""

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        return ModelOutput(outputs={"t": next(iter(inputs.values()))})


class NotAModel(nn.Module):
    """Something a `_target_` may point at by mistake: it answers nothing a run can ask a model for."""

    def forward(self, inputs: Mapping[str, TensorTree]) -> None:
        return None
