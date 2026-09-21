"""The model section becomes a graph: a backbone from its declaration, one head per task, sized from both."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from src.config import ComponentConfig, HeadConfig, ModelConfig
from src.core import Axis, ModelOutput, Stream, TensorShape, TensorTree
from src.models import Backbone, CompositeModel, Model
from src.models.backbones.multiencoder import MultiEncoderBackbone
from src.models.base import Encoded
from src.models.build import build_head, build_model
from src.models.heads import ConvHead, ExpandedHead, LinearHead, Mlp, StackedHeads
from src.models.necks.projector import Projector
from tests.unit.models.conftest import MAP_WIDTH, NARROW_WIDTH, POOLED_WIDTH, Encoder, Sentences

CLASSES = TensorShape(axes=(Axis.CLASSES,), sizes=(3,))
DENSE = TensorShape(axes=(Axis.CLASSES, Axis.HEIGHT, Axis.WIDTH), sizes=(3, None, None))
SCALAR = TensorShape(axes=(), sizes=())
ENCODER = ComponentConfig(_target_="tests.unit.models.conftest.Encoder")
TOWERS = ComponentConfig(_target_="tests.unit.models.test_build.TwoTowers")
BROUGHT = ComponentConfig(name="projector", width=6, stream="first")


def composite(**overrides: Any) -> ModelConfig:
    return ModelConfig(name="composite", backbone=ENCODER, **overrides)


def head(name: str = "linear", stream: str | list[str] = Stream.POOLED, **params: Any) -> HeadConfig:
    return HeadConfig(name=name, stream=stream, **params)


def paired() -> MultiEncoderBackbone:
    """Two towers of deliberately different widths, which is what a head over both is sized by."""
    return MultiEncoderBackbone({"image": Encoder(), "text": Sentences()})


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

    def test_a_head_declared_over_several_streams_is_one_of_it_per_stream_sized_by_each(self) -> None:
        """A pair of towers is read by a pair of heads, and neither width is written in the declaration.

        Declared the other way round from how the towers were, and deliberately not in alphabetical
        order: what the schema promises is that the order *written* is the order the answers arrive in,
        and every specimen that happened to be alphabetical left that promise resting on nothing.
        """
        built = build_head("t", head("linear", ["text_pooled", "image_pooled"]), CLASSES, Encoded(paired()))

        assert isinstance(built.head, StackedHeads)
        assert [projection(part).in_features for part in built.head.heads.values()] == [NARROW_WIDTH, POOLED_WIDTH]
        assert built.streams == ("text_pooled", "image_pooled")

    def test_a_head_over_one_stream_is_that_head_rather_than_a_stack_holding_it(self) -> None:
        """Every run this framework has is this one; reading a pair is what the other shape is for."""
        built = build_head("t", head("linear", Stream.POOLED), CLASSES, Encoded(Encoder()))

        assert isinstance(built.head, LinearHead)

    def test_a_task_whose_output_names_no_width_is_refused_rather_than_given_one(self) -> None:
        """A head makes some number of values per position; a shape naming none cannot size one.

        Answering 1 would be a guess that builds, trains and reports — the silent fallback this
        framework treats as a defect — so a shape it cannot read is named instead.
        """
        with pytest.raises(ValueError, match="age"):
            build_model(composite(), {"age": head()}, {"age": SCALAR})

    def test_a_task_whose_output_names_two_widths_is_refused_by_name(self) -> None:
        """Which of them a head is built at is not the framework's to pick; both are named."""
        both = TensorShape(axes=(Axis.CLASSES, Axis.CHANNELS), sizes=(3, 8))

        with pytest.raises(ValueError, match="classes, channels"):
            build_model(composite(), {"t": head()}, {"t": both})

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


class TestNeck:
    """What a run puts between its backbone and the heads that read it."""

    def test_a_head_over_a_brought_stream_is_sized_from_the_width_the_neck_declared(self) -> None:
        """The whole of what a neck is for: the head reads the number the run wrote down rather than the
        one the library chose, and nothing in the head's own declaration says so.

        The children are asserted beside it because they are the contract the wrapper broke: a path a
        `freeze` or an `adapter` writes goes on naming what it named before the neck was declared.
        """
        declared = composite(neck=ComponentConfig(name="projector", width=6, stream=Stream.POOLED))

        model = build_model(declared, {"t": head()}, {"t": CLASSES})

        assert isinstance(model, CompositeModel)
        assert projection(model.heads["t"]).in_features == 6
        assert [name for name, _ in model.named_children()] == ["backbone", "neck", "heads"]

    def test_a_neck_is_built_from_the_words_a_config_writes_rather_than_the_types_it_cannot(self) -> None:
        """`norm` arrives from a config as a string and `hidden_features` as a list, and `instantiate` is
        untyped passthrough that `make typecheck` sees nothing of. This is the one place the
        declaration's own vocabulary is held against what the neck accepts."""
        declared = composite(
            neck=ComponentConfig(
                name="projector", width=6, stream=Stream.POOLED, hidden_features=[10], norm="layer_norm"
            )
        )

        model = build_model(declared, {"t": head()}, {"t": CLASSES})

        assert isinstance(model, CompositeModel)
        assert isinstance(model.neck, Projector) and isinstance(model.neck.norm, nn.LayerNorm)
        assert projection(model.heads["t"]).in_features == 6

    def test_a_width_no_layer_could_answer_with_is_refused_in_the_name_of_the_position(self) -> None:
        """A neck builds a stack it did not write, and the class that owns that refusal has never heard
        of `model.neck`: a reader who is told `hidden_features` is wrong still has to find which of a
        run's sections wrote it. The position is added by whoever knows it, as `tasks.<task>.head` is."""
        declared = composite(
            neck=ComponentConfig(name="projector", width=6, stream=Stream.POOLED, hidden_features=[4, 0])
        )

        with pytest.raises(ValueError, match=r"model\.neck.*at least one"):
            build_model(declared, {"t": head()}, {"t": CLASSES})

    def test_the_libraries_own_head_over_a_stream_a_neck_replaced_is_refused_by_name(self) -> None:
        """That classifier is sized for the features this run put a neck in front of. Built anyway it
        would read a feature space that is gone, and — since a projection happens to fit whatever it is
        handed once the widths line up — the run would train and report on a number that means nothing.
        """
        declared = ModelConfig(name="composite", backbone=TOWERS, neck=BROUGHT)

        with pytest.raises(ValueError, match="replaced"):
            build_model(declared, {"t": head("native", "first")}, {"t": CLASSES})

    def test_the_libraries_own_head_over_a_stream_the_neck_passed_through_is_built_as_it_always_was(self) -> None:
        """The rule is per stream and not per run: a neck that brought one tower's features changed
        nothing about the tower beside it, so the library's head over that one is still its head.

        On a double because no family shipped here can be asked this — each publishes either one stream
        or several with no classifier of its own — and the arrangement is what a two-headed network from
        somebody's library leaves behind.
        """
        declared = ModelConfig(name="composite", backbone=TOWERS, neck=BROUGHT)

        model = build_model(declared, {"t": head("native", "second")}, {"t": CLASSES})

        assert isinstance(model, CompositeModel)
        assert projection(model.heads["t"]).in_features == NARROW_WIDTH

    def test_a_run_that_declares_no_neck_registers_none_and_writes_the_checkpoint_it_always_wrote(self) -> None:
        """Measured: an attribute left `None` is inert — it reaches neither `state_dict` nor
        `named_children` — so the position costs a run that does not declare one exactly nothing."""
        model = build_model(composite(), {"t": head()}, {"t": CLASSES})

        assert isinstance(model, CompositeModel)
        assert model.neck is None
        assert [name for name, _ in model.named_children()] == ["backbone", "heads"]
        assert not [name for name in model.state_dict() if name.startswith("neck")]

    def test_a_stream_nobody_publishes_is_refused_in_the_name_of_whichever_published_last(self) -> None:
        """A refusal that reads a list of streams back names whoever wrote that list, and with a neck
        declared that is the neck: the backbone's own list is the one the run stopped reading the moment
        it put something after it, so naming the backbone would send a reader to the wrong class.
        """
        backbone = Encoder()
        brought = Projector(backbone_shapes=backbone.feature_shapes, width=6, stream=Stream.POOLED)

        with pytest.raises(ValueError, match="but Projector publishes"):
            build_head("t", head(stream="absent"), CLASSES, Encoded(backbone, brought))

    def test_a_head_cannot_be_sized_from_a_stream_another_head_publishes(self) -> None:
        """A head's hidden width is a stream a term may compare and never one a head may read: heads are
        sized from what the encoding half published, and a head reading a head would be an order nobody
        declared. Held by the refusal that is already here, because a head's streams never enter
        `Encoded.published` at all — this is what pins that they do not.
        """
        with pytest.raises(ValueError, match="reads 'species_hidden_0', but Encoder publishes"):
            build_head("t", head(stream="species_hidden_0"), CLASSES, Encoded(Encoder()))


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
            pytest.param(
                ModelConfig(
                    _target_="tests.unit.models.test_build.Whole",
                    neck=ComponentConfig(name="projector", width=8),
                ),
                "reads its own features",
                id="a whole model with a neck",
            ),
            pytest.param(
                ModelConfig(
                    name="composite",
                    backbone=ENCODER,
                    neck=ComponentConfig(_target_="tests.unit.models.test_build.NotANeck"),
                ),
                "not a Neck",
                id="a neck position holding something that publishes nothing",
            ),
        ],
    )
    def test_refuses_a_model_section_that_contradicts_itself(self, declared: ModelConfig, reason: str) -> None:
        with pytest.raises((ValueError, TypeError), match=reason):
            build_model(declared, {"t": head()}, {"t": CLASSES})


class TwoTowers(Backbone):
    """Two pooled streams, each with a classifier of its own, which is what a two-headed network leaves.

    Here rather than in `conftest` because it is the one arrangement no shipped family has: `timm` and
    `hf_text` publish one stream each, `smp` publishes two that are spatial, and `multiencoder` publishes
    two and offers no head of its own. Only over a family like this do a neck over one stream and the
    library's own head over the other meet in one run.
    """

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        return {
            "first": TensorShape(axes=(Axis.CHANNELS,), sizes=(POOLED_WIDTH,)),
            "second": TensorShape(axes=(Axis.CHANNELS,), sizes=(NARROW_WIDTH,)),
        }

    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        raise NotImplementedError

    def native_head(self, stream: str, out_features: int) -> nn.Module | None:
        return nn.Linear(POOLED_WIDTH if stream == "first" else NARROW_WIDTH, out_features)


class Whole(Model):
    """A network reached by import path: it owns its heads, so no task's head is built for it."""

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        return ModelOutput(outputs={"t": next(iter(inputs.values()))})


class NotANeck(nn.Module):
    """Something a `_target_` in the neck position may point at: it takes the streams and brings none.

    Takes them, because a neck that did not would be refused a step earlier by the framework naming the
    argument it tried to hand over — and that refusal is not the one this stands for.
    """

    def __init__(self, backbone_shapes: Mapping[str, TensorShape]) -> None:
        super().__init__()
        self.backbone_shapes = backbone_shapes

    def forward(self, features: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        return features


class NotAModel(nn.Module):
    """Something a `_target_` may point at by mistake: it answers nothing a run can ask a model for."""

    def forward(self, inputs: Mapping[str, TensorTree]) -> None:
        return None


CARRIED_ROW, CARRIED_BIAS = 0.5, 0.25


class Started(Encoder):
    """An encoder built from a weight file that carried a classifier, which is what `checkpoint_path` leaves.

    The carried tensors are what the shipped backbones hand back: everything the file named that a
    headless graph had no place for. Written out here rather than loaded from one, because what this is
    about is the builder, and a real file would only be a slower way to say the same shapes.
    """

    def __init__(self, rows: int = 2, width: int = POOLED_WIDTH) -> None:
        super().__init__()
        self.carried_head = {
            "fc.weight": torch.full((rows, width), CARRIED_ROW),
            "fc.bias": torch.full((rows,), CARRIED_BIAS),
        }


class TestAHeadDeclaredWithItsOwnFile:
    """Weights prepared for exactly this head, rather than a classifier a backbone's file happened to carry."""

    def built(self, tmp_path: Path, holds: Mapping[str, Tensor], **declared: Any) -> nn.Module:
        kept = tmp_path / "head.pt"
        torch.save(dict(holds), kept)
        return build_head("label", head(checkpoint_path=str(kept), **declared), CLASSES, Encoded(Encoder())).head

    def tail(self, hidden_features: list[int]) -> Mapping[str, Tensor]:
        """A head of the declared shape, holding numbers no freshly built one would have."""
        made = Mlp(in_features=POOLED_WIDTH, out_features=3, hidden_features=hidden_features)
        return {name: torch.full_like(value, 0.5) for name, value in made.state_dict().items()}

    def test_a_head_declared_with_a_file_holds_the_weights_that_file_names(self, tmp_path: Path) -> None:
        """A head that started from a file and one that started from `seed` are the same object to every
        reader below it; what separates them is only whether the file's numbers are in it."""
        written = self.tail([4])

        built = self.built(tmp_path, written, name="mlp", hidden_features=[4])

        assert all(torch.equal(value, written[name]) for name, value in built.state_dict().items())

    def test_a_file_is_read_beside_a_head_that_names_no_such_argument(self, tmp_path: Path) -> None:
        """The file is a position the builder fills, not an argument every head has to accept: `linear`
        takes two widths and nothing else, and declaring a file beside it still builds."""
        made = LinearHead(in_features=POOLED_WIDTH, out_features=3)

        assert self.built(tmp_path, made.state_dict())

    @pytest.mark.parametrize(
        ("holding", "refused"),
        [
            pytest.param("another head", "does not hold this head", id="the names of some other head"),
            pytest.param("another width", r"layers\.0\.bias \[9\]", id="this head's names at another width"),
        ],
    )
    def test_a_file_that_is_not_this_head_is_refused_while_the_run_is_assembled(
        self, tmp_path: Path, holding: str, refused: str
    ) -> None:
        """A head half from a file is not that file's head, and a run that filled two layers of three
        reports every number it prints as though it had started warm."""
        holds = LinearHead(POOLED_WIDTH, 3).state_dict() if holding == "another head" else self.tail([9])

        with pytest.raises(ValueError, match=refused):
            self.built(tmp_path, holds, name="mlp", hidden_features=[4])


class TestStartedFromAFile:
    """A file's classifier reaches the head the run declared, and the class space is allowed to have grown."""

    def built(self, task: str = "label", classes: int = 3, **carried: Any) -> nn.Module:
        backbone = Started(**carried)
        return build_head(task, head(), TensorShape(axes=(Axis.CLASSES,), sizes=(classes,)), Encoded(backbone)).head

    def test_a_file_carrying_every_class_the_task_declares_fills_the_head_it_built(self) -> None:
        """Nothing grew, so nothing is appended: the declared head, with the rows the file already had."""
        built = self.built(classes=3, rows=3)

        assert isinstance(built, LinearHead)
        assert torch.equal(built.projection.weight, torch.full((3, POOLED_WIDTH), CARRIED_ROW))

    def test_a_task_declaring_more_classes_than_the_file_keeps_what_was_learned_and_appends_the_rest(self) -> None:
        """The declared vocabulary pins index to name, so the carried rows go in at the indices they had."""
        built = self.built(classes=5, rows=2)

        assert isinstance(built, ExpandedHead)
        assert torch.equal(projection(built.base).weight, torch.full((2, POOLED_WIDTH), CARRIED_ROW))
        assert projection(built.novel).weight.shape == (3, POOLED_WIDTH)

    def test_narrowing_the_class_space_is_refused_because_nobody_said_which_classes_stay(self) -> None:
        """Dropping rows is a mapping, and a transplant that guessed one would report under the wrong names."""
        with pytest.raises(ValueError, match="5"):
            self.built(classes=3, rows=5)

    def test_a_classifier_read_from_another_feature_space_is_refused(self) -> None:
        """Rows of the right count over the wrong width are not this network's classifier at all."""
        with pytest.raises(ValueError, match="no part"):
            self.built(classes=3, rows=3, width=POOLED_WIDTH + 1)

    def test_a_head_only_half_of_which_the_file_fills_is_refused(self) -> None:
        """The rule the rest of this framework keeps: what a file does not carry would be a reading of `seed`."""
        backbone = Started(rows=3)
        backbone.carried_head = {"fc.weight": backbone.carried_head["fc.weight"]}

        with pytest.raises(ValueError, match=r"projection\.bias"):
            build_head("label", head(), TensorShape(axes=(Axis.CLASSES,), sizes=(3,)), Encoded(backbone))

    def test_a_file_carrying_more_than_one_head_is_refused_rather_than_read_for_one(self) -> None:
        """smp writes a segmentation head and an auxiliary classifier, and a file may hold both.

        Neither is the other's, and nothing in the file says which this task continues — so the run is
        refused by name rather than started from rows trained to answer another question.
        """
        backbone = Started(rows=3)
        backbone.carried_head = {**backbone.carried_head, "aux.weight": torch.zeros(5, POOLED_WIDTH)}

        with pytest.raises(ValueError, match="more than one head"):
            build_head("label", head(), TensorShape(axes=(Axis.CLASSES,), sizes=(3,)), Encoded(backbone))

    def test_a_backbone_that_started_from_no_file_builds_the_head_as_declared(self) -> None:
        """Every ordinary run: nothing was carried, so nothing is transplanted and nothing is checked."""
        built = build_head("label", head(), TensorShape(axes=(Axis.CLASSES,), sizes=(3,)), Encoded(Encoder())).head

        assert isinstance(built, LinearHead) and built.projection.weight.shape == (3, POOLED_WIDTH)


def test_a_carried_classifier_with_more_than_one_head_to_land_in_is_refused_by_name() -> None:
    """One file carries one classifier, and two tasks reading it would both claim the same rows.

    Named by the weights rather than by the key that brought them: where `checkpoint_path` is written
    depends on where the backbone sits, and a wrapped one is declared at `model.backbone.backbone` —
    so a message spelling the plain path out would send that run to edit a key it does not have.
    """
    shapes = {"a": TensorShape(axes=(Axis.CLASSES,), sizes=(3,)), "b": TensorShape(axes=(Axis.CLASSES,), sizes=(3,))}
    named = "weights this backbone started from carry one classifier, and this run declares a, b"

    with pytest.raises(ValueError, match=named):
        build_model(
            ModelConfig(name="composite", backbone=ComponentConfig(_target_=f"{__name__}.Started")),
            {"a": head(), "b": head()},
            shapes,
        )


class TestACarriedClassifierUnderANeck:
    """Rows a weight file carried were read off the features the library published, and a neck replaces them."""

    def test_a_head_over_a_brought_stream_starts_fresh_and_the_run_is_told(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Said rather than refused: the file is named for the encoder's weights and they arrive either
        way, but a warm start a run asked for and silently did not get is the defect this says out loud.
        """
        backbone = Started()
        brought = Projector(backbone_shapes=backbone.feature_shapes, width=6, stream=Stream.POOLED)

        with caplog.at_level(logging.INFO):
            built = build_head("t", head(), CLASSES, Encoded(backbone, brought))

        said = "\n".join(caplog.messages)
        assert projection(built.head).in_features == 6
        assert "starts fresh" in said
        assert f"brought from {POOLED_WIDTH} to 6" in said, said

    def test_a_head_over_a_stream_the_neck_passed_through_starts_from_the_rows_as_it_always_did(self) -> None:
        """Per stream and not per run: a neck that brought the pooled vector left the map beside it
        alone, so a head over the map reads the file exactly as it read it before any neck existed."""
        backbone = Started()
        backbone.carried_head = {
            "seg.weight": torch.full((3, MAP_WIDTH, 1, 1), CARRIED_ROW),
            "seg.bias": torch.full((3,), CARRIED_BIAS),
        }
        brought = Projector(backbone_shapes=backbone.feature_shapes, width=6, stream=Stream.POOLED)

        built = build_head("t", head("conv", Stream.DECODER), DENSE, Encoded(backbone, brought))

        assert torch.equal(projection(built.head).weight, torch.full((3, MAP_WIDTH, 1, 1), CARRIED_ROW))

    def test_two_heads_over_a_brought_stream_are_not_two_claimants_to_rows_neither_could_hold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A file carrying one classifier is refused a run of several tasks, because which of them it was
        trained to answer is written nowhere. Over a stream a neck brought there is no such question: the
        rows were read off a width that is gone, so no head could hold them however few there were, and
        refusing sends the run to drop a `checkpoint_path` it wants for the encoder's own weights.
        """
        declared = ModelConfig(
            name="composite",
            backbone=ComponentConfig(_target_=f"{__name__}.Started"),
            neck=ComponentConfig(name="projector", width=6, stream=Stream.POOLED),
        )

        with caplog.at_level(logging.INFO):
            built = build_model(declared, {"a": head(), "b": head()}, {"a": CLASSES, "b": CLASSES})

        assert isinstance(built, CompositeModel)
        assert [projection(built.heads[task]).in_features for task in ("a", "b")] == [6, 6]
        assert "starts fresh" in "\n".join(caplog.messages)
