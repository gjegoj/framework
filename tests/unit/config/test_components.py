"""One grammar for every component: ``name`` or ``_target_``, every other key a constructor argument."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.config import ComponentConfig, DistilledLossConfig, HeadConfig, LearnerConfig, ModelConfig

STREAMS_REFUSED = [
    pytest.param(" pooled", id="padded"),
    pytest.param(["image_pooled", " text_pooled"], id="padded among several"),
    pytest.param(["pooled", "pooled"], id="one feature named twice"),
    pytest.param([], id="a list naming nothing"),
]
"""One table for the two declarations that name a backbone's features, because the rule they read is one."""


@pytest.mark.parametrize(
    ("declared", "name", "target", "params"),
    [
        pytest.param("cross_entropy", "cross_entropy", None, {}, id="bare name"),
        pytest.param({"name": "dice", "smooth": 1.0}, "dice", None, {"smooth": 1.0}, id="name with arguments"),
        pytest.param({"_target_": "my.Loss", "gamma": 2}, None, "my.Loss", {"gamma": 2}, id="import path"),
        pytest.param(
            {"name": "x", "inner": {"_target_": "my.Inner"}},
            "x",
            None,
            {"inner": {"_target_": "my.Inner"}},
            id="nested stays raw",
        ),
    ],
)
def test_reads_every_spelling_the_same_way(
    declared: Any, name: str | None, target: str | None, params: dict[str, Any]
) -> None:
    component = ComponentConfig.model_validate(declared)

    assert (component.name, component.import_path, component.params) == (name, target, params)
    assert component.spelled == (name or target)


@pytest.mark.parametrize(
    "declared",
    [
        pytest.param({}, id="neither"),
        pytest.param({"name": "a", "_target_": "b"}, id="both"),
        pytest.param({"name": ""}, id="blank name"),
        pytest.param({"name": "a", "_partial_": True}, id="hydra meta key"),
        pytest.param({"name": "a", "_args_": [1]}, id="positional arguments"),
        pytest.param({"name": "a", "import_path": "b"}, id="alias spelled by field name"),
        pytest.param(42, id="not a mapping"),
    ],
)
def test_refuses_an_ambiguous_or_foreign_declaration(declared: Any) -> None:
    with pytest.raises(ValidationError):
        ComponentConfig.model_validate(declared)


class TestHead:
    """A head declares the features it reads: one, or several where its task is learned over a pair of them."""

    @pytest.mark.parametrize(
        ("declared", "streams"),
        [
            pytest.param({"name": "linear", "stream": "pooled"}, ("pooled",), id="one"),
            pytest.param(
                {"name": "linear", "stream": ["image_pooled", "text_pooled"]},
                ("image_pooled", "text_pooled"),
                id="several, in the order they were written",
            ),
            pytest.param({"name": "native"}, (), id="none, for the task's own default to fill in"),
        ],
    )
    def test_reads_one_name_or_several_the_same_way(self, declared: Any, streams: tuple[str, ...]) -> None:
        """Whichever shape the declaration took, what builds the head reads one: `stream: pooled` is a list of one."""
        assert HeadConfig.model_validate(declared).streams == streams

    @pytest.mark.parametrize("stream", STREAMS_REFUSED)
    def test_refuses_anything_but_distinct_names_of_features(self, stream: Any) -> None:
        """A feature named twice would build two heads over one stream, which can only be a slip of the pen."""
        with pytest.raises(ValidationError):
            HeadConfig(name="linear", stream=stream)


class TestModel:
    """A network by name, and the child positions a composite fills from their own registries."""

    def test_a_neck_is_a_position_of_its_own_and_never_reaches_the_model_s_constructor(self) -> None:
        """One declaration cannot be two statements of one thing: what the builder resolves and hands
        over ready is not also a keyword the model family would have to accept and know how to read."""
        declared = ModelConfig.model_validate(
            {"name": "composite", "backbone": {"name": "timm"}, "neck": {"name": "projector", "width": 512}}
        )

        assert declared.neck is not None
        assert declared.neck.params == {"width": 512}
        assert declared.params == {}


class TestDistilledTerm:
    """One term of what a run learns from a second network, and which of its answers the term compares."""

    def test_a_term_that_names_no_stream_compares_the_answers(self) -> None:
        """Which is what distilling here has always meant, so every config written before this key
        existed goes on meaning what it meant."""
        assert DistilledLossConfig.model_validate({"loss": "mse"}).streams == ()

    @pytest.mark.parametrize(
        ("declared", "streams"),
        [
            pytest.param({"loss": "mse", "stream": "pooled"}, ("pooled",), id="one"),
            pytest.param(
                {"loss": "mse", "stream": ["image_pooled", "text_pooled"]},
                ("image_pooled", "text_pooled"),
                id="several, in the order they were written",
            ),
        ],
    )
    def test_names_one_stream_or_several_the_same_way_a_head_does(
        self, declared: Any, streams: tuple[str, ...]
    ) -> None:
        """The same word in the same shape: it is the same question about the same names a backbone publishes."""
        assert DistilledLossConfig.model_validate(declared).streams == streams

    @pytest.mark.parametrize("stream", STREAMS_REFUSED)
    def test_refuses_anything_but_distinct_names_of_features(self, stream: Any) -> None:
        """One rule about stream names, read here and by a head; a second copy would be free to drift."""
        with pytest.raises(ValidationError):
            DistilledLossConfig.model_validate({"loss": "mse", "stream": stream})


class TestLearner:
    """The algorithm a run trains by, and what it is declared to learn from a second network."""

    def test_declares_one_objective_or_a_weighted_list_of_them(self) -> None:
        """The grammar `tasks.<name>.loss` already uses, because it is the same question asked of a
        second network: one term, or several with the shares they are worth."""
        one = LearnerConfig.model_validate({"name": "distillation", "loss": {"name": "kullback_leibler"}})
        several = LearnerConfig.model_validate(
            {
                "name": "distillation",
                "loss": [
                    {"loss": {"name": "kullback_leibler", "temperature": 3.0}},
                    {"loss": "mse", "weight": 5.0, "stream": "pooled"},
                ],
            }
        )

        assert isinstance(one.loss, ComponentConfig)
        assert isinstance(several.loss, list)
        assert [term.streams for term in several.loss] == [(), ("pooled",)]
        assert [term.weight for term in several.loss] == [1.0, 5.0]

    def test_a_stream_written_beside_a_single_objective_is_refused_where_the_list_would_carry_it(self) -> None:
        """One objective is the loss itself, and every other key beside it is that loss's own argument —
        so a `stream` written there would reach the constructor of a loss that has no such parameter,
        and the run would compare answers while its declaration says features.
        """
        with pytest.raises(ValidationError, match="stream"):
            LearnerConfig.model_validate({"name": "distillation", "loss": {"name": "mse", "stream": "pooled"}})

    def test_a_typed_position_never_reaches_the_constructor_s_own_arguments(self) -> None:
        """One declaration cannot be two statements of one thing: what the builder resolves is not also
        handed to the learner as a keyword it would have to accept."""
        declared = LearnerConfig.model_validate(
            {"name": "distillation", "weight": 2.0, "loss": [{"loss": "mse", "stream": "pooled"}]}
        )

        assert declared.params == {"weight": 2.0}
