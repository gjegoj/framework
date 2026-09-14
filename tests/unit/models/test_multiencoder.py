"""Several networks side by side: each reads the input it was declared for, under a name of its own."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import Tensor, nn

from src.core import Stream, TensorShape, TensorTree
from src.models import Backbone
from src.models.backbones.multiencoder import MultiEncoderBackbone
from tests.unit.models.conftest import MAP, NARROW, SIDE, VECTOR, Encoder, Sentences

BATCH = 2


def towers() -> dict[str, Backbone]:
    return {"image": Encoder(), "text": Sentences()}


def batch() -> dict[str, TensorTree]:
    """One batch carrying both inputs, since a pairing reads each with the tower declared for it."""
    return {
        "image": torch.arange(BATCH * 3 * SIDE * SIDE, dtype=torch.float32).reshape(BATCH, 3, SIDE, SIDE),
        "text": torch.arange(BATCH * 4, dtype=torch.float32).reshape(BATCH, 4),
    }


class Publishes(Backbone):
    """A tower publishing whatever name it is told to, for the one rule that is about names and nothing else."""

    def __init__(self, *streams: str) -> None:
        super().__init__()
        self.streams = streams

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        return dict.fromkeys(self.streams, VECTOR)

    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        return dict.fromkeys(self.streams, torch.zeros(1))


def test_every_stream_is_published_under_the_tower_that_made_it() -> None:
    """Both towers publish `pooled`, so what tells the two apart is the name the declaration gave each.

    The shapes themselves pass through untouched, widths included: what a head over one tower is built
    from is what that tower alone publishes, which is why the two here are deliberately not equal.
    """
    published = MultiEncoderBackbone(towers()).feature_shapes

    assert dict(published) == {"image_pooled": VECTOR, "image_decoder": MAP, "text_pooled": NARROW}


def test_each_tower_answers_from_the_input_it_reads_and_the_answers_arrive_side_by_side() -> None:
    """One pass gives every stream a head could be built over, each exactly what its own tower answers."""
    encoders = towers()

    answered = MultiEncoderBackbone(encoders)(batch())

    assert set(answered) == {"image_pooled", "image_decoder", "text_pooled"}
    assert torch.equal(answered["text_pooled"], encoders["text"](batch())[Stream.POOLED])
    assert torch.equal(answered["image_pooled"], encoders["image"](batch())[Stream.POOLED])


def test_two_towers_whose_streams_would_answer_to_one_name_are_refused_rather_than_one_replacing_the_other() -> None:
    """A name arriving twice would leave one tower's features silently standing in for the other's."""
    with pytest.raises(ValueError, match="one_two_three"):
        MultiEncoderBackbone({"one": Publishes("two_three"), "one_two": Publishes("three")})


@pytest.mark.parametrize(
    ("encoders", "refusal"),
    [
        pytest.param({"image": nn.Linear(2, 2)}, TypeError, id="not a backbone"),
        pytest.param({}, ValueError, id="nothing to pair"),
    ],
)
def test_a_pairing_that_could_publish_nothing_a_head_reads_is_refused_where_it_is_declared(
    encoders: dict[str, Backbone], refusal: type[Exception]
) -> None:
    with pytest.raises(refusal, match="encoders"):
        MultiEncoderBackbone(encoders)


def test_a_pairing_carries_no_classifier_of_its_own_however_much_its_towers_started_from() -> None:
    """Which of a pairing's feature spaces a file's classifier answers for is not written anywhere.

    Held back rather than handed on: passing them up would put one tower's classifier in front of a
    head that may be reading another's, and where the two happen to share a width nothing would notice.
    The wrapper that draws views does hand them on, because there a head reads the one space they came
    from; the difference between the two is exactly that.
    """
    carrying = Encoder()
    carrying.carried_head = {"fc.weight": torch.zeros(2, 8)}

    assert MultiEncoderBackbone({"image": carrying, "text": Sentences()}).carried_head == {}


def test_a_tower_whose_file_carried_a_classifier_hears_that_those_rows_stay_where_they_were_read(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A warm start and a fresh one are told apart here, because the tower's own line cannot tell them.

    What a tower prints is `n were held back`, which is the line an ordinary run prints too — and there
    those rows go on to start a head, named by `build_head`. In a pairing nothing reads them at all, so
    without a word from the pairing itself the run that grew a class space and the run that did not
    read exactly alike.
    """
    carrying = Encoder()
    carrying.carried_head = {"fc.weight": torch.zeros(2, 8)}

    with caplog.at_level("INFO"):
        MultiEncoderBackbone({"image": carrying, "text": Sentences()})

    assert any("image" in one.message and "left where they were read" in one.message for one in caplog.records)
