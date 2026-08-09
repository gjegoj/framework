"""``TaskTopology`` contracts: head construction, stream choice, pairing support."""

from __future__ import annotations

import torch

from src.core import Objective, Stream, Topology
from src.models import ConvHead, IdentityHead, LinearHead
from src.tasks import (
    DenseTopology,
    GlobalTopology,
    InstancesTopology,
    MultiStreamTopology,
    MultiViewTopology,
)
from src.tasks.registry import topology_registry


def test_registry_covers_the_implemented_topologies() -> None:
    assert set(topology_registry) == {
        Topology.GLOBAL,
        Topology.DENSE,
        Topology.MULTISTREAM,
        Topology.MULTIVIEW,
        Topology.INSTANCES,
    }


def test_global_builds_a_linear_head_of_the_requested_size() -> None:
    head = GlobalTopology().build_head(in_features=8, out_features=3)

    assert isinstance(head, LinearHead)
    assert head(torch.zeros(2, 8)).shape == (2, 3)


def test_global_reads_the_features_stream() -> None:
    assert GlobalTopology().stream == Stream.FEATURES


def test_global_supports_every_objective() -> None:
    """Metric learning included: an ArcFace proxy judges one embedding per sample."""
    topology = GlobalTopology()

    assert all(topology.supports(objective) for objective in Objective)


def test_global_serves_metric_learning_with_an_identity_head() -> None:
    """No width asked for is the metric contract — the embedding is the output.

    Spelled as ``0`` this was a sentinel whose meaning lived in the reader: one file
    returned it and another decoded it with ``> 0``, and neither said what it stood for.
    """
    head = GlobalTopology().build_head(in_features=16, out_features=None)

    assert isinstance(head, IdentityHead)


def test_multistream_reads_the_embeddings_stream_through_identity() -> None:
    topology = MultiStreamTopology()

    assert topology.stream == Stream.EMBEDDINGS
    assert isinstance(topology.build_head(in_features=8, out_features=0), IdentityHead)


def test_multistream_pairs_only_with_metric_learning() -> None:
    topology = MultiStreamTopology()

    assert topology.supports(Objective.METRIC)
    assert not topology.supports(Objective.MULTICLASS)


def test_multiview_mirrors_multistream_behaviour() -> None:
    topology = MultiViewTopology()

    assert topology.stream == Stream.EMBEDDINGS
    assert isinstance(topology.build_head(in_features=8, out_features=0), IdentityHead)
    assert topology.supports(Objective.METRIC)
    assert not topology.supports(Objective.CONTINUOUS)


def test_dense_reads_the_decoder_stream() -> None:
    assert DenseTopology().stream == Stream.DECODER


def test_dense_builds_a_conv_head_preserving_spatial_dims() -> None:
    head = DenseTopology().build_head(in_features=16, out_features=3)

    assert isinstance(head, ConvHead)
    assert head(torch.zeros(2, 16, 8, 8)).shape == (2, 3, 8, 8)


def test_dense_rejects_metric_learning() -> None:
    topology = DenseTopology()

    assert not topology.supports(Objective.METRIC)
    assert topology.supports(Objective.MULTICLASS)
    assert topology.supports(Objective.BINARY)


def test_a_per_instance_task_declares_that_nothing_composes_its_head() -> None:
    """Its assigner, its anchors and its loss are one design, and this framework composes
    none of them. Building something anyway would put a linear layer where a detection
    head belongs, and the run would fail on a shape far from the declaration that caused it.

    Declared beside ``supports`` rather than thrown from ``build_head``: the two are one
    question — can this framework serve this task? — and the builder asks them together,
    before anything is built. A refusal inside ``build_head`` lived in a method the
    builder was never meant to reach, which is a promise the base class makes and this
    subclass breaks. ``test_a_per_instance_task_is_refused_where_the_decision_is_taken``
    in ``test_builder.py`` is where the sentence a user reads is pinned.
    """
    assert not InstancesTopology().composes_head
    assert GlobalTopology().composes_head


def test_a_per_instance_task_is_supervised_as_one_of_n_classes() -> None:
    """The box is geometry the topology carries; the class is one of N like any other.

    Pairing it with a regression or metric objective would declare a task nothing can
    serve, and the refusal is cheaper here than at the first batch.
    """
    topology = InstancesTopology()

    assert topology.supports(Objective.MULTICLASS)
    assert not topology.supports(Objective.CONTINUOUS)
    assert not topology.supports(Objective.METRIC)
