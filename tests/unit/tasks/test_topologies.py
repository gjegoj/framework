"""``TaskTopology`` contracts: head construction, stream choice, pairing support."""

from __future__ import annotations

import torch

from src.core import InputTopology, Objective, OutputTopology, Stream
from src.models import ConvHead, IdentityHead, LinearHead
from src.tasks import DenseTopology, GlobalTopology, InstancesTopology
from src.tasks.registry import topology_registry


def test_registry_covers_the_implemented_topologies() -> None:
    assert set(topology_registry) == {
        OutputTopology.GLOBAL,
        OutputTopology.DENSE,
        OutputTopology.INSTANCES,
    }


def test_global_builds_a_linear_head_of_the_requested_size() -> None:
    head = GlobalTopology().build_head(in_features=8, out_features=3)

    assert isinstance(head, LinearHead)
    assert head(torch.zeros(2, 8)).shape == (2, 3)


def test_the_stream_is_a_joint_decision_of_output_and_input() -> None:
    """One vector off FEATURES when one encoder made it, off EMBEDDINGS when views did."""
    topology = GlobalTopology()

    assert topology.stream(InputTopology.SINGLE) == Stream.FEATURES
    assert topology.stream(InputTopology.MULTIVIEW) == Stream.EMBEDDINGS
    assert topology.stream(InputTopology.MULTISTREAM) == Stream.EMBEDDINGS


def test_global_with_a_single_input_supports_every_objective() -> None:
    """Metric learning included: an ArcFace proxy judges one embedding per sample."""
    topology = GlobalTopology()

    assert all(topology.supports(objective, InputTopology.SINGLE) for objective in Objective)


def test_stacked_inputs_are_supervised_by_comparison_only() -> None:
    """Stacked views have no per-sample labels to project onto."""
    topology = GlobalTopology()

    assert topology.supports(Objective.METRIC, InputTopology.MULTISTREAM)
    assert topology.supports(Objective.METRIC, InputTopology.MULTIVIEW)
    assert not topology.supports(Objective.MULTICLASS, InputTopology.MULTIVIEW)
    assert not topology.supports(Objective.CONTINUOUS, InputTopology.MULTISTREAM)


def test_global_serves_metric_learning_with_an_identity_head() -> None:
    """No width asked for is the metric contract — the embedding is the output.

    Spelled as ``0`` this was a sentinel whose meaning lived in the reader: one file
    returned it and another decoded it with ``> 0``, and neither said what it stood for.
    """
    head = GlobalTopology().build_head(in_features=16, out_features=None)

    assert isinstance(head, IdentityHead)


def test_dense_reads_the_decoder_stream_whatever_the_input() -> None:
    dense = DenseTopology()

    assert dense.stream(InputTopology.SINGLE) == Stream.DECODER


def test_dense_builds_a_conv_head_preserving_spatial_dims() -> None:
    head = DenseTopology().build_head(in_features=16, out_features=3)

    assert isinstance(head, ConvHead)
    assert head(torch.zeros(2, 16, 8, 8)).shape == (2, 3, 8, 8)


def test_dense_rejects_metric_learning() -> None:
    dense = DenseTopology()

    assert not dense.supports(Objective.METRIC, InputTopology.SINGLE)
    assert dense.supports(Objective.MULTICLASS, InputTopology.SINGLE)
    assert dense.supports(Objective.BINARY, InputTopology.SINGLE)


def test_a_dense_output_refuses_stacked_inputs_whatever_the_objective() -> None:
    """A decoder decodes one image's map — there is nothing dense about a stack of views."""
    dense = DenseTopology()

    for objective in Objective:
        assert not dense.supports(objective, InputTopology.MULTIVIEW), objective


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


def test_an_instances_output_is_single_input_multiclass_only() -> None:
    """The box is geometry the topology carries; the class is one of N like any other.

    Pairing it with a regression or metric objective — or a stacked input — would
    declare a task nothing can serve, and the refusal is cheaper here than at the
    first batch.
    """
    instances = InstancesTopology()

    assert instances.supports(Objective.MULTICLASS, InputTopology.SINGLE)
    assert not instances.supports(Objective.CONTINUOUS, InputTopology.SINGLE)
    assert not instances.supports(Objective.METRIC, InputTopology.SINGLE)
    assert not instances.supports(Objective.MULTICLASS, InputTopology.MULTISTREAM)
