"""``TaskTopology`` contracts: head construction, stream choice, pairing support."""

from __future__ import annotations

import pytest
import torch

from src.core import InputTopology, Objective, OutputTopology, Stream
from src.models import ConvHead, IdentityHead, LinearHead
from src.tasks import DenseTopology, GlobalTopology, InstancesTopology
from src.tasks.registry import topology_registry
from src.tasks.topologies import TaskTopology


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


def test_the_streams_are_a_joint_decision_of_output_and_input() -> None:
    """One vector off FEATURES when one encoder made it, off EMBEDDINGS when views did."""
    topology = GlobalTopology()

    assert topology.streams(InputTopology.SINGLE) == (Stream.FEATURES,)
    assert topology.streams(InputTopology.MULTIVIEW) == (Stream.EMBEDDINGS,)
    assert topology.streams(InputTopology.MULTISTREAM) == (Stream.EMBEDDINGS,)


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
    assert DenseTopology().streams(InputTopology.SINGLE) == (Stream.DECODER,)


def test_instances_defer_the_streams_to_the_backbones_pyramid() -> None:
    """Which levels, how many and in what order is the backbone's fact, never the topology's."""
    assert InstancesTopology().streams(InputTopology.SINGLE) is None


@pytest.mark.parametrize("topology", [GlobalTopology(), DenseTopology()])
def test_a_single_stream_topology_refuses_pyramid_widths_by_name(topology: TaskTopology) -> None:
    """Sized from three widths, a linear or conv head would silently pick one of them."""
    with pytest.raises(ValueError, match=f"{type(topology).__name__} reads one stream"):
        topology.build_head(in_features=(64, 128, 256), out_features=3)


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


def test_a_per_instance_task_declares_that_the_backbones_head_serves() -> None:
    """The framework composes no detection head, so the builder takes the native one
    without being asked; ``test_builder.py`` pins what happens when there is none."""
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
