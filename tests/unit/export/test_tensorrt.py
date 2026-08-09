"""The TensorRT backend, as far as a machine without TensorRT can judge it.

NVIDIA publishes no macOS build, so the engine build and its parity check run
only on a CUDA node — and the tests that need one say so by skipping rather than
by pretending. What is testable anywhere is everything that happens *before* the
library is touched: the precision, the batch profile, and the message a machine
without the package receives. Those are also the parts a user meets first.
"""

from __future__ import annotations

import pytest

from src.export import TensorRtExporter, exporter_registry
from src.export.backends.tensorrt import require_tensorrt

try:
    require_tensorrt()
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

needs_tensorrt = pytest.mark.skipif(not HAS_TENSORRT, reason="TensorRT is not installed on this machine")
without_tensorrt = pytest.mark.skipif(HAS_TENSORRT, reason="TensorRT is installed, so nothing is missing")


def test_the_format_is_registered_even_where_it_cannot_run() -> None:
    """A config naming an unavailable format must fail on the library, not on an unknown key."""
    assert "tensorrt" in exporter_registry


@without_tensorrt
def test_a_missing_library_names_the_command_that_fixes_it() -> None:
    """It is deliberately not a declared dependency, so the error carries what pyproject cannot."""
    with pytest.raises(ImportError, match="uv pip install tensorrt"):
        require_tensorrt()


@without_tensorrt
def test_construction_reports_the_missing_library_at_assembly() -> None:
    """Failing while the experiment is assembled beats failing after an hour of training."""
    with pytest.raises(ImportError, match="tensorrt"):
        TensorRtExporter()


def test_an_unknown_precision_is_refused_before_the_library_is_touched() -> None:
    """Cheap arguments are checked first, so a typo is answered on any machine."""
    with pytest.raises(ValueError, match="Precision must be one of fp16, fp32"):
        TensorRtExporter(precision="f16")  # type: ignore[arg-type]


@pytest.mark.parametrize(("smallest", "tuned", "largest"), [(4, 2, 8), (0, 1, 8), (1, 9, 8)])
def test_an_incoherent_batch_profile_is_refused(smallest: int, tuned: int, largest: int) -> None:
    """TensorRT would accept min above max and build an engine nothing can feed."""
    with pytest.raises(ValueError, match="min <= opt <= max"):
        TensorRtExporter(min_batch=smallest, opt_batch=tuned, max_batch=largest)


@needs_tensorrt
def test_half_precision_carries_a_looser_tolerance_than_full() -> None:
    """An fp16 engine drifts where a traced graph does not; the format knows its own numerics."""
    assert TensorRtExporter(precision="fp16").atol > TensorRtExporter(precision="fp32").atol


@needs_tensorrt
def test_declared_tolerances_win_over_the_precision_default() -> None:
    """A deployment that measured its own bound must be able to state it."""
    assert TensorRtExporter(precision="fp16", atol=1e-4).atol == pytest.approx(1e-4)
