"""``report_metric``: routed by geometry, drawn only when identified."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.core import Curve, Matrix
from src.core.reporting import report_metric


class RecordingLogger:
    """Fake backend satisfying both artifact protocols structurally."""

    def __init__(self) -> None:
        self.matrices: list[tuple[str, Matrix]] = []
        self.curves: list[tuple[str, Curve]] = []

    def log_matrix(self, title: str, matrix: Matrix, iteration: int) -> None:
        self.matrices.append((title, matrix))

    def log_curve(self, title: str, curve: Curve, iteration: int) -> None:
        self.curves.append((title, curve))


def _reject(key: str, value: Any) -> None:
    raise AssertionError(f"Scalar path must not be taken for '{key}'.")


def test_a_scalar_takes_the_plain_path() -> None:
    logged: dict[str, Any] = {}
    report_metric(
        "val/label/f1", torch.tensor(0.5), scalar_log=logged.__setitem__, logger=None, step=0, class_names=None
    )

    assert set(logged) == {"val/label/f1"}


def test_a_vector_logs_its_mean_and_one_scalar_per_class() -> None:
    logged: dict[str, float] = {}
    report_metric(
        "val/label/f1",
        torch.tensor([0.5, 0.7]),
        scalar_log=lambda key, value: logged.__setitem__(key, float(value)),
        logger=None,
        step=0,
        class_names=["cat", "dog"],
    )

    assert logged == {
        "val/label/f1/mean": pytest.approx(0.6),
        "val/label/f1/cat": pytest.approx(0.5),
        "val/label/f1/dog": pytest.approx(0.7),
    }


def test_missing_names_fall_back_to_indexed_labels() -> None:
    logged: dict[str, float] = {}
    report_metric(
        "val/label/f1",
        torch.tensor([0.5, 0.7]),
        scalar_log=lambda key, value: logged.__setitem__(key, float(value)),
        logger=None,
        step=0,
        class_names=None,
    )

    assert set(logged) == {"val/label/f1/mean", "val/label/f1/class0", "val/label/f1/class1"}


def test_a_name_length_mismatch_warns_and_falls_back() -> None:
    logged: dict[str, float] = {}
    with pytest.warns(UserWarning, match="val/label/f1"):
        report_metric(
            "val/label/f1",
            torch.tensor([0.5, 0.7, 0.9]),
            scalar_log=lambda key, value: logged.__setitem__(key, float(value)),
            logger=None,
            step=0,
            class_names=["cat", "dog"],
        )

    assert "val/label/f1/class2" in logged


def test_a_matrix_entity_reaches_the_port_with_class_labels_filled() -> None:
    backend = RecordingLogger()
    report_metric(
        "val/label/cm",
        Matrix(value=torch.eye(2), xaxis="Predicted", yaxis="True"),
        scalar_log=_reject,
        logger=backend,
        step=0,
        class_names=["cat", "dog"],
    )

    title, matrix = backend.matrices[0]
    assert title == "val/label/cm"
    assert matrix.labels == ("cat", "dog")
    assert (matrix.xaxis, matrix.yaxis) == ("Predicted", "True")


def test_translator_set_labels_are_never_overwritten() -> None:
    """A matrix whose axes mean something else keeps its own names; task context stays out."""
    backend = RecordingLogger()
    own = Matrix(value=torch.eye(2), xaxis="Bin", yaxis="Bin", labels=("low", "high"))

    report_metric("val/label/m", own, scalar_log=_reject, logger=backend, step=0, class_names=["cat", "dog"])

    assert backend.matrices[0][1].labels == ("low", "high")


def test_a_matrix_without_a_capable_backend_is_skipped_quietly() -> None:
    """The default table carries a confusion matrix; a CSV run must not warn every epoch."""
    matrix = Matrix(value=torch.eye(2), xaxis="Predicted", yaxis="True")
    report_metric("val/label/cm", matrix, scalar_log=_reject, logger=object(), step=0, class_names=None)


def test_a_raw_matrix_warns_instead_of_wearing_class_names() -> None:
    """An unidentified 2-D value must not be drawn with labels it may not have."""
    with pytest.warns(UserWarning, match="val/label/corr"):
        report_metric(
            "val/label/corr", torch.eye(2), scalar_log=_reject, logger=RecordingLogger(), step=0, class_names=None
        )


def test_a_raw_curve_tuple_warns_instead_of_guessing_its_orientation() -> None:
    """PR and ROC tuples are mirror images; drawing an unidentified one would lie quietly."""
    with pytest.warns(UserWarning, match="val/label/pr"):
        report_metric(
            "val/label/pr",
            (torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.5])),
            scalar_log=_reject,
            logger=RecordingLogger(),
            step=0,
            class_names=None,
        )


def test_a_curve_is_completed_with_series_names_and_passed_whole() -> None:
    backend = RecordingLogger()
    curve = Curve(
        x=(torch.tensor([0.0]), torch.tensor([0.1])),
        y=(torch.tensor([1.0]), torch.tensor([0.9])),
        xaxis="Recall",
        yaxis="Precision",
    )

    report_metric("val/label/pr", curve, scalar_log=_reject, logger=backend, step=0, class_names=["cat", "dog"])

    title, completed = backend.curves[0]
    assert title == "val/label/pr"
    assert completed.series == ("cat", "dog")
    assert (completed.xaxis, completed.yaxis) == ("Recall", "Precision")


def test_a_positive_only_curve_takes_the_positive_class_name() -> None:
    backend = RecordingLogger()
    curve = Curve(
        x=(torch.tensor([0.0, 1.0]),),
        y=(torch.tensor([1.0, 0.5]),),
        xaxis="FPR",
        yaxis="TPR",
        positive_only=True,
    )

    report_metric("val/label/roc", curve, scalar_log=_reject, logger=backend, step=0, class_names=["neg", "pos"])

    assert backend.curves[0][1].series == ("pos",)


def test_translator_set_series_are_never_overwritten() -> None:
    backend = RecordingLogger()
    curve = Curve(x=(torch.tensor([0.0]),), y=(torch.tensor([1.0]),), xaxis="x", yaxis="y", series=("own",))

    report_metric("val/label/c", curve, scalar_log=_reject, logger=backend, step=0, class_names=["cat", "dog"])

    assert backend.curves[0][1].series == ("own",)


def test_a_curve_without_a_capable_backend_is_skipped_quietly() -> None:
    curve = Curve(x=(torch.tensor([0.0]),), y=(torch.tensor([1.0]),), xaxis="x", yaxis="y")
    report_metric("val/label/pr", curve, scalar_log=_reject, logger=object(), step=0, class_names=None)


def test_an_unknown_geometry_warns_and_names_the_key() -> None:
    with pytest.warns(UserWarning, match="val/label/odd"):
        report_metric("val/label/odd", torch.zeros(2, 2, 2), scalar_log=_reject, logger=None, step=0, class_names=None)
