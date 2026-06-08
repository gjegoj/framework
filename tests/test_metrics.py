"""Unit tests for the typed metric handler chain (M4.2 + curve/matrix axes)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from src.core.ports import PlotLogger
from src.metrics.handlers import (
    DEFAULT_METRIC_HANDLERS,
    CurveMetricHandler,
    MatrixMetricHandler,
    MetricHandler,
    MetricLogContext,
    ScalarMetricHandler,
    VectorMetricHandler,
    dispatch,
)
from src.metrics.registry import curve_specs, matrix_axes

# ------------------------------------------------------------------ fixtures


class FakePlotLogger(PlotLogger):
    """Test double that records ``log_matrix`` and ``log_curve`` calls."""

    def __init__(self) -> None:
        self.matrix_calls: list[dict[str, Any]] = []
        self.curve_calls: list[dict[str, Any]] = []

    def log_matrix(
        self,
        title: str,
        matrix: torch.Tensor,
        iteration: int,
        labels: list[str] | None = None,
        xaxis: str | None = None,
        yaxis: str | None = None,
    ) -> None:
        self.matrix_calls.append(
            {"title": title, "matrix": matrix, "iteration": iteration, "labels": labels, "xaxis": xaxis, "yaxis": yaxis}
        )

    def log_curve(
        self,
        title: str,
        x: torch.Tensor,
        y: torch.Tensor,
        iteration: int,
        series: str = "curve",
        xaxis: str | None = None,
        yaxis: str | None = None,
    ) -> None:
        self.curve_calls.append(
            {"title": title, "x": x, "y": y, "iteration": iteration, "series": series, "xaxis": xaxis, "yaxis": yaxis}
        )


def _ctx(
    plot_logger: Any = None,
    class_names: list[str] | None = None,
    metric_name: str | None = None,
) -> tuple[MetricLogContext, MagicMock]:
    log_scalar: MagicMock = MagicMock()
    ctx = MetricLogContext(
        log_scalar=log_scalar,
        logger=plot_logger,
        step=3,
        class_names=class_names,
        metric_name=metric_name,
    )
    return ctx, log_scalar


# ------------------------------------------------------------------ can_handle


class TestCanHandle:
    def test_scalar_handler_accepts_0d_tensor(self) -> None:
        assert ScalarMetricHandler().can_handle(torch.tensor(0.5))

    def test_scalar_handler_accepts_plain_float(self) -> None:
        assert ScalarMetricHandler().can_handle(0.5)

    def test_scalar_handler_rejects_1d_tensor(self) -> None:
        assert not ScalarMetricHandler().can_handle(torch.tensor([0.5, 0.6]))

    def test_vector_handler_accepts_1d_tensor(self) -> None:
        assert VectorMetricHandler().can_handle(torch.tensor([0.5, 0.6, 0.7]))

    def test_vector_handler_rejects_0d_tensor(self) -> None:
        assert not VectorMetricHandler().can_handle(torch.tensor(0.5))

    def test_vector_handler_rejects_2d_tensor(self) -> None:
        assert not VectorMetricHandler().can_handle(torch.zeros(3, 3))

    def test_matrix_handler_accepts_2d_tensor(self) -> None:
        assert MatrixMetricHandler().can_handle(torch.zeros(3, 3))

    def test_matrix_handler_rejects_1d_tensor(self) -> None:
        assert not MatrixMetricHandler().can_handle(torch.tensor([0.5, 0.6]))

    def test_curve_handler_accepts_3tuple(self) -> None:
        t = torch.zeros(5)
        assert CurveMetricHandler().can_handle((t, t, t))

    def test_curve_handler_rejects_2tuple(self) -> None:
        t = torch.zeros(5)
        assert not CurveMetricHandler().can_handle((t, t))

    def test_curve_handler_rejects_tensor(self) -> None:
        assert not CurveMetricHandler().can_handle(torch.zeros(3, 3))

    def test_no_overlap_for_0d(self) -> None:
        v = torch.tensor(0.5)
        matches = [h for h in DEFAULT_METRIC_HANDLERS if h.can_handle(v)]
        assert len(matches) == 1 and isinstance(matches[0], ScalarMetricHandler)

    def test_no_overlap_for_1d(self) -> None:
        v = torch.tensor([0.1, 0.2])
        matches = [h for h in DEFAULT_METRIC_HANDLERS if h.can_handle(v)]
        assert len(matches) == 1 and isinstance(matches[0], VectorMetricHandler)

    def test_no_overlap_for_2d(self) -> None:
        v = torch.zeros(2, 2)
        matches = [h for h in DEFAULT_METRIC_HANDLERS if h.can_handle(v)]
        assert len(matches) == 1 and isinstance(matches[0], MatrixMetricHandler)

    def test_no_overlap_for_3tuple(self) -> None:
        t = torch.zeros(5)
        matches = [h for h in DEFAULT_METRIC_HANDLERS if h.can_handle((t, t, t))]
        assert len(matches) == 1 and isinstance(matches[0], CurveMetricHandler)


# ------------------------------------------------------------------ handle


class TestScalarMetricHandler:
    def test_calls_log_scalar_with_tensor(self) -> None:
        ctx, mock = _ctx()
        ScalarMetricHandler().handle("acc", torch.tensor(0.9), ctx)
        mock.assert_called_once_with("acc", torch.tensor(0.9))

    def test_calls_log_scalar_with_float(self) -> None:
        ctx, mock = _ctx()
        ScalarMetricHandler().handle("loss", 0.42, ctx)
        mock.assert_called_once_with("loss", 0.42)


class TestVectorMetricHandler:
    def test_logs_mean_at_mean_subkey(self) -> None:
        ctx, mock = _ctx()
        v = torch.tensor([0.4, 0.6])
        VectorMetricHandler().handle("f1", v, ctx)
        first_call = mock.call_args_list[0]
        key, val = first_call[0]
        assert key == "f1/mean"
        assert val.item() == pytest.approx(0.5)

    def test_logs_per_class_fallback_indices(self) -> None:
        ctx, mock = _ctx()
        v = torch.tensor([0.3, 0.7, 0.5])
        VectorMetricHandler().handle("iou", v, ctx)
        calls = {args[0][0]: args[0][1].item() for args in mock.call_args_list}
        assert "iou/mean" in calls
        assert "iou/class0" in calls
        assert "iou/class1" in calls
        assert "iou/class2" in calls
        assert calls["iou/class0"] == pytest.approx(0.3)

    def test_logs_per_class_with_names(self) -> None:
        ctx, mock = _ctx(class_names=["cat", "dog", "cow"])
        v = torch.tensor([0.3, 0.7, 0.5])
        VectorMetricHandler().handle("f1", v, ctx)
        calls = {args[0][0]: args[0][1].item() for args in mock.call_args_list}
        assert "f1/mean" in calls
        assert "f1/cat" in calls
        assert "f1/dog" in calls
        assert "f1/cow" in calls
        assert "f1/class0" not in calls
        assert calls["f1/cat"] == pytest.approx(0.3)

    def test_total_call_count(self) -> None:
        ctx, mock = _ctx()
        v = torch.tensor([0.1, 0.2, 0.3])
        VectorMetricHandler().handle("metric", v, ctx)
        assert mock.call_count == 4  # mean + 3 per-class


class TestMatrixMetricHandler:
    def test_calls_log_matrix_when_plot_logger(self) -> None:
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake)
        m = torch.eye(3)
        MatrixMetricHandler().handle("conf_mat", m, ctx)
        assert len(fake.matrix_calls) == 1
        call = fake.matrix_calls[0]
        assert call["title"] == "conf_mat"
        assert call["iteration"] == 3
        assert torch.equal(call["matrix"], m)
        assert call["labels"] is None

    def test_passes_class_names_as_labels(self) -> None:
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake, class_names=["cat", "dog"])
        MatrixMetricHandler().handle("conf_mat", torch.eye(2), ctx)
        assert fake.matrix_calls[0]["labels"] == ["cat", "dog"]

    def test_passes_axes_from_registry(self) -> None:
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake, metric_name="confusion_matrix")
        MatrixMetricHandler(matrix_axes).handle("conf_mat", torch.eye(2), ctx)
        call = fake.matrix_calls[0]
        assert call["xaxis"] == "Predicted"
        assert call["yaxis"] == "True"

    def test_axes_none_when_not_in_registry(self) -> None:
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake, metric_name="unknown_matrix")
        MatrixMetricHandler(matrix_axes).handle("m", torch.eye(2), ctx)
        call = fake.matrix_calls[0]
        assert call["xaxis"] is None
        assert call["yaxis"] is None

    def test_no_log_scalar_called_for_matrix(self) -> None:
        fake = FakePlotLogger()
        ctx, mock = _ctx(plot_logger=fake)
        MatrixMetricHandler().handle("conf_mat", torch.eye(2), ctx)
        mock.assert_not_called()

    def test_noop_when_logger_is_none(self) -> None:
        ctx, mock = _ctx(plot_logger=None)
        MatrixMetricHandler().handle("conf_mat", torch.eye(3), ctx)
        mock.assert_not_called()

    def test_noop_when_logger_is_not_plot_logger(self) -> None:
        ctx, mock = _ctx(plot_logger=MagicMock())
        MatrixMetricHandler().handle("conf_mat", torch.eye(3), ctx)
        mock.assert_not_called()


class TestCurveMetricHandler:
    def _make_multiclass_pr(self, n_classes: int = 3, n_points: int = 9) -> tuple[Any, Any, Any]:
        precision = [torch.rand(n_points) for _ in range(n_classes)]
        recall = [torch.rand(n_points) for _ in range(n_classes)]
        thresholds = [torch.rand(n_points - 1) for _ in range(n_classes)]
        return precision, recall, thresholds

    def test_calls_log_curve_per_class_multiclass(self) -> None:
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake, metric_name="precision_recall_curve")
        prec, rec, the = self._make_multiclass_pr(n_classes=3)
        CurveMetricHandler(curve_specs).handle("pr", (prec, rec, the), ctx)
        assert len(fake.curve_calls) == 3

    def test_curve_axes_from_pr_spec(self) -> None:
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake, metric_name="precision_recall_curve")
        prec, rec, the = self._make_multiclass_pr(n_classes=2)
        CurveMetricHandler(curve_specs).handle("pr", (prec, rec, the), ctx)
        call = fake.curve_calls[0]
        assert call["xaxis"] == "Recall"
        assert call["yaxis"] == "Precision"

    def test_pr_curve_x_is_recall_y_is_precision(self) -> None:
        # PR spec: x_idx=1 (recall), y_idx=0 (precision)
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake, metric_name="precision_recall_curve")
        prec = [torch.tensor([0.9, 0.8])]
        rec = [torch.tensor([0.1, 0.5])]
        the = [torch.tensor([0.5])]
        CurveMetricHandler(curve_specs).handle("pr", (prec, rec, the), ctx)
        call = fake.curve_calls[0]
        assert torch.equal(call["x"], rec[0])
        assert torch.equal(call["y"], prec[0])

    def test_roc_curve_x_is_fpr_y_is_tpr(self) -> None:
        # ROC spec: x_idx=0 (fpr), y_idx=1 (tpr)
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake, metric_name="roc")
        fpr = [torch.tensor([0.0, 0.2, 1.0])]
        tpr = [torch.tensor([0.0, 0.8, 1.0])]
        the = [torch.tensor([1.0, 0.5])]
        CurveMetricHandler(curve_specs).handle("roc", (fpr, tpr, the), ctx)
        call = fake.curve_calls[0]
        assert torch.equal(call["x"], fpr[0])
        assert torch.equal(call["y"], tpr[0])
        assert call["xaxis"] == "FPR"
        assert call["yaxis"] == "TPR"

    def test_series_uses_class_names(self) -> None:
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake, class_names=["cat", "dog"], metric_name="precision_recall_curve")
        prec, rec, the = self._make_multiclass_pr(n_classes=2)
        CurveMetricHandler(curve_specs).handle("pr", (prec, rec, the), ctx)
        series = [c["series"] for c in fake.curve_calls]
        assert series == ["cat", "dog"]

    def test_series_fallback_indices(self) -> None:
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake, metric_name="precision_recall_curve")
        prec, rec, the = self._make_multiclass_pr(n_classes=3)
        CurveMetricHandler(curve_specs).handle("pr", (prec, rec, the), ctx)
        series = [c["series"] for c in fake.curve_calls]
        assert series == ["class0", "class1", "class2"]

    def test_binary_input_normalized_to_single_curve(self) -> None:
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake, metric_name="precision_recall_curve")
        # Binary: elements are Tensors, not lists
        prec = torch.rand(10)
        rec = torch.rand(10)
        the = torch.rand(9)
        CurveMetricHandler(curve_specs).handle("pr", (prec, rec, the), ctx)
        assert len(fake.curve_calls) == 1

    def test_generic_axes_when_metric_not_in_registry(self) -> None:
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake, metric_name="unknown_curve")
        t = torch.rand(5)
        CurveMetricHandler(curve_specs).handle("c", (t, t, t), ctx)
        call = fake.curve_calls[0]
        assert call["xaxis"] == "x"
        assert call["yaxis"] == "y"

    def test_noop_when_logger_is_none(self) -> None:
        ctx, mock = _ctx(plot_logger=None)
        t = torch.rand(5)
        CurveMetricHandler(curve_specs).handle("pr", (t, t, t), ctx)
        mock.assert_not_called()

    def test_noop_when_logger_is_not_plot_logger(self) -> None:
        ctx, mock = _ctx(plot_logger=MagicMock())
        t = torch.rand(5)
        CurveMetricHandler(curve_specs).handle("pr", (t, t, t), ctx)
        mock.assert_not_called()


# ------------------------------------------------------------------ dispatch


class TestDispatch:
    def test_dispatch_routes_scalar(self) -> None:
        ctx, mock = _ctx()
        dispatch("acc", torch.tensor(0.9), ctx)
        mock.assert_called_once()

    def test_dispatch_routes_vector(self) -> None:
        ctx, mock = _ctx()
        dispatch("f1", torch.tensor([0.5, 0.7]), ctx)
        assert mock.call_count == 3  # mean + 2 classes

    def test_dispatch_routes_matrix_to_plot_logger(self) -> None:
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake)
        dispatch("cm", torch.zeros(3, 3), ctx)
        assert len(fake.matrix_calls) == 1

    def test_dispatch_routes_tuple_to_curve_handler(self) -> None:
        fake = FakePlotLogger()
        ctx, _ = _ctx(plot_logger=fake, metric_name="precision_recall_curve")
        prec = [torch.rand(5) for _ in range(2)]
        rec = [torch.rand(5) for _ in range(2)]
        the = [torch.rand(4) for _ in range(2)]
        dispatch("pr", (prec, rec, the), ctx)
        assert len(fake.curve_calls) == 2

    def test_dispatch_warns_for_unhandled_ndim(self) -> None:
        ctx, _ = _ctx()
        v = torch.zeros(2, 2, 2)  # 3-D
        with pytest.warns(UserWarning, match="No handler"):
            dispatch("weird", v, ctx)

    def test_dispatch_uses_custom_handler_tuple(self) -> None:
        ctx, _ = _ctx()
        custom = MagicMock(spec=MetricHandler)
        custom.can_handle.return_value = True
        dispatch("x", torch.tensor(1.0), ctx, handlers=(custom,))
        custom.handle.assert_called_once()

    def test_fake_plot_logger_is_plot_logger_instance(self) -> None:
        assert isinstance(FakePlotLogger(), PlotLogger)


# ------------------------------------------------------------------ registry specs


class TestRegistrySpecs:
    def test_matrix_axes_has_confusion_matrix(self) -> None:
        assert "confusion_matrix" in matrix_axes
        assert matrix_axes["confusion_matrix"] == ("Predicted", "True")

    def test_curve_specs_has_pr_curve(self) -> None:
        spec = curve_specs["precision_recall_curve"]
        assert spec.xaxis == "Recall"
        assert spec.yaxis == "Precision"
        assert spec.x_idx == 1
        assert spec.y_idx == 0

    def test_curve_specs_has_roc(self) -> None:
        spec = curve_specs["roc"]
        assert spec.xaxis == "FPR"
        assert spec.yaxis == "TPR"
        assert spec.x_idx == 0
        assert spec.y_idx == 1
