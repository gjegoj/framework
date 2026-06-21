"""Unit tests for the visualization service (IR, annotators, renderer)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.core.entities import Task, TaskStepView
    from src.visualization.entities import SampleView


class TestVisualizationEntities:
    def test_classification_carries_label_and_confidence(self) -> None:
        from src.visualization.entities import Classification

        label = Classification(label="cat", confidence=0.91)
        assert label.label == "cat"
        assert label.confidence == 0.91

    def test_classification_confidence_optional(self) -> None:
        from src.visualization.entities import Classification

        assert Classification(label="cat").confidence is None

    def test_classifications_holds_items(self) -> None:
        from src.visualization.entities import Classification, Classifications

        multi = Classifications(items=[Classification("a"), Classification("b", 0.5)])
        assert [c.label for c in multi.items] == ["a", "b"]

    def test_sample_view_defaults(self) -> None:
        from src.visualization.entities import SampleView

        sample = SampleView(image=np.zeros((4, 4, 3), dtype=np.uint8))
        assert sample.image.shape == (4, 4, 3)
        assert sample.fields == {}
        assert sample.tags == []
        assert sample.metadata == {}


class TestMasks:
    def _square(self) -> "np.ndarray":
        import numpy as np

        mask = np.zeros((20, 20), dtype=bool)
        mask[6:14, 6:14] = True
        return mask

    def _decode(self, uri: str) -> "np.ndarray":
        import base64
        import io

        import numpy as np
        from PIL import Image

        raw = base64.b64decode(uri.split(",", 1)[1])
        return np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))

    def test_returns_png_data_uri(self) -> None:
        from src.visualization.masks import mask_overlay_uri

        uri = mask_overlay_uri(self._square(), (67, 99, 216))
        assert uri.startswith("data:image/png;base64,")

    def test_fill_inside_and_black_outer_border(self) -> None:
        from src.visualization.masks import mask_overlay_uri

        rgba = self._decode(mask_overlay_uri(self._square(), (67, 99, 216)))
        assert rgba[10, 10, 3] > 0 and rgba[10, 10, 3] < 200  # translucent fill in the interior
        assert tuple(rgba[10, 10, :3]) == (67, 99, 216)  # fill is the class color
        assert rgba[5, 10, 3] > 200 and tuple(rgba[5, 10, :3]) == (0, 0, 0)  # outer rim is black

    def test_border_is_symmetric_all_sides(self) -> None:
        from src.visualization.masks import mask_overlay_uri

        rgba = self._decode(mask_overlay_uri(self._square(), (67, 99, 216)))
        # outer black rim present on top AND bottom AND left AND right (no shadow offset)
        assert rgba[5, 10, 3] > 200 and rgba[14, 10, 3] > 200
        assert rgba[10, 5, 3] > 200 and rgba[10, 14, 3] > 200


class TestColors:
    def test_task_palette_distinct_per_class(self) -> None:
        from src.visualization.colors import task_palette

        palette = task_palette("scene", ["sky", "road", "car", "tree"])
        assert len(set(palette.values())) == 4  # all distinct
        assert all(v.startswith("#") and len(v) == 7 for v in palette.values())

    def test_task_palette_reproducible_and_order_independent(self) -> None:
        from src.visualization.colors import task_palette

        a = task_palette("scene", ["car", "sky", "road"])
        b = task_palette("scene", ["road", "car", "sky"])
        assert a == b  # sorted internally → input order does not matter

    def test_task_palette_per_task_gamut_differs(self) -> None:
        from src.visualization.colors import task_palette

        scene = task_palette("scene", ["car"])
        species = task_palette("species", ["car"])
        assert scene["car"] != species["car"]  # per-task hue offset

    def test_hex_to_rgb_roundtrip(self) -> None:
        from src.visualization.colors import hex_to_rgb

        assert hex_to_rgb("#4363d8") == (0x43, 0x63, 0xD8)


class TestRegressionEntities:
    def test_component_carries_name_value_error(self) -> None:
        from src.visualization.entities import RegressionComponent

        c = RegressionComponent(name="height", value=1.78, error=0.05)
        assert c.name == "height" and c.value == 1.78 and c.error == 0.05

    def test_component_error_optional(self) -> None:
        from src.visualization.entities import RegressionComponent

        assert RegressionComponent(name="", value=3.4).error is None

    def test_regression_holds_components(self) -> None:
        from src.visualization.entities import Regression, RegressionComponent

        reg = Regression([RegressionComponent("", 3.4), RegressionComponent("", 5.0, 0.2)])
        assert [c.value for c in reg.components] == [3.4, 5.0]

    def test_regression_is_a_label(self) -> None:
        from src.visualization.entities import Label, Regression

        assert isinstance(Regression(), Label)


class TestSegmentationEntities:
    def test_segmentation_class_holds_name_and_mask(self) -> None:
        import numpy as np

        from src.visualization.entities import SegmentationClass

        mask = np.zeros((4, 4), dtype=bool)
        sc = SegmentationClass(name="road", mask=mask)
        assert sc.name == "road" and sc.mask.shape == (4, 4)

    def test_segmentation_holds_classes_and_is_label(self) -> None:
        import numpy as np

        from src.visualization.entities import Label, Segmentation, SegmentationClass

        seg = Segmentation([SegmentationClass("road", np.zeros((2, 2), dtype=bool))])
        assert [c.name for c in seg.classes] == ["road"]
        assert isinstance(seg, Label)


class TestAnnotators:
    def _task(self, objective: str = "multiclass") -> "Task":
        from dataclasses import replace

        from src.core.entities import Task
        from src.tasks.presets import classification

        names = ["cat", "cow", "dog"]
        task = replace(classification("species", num_classes=3, objective=objective), class_names=names)
        assert isinstance(task, Task)
        return task

    def _view(self, preds: list[list[float]], target: list[object]) -> "TaskStepView":
        import torch

        from src.core.entities import TaskStepView

        return TaskStepView(preds=torch.tensor(preds), metric_target=torch.tensor(target))

    def test_registry_keyed_by_axes(self) -> None:
        from src.tasks.taxonomy import Objective, Topology
        from src.visualization.annotators import annotators, axes_key

        assert axes_key(Topology.GLOBAL, Objective.MULTICLASS) in annotators
        assert axes_key(Topology.GLOBAL, Objective.MULTILABEL) in annotators

    def test_classification_annotator_adds_gt_pred_and_tag(self) -> None:
        import numpy as np

        from src.visualization.annotators import ClassificationAnnotator
        from src.visualization.entities import Classification, SampleView

        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        view = self._view(preds=[[0.1, 0.2, 0.7]], target=[2])  # softmax-like, pred=dog, gt=dog
        ClassificationAnnotator().annotate(sample, self._task(), view, index=0)

        gt = sample.fields["species_gt"]
        pred = sample.fields["species_pred"]
        assert isinstance(gt, Classification) and gt.label == "dog"
        assert isinstance(pred, Classification) and pred.label == "dog"
        assert abs((pred.confidence or 0.0) - 0.7) < 1e-6
        assert "species:correct" in sample.tags

    def test_classification_annotator_wrong_tag(self) -> None:
        import numpy as np

        from src.visualization.annotators import ClassificationAnnotator
        from src.visualization.entities import SampleView

        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        view = self._view(preds=[[0.8, 0.1, 0.1]], target=[2])  # pred=cat, gt=dog
        ClassificationAnnotator().annotate(sample, self._task(), view, index=0)
        assert "species:wrong" in sample.tags

    def test_binary_annotator_thresholds_positive_class(self) -> None:
        from dataclasses import replace

        import numpy as np

        from src.tasks.presets import classification
        from src.visualization.annotators import BinaryClassificationAnnotator
        from src.visualization.entities import Classification, SampleView

        task = replace(classification("species", num_classes=2, objective="binary"), class_names=["cat", "dog"])
        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        # sigmoid P(positive=dog) = 0.8 → pred dog; gt = 1 (dog)
        view = self._view(preds=[[0.8]], target=[[1]])
        BinaryClassificationAnnotator().annotate(sample, task, view, index=0)

        gt = sample.fields["species_gt"]
        pred = sample.fields["species_pred"]
        assert isinstance(gt, Classification) and gt.label == "dog"
        assert isinstance(pred, Classification) and pred.label == "dog"
        assert abs((pred.confidence or 0.0) - 0.8) < 1e-6
        assert "species:correct" in sample.tags

    def test_binary_annotator_below_threshold_is_negative_class(self) -> None:
        from dataclasses import replace

        import numpy as np

        from src.tasks.presets import classification
        from src.visualization.annotators import BinaryClassificationAnnotator
        from src.visualization.entities import Classification, SampleView

        task = replace(classification("species", num_classes=2, objective="binary"), class_names=["cat", "dog"])
        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        # P(dog) = 0.3 → pred cat with confidence 0.7; gt = 1 (dog) → wrong
        view = self._view(preds=[[0.3]], target=[[1]])
        BinaryClassificationAnnotator().annotate(sample, task, view, index=0)

        pred = sample.fields["species_pred"]
        assert isinstance(pred, Classification) and pred.label == "cat"
        assert abs((pred.confidence or 0.0) - 0.7) < 1e-6
        assert "species:wrong" in sample.tags

    def test_binary_registered_for_global_binary(self) -> None:
        from src.tasks.taxonomy import Objective, Topology
        from src.visualization.annotators import annotators, axes_key

        assert axes_key(Topology.GLOBAL, Objective.BINARY) in annotators

    def test_multilabel_annotator_builds_classifications(self) -> None:
        import numpy as np

        from src.visualization.annotators import MultilabelAnnotator
        from src.visualization.entities import Classifications, SampleView

        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        # sigmoid preds: cat=0.9, cow=0.1, dog=0.8 ; gt multi-hot: cat=1, dog=1
        view = self._view(preds=[[0.9, 0.1, 0.8]], target=[[1.0, 0.0, 1.0]])
        MultilabelAnnotator(threshold=0.5).annotate(sample, self._task("multilabel"), view, index=0)

        gt = sample.fields["species_gt"]
        pred = sample.fields["species_pred"]
        assert isinstance(gt, Classifications) and {c.label for c in gt.items} == {"cat", "dog"}
        assert isinstance(pred, Classifications) and {c.label for c in pred.items} == {"cat", "dog"}


class TestRegressionAnnotator:
    def _task(self) -> "Task":
        from src.core.entities import Task
        from src.tasks.presets import regression

        task = regression("age", num_classes=1)
        assert isinstance(task, Task)
        return task

    def _view(self, preds: list[list[float]], target: list[list[float]]) -> "TaskStepView":
        import torch

        from src.core.entities import TaskStepView

        return TaskStepView(preds=torch.tensor(preds), metric_target=torch.tensor(target))

    def test_registered_for_global_continuous(self) -> None:
        from src.tasks.taxonomy import Objective, Topology
        from src.visualization.annotators import annotators, axes_key

        assert axes_key(Topology.GLOBAL, Objective.CONTINUOUS) in annotators

    def test_scalar_gt_pred_and_signed_error(self) -> None:
        import numpy as np

        from src.visualization.annotators import RegressionAnnotator
        from src.visualization.entities import Regression, SampleView

        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        view = self._view(preds=[[3.51]], target=[[3.42]])
        RegressionAnnotator().annotate(sample, self._task(), view, index=0)

        gt = sample.fields["age_gt"]
        pred = sample.fields["age_pred"]
        assert isinstance(gt, Regression) and isinstance(pred, Regression)
        assert gt.components[0].name == "" and abs(gt.components[0].value - 3.42) < 1e-5
        assert gt.components[0].error is None
        assert abs(pred.components[0].value - 3.51) < 1e-5
        assert abs((pred.components[0].error or 0.0) - 0.09) < 1e-5  # signed pred - gt


class TestSegmentationAnnotator:
    def _task(self) -> "Task":
        from src.core.entities import Task
        from src.tasks.presets import segmentation

        task = segmentation("mask", num_classes=3)
        assert isinstance(task, Task)
        return task

    def _view(self, preds: object, target: object) -> "TaskStepView":
        from src.core.entities import TaskStepView

        return TaskStepView(preds=preds, metric_target=target)  # type: ignore[arg-type]

    def test_registered_for_dense_multiclass(self) -> None:
        from src.tasks.taxonomy import Objective, Topology
        from src.visualization.annotators import annotators, axes_key

        assert axes_key(Topology.DENSE, Objective.MULTICLASS) in annotators

    def test_builds_per_class_masks_from_logits_and_label_map(self) -> None:
        import numpy as np
        import torch

        from src.visualization.annotators import SegmentationAnnotator
        from src.visualization.entities import SampleView, Segmentation

        # preds [B, C=3, H=4, W=4] softmax-ish; argmax over C gives the pred label map
        probs = torch.zeros(1, 3, 4, 4)
        probs[0, 0, :2, :] = 0.9  # top half class 0
        probs[0, 1, 2:, :] = 0.9  # bottom half class 1
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        target[0, 2:, :] = 1  # gt: bottom half class 1

        sample = SampleView(image=np.zeros((4, 4, 3), dtype=np.uint8))
        SegmentationAnnotator().annotate(sample, self._task(), self._view(probs, target), index=0)

        gt = sample.fields["mask_gt"]
        pred = sample.fields["mask_pred"]
        assert isinstance(gt, Segmentation) and isinstance(pred, Segmentation)
        gt_classes = {c.name for c in gt.classes}
        assert gt_classes == {"0", "1"}  # both present in gt
        road = next(c for c in gt.classes if c.name == "1")
        assert road.mask.shape == (4, 4) and bool(road.mask[3, 0])  # bottom half is class 1

    def test_ignore_index_skips_class(self) -> None:
        import numpy as np
        import torch

        from src.visualization.annotators import SegmentationAnnotator
        from src.visualization.entities import SampleView, Segmentation

        probs = torch.zeros(1, 3, 4, 4)
        probs[0, 0, :2, :] = 0.9
        probs[0, 1, 2:, :] = 0.9
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        target[0, 2:, :] = 1

        sample = SampleView(image=np.zeros((4, 4, 3), dtype=np.uint8))
        SegmentationAnnotator(ignore_index=0).annotate(sample, self._task(), self._view(probs, target), index=0)

        gt = sample.fields["mask_gt"]
        assert isinstance(gt, Segmentation)
        assert {c.name for c in gt.classes} == {"1"}  # class 0 (background) skipped


class TestMultilabelSegmentationAnnotator:
    def _task(self) -> "Task":
        from src.core.entities import Task
        from src.tasks.presets import segmentation

        task = segmentation("mask", num_classes=2, objective="multilabel")
        assert isinstance(task, Task)
        return task

    def _view(self, preds: object, target: object) -> "TaskStepView":
        from src.core.entities import TaskStepView

        return TaskStepView(preds=preds, metric_target=target)  # type: ignore[arg-type]

    def test_registered_for_dense_multilabel(self) -> None:
        from src.tasks.taxonomy import Objective, Topology
        from src.visualization.annotators import annotators, axes_key

        assert axes_key(Topology.DENSE, Objective.MULTILABEL) in annotators

    def test_overlapping_masks_from_per_channel_threshold(self) -> None:
        import numpy as np
        import torch

        from src.visualization.annotators import MultilabelSegmentationAnnotator
        from src.visualization.entities import SampleView, Segmentation

        # preds [B, C=2, H=4, W=4] sigmoid; gt [B, C=2, H=4, W=4] multi-hot (classes overlap)
        probs = torch.zeros(1, 2, 4, 4)
        probs[0, 0] = 0.9  # class 0 active everywhere
        probs[0, 1, :2, :2] = 0.9  # class 1 active only top-left → overlaps class 0
        gt = torch.zeros(1, 2, 4, 4)
        gt[0, 0] = 1.0
        gt[0, 1, :2, :2] = 1.0

        sample = SampleView(image=np.zeros((4, 4, 3), dtype=np.uint8))
        MultilabelSegmentationAnnotator(threshold=0.5).annotate(sample, self._task(), self._view(probs, gt), index=0)

        pred = sample.fields["mask_pred"]
        assert isinstance(pred, Segmentation)
        masks = {c.name: c.mask for c in pred.classes}
        assert set(masks) == {"0", "1"}
        assert bool(masks["0"][0, 0]) and bool(masks["1"][0, 0])  # same pixel in BOTH classes
        assert bool(masks["0"][3, 3]) and not bool(masks["1"][3, 3])  # class 1 absent bottom-right

    def test_threshold_drops_low_confidence_class(self) -> None:
        import numpy as np
        import torch

        from src.visualization.annotators import MultilabelSegmentationAnnotator
        from src.visualization.entities import SampleView, Segmentation

        probs = torch.zeros(1, 2, 4, 4)
        probs[0, 0] = 0.9  # present
        probs[0, 1] = 0.3  # below threshold everywhere → absent
        gt = torch.zeros(1, 2, 4, 4)
        gt[0, 0] = 1.0

        sample = SampleView(image=np.zeros((4, 4, 3), dtype=np.uint8))
        MultilabelSegmentationAnnotator(threshold=0.5).annotate(sample, self._task(), self._view(probs, gt), index=0)

        pred = sample.fields["mask_pred"]
        assert isinstance(pred, Segmentation)
        assert {c.name for c in pred.classes} == {"0"}  # class 1 thresholded out


class TestPipeline:
    def test_build_sample_views_annotates_each_image(self) -> None:
        import numpy as np

        from src.visualization.entities import Classification
        from src.visualization.pipeline import build_sample_views

        annotator_tests = TestAnnotators()
        task = annotator_tests._task()
        views = {
            "species": annotator_tests._view(
                preds=[[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]],
                target=[2, 2],
            )
        }
        images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        samples = build_sample_views(images, [task], views)

        assert len(samples) == 2
        gt = samples[0].fields["species_gt"]
        pred = samples[0].fields["species_pred"]
        assert isinstance(gt, Classification) and gt.label == "dog"
        assert isinstance(pred, Classification)
        assert samples[1].tags == ["species:wrong"]

    def test_build_sample_views_threads_sources(self) -> None:
        import numpy as np

        from src.visualization.pipeline import build_sample_views

        images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        samples = build_sample_views(images, [], {}, sources=["a.jpg", "b.jpg"])
        assert [sample.source for sample in samples] == ["a.jpg", "b.jpg"]

    def test_build_sample_views_source_none_by_default(self) -> None:
        import numpy as np

        from src.visualization.pipeline import build_sample_views

        assert build_sample_views(np.zeros((1, 4, 4, 3), dtype=np.uint8), [], {})[0].source is None


class TestPipelineRegression:
    def test_regression_task_is_annotated_not_skipped(self) -> None:
        import numpy as np
        import torch

        from src.core.entities import Task, TaskStepView
        from src.tasks.presets import regression
        from src.visualization.entities import Regression
        from src.visualization.pipeline import build_sample_views

        task = regression("age", num_classes=1)
        assert isinstance(task, Task)
        views = {"age": TaskStepView(preds=torch.tensor([[3.51], [7.0]]), metric_target=torch.tensor([[3.42], [5.0]]))}
        images = np.zeros((2, 4, 4, 3), dtype=np.uint8)

        samples = build_sample_views(images, [task], views)
        assert isinstance(samples[0].fields["age_gt"], Regression)
        assert isinstance(samples[1].fields["age_pred"], Regression)


class TestAssets:
    def test_css_generalizes_overlay_toggle_and_cover_zone(self) -> None:
        from pathlib import Path

        import src.visualization as viz_pkg

        css = (Path(viz_pkg.__file__).parent / "assets" / "grid.css").read_text(encoding="utf-8")
        assert "flex-wrap" in css and "border-radius" in css
        assert ".layer.hidden" in css  # any overlay (chip/mask) toggles
        assert ".cover .mask" in css  # full-cell mask layers in the cover zone
        assert ".caret:hover" in css  # bigger, hoverable disclosure caret

    def test_js_toggles_layer_not_just_chip(self) -> None:
        from pathlib import Path

        import src.visualization as viz_pkg

        js = (Path(viz_pkg.__file__).parent / "assets" / "grid.js").read_text(encoding="utf-8")
        assert "refreshGroups" in js and "setHidden" in js
        assert ".layer[data-key=" in js  # generalized from .chip
        assert "cloneNode" in js  # lightbox clones a cell into the zoom view
        assert ".grid > .cell" in js


class TestFieldHelpers:
    def test_field_key_joins_with_double_colon(self) -> None:
        from src.visualization.renderer import field_key

        assert field_key("species", "pred", "dog") == "species::pred::dog"

    def test_render_chip_filled_carries_layer_class_and_key(self) -> None:
        from src.visualization.renderer import render_chip

        chip = render_chip("species::gt::cat", "cat", "#abcdef", filled=True)
        assert 'class="layer chip gt"' in chip
        assert 'data-key="species::gt::cat"' in chip
        assert "background:#abcdef" in chip

    def test_render_chip_outlined_uses_border_color(self) -> None:
        from src.visualization.renderer import render_chip

        chip = render_chip("species::pred::dog", "dog 0.73", "#abcdef", filled=False)
        assert 'class="layer chip pred"' in chip
        assert "border-color:#abcdef" in chip

    def test_render_chip_truncates_long_text_keeps_full_in_title(self) -> None:
        from src.visualization.renderer import render_chip

        chip = render_chip("t::gt::x", "a" * 40, "#000000", filled=True, max_chars=10)
        assert "…" in chip
        assert "a" * 40 in chip  # full text preserved in title

    def test_render_chip_data_full_carries_untruncated_text(self) -> None:
        """The lightbox swaps in ``data-full``; it must hold the full (untruncated) name."""
        from src.visualization.renderer import render_chip

        chip = render_chip("t::gt::x", "staffordshire_bull_terrier", "#000000", filled=True, max_chars=10)
        assert 'data-full="staffordshire_bull_terrier"' in chip
        assert "…" in chip  # the visible text is still the compact truncated form


class TestLabelRenderers:
    def test_registry_has_all_label_types(self) -> None:
        from src.visualization.entities import Classification, Classifications, Regression
        from src.visualization.renderer import label_renderers

        for name in (Classification.__name__, Classifications.__name__, Regression.__name__):
            assert name in label_renderers

    def test_classification_renders_one_chip_item(self) -> None:
        from src.visualization.entities import Classification
        from src.visualization.renderer import ClassificationLabelRenderer, FieldContext

        ctx = FieldContext("species", "pred", colors={"dog": "#abcdef"})
        items = ClassificationLabelRenderer().render_field(ctx, Classification("dog", 0.73))
        assert len(items) == 1
        item = items[0]
        assert item.data_key == "species::pred::dog"
        assert item.zone == "chips" and item.filled is False
        assert item.color == "#abcdef"  # color comes from the per-task palette in ctx
        assert "dog 0.73" in item.overlay_html  # confidence on pred

    def test_leaves_for_each_label_type(self) -> None:
        import numpy as np

        from src.visualization.entities import (
            Classification,
            Classifications,
            Regression,
            RegressionComponent,
            Segmentation,
            SegmentationClass,
        )
        from src.visualization.renderer import (
            ClassificationLabelRenderer,
            ClassificationsLabelRenderer,
            RegressionLabelRenderer,
            SegmentationLabelRenderer,
        )

        assert ClassificationLabelRenderer().leaves(Classification("dog")) == ["dog"]
        assert ClassificationsLabelRenderer().leaves(Classifications([Classification("a"), Classification("b")])) == [
            "a",
            "b",
        ]
        assert RegressionLabelRenderer().leaves(Regression([RegressionComponent("", 1.0)])) == ["value"]
        assert SegmentationLabelRenderer().leaves(
            Segmentation([SegmentationClass("road", np.zeros((2, 2), dtype=bool))])
        ) == ["road"]

    def test_segmentation_renders_cover_mask_items(self) -> None:
        import numpy as np

        from src.visualization.entities import Segmentation, SegmentationClass
        from src.visualization.renderer import FieldContext, SegmentationLabelRenderer

        mask = np.zeros((6, 6), dtype=bool)
        mask[1:5, 1:5] = True
        gt = SegmentationLabelRenderer().render_field(
            FieldContext("scene", "gt", colors={"road": "#4363d8"}),
            Segmentation([SegmentationClass("road", mask)]),
        )
        assert len(gt) == 1
        item = gt[0]
        assert item.zone == "cover" and item.filled is True
        assert 'class="layer mask" data-key="scene::gt::road"' in item.overlay_html
        assert "data:image/png;base64," in item.overlay_html
        assert item.color == "#4363d8"

    def test_classification_gt_has_no_confidence_and_is_filled(self) -> None:
        from src.visualization.entities import Classification
        from src.visualization.renderer import ClassificationLabelRenderer, FieldContext

        (item,) = ClassificationLabelRenderer().render_field(FieldContext("species", "gt"), Classification("cat"))
        assert item.filled is True and ">cat<" in item.overlay_html

    def test_classifications_render_one_item_each(self) -> None:
        from src.visualization.entities import Classification, Classifications
        from src.visualization.renderer import ClassificationsLabelRenderer, FieldContext

        labels = Classifications([Classification("a"), Classification("b", 0.5)])
        items = ClassificationsLabelRenderer().render_field(FieldContext("tags", "pred"), labels)
        assert [it.data_key for it in items] == ["tags::pred::a", "tags::pred::b"]

    def test_regression_scalar_neutral_color_and_signed_delta(self) -> None:
        from src.visualization.colors import REGRESSION_COLOR
        from src.visualization.entities import Regression, RegressionComponent
        from src.visualization.renderer import FieldContext, RegressionLabelRenderer

        gt = RegressionLabelRenderer().render_field(
            FieldContext("age", "gt"), Regression([RegressionComponent("", 3.42)])
        )
        pred = RegressionLabelRenderer().render_field(
            FieldContext("age", "pred"), Regression([RegressionComponent("", 3.51, error=0.09)])
        )
        assert gt[0].data_key == "age::gt::value" and gt[0].color == REGRESSION_COLOR
        assert ">3.42<" in gt[0].overlay_html  # gt: bare number, no delta
        assert "3.51 Δ+0.09" in pred[0].overlay_html  # pred: signed delta as text
        assert "Δ" not in gt[0].overlay_html

    def test_regression_negative_error_shows_minus(self) -> None:
        from src.visualization.entities import Regression, RegressionComponent
        from src.visualization.renderer import FieldContext, RegressionLabelRenderer

        (item,) = RegressionLabelRenderer().render_field(
            FieldContext("age", "pred"), Regression([RegressionComponent("", 5.2, error=-1.8)])
        )
        assert "5.20 Δ-1.80" in item.overlay_html

    def test_regression_named_component_is_prefixed(self) -> None:
        from src.visualization.entities import Regression, RegressionComponent
        from src.visualization.renderer import FieldContext, RegressionLabelRenderer

        (item,) = RegressionLabelRenderer().render_field(
            FieldContext("body", "gt"), Regression([RegressionComponent("height", 1.78)])
        )
        assert item.data_key == "body::gt::height" and "height 1.78" in item.overlay_html


class TestHtmlRenderer:
    def _sample(self) -> "SampleView":
        import numpy as np

        from src.visualization.entities import (
            Classification,
            Classifications,
            Regression,
            RegressionComponent,
            SampleView,
        )

        return SampleView(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            fields={
                "species_gt": Classification("cat"),
                "species_pred": Classification("dog", confidence=0.73),
                "tags_gt": Classifications([Classification("indoor")]),
                "tags_pred": Classifications([Classification("indoor", 0.9), Classification("small", 0.6)]),
                "age_gt": Regression([RegressionComponent("", 3.42)]),
                "age_pred": Regression([RegressionComponent("", 3.51, error=0.09)]),
            },
        )

    def test_render_returns_self_contained_html(self) -> None:
        from src.visualization.renderer import HtmlRenderer

        out = HtmlRenderer().render([self._sample()], title="samples/val")
        assert out.startswith("<!DOCTYPE html>")
        assert "data:image/png;base64," in out
        assert "flex-wrap" in out and "refreshGroups" in out
        assert "samples/val" in out

    def test_chips_carry_layer_class_and_keys(self) -> None:
        from src.visualization.colors import task_palette
        from src.visualization.renderer import HtmlRenderer

        out = HtmlRenderer().render([self._sample()], title="t")
        palette = task_palette("species", ["cat", "dog"])  # present species classes in the sample
        assert 'class="layer chip gt" data-key="species::gt::cat"' in out
        assert 'class="layer chip pred" data-key="species::pred::dog"' in out
        assert f"background:{palette['cat']}" in out  # gt filled, per-task color
        assert f"border-color:{palette['dog']}" in out  # pred outlined, per-task color

    def test_regression_chip_neutral_and_delta_in_cell(self) -> None:
        from src.visualization.colors import REGRESSION_COLOR
        from src.visualization.renderer import HtmlRenderer

        out = HtmlRenderer().render([self._sample()], title="t")
        assert 'class="layer chip gt" data-key="age::gt::value"' in out
        assert f"background:{REGRESSION_COLOR}" in out
        assert "3.51 Δ+0.09" in out
        assert "✓" not in out and "✗" not in out

    def test_sidebar_groups_classes_and_regression_leaf(self) -> None:
        from src.visualization.renderer import HtmlRenderer

        out = HtmlRenderer().render([self._sample()], title="t")
        assert 'class="grp" data-prefix="species::"' in out
        assert 'class="grp" data-prefix="age::pred::"' in out
        assert 'class="cls" data-key="species::pred::dog"' in out
        assert 'class="cls" data-key="age::pred::value"' in out

    def test_cell_has_cover_zone(self) -> None:
        from src.visualization.renderer import HtmlRenderer

        out = HtmlRenderer().render([self._sample()], title="t")
        assert '<div class="cover">' in out

    def test_cell_path_source_is_copy_button(self) -> None:
        import numpy as np

        from src.visualization.entities import SampleView
        from src.visualization.renderer import HtmlRenderer

        sample = SampleView(image=np.zeros((8, 8, 3), dtype=np.uint8), source="data/images/42.jpg")
        out = HtmlRenderer().render([sample], title="t")
        assert 'class="src copy"' in out  # a local path copies, not opens
        assert 'data-copy="data/images/42.jpg"' in out  # full path copied to clipboard
        assert ">📋 42.jpg</button>" in out  # basename shown, clipboard icon
        assert 'onclick="copySource(this, event)"' in out

    def test_cell_url_source_is_open_link(self) -> None:
        import numpy as np

        from src.visualization.entities import SampleView
        from src.visualization.renderer import HtmlRenderer

        sample = SampleView(image=np.zeros((8, 8, 3), dtype=np.uint8), source="https://files.x/img/42.jpg")
        out = HtmlRenderer().render([sample], title="t")
        assert 'class="src" href="https://files.x/img/42.jpg" target="_blank"' in out  # a URL opens
        assert ">🔗 42.jpg</a>" in out  # link icon, basename
        assert 'onclick="event.stopPropagation()"' in out  # click opens, but doesn't zoom

    def test_cell_omits_source_link_when_absent(self) -> None:
        from src.visualization.renderer import HtmlRenderer

        assert 'class="src"' not in HtmlRenderer().render([self._sample()], title="t")

    def test_includes_lightbox_scaffold(self) -> None:
        from src.visualization.renderer import HtmlRenderer

        out = HtmlRenderer().render([self._sample()], title="t")
        assert 'id="lb"' in out and 'id="lb-holder"' in out  # click-to-zoom overlay present

    def test_segmentation_masks_render_in_cover_with_sidebar(self) -> None:
        import numpy as np

        from src.visualization.entities import SampleView, Segmentation, SegmentationClass
        from src.visualization.renderer import HtmlRenderer

        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        sample = SampleView(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            fields={
                "scene_gt": Segmentation([SegmentationClass("road", mask)]),
                "scene_pred": Segmentation([SegmentationClass("road", mask)]),
            },
        )
        out = HtmlRenderer().render([sample], title="t")
        assert '<div class="cover"><img class="layer mask" data-key="scene::gt::road"' in out
        assert 'data-key="scene::pred::road"' in out
        assert 'class="cls" data-key="scene::pred::road"' in out  # sidebar leaf for the class

    def test_render_escapes_dynamic_text(self) -> None:
        import numpy as np

        from src.visualization.entities import Classification, SampleView
        from src.visualization.renderer import HtmlRenderer

        sample = SampleView(image=np.zeros((4, 4, 3), dtype=np.uint8), fields={"a<b_gt": Classification("x&y")})
        out = HtmlRenderer().render([sample], title="<title>")
        assert "&lt;title&gt;" in out and "x&amp;y" in out

    def test_render_handles_empty(self) -> None:
        from src.visualization.renderer import HtmlRenderer

        out = HtmlRenderer().render([], title="empty")
        assert isinstance(out, str) and "empty" in out and 'class="layer chip' not in out
