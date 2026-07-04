"""Label renderers and the self-contained HTML grid renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.visualization.entities import SampleView


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

        from src.visualization.entities import SampleView
        from src.visualization.renderer import HtmlRenderer

        sample = SampleView(image=np.zeros((8, 8, 3), dtype=np.uint8), source="data/images/42.jpg")
        out = HtmlRenderer().render([sample], title="t")
        assert 'class="src copy"' in out  # a local path copies, not opens
        assert 'data-copy="data/images/42.jpg"' in out  # full path copied to clipboard
        assert ">📋 42.jpg</button>" in out  # basename shown, clipboard icon
        assert 'onclick="copySource(this, event)"' in out

    def test_cell_url_source_is_open_link(self) -> None:

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

        from src.visualization.entities import Classification, SampleView
        from src.visualization.renderer import HtmlRenderer

        sample = SampleView(image=np.zeros((4, 4, 3), dtype=np.uint8), fields={"a<b_gt": Classification("x&y")})
        out = HtmlRenderer().render([sample], title="<title>")
        assert "&lt;title&gt;" in out and "x&amp;y" in out

    def test_render_handles_empty(self) -> None:
        from src.visualization.renderer import HtmlRenderer

        out = HtmlRenderer().render([], title="empty")
        assert isinstance(out, str) and "empty" in out and 'class="layer chip' not in out
