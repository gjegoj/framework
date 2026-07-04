"""Visualization IR entities: labels, masks, colors, assets, field helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


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
