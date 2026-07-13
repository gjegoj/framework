"""Unit tests for the model layer (offline, pretrained=False)."""

import pytest
import torch

from src.core.entities import HeadSpec
from src.core.keys import DECODER, ENCODER_LAST, IMAGE, POOLED
from src.models import CompositeModel, build_composite_model
from src.models.backbones import DinoDptBackbone, EmbeddingBackbone, SmpBackbone, TimmBackbone
from src.models.registry import backbones, head_builders


def _images(batch: int = 2) -> dict[str, torch.Tensor]:
    return {IMAGE: torch.randn(batch, 3, 32, 32)}


class TestTimmBackbone:
    def test_pooled_stream_and_dim(self) -> None:
        backbone = TimmBackbone("resnet18", pretrained=False)
        features = backbone(_images(2))
        assert features[POOLED].shape == (2, 512)
        assert backbone.feature_dim(POOLED) == 512

    def test_unknown_stream_raises(self) -> None:
        backbone = TimmBackbone("resnet18", pretrained=False)
        with pytest.raises(KeyError, match="exposes only 'pooled'"):
            backbone.feature_dim("decoder")


class TestEmbeddingBackbone:
    def test_passthrough_stream_and_dim(self) -> None:
        backbone = EmbeddingBackbone(embedding_dim=16)
        vectors = {IMAGE: torch.randn(2, 16)}
        features = backbone(vectors)
        assert features[POOLED].shape == (2, 16)
        assert backbone.feature_dim(POOLED) == 16

    def test_reads_configured_input_key(self) -> None:
        backbone = EmbeddingBackbone(embedding_dim=8, input_key="embedding")
        features = backbone({"embedding": torch.randn(3, 8)})
        assert features[POOLED].shape == (3, 8)

    def test_unknown_stream_raises(self) -> None:
        backbone = EmbeddingBackbone(embedding_dim=16)
        with pytest.raises(KeyError, match="exposes only 'pooled'"):
            backbone.feature_dim("decoder")

    def test_builds_from_config_ignoring_name_and_pretrained(self) -> None:
        from src.composition.wiring import build_backbone
        from src.config.schema import BackboneConfig

        cfg = BackboneConfig(kind="embedding", name="identity", embedding_dim=8, input_key="embedding")
        backbone = build_backbone(cfg)
        assert isinstance(backbone, EmbeddingBackbone)
        assert backbone.feature_dim(POOLED) == 8


class TestConvHead:
    def test_preserves_spatial_dims_maps_channels(self) -> None:
        head = head_builders.create("conv", in_features=16, out_features=4)
        features = torch.randn(2, 16, 24, 24)  # [B, D, H, W]
        logits = head(features)
        assert logits.shape == (2, 4, 24, 24)  # [B, C, H, W]


class TestSmpBackbone:
    def test_exposes_encoder_last_and_decoder_streams(self) -> None:
        backbone = SmpBackbone(name="unet", encoder_name="resnet18", pretrained=False)
        features = backbone(_images(2))  # 32x32 input (divisible by 32)
        assert set(features.streams) == {ENCODER_LAST, DECODER}
        assert features[DECODER].shape[0] == 2
        assert features[DECODER].shape[2:] == (32, 32)  # full input resolution
        assert features[DECODER].shape[1] == backbone.feature_dim(DECODER)
        enc = features[ENCODER_LAST]
        assert enc.ndim == 4, "encoder_last must be [B, D, H, W]"
        assert enc.shape[0] == 2
        assert enc.shape[1] == backbone.feature_dim(ENCODER_LAST)

    def test_unknown_stream_raises_with_listing(self) -> None:
        backbone = SmpBackbone(encoder_name="resnet18", pretrained=False)
        with pytest.raises(KeyError, match="encoder_last"):
            backbone.feature_dim("tokens")

    def test_native_head_encoder_last_returns_classification_head(self) -> None:
        backbone = SmpBackbone(encoder_name="resnet18", pretrained=False)
        in_features = backbone.feature_dim(ENCODER_LAST)
        head = backbone.native_head(ENCODER_LAST, in_features, out_features=5)
        assert head is not None
        x = torch.randn(2, in_features, 4, 4)
        out = head(x)
        assert out.shape == (2, 5)

    def test_native_head_unknown_key_returns_none(self) -> None:
        backbone = SmpBackbone(encoder_name="resnet18", pretrained=False)
        assert backbone.native_head("pooled", 512, 3) is None

    def test_segmentation_model_produces_per_pixel_logits(self) -> None:
        backbone = SmpBackbone(encoder_name="resnet18", pretrained=False)
        specs = {"mask": HeadSpec(kind="conv", out_features=4, feature_key=DECODER)}
        model = build_composite_model(backbone, specs)
        output = model(_images(2))
        assert output.task_logits["mask"].shape == (2, 4, 32, 32)  # [B, C, H, W]

    def test_multitask_segmentation_and_classification(self) -> None:
        backbone = SmpBackbone(encoder_name="resnet18", pretrained=False)
        specs = {
            "mask": HeadSpec(kind="conv", out_features=3, feature_key=DECODER, prefer_native=True),
            "label": HeadSpec(kind="linear", out_features=5, feature_key=ENCODER_LAST, prefer_native=True),
        }
        model = build_composite_model(backbone, specs)
        output = model(_images(2))
        assert output.task_logits["mask"].shape == (2, 3, 32, 32)
        assert output.task_logits["label"].shape == (2, 5)

    def test_explicit_encoder_weights_overrides_pretrained(self) -> None:
        # An explicit encoder_weights must override pretrained's default, not collide with the
        # positional argument (None here also avoids any weight download).
        backbone = SmpBackbone(name="unet", encoder_name="resnet18", pretrained=True, encoder_weights=None)
        assert set(backbone(_images(2)).streams) == {ENCODER_LAST, DECODER}


class TestDinoDptPatch:
    @staticmethod
    def _stub_smp_build(monkeypatch: pytest.MonkeyPatch, encoder: object) -> None:
        """Stub SmpBackbone.__init__ so DinoDptBackbone builds without a real smp model."""

        def fake_init(self: SmpBackbone, **kwargs: object) -> None:
            self._encoder = encoder

        monkeypatch.setattr(SmpBackbone, "__init__", fake_init)

    def test_init_norms_prefix_token_intermediates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded: list[dict[str, object]] = []

        class _Model:
            def forward_intermediates(
                self, x: torch.Tensor, **kwargs: object
            ) -> list[tuple[torch.Tensor, torch.Tensor]]:
                recorded.append(kwargs)
                return [(torch.zeros(1), torch.ones(1)), (torch.zeros(1), torch.ones(1))]

        class _Encoder:
            has_prefix_tokens = True
            _output_indices = [2, 5]
            model = _Model()

        self._stub_smp_build(monkeypatch, _Encoder())
        backbone = DinoDptBackbone(encoder_name="x")
        features, prefix = backbone._encoder._forward_with_prefix_tokens(torch.zeros(1))

        assert recorded[0]["norm"] is True
        assert recorded[0]["return_prefix_tokens"] is True
        assert len(features) == 2 and len(prefix) == 2

    def test_init_noop_without_prefix_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Encoder:
            has_prefix_tokens = False

        self._stub_smp_build(monkeypatch, _Encoder())
        backbone = DinoDptBackbone(encoder_name="x")
        assert not hasattr(backbone._encoder, "_forward_with_prefix_tokens")


class TestRegistries:
    def test_builtins_registered(self) -> None:
        assert "timm" in backbones
        assert "smp" in backbones
        assert "dino_dpt" in backbones
        assert "linear" in head_builders
        assert "conv" in head_builders


class TestCompositeModel:
    def test_single_head_forward(self) -> None:
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        specs = {"label": HeadSpec(kind="linear", out_features=3, feature_key=POOLED)}
        model = build_composite_model(backbone, specs)

        assert isinstance(model, CompositeModel)
        output = model(_images(4))
        assert set(output.task_logits) == {"label"}
        assert output.task_logits["label"].shape == (4, 3)
        assert output.features[POOLED].shape == (4, 512)

    def test_multi_head_shares_backbone(self) -> None:
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        specs = {
            "species": HeadSpec(kind="linear", out_features=3),
            "age": HeadSpec(kind="linear", out_features=1),
        }
        model = build_composite_model(backbone, specs)
        output = model(_images(2))
        assert output.task_logits["species"].shape == (2, 3)
        assert output.task_logits["age"].shape == (2, 1)
        # One shared backbone instance, two heads.
        assert len(model.heads) == 2


class TestDinoDptStaticRope:
    """Device-portability pin: statically-built DINOv3 traces its ROPE from a buffer.

    With timm's default ``dynamic_img_size=True`` the rotary embedding is recomputed per
    forward and ``torch.jit.trace`` bakes its coordinate grid as trace-device constants —
    the artifact then breaks on ``.to("cuda")``. Built statically (the dinov3_dpt config),
    the embedding lives in the ``pos_embed_cached`` registered buffer, which ``.to()`` moves.
    """

    def test_static_build_traces_rope_from_registered_buffer(self) -> None:
        import torch

        from src.models.registry import backbones

        backbone = backbones.create(
            "dino_dpt",
            encoder_name="tu-vit_tiny_patch16_dinov3_qkvb",
            pretrained=False,
            encoder_depth=4,
            encoder_output_indices=[2, 5, 8, 11],
            decoder_intermediate_channels=[16, 32, 64, 64],
            decoder_fusion_channels=16,
            global_pool="token",
            dynamic_img_size=False,
            img_size=64,
        )
        encoder = backbone._encoder  # noqa: SLF001 — pinning smp/timm internals on purpose
        assert isinstance(encoder, torch.nn.Module)
        encoder_model = encoder.model
        assert isinstance(encoder_model, torch.nn.Module)
        assert encoder_model.dynamic_img_size is False
        rope = encoder_model.rope
        assert isinstance(rope, torch.nn.Module)
        assert "pos_embed_cached" in dict(rope.named_buffers())

        class _DecoderOnly(torch.nn.Module):
            def __init__(self, wrapped: torch.nn.Module) -> None:
                super().__init__()
                self.wrapped = wrapped

            def forward(self, image: torch.Tensor) -> torch.Tensor:
                decoder_stream: torch.Tensor = self.wrapped({"image": image})["decoder"]
                return decoder_stream

        module = _DecoderOnly(backbone).eval()
        with torch.no_grad():
            traced = torch.jit.trace(module, torch.randn(1, 3, 64, 64))
        graph = str(traced.inlined_graph)
        assert "pos_embed_cached" in graph  # ROPE read from the movable buffer, not baked constants
