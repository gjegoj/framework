# Backbones

Select the backbone group in `defaults` or override it:

```yaml
defaults:
  - backbone: resnet18    # configs/backbone/resnet18.yaml
```

| Group file | Architecture | Kind |
|---|---|---|
| `resnet18.yaml` | timm ResNet-18 | `timm` |
| `smp_unet.yaml` | smp U-Net (ResNet-34 encoder) | `smp` |
| `smp_dpt.yaml` | smp DPT | `smp` |
| `dinov3_dpt.yaml` | DINOv3 ViT encoder + DPT decoder | `dino_dpt` |
| `embedding.yaml` | precomputed feature vectors (no encoder) | `embedding` |

| Kind | Use for |
|---|---|
| `timm` | any timm classifier / encoder (global tasks) |
| `smp` | segmentation & multi-task (two spatial streams) |
| `dino_dpt` | DINOv3/ViT encoder with a DPT decoder (dense + global multi-task) |
| `embedding` | precomputed embeddings modality |
| `multi` | N named encoders for MULTISTREAM (dual-encoder / CLIP-style) |

**timm backbone** (any model from the timm registry):

```yaml
backbone:
  kind: timm
  name: efficientnet_b3
  pretrained: true
```

**smp backbone** for segmentation or multi-task:

```yaml
backbone:
  kind: smp
  name: unet
  encoder_name: resnet34
  pretrained: true
```

SMP exposes two feature streams:

| Key | Shape | Use for |
|---|---|---|
| `decoder` | `[B, D, H, W]` | segmentation head (default for `segmentation` preset) |
| `encoder_last` | `[B, D, H, W]` | classification head with SMP's internal pooling |

For **multi-task on a single smp backbone**, set `feature_key` per task:

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    num_classes: 3
    # feature_key: decoder  ← default, no need to write

  label:
    preset: classification
    target: label
    class_mapping: {0: cat, 1: dog}
    feature_key: encoder_last   # explicit: use encoder output, not decoder
```
