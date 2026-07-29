# Recipes

```yaml
# configs/experiment/classify_pets.yaml
# @package _global_
defaults:
  - override /model: resnet18
  - override /callbacks: default

project: pets
epochs: 20
batch_size: 32
image_size: [224, 224]
lr: 1.0e-3

data:
  sources: data/pets.csv
  inputs: image_path
  split: {train: 0.8, val: 0.1, test: 0.1}

tasks:
  species:
    preset: classification
    target: species
    class_mapping: {0: cat, 1: dog, 2: rabbit}
```

```bash
uv run python main.py +experiment=classify_pets
```



```yaml
# configs/experiment/multitask.yaml
# @package _global_
defaults:
  - override /model: smp_unet
  - override /callbacks: default

project: multitask-demo
epochs: 30
batch_size: 8
image_size: [512, 512]
lr: 1.0e-3

data:
  sources: data/annotations.csv
  inputs: image_path
  split: {train: 0.8, val: 0.1, test: 0.1}

tasks:
  mask:
    preset: segmentation
    target: mask_path
    class_mapping: {0: background, 1: defect, 2: edge}
    loss: {name: weighted_sum, losses: {cross_entropy: 1.0, dice: 1.0}}

  label:
    preset: classification
    target: label
    class_mapping: {0: ok, 1: defective}
    feature_key: encoder_last
    optimizer:
      lr: 5.0e-4
```



```yaml
# configs/experiment/align.yaml
# @package _global_
defaults:
  - override /model: ...        # a `multi` backbone (image + other encoder)
  - override /callbacks: default
  - override /scheduler: cosine

project: align
epochs: 50
batch_size: 64
lr: 3.0e-4

data:
  sources: data/pairs.csv
  inputs:
    image: image_path
    other: other_path
  split: {train: 0.9, val: 0.1}

tasks:
  align:
    preset: contrastive          # MULTISTREAM + InfoNCE
    target: image                # structural target
    dim: 256
    loss: siglip                 # or info_nce
```



```yaml
# configs/experiment/finetune.yaml
# @package _global_
defaults:
  - override /callbacks: default

project: finetune
epochs: 40
lr: 5.0e-4

# Extend the default callback set with freeze.
# Declare freeze before checkpoint so unfreezing runs before the save decision.
callbacks:
  freeze:
    targets: [model.backbone]
    unfreeze_at: 0.25   # unfreeze after 25% of epochs

# ... data and tasks as usual
```



```yaml
# configs/experiment/distill.yaml
# @package _global_
defaults:
  - override /model: resnet18       # the STUDENT backbone (small)
  - override /callbacks: default
  - distillation: kl

project: distill-demo
epochs: 30
batch_size: 32
image_size: [256, 256]
lr: 1.0e-3

distillation:
  teachers:
    - model: {kind: dino_dpt, name: dpt, encoder_name: ...}      # the TEACHER architecture
      ckpt_path: runs/teacher/checkpoints/best.ckpt              # its trained weights (EMA-aware)
  temperature: 2.0
  weight: 0.7

# data and tasks exactly as in a normal run — the hard loss is untouched
```

Watch `loss/train/<task>/kl` next to the hard components; val loss stays a pure task loss,
so the best-checkpoint choice is comparable with the non-distilled baseline.



```yaml
# configs/experiment/lora_finetune.yaml
# @package _global_
defaults:
  - override /model: dinov3_dpt
  - override /callbacks: default
  - lora: vit

project: lora-demo
epochs: 20
lr: 1.0e-3

init_ckpt_path: runs/pretrained/full.ckpt   # optional: seed the frozen base from a full checkpoint

lora:
  target_modules: [qkv, proj, fc1, fc2]
  rank: 8
  alpha: 16

# data and tasks as usual; heads train fully, backbone trains only through adapters.
# Checkpoints hold adapters+heads only; export merges them into one plain artifact.
```



```yaml
# configs/experiment/debug_quick.yaml
# @package _global_
defaults:
  - override /trainer: cpu_smoke
  - override /dataloader: debug
  - override /callbacks: none
  - override /logger: none

epochs: 2
batch_size: 4
image_size: [64, 64]

data:
  max_samples: 100

# ... tasks as usual
```


---
