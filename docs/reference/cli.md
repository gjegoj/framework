# CLI reference

Override any config value directly — standard Hydra syntax:

```bash
# change a scalar
uv run python main.py epochs=5 batch_size=64

# swap a config group
uv run python main.py 'defaults=[{override /backbone: smp_unet}]'

# swap dataloader / scheduler / export presets
uv run python main.py dataloader=performance scheduler=cosine export=all

# turn on training regimes
uv run python main.py lora=vit
uv run python main.py distillation=kl 'distillation.teachers.0.ckpt_path=runs/.../teacher.ckpt'

# load a full experiment override
uv run python main.py +experiment=classify_pets

# combine experiment + group swap
uv run python main.py +experiment=classify_pets 'defaults=[{override /logger: clearml}]'

# disable a callback (deletes the key from the dict)
uv run python main.py 'defaults=[{override /callbacks: default}]' '~callbacks.ema'

# per-task LR from CLI
uv run python main.py 'tasks.mask.optimizer.lr=1e-5'

# train only / eval-only / skip export
uv run python main.py run_test=false run_export=false
uv run python main.py run_train=false ckpt_path=runs/.../epoch=11.ckpt
```


---
