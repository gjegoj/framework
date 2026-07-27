# Export

After `fit`/`test`, the model is exported for deployment (gated by `run_export`). Export is
a config group (`onnx` · `torchscript` · `tensorrt` · `all`); targets are a per-format list, so one
run can emit several formats:

```yaml
defaults:
  - export: onnx        # or: torchscript · tensorrt · all

export:
  targets:
    - {format: onnx, opset_version: 17, dynamic_batch: true, simplify: true}
    - {format: torchscript, method: trace}
  combined: true          # one graph: image → all task logits
  split_components: false # also write backbone + each head as separate files
  output_dir: null        # defaults to {save_dir}/export
```

Each format validates its own option surface at `load_config` time (a misplaced
`opset_version` under `torchscript` fails immediately). Every target is **verified**: the
written artifact is re-run and its outputs compared to the source model within tolerance —

```yaml
export:
  targets:
    - {format: onnx, verify_outputs: true, atol: 1.0e-4, rtol: 1.0e-3}
```

A rich table reports per-output abs/rel error and a pass/fail verdict. Disable export with
`run_export: false` or an empty `targets` list.

**TensorRT.** The `tensorrt` target compiles straight from the PyTorch graph via torch-tensorrt
(no ONNX intermediate) to a serialized engine (`model_*.plan`) written to `{save_dir}/export/`.
The `shapes` profile references `image_size` instead of hardcoding H/W:

```yaml
defaults:
  - export: tensorrt

export:
  targets:
    - format: tensorrt
      precision: fp16          # or fp32
      atol: 1.0e-2             # fp16 needs a looser parity tolerance
      shapes:                  # min/opt/max optimization profile (drop it → batch 1/4/8)
        min: [1, 3, "${image_size.0}", "${image_size.1}"]
        opt: [4, 3, "${image_size.0}", "${image_size.1}"]
        max: [8, 3, "${image_size.0}", "${image_size.1}"]
```

> CUDA-only: a `.plan` engine is hardware + TensorRT-version specific, so build it on a node
> matching your Triton deployment. Install the optional backend once:
> `uv add --optional export-trt torch-tensorrt tensorrt`.
