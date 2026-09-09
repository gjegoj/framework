---
status: accepted
date: 2026-09-07
---

# A model that arrives whole is any `Model` reached by `_target_`

The framework had two kinds of run, told apart by the name in `config.model`: a
*composed* one (a backbone from `backbone_registry`, one head per task) and a *vendor*
one (`YoloModel` with `YoloDataModule`, found in `vendor_model_registry`), with a table
of what the second could not serve, a second registry for its pipeline, and a port
method (`DataModule.collate`) that existed for its batching. We retire the vendor kind
of run. A model that owns its head, loss and decoding is reached like any other
component — `model: {_target_: my_pkg.MyDetector, ...}` — and `build_model` returns a
built `Model` as it is; only a `Backbone` is composed. What such a model cannot serve
is refused where the need arises, by type: adapters and distillation require a
`CompositeModel`, and say so.

## Why

- One kind of run, one grammar: extension is a class plus a config line, with no
  registry to join and no refusal table to extend.
- `ultralytics` stops being a *family* and becomes a *library behind ports*, exactly
  as timm and smp are: imported only inside the detection family's own modules (its
  backbone and head today, its criterion and decoder with P0.3b). The dependency stays.
- The vendor path forfeited the framework's guarantees (declared transforms, the loader
  cache, honest loss parts, EMA and the other training practices as declared features);
  keeping it meant carrying two of everything through every refactor.

## Consequences

- Deleted: `src/models/yolo.py`, `src/data/datamodules/yolo.py`, both vendor
  registries, the vendor block of `build.py`, `DataModule.collate`, the legality of
  `inputs: {}`, `configs/model/yolov8n.yaml`, `configs/experiment/examples/detection.yaml`
  (until P0.3b), 35 tests. Kept: `UltralyticsBackbone`, `DetectHead`, the whole
  detection data path, and their tests.
- Between this decision and P0.3b no detector trains: the `Detection` kind refuses by
  name, at construction, without pointing anywhere. `-seg`/`-pose`, the NMS-free lines
  and RT-DETR return only as head adapters of their own, when someone needs them.
- A whole model is built from its declaration alone: nothing composes it, so the sizes it
  needs are its own to declare (`num_classes: 3`), and no fact is matched into its signature.
- The port defaults a whole model relies on (`criterion_of` → `None`,
  `task_parameters` → nothing, `statistics` → empty) stay; their docstrings name a
  model that owns its loss, not a vendor.
