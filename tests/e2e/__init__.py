"""Whole runs, from a declaration on disk to what the run leaves behind.

Three shapes live here, and the difference is the *path* each one exercises rather than a
matter of taste:

- **Config-driven** — ``disk_config(dataset_root, <the section this test is about>)``,
  then ``assemble`` and ``run``. Every one of these looks the same, and the only thing
  visible in the test is what makes it different: ``adapters``, ``export``,
  ``distillation``, a callback. This is the shape to copy for a new one.
- **YAML-driven** — a real YAML string composed through OmegaConf. The subject *is*
  interpolation and Hydra's own composition, which a mapping cannot exercise: a config
  built in Python has no ``${lr}`` to resolve.
- **Hand-assembled** — the objects built directly and Lightning driven by hand, for the
  runs whose subject is a wiring assembly does not produce: a stand-in backbone, a
  schema the config grammar has no way to spell.

The dataset is always ``dataset_root`` or the ``tests.support.datasets`` primitives, so
no test writes image files of its own.
"""
