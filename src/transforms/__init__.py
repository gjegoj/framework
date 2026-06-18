"""Transforms: per-sample input augmentation and cross-sample batch transforms.

Kept deliberately import-light: submodules are imported directly
(``src.transforms.sample`` for the per-sample ``Transform``/Albumentations,
``src.transforms.batch`` for batch transforms) so that pulling the per-sample
transform into the data layer does not drag in torchvision/tasks via the batch
package.
"""
