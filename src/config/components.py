"""The one grammar for configuring a named component."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ComponentConfig(BaseModel):
    """Which component to build and with which constructor arguments.

    Three interchangeable forms::

        loss: cross_entropy                                  # registry name
        loss: {name: cross_entropy, label_smoothing: 0.1}    # registry name + params
        loss: {_target_: my_pkg.FocalLoss, gamma: 2.0}       # import path + params

    Exactly one of ``name`` or ``_target_`` must be set; every other key becomes a
    constructor argument. ``_target_`` means what Hydra means by it; Hydra's other
    meta-keys are *rejected* rather than ignored, because silently dropping ``_partial_``
    would give a user an instance where they asked for a factory.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    TARGET_KEY: ClassVar[str] = "_target_"

    name: str | None = Field(None, description="Registry key of the component, e.g. 'adamw', 'cross_entropy'.")
    target: str | None = Field(
        None,
        alias=TARGET_KEY,
        description="Dotted import path to a class or function, for anything the registry does not carry.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_a_bare_name(cls, value: object) -> object:
        """``loss: cross_entropy`` is the same declaration as ``loss: {name: cross_entropy}``."""
        return {"name": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def _require_one_reference(self) -> ComponentConfig:
        """Neither form names a component; both leave it ambiguous which one was meant."""
        if (self.name is None) == (self.target is None):
            raise ValueError(f"Set exactly one of 'name' or '{self.TARGET_KEY}'.")
        return self

    @model_validator(mode="after")
    def _reject_other_meta_keys(self) -> ComponentConfig:
        """Silently dropping ``_partial_`` would hand back an instance where a factory was asked for."""
        reserved = sorted(key for key in (self.model_extra or {}) if key.startswith("_"))
        if reserved:
            raise ValueError(
                f"Unsupported reserved keys: {', '.join(reserved)}. "
                f"Only '{self.TARGET_KEY}' is meaningful here: recursion is always on, and "
                "partial construction is decided in code, not in config."
            )
        return self

    @property
    def params(self) -> dict[str, Any]:
        """Constructor arguments: every key beyond ``name``/``_target_``."""
        return dict(self.model_extra or {})


ModelConfig = ComponentConfig
"""The model to build: a registry name ('timm', 'smp') or an import path, plus its arguments.

Here rather than beside one section because two of them share it: the experiment's
own model and a distillation teacher's — the same declaration, built through the
same registry, so the two cannot drift into different shapes.
"""


TransformConfig = ComponentConfig
"""A sample pipeline to build — the same grammar, named at its points of use.

Declared here rather than beside one section because two of them share it: an
experiment's per-stage transforms and a source's own.
"""


MetricConfig = ComponentConfig
"""One metric to build: a registry name ('accuracy', 'f1') or an import path.

Declared under the label it logs as — the key says *where* it logs, the value
says *what* it is, always explicitly. Here rather than beside one section
because two of them share it: a task's declared metrics and the preset table.
"""
