"""Declaration validation only; runtime components receive values and ready objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from src.core.entities import validate_classes


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=False)
    name: str | None = Field(None, min_length=1)
    import_path: str | None = Field(None, alias="_target_", min_length=1)

    @model_validator(mode="before")
    @classmethod
    def shorthand(cls, value: Any) -> Any:
        return {"name": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def one_selector(self) -> ComponentConfig:
        if (self.name is None) == (self.import_path is None):
            raise ValueError("Declare exactly one of name or _target_.")
        for key in self.model_extra or {}:
            if key in {"_args_", "_partial_", "_recursive_", "_convert_", "import_path"}:
                raise ValueError(f"Unsupported component key: {key}.")
        return self

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class ClassFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: str = Field(min_length=1)


class WeightedLossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    loss: ComponentConfig
    weight: float = Field(1.0, ge=0, allow_inf_nan=False)
    log_name: str | None = Field(None, min_length=1)


class HeadConfig(ComponentConfig):
    input: str | list[str] | None = None

    @field_validator("input")
    @classmethod
    def selected_inputs(cls, value: str | list[str] | None) -> str | list[str] | None:
        names = [value] if isinstance(value, str) else value
        if names is not None and (
            not names or any(not name.strip() for name in names) or len(set(names)) != len(names)
        ):
            raise ValueError("Head inputs require nonblank, distinct feature names.")
        return value


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: ComponentConfig
    target: str | None = None
    classes: dict[int, str] | ClassFile | None = None
    head: HeadConfig | None = None
    output: str | None = Field(None, min_length=1)
    loss: ComponentConfig | list[WeightedLossConfig] | None = None
    target_encoder: ComponentConfig | None = None
    metrics: dict[str, ComponentConfig] | None = None
    weight: float = Field(1.0, gt=0, allow_inf_nan=False)
    lr: float | None = Field(None, gt=0, allow_inf_nan=False)

    @field_validator("classes", mode="before")
    @classmethod
    def explicit_classes(cls, value: Any) -> Any:
        if value is None or isinstance(value, ClassFile):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(  # noqa: TRY004 -- Pydantic wraps ValueError, not TypeError.
                "Classes require an index-to-name mapping or a file declaration."
            )
        if "file" in value:
            return ClassFile.model_validate(value)
        normalized: dict[int, str] = {}
        for key, name in value.items():
            if type(key) is int:
                index = key
            elif isinstance(key, str) and key.isascii() and key.isdecimal():
                index = int(key)
            else:
                raise ValueError(f"Invalid class index: {key!r}.")
            if index in normalized:
                raise ValueError(f"Colliding class index: {key!r}.")
            normalized[index] = name
        validate_classes(normalized)
        return normalized

    @model_validator(mode="after")
    def task_connections(self) -> TaskConfig:
        if self.head is not None and self.output is not None:
            raise ValueError("head and output are mutually exclusive.")
        validate_losses(self.loss)
        return self


def validate_losses(loss: ComponentConfig | list[WeightedLossConfig] | None) -> None:
    if isinstance(loss, list):
        if not loss or not any(item.weight > 0 for item in loss):
            raise ValueError("A loss list needs at least one positive weight.")
        explicit = [item.log_name for item in loss if item.log_name is not None]
        if len(set(explicit)) != len(explicit):
            raise ValueError("Explicit loss log names must be distinct.")


class AdapterConfig(ComponentConfig):
    module: str = Field(description="Path inside this model; empty string explicitly selects the whole model.")

    @model_validator(mode="after")
    def module_path(self) -> AdapterConfig:
        if self.module and any(not part.strip() or part.strip() != part for part in self.module.split(".")):
            raise ValueError("Adapter module must be a dotted module path, or empty for the whole model.")
        return self


def adapter_list(value: Any) -> Any:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


AdaptersConfig = Annotated[list[AdapterConfig], BeforeValidator(adapter_list)]
