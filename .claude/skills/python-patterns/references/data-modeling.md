# Data Modeling

How to hold data: `dataclass` for domain/application objects, `Pydantic` at I/O
boundaries, `NamedTuple` for tiny immutable records, `ABC` for owned interfaces,
`Enum` for closed value sets. The unifying rule: keep validation/coercion at the
edges and pass plain, predictable objects through the core.

## Contents
- [dataclass vs Pydantic — the boundary rule](#dataclass-vs-pydantic--the-boundary-rule)
- [dataclass options](#dataclass-options)
- [NamedTuple](#namedtuple)
- [ABC for owned interfaces](#abc-for-owned-interfaces)
- [Enum and StrEnum](#enum-and-strenum)
- [The mutable-default-argument trap](#the-mutable-default-argument-trap)

## dataclass vs Pydantic — the boundary rule

| Layer | Tool | Why |
|-------|------|-----|
| Domain / application logic | `@dataclass` | stdlib, no hidden coercion, trivial to build in tests |
| Incoming / outgoing data (HTTP, config, JSON) | `Pydantic BaseModel` | validation, coercion, JSON schema, clear errors at the edge |

Validate with Pydantic at the boundary, then convert to dataclasses for the core
so the rest of the code never deals with coercion surprises:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, EmailStr, Field


class CreateUserBody(BaseModel):                 # edge: validate + coerce
    name: str = Field(min_length=1, max_length=200)
    email: EmailStr


@dataclass(frozen=True, slots=True)              # core: plain immutable value object
class User:
    id: str
    name: str
    email: str
    created_at: datetime


def register(body: CreateUserBody) -> User:
    return User(
        id=str(uuid4()),
        name=body.name.strip(),
        email=str(body.email),
        created_at=datetime.now(tz=UTC),
    )
```

Rules of thumb:
- Do **not** pass `BaseModel` instances into domain logic — convert at the edge.
- Keep `__post_init__` for **light** invariants only (a simple guard `raise`);
  cross-field validation with coercion belongs in Pydantic.

## dataclass options

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    host: str
    port: int = 8080
    tags: list[str] = field(default_factory=list)   # never a mutable literal default
    _cache: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
```

- `frozen=True` — immutable and hashable; the default for value objects.
- `slots=True` (3.10+) — drops the per-instance `__dict__` (~50 bytes less, faster
  attribute access). Combine with `frozen=True` for a compact value object.
  Replaces hand-written `__slots__`.
- `kw_only=True` — forces keyword arguments at the call site; prevents positional
  mix-ups and lets you add fields without ordering constraints.
- `field(default_factory=...)` — for any mutable default (list/dict/set).
- `field(repr=False, compare=False)` — exclude a field from `__repr__`/`__eq__`.

## NamedTuple

A lightweight immutable record with tuple semantics (unpackable, hashable,
usable as a dict key). Good for small fixed tuples returned from functions:

```python
from typing import NamedTuple


class Coordinate(NamedTuple):
    lat: float
    lon: float


location = Coordinate(51.5, -0.12)
lat, lon = location          # unpacks like a tuple; also valid as a dict key
```

Prefer `@dataclass(frozen=True, slots=True)` when you need methods, mutability,
or a clear "this is an object, not a tuple" signal. `NamedTuple` shines for
multi-value returns where tuple unpacking reads well.

## ABC for owned interfaces

Use `ABC` + `@abstractmethod` when you **own** the hierarchy and want to enforce
overrides or provide shared default implementations:

```python
from __future__ import annotations

from abc import ABC, abstractmethod


class DataSource(ABC):
    @abstractmethod
    def read(self, path: str) -> list[dict]:
        """Read one source into records. Subclasses implement per format."""

    def read_all(self, paths: list[str]) -> list[dict]:   # shared default
        records: list[dict] = []
        for path in paths:
            records.extend(self.read(path))
        return records
```

Use a `Protocol` (see `type-system.md`) when you do **not** own the types or want
structural typing across unrelated classes — the difference is ownership and
whether you want shared behaviour.

## Enum and StrEnum

A closed set of named values. `StrEnum` (3.11+) members compare equal to their
string value, which serialises cleanly into configs and logs:

```python
from enum import Enum, StrEnum, auto


class Color(Enum):
    RED = auto()
    GREEN = auto()


class Stage(StrEnum):
    TRAIN = "train"          # Stage.TRAIN == "train" is True
    VAL = "val"
```

Prefer an enum over bare string/int "magic" constants wherever a parameter has a
fixed set of valid values (it documents the options and lets `mypy` check them).

## The mutable-default-argument trap

A default argument is evaluated **once**, at function definition — so a mutable
default is shared across every call, accumulating state:

```python
# Bug: the same list is reused on every call.
def append(item: int, into: list[int] = []) -> list[int]:
    into.append(item)
    return into


# Fix: use None as the sentinel and create a fresh object inside.
def append(item: int, into: list[int] | None = None) -> list[int]:
    if into is None:
        into = []
    into.append(item)
    return into
```

The same hazard applies to dataclass fields — that is why mutable dataclass
defaults must go through `field(default_factory=...)`.
