---
name: python-patterns
description: Python-specific standards for any Python 3.12+ project: `from __future__ import annotations`, type system (PEP 695, Protocol, TYPE_CHECKING, collections.abc, @override), data modeling (dataclass vs Pydantic, slots, ABC), Google-style docstrings with Parameters:, error handling with exception chaining, context managers, and key patterns (parameterized decorators, sentinel, generators). Use when writing, reviewing, or refactoring Python code.
disable-model-invocation: false
---

# Python Patterns

Modern Python standards for any 3.12+ project. Does not duplicate what `clean-code` (naming, function design, testing) and `refactoring-patterns` (Extract Method, guard clauses, CQS) already cover — focus here is Python-specific idioms.

## Conventions

Every Python file starts with:

```python
from __future__ import annotations
```

This stores all annotations as strings at runtime, enabling forward references without quotes. Still required in Python 3.12 — PEP 649 (native lazy evaluation) arrives in Python 3.14.

- Python 3.12+, line length 120
- `ruff format` + `ruff check --fix` (handles import sorting too)
- `mypy` (all public functions annotated)

## Type System

### Built-in generics and unions

Use built-in types in annotations, not `typing.List`/`typing.Dict`/`typing.Optional`:

```python
from __future__ import annotations

from typing import Any


def process(items: list[str], mapping: dict[str, Any]) -> tuple[str, ...]:
    return tuple(items)


def find(user_id: str) -> User | None:  # X | Y, not Optional[X]
    ...
```

### PEP 695 type aliases and generics (Python 3.12+)

```python
from __future__ import annotations

import json

type JSON = dict[str, JSON] | list[JSON] | str | int | float | bool | None


def parse(data: str) -> JSON:
    return json.loads(data)


def first[T](items: list[T]) -> T | None:
    return items[0] if items else None
```

### Protocol for duck typing

Use `Protocol` when you want structural subtyping (duck typing) without requiring inheritance:

```python
from __future__ import annotations

from typing import Protocol


class Renderable(Protocol):
    def render(self) -> str: ...


def render_all(items: list[Renderable]) -> str:
    return "\n".join(item.render() for item in items)
```

### TYPE_CHECKING for import cycles

Imports inside `if TYPE_CHECKING:` execute only during type analysis, not at runtime. Use for type-only imports that would create circular dependencies:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.user import User  # only for mypy, not imported at runtime


def process(callback: Callable[[User], None]) -> None:
    ...
```

### `collections.abc` for argument types

Accept the widest contract — use `collections.abc` abstract types for **arguments**, concrete types for **return values**:

```python
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence


def process_items(items: Sequence[str]) -> list[str]:
    return [s.upper() for s in items]


def lookup(config: Mapping[str, str], key: str) -> str | None:
    return config.get(key)


def apply(fn: Callable[[int], int], values: Iterator[int]) -> list[int]:
    return [fn(v) for v in values]
```

`Sequence` accepts lists, tuples, strings. `Mapping` accepts dicts, `defaultdict`, etc. This makes functions usable with more types without changing their contract.

### `@override` for explicit method contracts

```python
from __future__ import annotations

from typing import override


class Base:
    def compute(self, x: int) -> int:
        return x


class Derived(Base):
    @override  # mypy errors if signature doesn't match Base.compute
    def compute(self, x: int) -> int:
        return x * 2
```

## Data Modeling

### dataclass vs Pydantic

| Layer | Tool | Why |
|-------|------|-----|
| Domain / application logic | `@dataclass` | stdlib, no hidden coercion, trivial to construct in tests |
| Incoming / outgoing data | `Pydantic BaseModel` | validation, coercion, JSON schema, clear errors at boundaries |

Flow: Pydantic validates at the edge → map to dataclasses inside services → Pydantic again for outgoing responses when shape needs to be stable.

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, EmailStr, Field


# Edge: incoming request body (Pydantic validates/coerces)
class CreateUserBody(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: EmailStr


# Domain: internal object (dataclass, no coercion surprises)
@dataclass(frozen=True)
class User:
    id: str
    name: str
    email: str
    created_at: datetime


def register(body: CreateUserBody) -> User:
    """Create a domain user from a validated API payload.

    Parameters:
        body (CreateUserBody): Already validated by Pydantic.

    Returns:
        User: Immutable domain object for persistence and use cases.
    """
    return User(
        id=str(uuid4()),
        name=body.name.strip(),
        email=str(body.email),
        created_at=datetime.now(tz=UTC),
    )
```

Rules of thumb:
- Do **not** pass `BaseModel` instances into domain logic; convert at the edge.
- Use `frozen=True` for value objects (immutable, hashable).
- Keep `__post_init__` for **light** invariants (a simple guard raise). Cross-field validation with coercion belongs in Pydantic.

### `@dataclass(slots=True)` (Python 3.10+)

Replaces manual `__slots__`; eliminates per-instance `__dict__` (~50 bytes less per object):

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Vector:
    x: float
    y: float
    z: float = 0.0
```

Combine with `frozen=True` for an immutable slot-based value object: `@dataclass(frozen=True, slots=True)`.

### NamedTuple for simple immutable records

When you need a lightweight immutable container with tuple semantics (unpackable, usable as dict key):

```python
from typing import NamedTuple


class Coordinate(NamedTuple):
    lat: float
    lon: float


loc = Coordinate(51.5, -0.12)
lat, lon = loc  # unpacking works; works as dict key too
```

Prefer `@dataclass(frozen=True)` if you need methods or slots.

### ABC for interfaces

Use `ABC + abstractmethod` when you own the hierarchy and want to enforce overrides or provide shared default implementations:

```python
from __future__ import annotations

from abc import ABC, abstractmethod


class DataSource(ABC):
    """Abstract port for reading records."""

    @abstractmethod
    def read(self, path: str) -> list[dict]: ...

    def read_all(self, paths: list[str]) -> list[dict]:
        result: list[dict] = []
        for p in paths:
            result.extend(self.read(p))
        return result


class CsvSource(DataSource):
    def read(self, path: str) -> list[dict]:
        ...
```

Use `Protocol` (see Type System) when you do *not* own the hierarchy or want structural typing across unrelated classes.

## Docstrings

Use **Google-style**. Key conventions:
- One-line summary, blank line, then optional narrative.
- `Parameters:` (not `Args:`): `param_name (type): description`.
- `Returns:` with type before the colon: `User | None: …`.
- `Raises:` and `Examples:` only when they add information not obvious from types or names.

```python
def fetch_user(user_id: str, *, include_inactive: bool = False) -> User | None:
    """Load a user record by identifier.

    Queries the primary store; does not cache. Callers that need caching
    should wrap this function.

    Parameters:
        user_id (str): Unique identifier (non-empty).
        include_inactive (bool): When True, include deactivated users.

    Returns:
        User | None: The matching user, or None if not found.

    Raises:
        ValueError: If ``user_id`` is empty or whitespace-only.
    """
    if not user_id.strip():
        raise ValueError("user_id must be non-empty")
    ...
```

| Section | When to include |
|---------|-----------------|
| Summary | Always (first line) |
| Body | Side effects, caveats, non-obvious behavior |
| Parameters | Non-trivial or non-obvious arguments |
| Returns | Non-obvious return value or multiple outcomes |
| Raises | Exceptions callers should handle |
| Examples | Non-obvious usage or edge cases |

## Error Handling

Catch specific exceptions. Always chain with `from e` to preserve the original traceback:

```python
from __future__ import annotations

import json


def load_config(path: str) -> Config:
    try:
        with open(path) as f:
            return Config.from_json(f.read())
    except FileNotFoundError as e:
        raise ConfigError(f"Config not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON: {path}") from e
```

Never use bare `except:` — it silently catches `KeyboardInterrupt`, `SystemExit`, and hides bugs. `except Exception:` is the widest acceptable catch.

### Custom exception hierarchy

```python
class AppError(Exception):
    """Base for all application errors."""


class ValidationError(AppError):
    """Input failed validation."""


class NotFoundError(AppError):
    """Requested resource does not exist."""
```

Keep the hierarchy shallow (2 levels is usually enough). Callers catch `AppError` broadly or specific subclasses as needed.

## Context Managers

### `@contextmanager` (functional form)

Prefer for stateless or one-shot resources. Code before `yield` is `__enter__`, after `yield` is `__exit__`:

```python
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def timer(label: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - start:.4f}s")


with timer("batch"):
    process()
```

### Class form

Use when you need state, or when `__enter__` should return `self` so the caller can use the bound object:

```python
class Transaction:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def __enter__(self) -> Transaction:
        self._conn.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        return False  # False = let exceptions propagate; True = suppress them
```

`__exit__` returning `False` (or `None`) lets exceptions propagate. Return `True` only to intentionally silence them (rare).

## Patterns

### Parameterized decorators

Three-level closure: outer function takes params → returns decorator → decorator wraps the function. The extra level is what trips people:

```python
from __future__ import annotations

import functools
from collections.abc import Callable


def retry(times: int, exceptions: tuple[type[Exception], ...] = (Exception,)) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == times - 1:
                        raise
        return wrapper
    return decorator


@retry(times=3, exceptions=(TimeoutError, ConnectionError))
def fetch(url: str) -> str:
    ...
```

### Sentinel for "no value" when `None` is valid

When `None` is a legitimate domain value, use a sentinel to distinguish "not provided" from "explicitly set to None":

```python
from __future__ import annotations

_MISSING = object()  # module-level sentinel — identity check only


def update_user(user_id: str, name: str | object = _MISSING) -> User:
    if name is not _MISSING:
        ...  # name was explicitly passed
```

For public APIs, an `enum` sentinel is cleaner to type:

```python
from enum import Enum, auto


class _Unset(Enum):
    token = auto()


UNSET = _Unset.token


def update(value: str | _Unset = UNSET) -> None:
    if value is not UNSET:
        ...
```

### Generator vs list

Return `Iterator` when the caller may not need all items, or the sequence is large — avoids materializing everything in memory:

```python
from __future__ import annotations

from collections.abc import Iterator


def read_lines(path: str) -> Iterator[str]:
    with open(path) as f:
        for line in f:
            yield line.strip()


for line in read_lines("large.log"):
    if "ERROR" in line:
        break  # stops after first match; rest of file never read
```

Return `list` when the caller always needs random access or the result is naturally finite and small.
