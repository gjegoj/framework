# Type System

Static typing for Python 3.12+. The goal is maximum information for `mypy` and
readers with minimum syntactic noise. Prefer built-in collection types and
`collections.abc`; reach into `typing` only for constructs that have no syntax
form — `Protocol`, `Any`, `TYPE_CHECKING`, `override`, `Self`, `Literal`,
`Final`, `cast`, `overload`. The legacy aliases (`List`, `Dict`, `Optional`,
`Union`, `TypeVar`, `typing.Callable`) are obsolete on 3.12 — see the last section.

## Contents
- [Built-in generics and unions](#built-in-generics-and-unions)
- [PEP 695 type aliases and generics](#pep-695-type-aliases-and-generics)
- [`collections.abc` for arguments](#collectionsabc-for-arguments)
- [Protocol — structural typing](#protocol--structural-typing)
- [`TYPE_CHECKING` — break import cycles](#type_checking--break-import-cycles)
- [`@override`](#override)
- [`Self`, `Literal`, `Final`, `TypedDict`](#self-literal-final-typeddict)
- [`Any` vs `object`](#any-vs-object)
- [What to avoid (legacy typing)](#what-to-avoid-legacy-typing)

## Built-in generics and unions

Use the built-in collection types and `|` unions directly in annotations:

```python
from __future__ import annotations

from typing import Any


def process(items: list[str], mapping: dict[str, Any]) -> tuple[str, ...]:
    return tuple(items)


def find(user_id: str) -> User | None:  # X | None, never Optional[X]
    ...
```

`list[str]`, `dict[str, int]`, `tuple[int, ...]`, `set[str]`, `frozenset[int]` —
all built-in. `X | Y` replaces `Union[X, Y]`; `X | None` replaces `Optional[X]`.

## PEP 695 type aliases and generics

The `type` statement defines aliases; the `[T]` syntax declares generic
functions and classes — no `TypeVar`/`TypeAlias` boilerplate, no module-level
type variables to import and keep in sync.

```python
from __future__ import annotations

import json

# Alias — lazily evaluated, so recursive references work without quotes.
type JSON = dict[str, JSON] | list[JSON] | str | int | float | bool | None


def parse(data: str) -> JSON:
    return json.loads(data)


# Generic function: the type parameter is scoped to the function.
def first[T](items: list[T]) -> T | None:
    return items[0] if items else None


# Generic class.
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()
```

Bounds and constraints live inline:

```python
def total[T: (int, float)](values: list[T]) -> T:  # constrained to int or float
    ...


def clamp[T: Comparable](value: T, low: T, high: T) -> T:  # upper bound
    ...
```

## `collections.abc` for arguments

Accept the widest contract you can. Annotate **arguments** with abstract types
from `collections.abc`, and return **concrete** types — callers should know
exactly what they get back, but you should accept anything that fits:

```python
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence


def upper_all(items: Sequence[str]) -> list[str]:      # accepts list, tuple, ...
    return [item.upper() for item in items]


def lookup(config: Mapping[str, str], key: str) -> str | None:  # accepts dict, ...
    return config.get(key)


def apply(fn: Callable[[int], int], values: Iterable[int]) -> list[int]:
    return [fn(value) for value in values]
```

`Sequence` accepts lists, tuples, and strings; `Mapping` accepts dicts and
`defaultdict`; `Iterable` accepts generators. `Callable` comes from
`collections.abc`, not `typing`.

## Protocol — structural typing

`Protocol` enables duck typing checked by `mypy`: a class satisfies it by having
the right shape, with no inheritance. Use it when you do **not** own the type
(third-party classes) or want to type across unrelated classes.

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable


class Renderable(Protocol):
    def render(self) -> str: ...


def render_all(items: list[Renderable]) -> str:
    return "\n".join(item.render() for item in items)


@runtime_checkable          # opt in to isinstance() checks (attribute presence only)
class Sized(Protocol):
    def __len__(self) -> int: ...
```

Use an `ABC` instead when you **own** the hierarchy and want to enforce
overrides or share default method implementations (see `data-modeling.md`).

## `TYPE_CHECKING` — break import cycles

Imports inside `if TYPE_CHECKING:` run only during type analysis, never at
runtime. Combined with `from __future__ import annotations` (which makes all
annotations strings), they break circular imports cleanly:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from myapp.models import User  # only for mypy; not imported at runtime


def on_login(callback: Callable[[User], None]) -> None:
    ...
```

## `@override`

Mark a method that is meant to override a base method. `mypy` then errors if the
base method is renamed or its signature drifts — catching a whole class of
silent bugs.

```python
from __future__ import annotations

from typing import override


class Base:
    def compute(self, x: int) -> int:
        return x


class Doubler(Base):
    @override
    def compute(self, x: int) -> int:
        return x * 2
```

## `Self`, `Literal`, `Final`, `TypedDict`

```python
from __future__ import annotations

from typing import Final, Literal, Self, TypedDict

MAX_RETRIES: Final = 3                       # reassignment is a type error


class Builder:
    def with_name(self, name: str) -> Self:  # returns the concrete subclass type
        self._name = name
        return self


def open_mode(mode: Literal["r", "w", "a"]) -> None:  # only these exact values
    ...


class Point(TypedDict):                      # the shape of a dict at the boundary
    x: int
    y: int
```

Reach for `TypedDict` only when a dict shape is fixed by an external contract
(JSON payloads); for your own data prefer a `dataclass` (see `data-modeling.md`).

## `Any` vs `object`

`Any` disables type checking — it is compatible with everything in both
directions, so it silently turns off `mypy`. Use it only at genuine boundaries
(untyped third-party returns, truly dynamic data). When you mean "any object but
I will inspect it before use", prefer `object` — `mypy` then forces a narrowing
check (`isinstance`) before you can call methods on it.

## What to avoid (legacy typing)

| Avoid (pre-3.10/3.12) | Use instead (3.12+) |
|---|---|
| `List[int]`, `Dict[str, int]` | `list[int]`, `dict[str, int]` |
| `Optional[X]` | `X \| None` |
| `Union[X, Y]` | `X \| Y` |
| `typing.Callable`, `typing.Iterable` | `collections.abc.Callable`, `...Iterable` |
| `TypeVar("T")` + `Generic[T]` | `def f[T]() -> ...`, `class C[T]:` |
| `TypeAlias` / `X: TypeAlias = ...` | `type X = ...` |
