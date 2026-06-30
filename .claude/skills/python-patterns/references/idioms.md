# Idioms

Everyday Python idioms: resource management, decorators, sentinels, lazy
iteration, and a catalogue of anti-patterns to avoid. These are the small,
high-frequency choices that make code read as idiomatic rather than translated
from another language.

## Contents
- [Naming conventions](#naming-conventions)
- [Context managers](#context-managers)
- [Decorators](#decorators)
- [Sentinel for "no value" when None is valid](#sentinel-for-no-value-when-none-is-valid)
- [Generators and comprehensions](#generators-and-comprehensions)
- [pathlib over os.path](#pathlib-over-ospath)
- [Small wins](#small-wins)
- [Anti-patterns](#anti-patterns)

## Naming conventions

These are the Python-specific *spellings* — the universal "names should reveal
intent / explicit over implicit" guidance belongs to `clean-code`, not here.

Casing:
- `snake_case` — functions, methods, variables, and module/package names.
- `PascalCase` — classes, `Protocol`s, `Enum`s, type aliases.
- `UPPER_SNAKE_CASE` — module-level constants.
- `self` / `cls` — always the first parameter of instance / class methods.

Underscores carry real meaning — don't use them decoratively:
- `_name` — "internal by convention" (a weak signal, not enforced); excluded from
  `from module import *`, and a hint to treat it as private API.
- `__name` (two leading underscores) — triggers **name mangling** inside a class
  (`__x` is rewritten to `_ClassName__x`), used to avoid attribute clashes in
  subclasses. It is *not* a stronger "private" — reach for it rarely.
- `name_` (one trailing underscore) — avoids clashing with a keyword or builtin
  (`class_`, `type_`, `id_`).
- `_` — a throwaway you don't intend to use (`for _ in range(3)`, `value, _ = pair`).
- `__dunder__` — reserved for Python's own protocols (`__init__`, `__enter__`,
  `__repr__`); never invent your own.

## Context managers

A context manager guarantees setup/teardown even on exceptions. Prefer the
`@contextmanager` decorator for stateless or one-shot resources; code before
`yield` is `__enter__`, code after is `__exit__`:

```python
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def timer(label: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter() - start:.4f}s")


with timer("batch"):
    process()
```

Use the class form when you need state, or when `__enter__` should return an
object the caller binds with `as`:

```python
class Transaction:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def __enter__(self) -> Transaction:
        self._connection.begin()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            self._connection.commit()
        else:
            self._connection.rollback()
        return False   # False/None → exceptions propagate; True → suppressed (rare)
```

Returning `False` from `__exit__` lets the exception propagate; return `True`
only to deliberately swallow it.

## Decorators

Wrap a function with `functools.wraps` so the wrapper keeps the original name,
docstring, and signature metadata:

```python
from __future__ import annotations

import functools
import time
from collections.abc import Callable


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            print(f"{func.__name__}: {time.perf_counter() - start:.4f}s")
    return wrapper
```

`[**P, R]` (PEP 695 `ParamSpec`) preserves the wrapped function's exact
signature for type checkers. A **parameterized** decorator adds a third closure
level — the outer function takes the arguments and returns the decorator:

```python
def retry(times: int, exceptions: tuple[type[Exception], ...] = (Exception,)) -> Callable:
    def decorator[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == times - 1:
                        raise
            raise AssertionError("unreachable")
        return wrapper
    return decorator


@retry(times=3, exceptions=(TimeoutError, ConnectionError))
def fetch(url: str) -> str:
    ...
```

## Sentinel for "no value" when None is valid

When `None` is a legitimate value, a module-level sentinel distinguishes "not
provided" from "explicitly None". Compare by identity (`is`):

```python
from enum import Enum


class _Missing(Enum):
    token = 0          # an enum sentinel types cleanly: `str | _Missing`


MISSING = _Missing.token


def update(name: str | _Missing = MISSING) -> None:
    if name is not MISSING:
        ...            # name was passed (possibly as None)
```

A plain `_MISSING = object()` also works but is harder to annotate; the enum
form is preferred for public APIs.

## Generators and comprehensions

Return an `Iterator` (a generator) when the caller may stop early or the sequence
is large — it avoids materialising everything in memory:

```python
from __future__ import annotations

from collections.abc import Iterator


def read_lines(path: str) -> Iterator[str]:
    with open(path) as file:
        for line in file:
            yield line.strip()


for line in read_lines("large.log"):
    if "ERROR" in line:
        break          # the rest of the file is never read
```

Return a `list` when the caller needs random access or the result is small and
finite. Use a generator **expression** (no brackets) to stream into a consumer
without building an intermediate list:

```python
total = sum(value * value for value in range(1_000_000))   # no temp list
```

Use a comprehension for simple map/filter; when it grows a second `for` or a
non-trivial condition, expand it into a loop for readability.

## pathlib over os.path

`pathlib.Path` is the modern, object-oriented file-path API — composable with `/`
and far more readable than string juggling:

```python
from pathlib import Path

config = Path.home() / ".config" / "app.toml"
if config.exists():
    text = config.read_text(encoding="utf-8")
for script in Path("scripts").glob("*.py"):
    ...
```

## Small wins

- **f-strings** for formatting; `f"{value!r}"` for debug repr, `f"{x:.2f}"` for precision.
- **`enumerate`** for index+item, **`zip(..., strict=True)`** to pair sequences and fail on length mismatch.
- **`str.join`** to build strings, never `+=` in a loop (quadratic).
- **Unpacking**: `first, *rest = items`; `a, b = b, a` to swap.
- **`dict`/`set` membership** (`x in mapping`) is O(1); scanning a list is O(n).

## Anti-patterns

```python
# Mutable default argument — shared across calls. Use None + create inside.
def f(items=[]): ...                # bug      →  def f(items=None): ...

# Identity vs equality for singletons.
if value == None: ...               # wrong    →  if value is None: ...

# Type checks: isinstance respects subclasses; type() == does not.
if type(obj) == list: ...           # brittle  →  if isinstance(obj, list): ...

# Bare except hides KeyboardInterrupt/SystemExit and real bugs.
try: ...
except: ...                         # never    →  except SpecificError as error: ...

# Wildcard import pollutes the namespace and breaks tooling.
from module import *                # avoid     →  from module import needed_name

# Manual resource handling — leaks on exception.
file = open(path); data = file.read(); file.close()   # →  with open(path) as file: ...

# String concatenation in a loop is O(n^2).
out = ""
for part in parts: out += part      # →  out = "".join(parts)
```
