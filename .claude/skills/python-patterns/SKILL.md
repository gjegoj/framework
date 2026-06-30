---
name: python-patterns
description: Modern Python 3.12+ idioms and standards across six areas — the type system (PEP 695 generics/aliases, Protocol, collections.abc, TYPE_CHECKING, @override), data modeling (dataclass vs Pydantic, slots/frozen, NamedTuple, ABC, Enum), Google-style docstrings, error handling (specific excepts, exception chaining, custom hierarchies, EAFP), everyday idioms (context managers, decorators, sentinels, generators, anti-patterns, naming/underscore conventions), the stdlib toolbox (functools caching/dispatch/partial, itertools grouping/combinatorics), logging (per-module loggers, lazy %-formatting, levels), and concurrency (threads vs multiprocessing vs asyncio, the GIL, concurrent.futures). Use this whenever writing, reviewing, or refactoring Python — including questions about type hints, dataclasses/Pydantic, docstrings, exceptions, decorators, generators, naming conventions, caching/memoization (lru_cache, cached_property), functools/itertools, logging (the logging module, %-style vs f-strings), or concurrency/parallelism (threading, multiprocessing, asyncio, GIL), even when the user doesn't explicitly name a "pattern."
disable-model-invocation: false
---

# Python Patterns

Modern Python (3.12+) idioms and standards. This complements `clean-code`
(naming, function design, testing) and `refactoring-patterns` (Extract Method,
guard clauses, CQS) — it does not repeat them. The focus here is Python-specific.

The detail lives in `references/` — read the one that fits the task:

| Topic | Reference | Read it for |
|---|---|---|
| Type system | [type-system.md](references/type-system.md) | annotations, PEP 695 generics/aliases, `Protocol`, `collections.abc`, `@override`, `TYPE_CHECKING` |
| Data modeling | [data-modeling.md](references/data-modeling.md) | dataclass vs Pydantic, slots/frozen, `NamedTuple`, `ABC`, `Enum`, the mutable-default trap |
| Docstrings | [docstrings.md](references/docstrings.md) | Google-style conventions; when each section earns its place |
| Error handling | [error-handling.md](references/error-handling.md) | specific excepts, chaining, custom hierarchy, EAFP, exception groups |
| Idioms | [idioms.md](references/idioms.md) | naming conventions, context managers, decorators, sentinels, generators, anti-patterns |
| Stdlib toolbox | [functools-itertools.md](references/functools-itertools.md) | caching (`@cache`/`@cached_property`), type dispatch (`@singledispatch`), `partial`; lazy iteration, grouping, combinatorics |
| Logging | [logging.md](references/logging.md) | one logger per module, lazy `%`-formatting, levels, exceptions, library vs app config |
| Concurrency | [concurrency.md](references/concurrency.md) | threads vs multiprocessing vs asyncio, the GIL, `concurrent.futures`, pitfalls |

## Always-on conventions

Every module starts with:

```python
from __future__ import annotations
```

It stores annotations as strings, enabling forward references without quotes and
cheaper imports. Still needed on 3.12 (native lazy annotations land in 3.14).

- Target **Python 3.12+**. Prefer modern syntax over the `typing` module:
  built-in generics (`list[str]`), `X | None`, PEP 695 `type` / `[T]`,
  `collections.abc`. Treat `typing.List` / `Optional` / `Union` / `TypeVar` as
  obsolete; reach into `typing` only for `Protocol`, `Any`, `TYPE_CHECKING`,
  `override`, `Self`, `Literal`, `Final`, `cast`.
- Format and lint with **`ruff format`** + **`ruff check --fix`** (sorts imports too).
- Type-check with **`mypy`**; annotate every public function.
- Google-style docstrings with a `Parameters:` block.

## Quick decision table

| Need | Reach for | Reference |
|---|---|---|
| Structural typing without inheritance | `Protocol` | type-system |
| Break a circular import (type-only) | `if TYPE_CHECKING:` | type-system |
| Reusable generic / type alias | PEP 695 `type X = ...` / `def f[T]` | type-system |
| Immutable value object | `@dataclass(frozen=True, slots=True)` | data-modeling |
| Validate external input | Pydantic at the edge → dataclass inside | data-modeling |
| Fixed set of named values | `Enum` / `StrEnum` | data-modeling |
| Re-raise but keep the cause | `raise X(...) from error` | error-handling |
| "Not provided" when `None` is valid | enum sentinel + `is` | idioms |
| Stream large data lazily | generator (`yield`) / generator expr | idioms |
| Guaranteed setup/teardown | context manager | idioms |
| Memoize a pure function | `@cache` / `@cached_property` | functools-itertools |
| Dispatch by argument type | `@singledispatch` (not `isinstance` chains) | functools-itertools |
| Group / window / combine iterables | `itertools` (groupby / pairwise / product) | functools-itertools |
| Python naming / underscore meaning | `snake_case`, `_private`, `__mangled` | idioms |
| Record an event (not `print`) | per-module logger + lazy `%`-args | logging |
| Speed up I/O-bound work | `ThreadPoolExecutor` / `asyncio` | concurrency |
| Speed up CPU-bound work | `ProcessPoolExecutor` | concurrency |
