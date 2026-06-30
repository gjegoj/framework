# functools & itertools

Two standard-library toolboxes that replace a lot of hand-written code:
`functools` for caching, type dispatch, and adapting callables; `itertools` for
lazy iteration, grouping, and combinatorics. Reaching for these is usually
clearer and faster than rolling your own loops — but each has a sharp edge worth
knowing.

## Contents
- [functools](#functools)
  - [@cache / @lru_cache](#cache--lru_cache)
  - [@cached_property](#cached_property)
  - [@singledispatch](#singledispatch)
  - [partial](#partial)
  - [reduce](#reduce)
  - [total_ordering](#total_ordering)
- [itertools](#itertools)
  - [Combine and slice](#combine-and-slice)
  - [groupby](#groupby)
  - [Sliding windows and infinite iterators](#sliding-windows-and-infinite-iterators)
  - [Combinatorics](#combinatorics)
- [When to reach for these](#when-to-reach-for-these)
- [Pitfalls](#pitfalls)

## functools

### @cache / @lru_cache

Memoize a pure function — cache results by arguments so repeated calls are free.
`@cache` (3.9+) is unbounded; `@lru_cache(maxsize=N)` evicts least-recently-used
once full:

```python
from __future__ import annotations

from functools import cache, lru_cache


@cache
def fib(n: int) -> int:                 # exponential recursion → linear with memoization
    return n if n < 2 else fib(n - 1) + fib(n - 2)


@lru_cache(maxsize=1024)
def geocode(address: str) -> tuple[float, float]:
    ...                                 # bounded cache for an expensive lookup
```

Arguments must be **hashable** (they form the cache key). Use this only for
*pure* functions — caching a function with side effects or time-dependent
results hides bugs.

### @cached_property

Compute an attribute once, on first access, then reuse the stored value — the
clean replacement for a `property` plus a manual `self._x` cache:

```python
from __future__ import annotations

from functools import cached_property


class Dataset:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    @cached_property
    def column_names(self) -> list[str]:
        return sorted({key for row in self._rows for key in row})   # computed once
```

Two caveats: it stores the value in the instance `__dict__`, so it is
**incompatible with `__slots__`** (a slotted class has no `__dict__`); and it is
**not thread-safe** — two threads racing the first access can both compute.

### @singledispatch

Function overloading by the type of the first argument — a clean alternative to a
chain of `isinstance` checks. Register an implementation per type; new types plug
in without editing the original function:

```python
from __future__ import annotations

from functools import singledispatch


@singledispatch
def serialize(value: object) -> str:
    raise TypeError(f"cannot serialize {type(value).__name__}")


@serialize.register
def _(value: int) -> str:
    return str(value)


@serialize.register
def _(value: list) -> str:
    return "[" + ", ".join(serialize(item) for item in value) + "]"
```

Use `singledispatchmethod` for methods. Dispatch is on the first argument only —
for multi-argument dispatch, a different design (Strategy, a dict of handlers) is
usually clearer.

### partial

Pre-bind arguments to produce a new callable. Clearer and more introspectable
than a `lambda` when you're only fixing some arguments:

```python
from __future__ import annotations

from functools import partial

to_int = partial(int, base=2)            # to_int("1010") == 10
rounded = sorted(values, key=partial(round, ndigits=2))
```

### reduce

`functools.reduce` folds a binary function over an iterable. It is rarely the
clearest option — a built-in or an explicit loop usually wins:

```python
from functools import reduce
import operator

product = reduce(operator.mul, values, 1)   # works, but...
# Prefer the obvious built-ins where they exist:
total = sum(values)
import math
product = math.prod(values)
everything = all(checks)
```

Reach for `reduce` only when no built-in fits and the accumulation is genuinely a
fold.

### total_ordering

Define `__eq__` and **one** ordering method (`__lt__`), and `@total_ordering`
fills in the rest (`__le__`, `__gt__`, `__ge__`):

```python
from __future__ import annotations

from functools import total_ordering


@total_ordering
class Version:
    def __init__(self, parts: tuple[int, ...]) -> None:
        self.parts = parts

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Version) and self.parts == other.parts

    def __lt__(self, other: Version) -> bool:
        return self.parts < other.parts
```

It trades a little speed for a lot less boilerplate; hand-write all four only in
a hot path. (`functools.wraps`, for writing decorators, is covered in
`idioms.md`.)

## itertools

`itertools` functions return **iterators** — lazy, single-pass, memory-cheap.
They compose well with generators (see `idioms.md`).

### Combine and slice

```python
import itertools

merged = itertools.chain(first, second)                 # lazy concatenation
flat = itertools.chain.from_iterable(list_of_lists)     # flatten one level
head = itertools.islice(stream, 10)                     # first 10 of an iterator
window = itertools.islice(stream, 5, 15)                # [5:15] — no negative indices
```

`islice` is how you "slice" a generator, which doesn't support `[a:b]`.

### groupby

Group **consecutive** items by a key. The decisive gotcha: it only groups runs
that are already adjacent, so you almost always **sort by the same key first**:

```python
import itertools

rows.sort(key=lambda row: row["category"])              # REQUIRED before groupby
for category, group in itertools.groupby(rows, key=lambda row: row["category"]):
    print(category, list(group))                        # consume `group` before the next step
```

Each `group` is a lazy sub-iterator that's invalidated when you advance to the
next group — materialise it (`list(group)`) if you need it later.

### Sliding windows and infinite iterators

```python
import itertools

for previous, current in itertools.pairwise(sequence):  # (s0,s1),(s1,s2),... (3.10+)
    delta = current - previous

ids = zip(itertools.count(1), names)                    # number items: (1, a), (2, b), ...
running = itertools.accumulate(values)                  # running totals
first_block = itertools.takewhile(lambda x: x > 0, stream)   # until the predicate fails
```

`count`, `cycle`, and `repeat` are infinite — always bound them with `islice`,
`zip`, or a `takewhile`, never iterate them to exhaustion.

### Combinatorics

Replace nested loops with declarative combinatorics:

```python
import itertools

itertools.product([0, 1], repeat=3)        # all 3-bit tuples
itertools.combinations(items, 2)           # unordered pairs, no repeats
itertools.permutations(items, 2)           # ordered pairs
```

## When to reach for these

| Need | Reach for |
|---|---|
| Cache results of a pure function | `@cache` / `@lru_cache` |
| Compute an attribute once per instance | `@cached_property` |
| Behaviour varies by argument type | `@singledispatch` (over `isinstance` chains) |
| Fix some arguments of a callable | `partial` (over `lambda`) |
| Concatenate / flatten / slice iterators | `chain` / `islice` |
| Group adjacent items by a key | `groupby` (sort first!) |
| Sliding pairs, running totals | `pairwise`, `accumulate` |
| All pairs / orderings / cartesian product | `combinations` / `permutations` / `product` |

## Pitfalls

- **`@cache` is unbounded** — it grows forever. Bound it with `lru_cache(maxsize=...)`
  when the argument space is large, and remember arguments must be hashable.
- **Caching instance methods leaks memory** — the cache holds `self` as part of
  the key, keeping every instance alive. For per-instance memoization use
  `@cached_property`; for a method, scope the cache carefully.
- **`@cached_property` clashes with `__slots__`** (needs `__dict__`) and isn't
  thread-safe.
- **`groupby` needs sorted input** — unsorted data yields fragmented groups. This
  is the single most common itertools bug.
- **itertools results are one-shot** — they're iterators: you can't index them,
  call `len()`, or iterate twice. Wrap in `list(...)` if you need any of that.
- **`reduce` hurts readability** — prefer `sum` / `math.prod` / `any` / `all` /
  `"".join`, or an explicit loop.
