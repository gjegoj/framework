# Error Handling

Errors are a separate concern from the happy path. Catch narrowly, preserve the
cause, model failures as a shallow exception hierarchy, and let exceptions
propagate unless you can genuinely handle them.

## Contents
- [Catch specific exceptions](#catch-specific-exceptions)
- [Chain with `from`](#chain-with-from)
- [Custom exception hierarchy](#custom-exception-hierarchy)
- [EAFP over LBYL](#eafp-over-lbyl)
- [`contextlib.suppress`](#contextlibsuppress)
- [Exception groups (`except*`)](#exception-groups-except)
- [Don't silently return None](#dont-silently-return-none)

## Catch specific exceptions

Catch the narrowest exception you can actually handle. A bare `except:` swallows
`KeyboardInterrupt` and `SystemExit` and hides bugs; `except Exception:` is the
widest acceptable catch, and even that should be deliberate.

```python
from __future__ import annotations

import json


def load_config(path: str) -> Config:
    try:
        with open(path) as file:
            return Config.from_json(file.read())
    except FileNotFoundError as error:
        raise ConfigError(f"Config not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON in {path}") from error
```

Keep the `try` block small — wrap only the lines that can raise the exception you
are catching, so you don't accidentally swallow an unrelated failure.

## Chain with `from`

When you re-raise as a different exception type, chain the original with
`from error` so the traceback shows the root cause. Use `from None` to
deliberately hide an irrelevant internal cause:

```python
try:
    value = registry[key]
except KeyError as error:
    raise LookupError(f"unknown key {key!r}") from error   # preserves the cause

try:
    return int(raw)
except ValueError:
    raise ConfigError(f"expected an integer, got {raw!r}") from None  # hide noise
```

## Custom exception hierarchy

Define exceptions by what the **caller** needs to distinguish, not by where they
were raised. Keep the tree shallow (two levels is usually enough): one base so
callers can catch the whole family, plus a few specific subclasses.

```python
class AppError(Exception):
    """Base for every error this application raises on purpose."""


class ValidationError(AppError):
    """Input failed validation."""


class NotFoundError(AppError):
    """A requested resource does not exist."""
```

Callers catch `AppError` to handle anything expected, or a specific subclass when
they can react to it. Carry context in the message or attributes
(`raise NotFoundError(f"user {user_id!r} not found")`) so logs are actionable.

## EAFP over LBYL

Python favours **EAFP** ("easier to ask forgiveness than permission") — attempt
the operation and handle the exception — over **LBYL** ("look before you leap"),
which checks preconditions first. EAFP is usually clearer and avoids a
time-of-check/time-of-use race (the condition can change between the check and
the use):

```python
# EAFP — attempt, then handle the specific failure.
try:
    return cache[key]
except KeyError:
    return compute(key)

# LBYL — a redundant lookup, and the key could vanish between check and use.
if key in cache:
    return cache[key]
return compute(key)
```

LBYL is fine when the check is cheap and there's no failure to "handle" (e.g. an
early `if not items: return []` guard clause).

## `contextlib.suppress`

When ignoring a specific exception is genuinely correct, `suppress` says so more
clearly than an empty `except` block:

```python
from contextlib import suppress

with suppress(FileNotFoundError):
    path.unlink()          # delete if present; do nothing if already gone
```

## Exception groups (`except*`)

When concurrent work raises several errors at once (e.g. an `asyncio.TaskGroup`),
they surface as an `ExceptionGroup`. Handle sub-types with `except*`, which can
match multiple groups in one statement:

```python
try:
    async with asyncio.TaskGroup() as group:
        for url in urls:
            group.create_task(fetch(url))
except* TimeoutError as group:
    log.warning("%d fetches timed out", len(group.exceptions))
except* ConnectionError as group:
    log.error("%d fetches failed to connect", len(group.exceptions))
```

## Don't silently return None

Returning `None` on failure forces every caller to remember a null check, and a
missing one crashes far from the cause. Prefer to **raise** for genuine errors,
or return an explicit `X | None` that is documented and obviously optional (e.g.
a `find_*` that may legitimately find nothing). For "absent collection", return
an empty `list`/`dict`, never `None`.
