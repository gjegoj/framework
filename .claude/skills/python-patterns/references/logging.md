# Logging

How to instrument code with the stdlib `logging` module: one logger per module,
lazy `%`-style messages, the right level, and tracebacks on errors. Logging is
the production replacement for `print` — it has levels, routing, and structure
that `print` cannot give you.

## Contents
- [One logger per module](#one-logger-per-module)
- [Lazy %-formatting (not f-strings)](#lazy-formatting-not-f-strings)
- [Levels — when to use which](#levels--when-to-use-which)
- [Logging exceptions](#logging-exceptions)
- [Libraries vs applications](#libraries-vs-applications)
- [Expensive debug payloads](#expensive-debug-payloads)
- [Don't log secrets](#dont-log-secrets)

## One logger per module

Create a module-level logger named after the module; never log on the root logger
and never use `print` for diagnostics. The `__name__` name gives a hierarchy
(`myapp.api.users`) that handlers and filters can target independently.

```python
import logging

log = logging.getLogger(__name__)


def charge(account_id: str, amount: int) -> None:
    log.info("charging account %s for %d", account_id, amount)
    ...
```

## Lazy %-formatting (not f-strings)

Pass the message as a `%`-style template plus arguments — do **not** pre-format
with an f-string or `+`:

```python
log.info("user %s did %s in %.2fs", user_id, action, elapsed)   # correct
log.info(f"user {user_id} did {action} in {elapsed:.2f}s")      # avoid
```

**Why it matters:** `logging` interpolates `msg % args` **lazily** — only if the
record is actually emitted (its level is enabled and a handler accepts it). With
an f-string, Python builds the final string **before** the call, every time — so
a `DEBUG` line that's filtered out at `INFO` still pays the formatting cost, and
any expensive `__str__`/`__repr__` still runs. In hot paths or debug logs inside
loops that is real, wasted work.

The lazy template is also what lets log aggregators group records by their
template (`"user %s did %s"`) instead of treating every interpolated string as a
unique event. Linters flag f-strings in logging calls for exactly this reason
(e.g. ruff's `G` rules, pylint's `logging-fstring-interpolation`).

## Levels — when to use which

| Level | Use for |
|---|---|
| `DEBUG` | Detailed diagnostics useful only when chasing a problem |
| `INFO` | Normal, expected milestones ("server started", "processed 1000 rows") |
| `WARNING` | Something unexpected but recoverable; the program continues |
| `ERROR` | An operation failed; a feature didn't work |
| `CRITICAL` | The program itself may not be able to continue |

Production usually runs at `INFO`; drop to `DEBUG` to investigate. The level is a
runtime knob — code logs at a fixed level, configuration decides what's shown.

## Logging exceptions

Inside an `except` block, `log.exception(...)` logs at `ERROR` **and** attaches
the active traceback — far better than `str(error)` alone, which throws the stack
away:

```python
try:
    process(item)
except ProcessingError:
    log.exception("failed to process item %s", item.id)   # message + full traceback
    raise
```

Outside an `except` block, or to attach a traceback at a non-error level, pass
`exc_info=True`: `log.warning("retrying after %s", reason, exc_info=True)`.

## Libraries vs applications

- **Applications** configure logging **once**, at the entrypoint, then everything
  else just calls `getLogger(__name__)`:

  ```python
  # app entrypoint
  import logging

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s %(name)s %(levelname)s %(message)s",
  )
  ```

- **Libraries must not configure logging** — no `basicConfig`, no handlers on the
  root logger, no setting levels. Configuration is the *application's* choice; a
  library only creates loggers and emits records. Attach a `NullHandler` to your
  package's top-level logger so a library used without configured logging stays
  silent instead of warning:

  ```python
  # mypackage/__init__.py
  import logging

  logging.getLogger(__name__).addHandler(logging.NullHandler())
  ```

## Expensive debug payloads

If *building* a debug message is itself costly (serialising a large object),
guard it so you don't pay even when `DEBUG` is disabled. `isEnabledFor` checks
the level without creating a record:

```python
if log.isEnabledFor(logging.DEBUG):
    log.debug("state dump: %s", expensive_serialize(state))
```

## Don't log secrets

Logs are persisted and shipped to aggregators where many people can read them.
Never log passwords, tokens, API keys, full PII, or raw request bodies that might
contain them. Mask at the call site — log the identifier, not the credential
(`log.info("authenticated as %s", user_id)`, never the token).
