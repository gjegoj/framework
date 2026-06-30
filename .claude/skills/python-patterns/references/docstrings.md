# Docstrings

Google-style docstrings. The discipline is to document what the **types and
names cannot** already say — intent, side effects, invariants, and the meaning
of edge cases — and to stay silent where the signature is self-explanatory.

## Format

- One-line imperative summary, then a blank line, then optional narrative.
- `Parameters:` (not `Args:`) — `name (type): description`.
- `Returns:` with the type before the colon — `User | None: ...`.
- `Raises:` and `Examples:` only when they add information not obvious from the
  types or names.

```python
def fetch_user(user_id: str, *, include_inactive: bool = False) -> User | None:
    """Load a user record by identifier.

    Queries the primary store directly; does not cache. Callers that need
    caching should wrap this function.

    Parameters:
        user_id (str): Unique identifier (non-empty).
        include_inactive (bool): When True, include deactivated users.

    Returns:
        User | None: The matching user, or None if no user has this id.

    Raises:
        ValueError: If ``user_id`` is empty or whitespace-only.
    """
    if not user_id.strip():
        raise ValueError("user_id must be non-empty")
    ...
```

## When to include each section

| Section | Include when |
|---------|--------------|
| Summary | Always (the first line) |
| Body | There are side effects, caveats, or non-obvious behaviour |
| Parameters | Arguments are non-trivial or their meaning isn't obvious from the name/type |
| Returns | The return value is non-obvious or has multiple outcomes (e.g. `\| None`) |
| Raises | The function raises exceptions callers are expected to handle |
| Examples | Usage is non-obvious or an edge case is worth pinning down |

## What not to do

- Don't restate the signature: `Parameters: x (int): the integer x` is noise.
- Don't document obvious returns: a `-> bool` named `is_valid` needs no `Returns:`.
- Don't let docstrings rot — update them in the same edit as the code. A wrong
  docstring is worse than none.
- Class docstrings describe the abstraction and its invariants; per-method
  docstrings describe behaviour. Don't repeat the class summary on every method.

The test: if removing a sentence loses no information a reader couldn't get from
the names and types, remove it.
