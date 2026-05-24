---
name: python-patterns
description: Use when writing, reviewing, or refactoring Python code in this project. Covers Pythonic idioms, PEP 8 (line length 120), type hints, Python 3.12+, dataclasses for application logic, Pydantic for I/O validation, and Google-style docstrings (parameter_name (type): ...).
disable-model-invocation: false
---

# Python Development Patterns

Idiomatic Python patterns and best practices for building robust, efficient, and maintainable applications in this project.

## When to Activate

- Writing new Python code
- Reviewing Python code
- Refactoring existing Python code
- Designing Python packages/modules

## Core Principles

### 1. Readability Counts

Python prioritizes readability. Code should be obvious and easy to understand.

```python
# Good: Clear and readable
def get_active_users(users: list[User]) -> list[User]:
    """Return only active users from the provided list."""
    return [user for user in users if user.is_active]


# Bad: Clever but confusing
def get_active_users(u):
    return [x for x in u if x.a]
```

### 2. Explicit is Better Than Implicit

Avoid magic; be clear about what your code does.

```python
# Good: Explicit configuration
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Bad: Hidden side effects
import some_module
some_module.setup()  # What does this do?
```

### 3. EAFP - Easier to Ask Forgiveness Than Permission

Python prefers exception handling over checking conditions.

```python
# Good: EAFP style
def get_value(dictionary: dict, key: str, default: Any = None) -> Any:
    try:
        return dictionary[key]
    except KeyError:
        return default

# Bad: LBYL (Look Before You Leap) style
def get_value(dictionary: dict, key: str, default: Any = None) -> Any:
    if key in dictionary:
        return dictionary[key]
    else:
        return default
```

## Docstrings (Google-style)

Use **Google-style** docstrings for modules, classes, and public functions. They stay readable in source and work well with Sphinx, MkDocs, and IDE tooltips.

**Conventions for this skill:**

- One-line summary, then a blank line, then optional extra narrative.
- Use **`Parameters`** (not `Args`) for the argument list.
- For **every** parameter, write **`parameter_name (type): description`**, where `(type)` matches the type hint in the signature (e.g. `user_id (str): ...`, `limit (int | None): ...`).
- For **`Returns`**, include the return type before the colon when it helps readers: e.g. `User | None: ...` or a short phrase if the type is obvious from annotations.
- Add **`Raises`** and **`Examples`** only when they add information not obvious from types or names.

```python
def fetch_user(user_id: str, *, include_inactive: bool = False) -> User | None:
    """Load a user record by identifier.

    Queries the primary store; does not cache. Callers that need caching should
    wrap this function.

    Parameters:
        user_id (str): Unique user identifier (non-empty).
        include_inactive (bool): When True, include users marked inactive.

    Returns:
        User | None: The matching user, or None if no row exists.

    Raises:
        ValueError: If ``user_id`` is empty or whitespace-only.
    """
    if not user_id.strip():
        raise ValueError("user_id must be non-empty")
    ...
```

```python
class UserRepository:
    """Persistence port for ``User`` aggregates."""

    def save(self, user: User) -> None:
        """Insert or update a user.

        Parameters:
            user (User): Domain user to persist; ``user.id`` may be assigned by the store.

        Raises:
            DuplicateEmailError: If another user already owns this email.
        """
        ...
```

**Quick checklist**

| Section | When to include |
|---------|-----------------|
| Summary | Always (first line) |
| Body | When behavior, side effects, or caveats need explanation |
| Parameters | Non-trivial or non-obvious arguments |
| Returns | Non-obvious return value or multiple outcomes |
| Raises | Specific exceptions callers should handle |
| Examples | Non-obvious usage or edge cases |

## Type Hints

### Basic Type Annotations

```python
from typing import Any

def process_user(
    user_id: str,
    data: dict[str, Any],
    active: bool = True,
) -> User | None:
    """Process a user and return the updated User or None."""
    if not active:
        return None
    return User(user_id, data)
```

### Built-in generic types (standard library)

Projects target **Python 3.12+**; use built-in generics (`list`, `dict`, `tuple`, `set`) in annotations instead of capitalized names from `typing`.

```python
def process_items(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}
```

### Type Aliases and Generics

Target is **Python 3.12+**: use the `type` statement for aliases and `X | Y` unions
instead of `typing.Union`. Use PEP 695 generic syntax for new generic functions/classes.

```python
import json
from typing import Any

# Type alias (PEP 695 `type` statement, Python 3.12+)
type JSON = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

def parse_json(data: str) -> JSON:
    return json.loads(data)

# Generic function (PEP 695 syntax, Python 3.12+)
def first[T](items: list[T]) -> T | None:
    """Return the first item or None if list is empty."""
    return items[0] if items else None
```

### Protocol-Based Duck Typing

```python
from typing import Protocol

class Renderable(Protocol):
    def render(self) -> str:
        """Render the object to a string."""

def render_all(items: list[Renderable]) -> str:
    """Render all items that implement the Renderable protocol."""
    return "\n".join(item.render() for item in items)
```

## Error Handling Patterns

### Specific Exception Handling

```python
# Good: Catch specific exceptions
def load_config(path: str) -> Config:
    try:
        with open(path) as f:
            return Config.from_json(f.read())
    except FileNotFoundError as e:
        raise ConfigError(f"Config file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in config: {path}") from e

# Bad: Bare except
def load_config(path: str) -> Config:
    try:
        with open(path) as f:
            return Config.from_json(f.read())
    except:
        return None  # Silent failure!
```

### Exception Chaining

```python
def process_data(data: str) -> Result:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as e:
        # Chain exceptions to preserve the traceback
        raise ValueError(f"Failed to parse data: {data}") from e
```

### Custom Exception Hierarchy

```python
class AppError(Exception):
    """Base exception for all application errors."""

class ValidationError(AppError):
    """Raised when input validation fails."""

class NotFoundError(AppError):
    """Raised when a requested resource is not found."""

# Usage
def get_user(user_id: str) -> User:
    user = db.find_user(user_id)
    if not user:
        raise NotFoundError(f"User not found: {user_id}")
    return user
```

## Context Managers

### Resource Management

```python
# Good: Using context managers
def process_file(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

# Bad: Manual resource management
def process_file(path: str) -> str:
    f = open(path, 'r')
    try:
        return f.read()
    finally:
        f.close()
```

### Custom Context Managers

```python
import time
from contextlib import contextmanager
from collections.abc import Iterator

@contextmanager
def timer(name: str) -> Iterator[None]:
    """Context manager to time a block of code."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name} took {elapsed:.4f} seconds")

# Usage
with timer("data processing"):
    process_large_dataset()
```

### Context Manager Classes

```python
class DatabaseTransaction:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        self.connection.begin_transaction()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        return False  # Don't suppress exceptions

# Usage
with DatabaseTransaction(conn):
    user = conn.create_user(user_data)
    conn.create_profile(user.id, profile_data)
```

## Comprehensions and Generators

### List Comprehensions

```python
from collections.abc import Iterable

# Good: List comprehension for simple transformations
names = [user.name for user in users if user.is_active]

# Bad: Manual loop
names = []
for user in users:
    if user.is_active:
        names.append(user.name)

# Complex comprehensions should be expanded
# Bad: Too complex
result = [x * 2 for x in items if x > 0 if x % 2 == 0]

# Good: Use a generator function
def filter_and_transform(items: Iterable[int]) -> list[int]:
    result = []
    for x in items:
        if x > 0 and x % 2 == 0:
            result.append(x * 2)
    return result
```

### Generator Expressions

```python
# Good: Generator for lazy evaluation
total = sum(x * x for x in range(1_000_000))

# Bad: Creates large intermediate list
total = sum([x * x for x in range(1_000_000)])
```

### Generator Functions

```python
from collections.abc import Iterator

def read_large_file(path: str) -> Iterator[str]:
    """Read a large file line by line."""
    with open(path) as f:
        for line in f:
            yield line.strip()

# Usage
for line in read_large_file("huge.txt"):
    process(line)
```

## Data Classes and Named Tuples

### `dataclasses` vs Pydantic (roles)

| Layer | Prefer | Why |
|-------|--------|-----|
| **Application / domain logic** | **`@dataclass`** | Stdlib, small overhead, easy to construct in tests, no hidden coercion. Use for entities, value objects, command/query records, internal results. |
| **Incoming / outgoing data** | **Pydantic v2** (`BaseModel`) | Validation, coercion (e.g. str -> int), JSON schema, clear errors at HTTP/API/config boundaries. |

**Flow:** validate and parse at the edge with Pydantic -> map to dataclasses (or primitives) inside services -> on the way out, build Pydantic response models from domain objects if you need a stable external shape.

```python
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, EmailStr, Field


# --- Edge: request/response (Pydantic) ---
class CreateUserBody(BaseModel):
    """JSON body for POST /users."""

    name: str = Field(min_length=1, max_length=200)
    email: EmailStr


class UserResponse(BaseModel):
    """JSON returned to clients."""

    id: str
    name: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}  # allow building from dataclass if field names match


# --- Core: domain / application (dataclass) ---
@dataclass(frozen=True)
class User:
    """Internal user aggregate - not tied to wire format."""

    id: str
    name: str
    email: str
    created_at: datetime


def register_user(body: CreateUserBody) -> User:
    """Create a user from a validated API payload.

    Parameters:
        body (CreateUserBody): Already validated/coerced by Pydantic (e.g. FastAPI).

    Returns:
        User: Domain object for persistence and use cases.
    """
    # body.email is already a validated email string
    return User(
        id=str(uuid4()),
        name=body.name.strip(),
        email=str(body.email),
        created_at=datetime.now(tz=UTC),
    )


def to_response(user: User) -> UserResponse:
    """Map domain user to an outgoing DTO.

    Parameters:
        user (User): Persisted domain user.

    Returns:
        UserResponse: Serializable response model.
    """
    return UserResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        created_at=user.created_at,
    )
```

**Pydantic-only edge example (config or webhook):**

```python
from pydantic import BaseModel, HttpUrl, field_validator


class WebhookPayload(BaseModel):
    """Incoming JSON from an external system."""

    event: str
    resource_url: HttpUrl

    @field_validator("event")
    @classmethod
    def event_must_be_known(cls, value: str) -> str:
        allowed = {"created", "updated", "deleted"}
        if value not in allowed:
            raise ValueError(f"event must be one of {allowed}")
        return value
```

**Rules of thumb**

- Do **not** pass `BaseModel` instances deep into domain rules; convert early to dataclasses or plain types.
- Use **`frozen=True`** on dataclasses when the object should be immutable (value objects).
- Keep **`__post_init__`** on dataclasses for **light** invariants only; rich rules or cross-field validation belong in Pydantic models at the boundary or in explicit domain functions.

### Data Classes

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class User:
    """User entity with automatic __init__, __repr__, and __eq__."""
    id: str
    name: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

# Usage
user = User(
    id="123",
    name="Alice",
    email="alice@example.com"
)
```

### Data Classes with light validation

Use **`__post_init__`** only for simple checks on dataclasses. Prefer **Pydantic** at I/O boundaries for coercion, field constraints, and nested models.

```python
@dataclass
class User:
    email: str
    age: int

    def __post_init__(self) -> None:
        if "@" not in self.email:
            raise ValueError(f"Invalid email: {self.email}")
        if not 0 <= self.age <= 150:
            raise ValueError(f"Invalid age: {self.age}")
```

### Named Tuples

```python
from typing import NamedTuple

class Point(NamedTuple):
    """Immutable 2D point."""
    x: float
    y: float

    def distance(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

# Usage
p1 = Point(0, 0)
p2 = Point(3, 4)
print(p1.distance(p2))  # 5.0
```

## Decorators

### Function Decorators

```python
import functools
import time
from collections.abc import Callable

def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)

# slow_function() prints: slow_function took 1.0012s
```

### Parameterized Decorators

```python
import functools
from collections.abc import Callable

def repeat(times: int):
    """Decorator to repeat a function multiple times."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(times):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat(times=3)
def greet(name: str) -> str:
    return f"Hello, {name}!"

# greet("Alice") returns ["Hello, Alice!", "Hello, Alice!", "Hello, Alice!"]
```

### Class-Based Decorators

```python
import functools
from collections.abc import Callable

class CountCalls:
    """Decorator that counts how many times a function is called."""
    def __init__(self, func: Callable):
        functools.update_wrapper(self, func)
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} has been called {self.count} times")
        return self.func(*args, **kwargs)

@CountCalls
def process():
    pass

# Each call to process() prints the call count
```

## Concurrency Patterns

### Threading for I/O-Bound Tasks

```python
import concurrent.futures

def fetch_url(url: str) -> str:
    """Fetch a URL (I/O-bound operation)."""
    import urllib.request
    with urllib.request.urlopen(url) as response:
        return response.read().decode()

def fetch_all_urls(urls: list[str]) -> dict[str, str]:
    """Fetch multiple URLs concurrently using threads."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(fetch_url, url): url for url in urls}
        results = {}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as e:
                results[url] = f"Error: {e}"
    return results
```

### Multiprocessing for CPU-Bound Tasks

```python
import concurrent.futures

def process_data(data: list[int]) -> int:
    """CPU-intensive computation."""
    return sum(x ** 2 for x in data)

def process_all(datasets: list[list[int]]) -> list[int]:
    """Process multiple datasets using multiple processes."""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_data, datasets))
    return results
```

### Async/Await for Concurrent I/O

```python
import asyncio

async def fetch_async(url: str) -> str:
    """Fetch a URL asynchronously."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def fetch_all(urls: list[str]) -> dict[str, str]:
    """Fetch multiple URLs concurrently."""
    tasks = [fetch_async(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return dict(zip(urls, results))
```

## Imports and Package Exports

### Import Conventions

```python
# Good: Import order - stdlib, third-party, local
import os
import sys
from pathlib import Path

import requests
from fastapi import FastAPI

from mypackage.models import User
from mypackage.utils import format_name

# Use ruff (or isort) for automatic import sorting.
```

### __init__.py for Package Exports

```python
# mypackage/__init__.py
"""mypackage - A sample Python package."""

__version__ = "1.0.0"

# Export main classes/functions at package level
from mypackage.models import User, Post
from mypackage.utils import format_name

__all__ = ["User", "Post", "format_name"]
```

## Memory and Performance

### Using __slots__ for Memory Efficiency

```python
# Bad: Regular class uses __dict__ (more memory)
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

# Good: __slots__ reduces memory usage
class Point:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
```

### Generator for Large Data

```python
from collections.abc import Iterator

# Bad: Returns full list in memory
def read_lines(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f]

# Good: Yields lines one at a time
def read_lines(path: str) -> Iterator[str]:
    with open(path) as f:
        for line in f:
            yield line.strip()
```

### Avoid String Concatenation in Loops

```python
# Bad: O(n^2) due to string immutability
result = ""
for item in items:
    result += str(item)

# Good: O(n) using join
result = "".join(str(item) for item in items)

# Good: Using StringIO for building
from io import StringIO

buffer = StringIO()
for item in items:
    buffer.write(str(item))
result = buffer.getvalue()
```

## Formatting & Tooling

- **Line length: 120.**
- Format and lint with **ruff** (`ruff format`, `ruff check --fix`); ruff also handles import sorting.
- Type-check with **mypy** (annotate all public functions; target Python 3.12+).

Tool versions and dependency declarations live in `pyproject.toml`, not in this skill.

## Quick Reference: Python Idioms

| Idiom | Description |
|-------|-------------|
| EAFP | Easier to Ask Forgiveness than Permission |
| Context managers | Use `with` for resource management |
| List comprehensions | For simple transformations |
| Generators | For lazy evaluation and large datasets |
| Type hints | Annotate function signatures |
| Dataclasses | Domain and application objects (default for internal logic) |
| Pydantic `BaseModel` | Validate/coerce incoming and outgoing data at boundaries |
| `__slots__` | For memory optimization |
| f-strings | For string formatting |
| `pathlib.Path` | For path operations |
| `enumerate` | For index-element pairs in loops |

## Anti-Patterns to Avoid

```python
# Bad: Mutable default arguments
def append_to(item, items=[]):
    items.append(item)
    return items

# Good: Use None and create new list
def append_to(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

# Bad: Checking type with type()
if type(obj) == list:
    process(obj)

# Good: Use isinstance
if isinstance(obj, list):
    process(obj)

# Bad: Comparing to None with ==
if value == None:
    process()

# Good: Use is
if value is None:
    process()

# Bad: from module import *
from os.path import *

# Good: Explicit imports
from os.path import join, exists

# Bad: Bare except
try:
    risky_operation()
except:
    pass

# Good: Specific exception
try:
    risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
```

**Remember**: Python code should be readable, explicit, and follow the principle of least surprise. When in doubt, prioritize clarity over cleverness.
