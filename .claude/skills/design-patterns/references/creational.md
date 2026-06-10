# Creational Patterns

Creational patterns deal with object creation — abstracting away the "how" so the rest of the code depends only on interfaces, not concrete classes.

---

## Factory Method

**Intent:** Define an interface for creating an object but let subclasses decide which class to instantiate. The creator calls its own factory method instead of `ClassName()` directly.

**Use when:**
- You don't know ahead of time the exact class you need to create.
- You want subclasses to extend which product gets created (Open/Closed).
- You want to centralise object construction logic (avoid scattered `if type == "x": return X()`).

**Don't use when:** There's only one concrete product and no need for extension — just call the constructor directly.

**Python notes:** The factory method is often a `@classmethod` or a standalone function. A `dict`-based registry (`{"csv": CsvParser, "json": JsonParser}`) is a common Pythonic alternative when the set of types is data-driven.

```python
from __future__ import annotations
from abc import ABC, abstractmethod


class Notifier(ABC):
    """Creator — declares the factory method."""

    @abstractmethod
    def create_channel(self) -> Channel:
        ...

    def notify(self, message: str) -> None:
        channel = self.create_channel()   # uses the factory method
        channel.send(message)


class EmailNotifier(Notifier):
    def create_channel(self) -> Channel:
        return EmailChannel()


class SlackNotifier(Notifier):
    def create_channel(self) -> Channel:
        return SlackChannel()


class Channel(ABC):
    @abstractmethod
    def send(self, message: str) -> None: ...


class EmailChannel(Channel):
    def send(self, message: str) -> None:
        print(f"Email: {message}")


class SlackChannel(Channel):
    def send(self, message: str) -> None:
        print(f"Slack: {message}")
```

**Pythonic registry variant** (no subclassing required):

```python
from __future__ import annotations

_CHANNELS: dict[str, type[Channel]] = {}

def register_channel(name: str):
    def decorator(cls: type[Channel]) -> type[Channel]:
        _CHANNELS[name] = cls
        return cls
    return decorator

def make_channel(name: str) -> Channel:
    cls = _CHANNELS.get(name)
    if cls is None:
        raise KeyError(f"Unknown channel: {name!r}. Known: {list(_CHANNELS)}")
    return cls()

@register_channel("email")
class EmailChannel(Channel):
    def send(self, message: str) -> None:
        print(f"Email: {message}")
```

---

## Abstract Factory

**Intent:** Produce families of related objects without specifying their concrete classes. The entire family is swapped at once by swapping the factory.

**Use when:**
- Your system needs to work with multiple families of products (e.g. light/dark UI theme, Windows/Mac widgets, SQL/NoSQL backends).
- You want to enforce that products from the same family are always used together.

**Don't use when:** You only have one product family — Factory Method is simpler.

**Python notes:** An `AbstractFactory` is often a Protocol or ABC with `create_*` methods. In simple cases a module with factory functions suffices.

```python
from __future__ import annotations
from abc import ABC, abstractmethod


# Abstract products
class Button(ABC):
    @abstractmethod
    def render(self) -> str: ...

class Checkbox(ABC):
    @abstractmethod
    def render(self) -> str: ...


# Abstract factory
class UIFactory(ABC):
    @abstractmethod
    def create_button(self) -> Button: ...
    @abstractmethod
    def create_checkbox(self) -> Checkbox: ...


# Concrete family: Web
class WebButton(Button):
    def render(self) -> str: return "<button>"

class WebCheckbox(Checkbox):
    def render(self) -> str: return "<input type=checkbox>"

class WebFactory(UIFactory):
    def create_button(self) -> Button: return WebButton()
    def create_checkbox(self) -> Checkbox: return WebCheckbox()


# Concrete family: Terminal
class TerminalButton(Button):
    def render(self) -> str: return "[ OK ]"

class TerminalCheckbox(Checkbox):
    def render(self) -> str: return "[x]"

class TerminalFactory(UIFactory):
    def create_button(self) -> Button: return TerminalButton()
    def create_checkbox(self) -> Checkbox: return TerminalCheckbox()


# Client — depends only on the abstract factory
def render_ui(factory: UIFactory) -> None:
    btn = factory.create_button()
    chk = factory.create_checkbox()
    print(btn.render(), chk.render())
```

---

## Builder

**Intent:** Construct complex objects step-by-step. The same construction process can produce different representations.

**Use when:**
- Construction has many optional parameters or steps (avoids telescoping constructors).
- You need to build the same product in multiple configurations (e.g. XML vs JSON report).
- The construction order matters or involves many side-effectful steps.

**Don't use when:** The object is simple — use `@dataclass` with defaults or keyword-only arguments instead.

**Python notes:** Method-chaining builders (`builder.set_x().set_y().build()`) are common. Python's keyword arguments often replace simple builders — reserve Builder for genuinely complex multi-step assembly.

```python
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class QueryResult:
    sql: str
    params: list
    timeout: int | None
    explain: bool


class QueryBuilder:
    """Step-by-step builder for a database query."""

    def __init__(self, table: str) -> None:
        self._table = table
        self._conditions: list[str] = []
        self._params: list = []
        self._timeout: int | None = None
        self._explain: bool = False

    def where(self, condition: str, *params) -> QueryBuilder:
        self._conditions.append(condition)
        self._params.extend(params)
        return self

    def timeout(self, seconds: int) -> QueryBuilder:
        self._timeout = seconds
        return self

    def explain(self) -> QueryBuilder:
        self._explain = True
        return self

    def build(self) -> QueryResult:
        where = " AND ".join(self._conditions)
        sql = f"SELECT * FROM {self._table}"
        if where:
            sql += f" WHERE {where}"
        return QueryResult(sql, self._params, self._timeout, self._explain)


# Usage
result = (
    QueryBuilder("users")
    .where("age > ?", 18)
    .where("active = ?", True)
    .timeout(30)
    .build()
)
```

---

## Prototype

**Intent:** Copy existing objects without making the code dependent on their classes. Delegate cloning to the objects themselves.

**Use when:**
- Object creation is expensive (DB lookup, network call, heavy computation) and cloning is cheaper.
- You need many objects that differ only slightly from a known "template" instance.
- You want to save/restore object state (also see Memento).

**Don't use when:** Object construction is cheap — just call the constructor.

**Python notes:** Use `copy.copy()` (shallow) or `copy.deepcopy()` (deep). Override `__copy__`/`__deepcopy__` to control which fields are copied. For dataclasses, `dataclasses.replace()` is often cleaner than a full clone.

```python
from __future__ import annotations
import copy
from dataclasses import dataclass, field, replace


@dataclass
class NetworkConfig:
    host: str
    port: int
    headers: dict[str, str] = field(default_factory=dict)
    retries: int = 3

    def clone(self, **overrides) -> NetworkConfig:
        """Return a shallow copy with optional field overrides."""
        return replace(self, **overrides)


base = NetworkConfig(host="api.example.com", port=443, headers={"Auth": "token"})
staging = base.clone(host="staging.example.com")
debug   = base.clone(retries=1)
```

For objects with mutable nested state that must be independent:

```python
import copy

class DeepConfig:
    def __init__(self, settings: dict) -> None:
        self.settings = settings

    def __deepcopy__(self, memo: dict) -> DeepConfig:
        new = DeepConfig(copy.deepcopy(self.settings, memo))
        memo[id(self)] = new
        return new
```

---

## Singleton

**Intent:** Ensure a class has only one instance and provide a global access point to it.

**Use when:**
- Exactly one shared resource is needed: config store, connection pool, logger, registry.
- You need lazy initialisation and controlled global access.

**Avoid when:** You're tempted to use Singleton just because "it's convenient" — it's a form of global state and makes testing hard. Prefer dependency injection.

**Python notes:** A module-level instance is the simplest Singleton — Python modules are singletons by nature. When you need a class-based Singleton (e.g. to allow subclassing or lazy init), a metaclass is thread-safe:

```python
from __future__ import annotations
import threading


class SingletonMeta(type):
    _instances: dict[type, object] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class AppConfig(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.debug: bool = False
        self.db_url: str = "sqlite://:memory:"


cfg1 = AppConfig()
cfg2 = AppConfig()
assert cfg1 is cfg2  # same instance
```

**Module-level pattern** (simpler, preferred for most cases):

```python
# config.py  — module IS the singleton
debug: bool = False
db_url: str = "sqlite://:memory:"

# usage: import config; config.db_url = "..."
```
