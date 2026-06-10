# Behavioral Patterns

Behavioral patterns deal with algorithms and the assignment of responsibilities between objects — how objects communicate and distribute work.

---

## Strategy

**Intent:** Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from the clients that use it.

**Use when:**
- You have multiple variants of an algorithm and want to switch between them at runtime or via configuration.
- You want to eliminate `if/elif` or `match` blocks that select behaviour based on type.
- Different subclasses differ only in their behaviour, not structure.

**Don't use when:** There are only two variants and they're simple — an `if` is clearer.

**Python notes:** In Python, a Strategy is often just a `Callable` — no class needed. Use a class-based Strategy only when the strategy needs to carry state.

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable


# --- Class-based Strategy (when state is needed) ---

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: list[int]) -> list[int]: ...


class BubbleSort(SortStrategy):
    def sort(self, data: list[int]) -> list[int]:
        lst = data[:]
        for i in range(len(lst)):
            for j in range(len(lst) - i - 1):
                if lst[j] > lst[j + 1]:
                    lst[j], lst[j + 1] = lst[j + 1], lst[j]
        return lst


class QuickSort(SortStrategy):
    def sort(self, data: list[int]) -> list[int]:
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        mid  = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + mid + self.sort(right)


class Sorter:
    def __init__(self, strategy: SortStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: SortStrategy) -> None:
        self._strategy = strategy

    def sort(self, data: list[int]) -> list[int]:
        return self._strategy.sort(data)


# --- Pythonic callable Strategy (no state needed) ---

type SortFn = Callable[[list[int]], list[int]]

def sort_with(data: list[int], fn: SortFn) -> list[int]:
    return fn(data)

# Usage: sort_with([3,1,2], sorted)
```

---

## Observer

**Intent:** Define a one-to-many dependency so that when one object changes state, all its dependents are notified and updated automatically.

**Use when:**
- A change in one object requires updating others, but you don't know how many at design time.
- Objects should be able to subscribe/unsubscribe without the publisher knowing about them.
- You're implementing event systems, reactive data flows, or model→view updates (MVC).

**Don't use when:** Subscribers are fixed and known — just call them directly.

**Python notes:** Python has no built-in event bus; the pattern is implemented explicitly. For GUI, use framework signals (`Qt signals`, `tkinter bind`). For async, consider `asyncio.Queue` or reactive libraries. Keep subscriber lists as `list` not `set` if order matters.

```python
from __future__ import annotations
from abc import ABC, abstractmethod


class Observer(ABC):
    @abstractmethod
    def update(self, event: str, data: object) -> None: ...


class EventEmitter:
    """Minimal observable — publisher side."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Observer]] = {}

    def subscribe(self, event: str, observer: Observer) -> None:
        self._subscribers.setdefault(event, []).append(observer)

    def unsubscribe(self, event: str, observer: Observer) -> None:
        self._subscribers.get(event, []).remove(observer)

    def emit(self, event: str, data: object = None) -> None:
        for obs in list(self._subscribers.get(event, [])):
            obs.update(event, data)


class StockMarket(EventEmitter):
    def __init__(self) -> None:
        super().__init__()
        self._price: float = 0.0

    @property
    def price(self) -> float:
        return self._price

    @price.setter
    def price(self, value: float) -> None:
        self._price = value
        self.emit("price_changed", value)


class AlertObserver(Observer):
    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def update(self, event: str, data: object) -> None:
        if isinstance(data, float) and data > self._threshold:
            print(f"Alert! Price {data} exceeded threshold {self._threshold}")


market = StockMarket()
market.subscribe("price_changed", AlertObserver(threshold=100.0))
market.price = 95.0
market.price = 105.0  # triggers alert
```

---

## Command

**Intent:** Encapsulate a request as an object, letting you parameterise clients with different requests, queue or log requests, and support undoable operations.

**Use when:**
- You need undo/redo functionality.
- You want to queue, schedule, or log operations.
- You need to parametrise an object with an operation (e.g. button → action mapping).
- You want to implement transactional behaviour (rollback on failure).

**Don't use when:** You just need to call a function — a `Callable` is sufficient.

**Python notes:** A Command is often a `@dataclass` with an `execute()` method. When no undo is needed, a plain `Callable` suffices. Use `dataclass` fields instead of a constructor for the command's parameters.

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Command(ABC):
    @abstractmethod
    def execute(self) -> None: ...
    @abstractmethod
    def undo(self) -> None: ...


@dataclass
class TextEditor:
    _text: str = ""

    @property
    def text(self) -> str:
        return self._text

    def insert(self, pos: int, s: str) -> None:
        self._text = self._text[:pos] + s + self._text[pos:]

    def delete(self, pos: int, length: int) -> None:
        self._text = self._text[:pos] + self._text[pos + length:]


@dataclass
class InsertCommand(Command):
    editor: TextEditor
    pos: int
    text: str

    def execute(self) -> None:
        self.editor.insert(self.pos, self.text)

    def undo(self) -> None:
        self.editor.delete(self.pos, len(self.text))


class CommandHistory:
    def __init__(self) -> None:
        self._history: list[Command] = []

    def execute(self, cmd: Command) -> None:
        cmd.execute()
        self._history.append(cmd)

    def undo(self) -> None:
        if self._history:
            self._history.pop().undo()


editor = TextEditor()
history = CommandHistory()
history.execute(InsertCommand(editor, 0, "Hello"))
history.execute(InsertCommand(editor, 5, " World"))
print(editor.text)   # Hello World
history.undo()
print(editor.text)   # Hello
```

---

## Template Method

**Intent:** Define the skeleton of an algorithm in a base class, deferring some steps to subclasses. Subclasses can override specific steps without changing the algorithm's overall structure.

**Use when:**
- Multiple classes share the same algorithm structure but differ in specific steps.
- You want to avoid code duplication across subclasses while allowing extension points.
- You're building a framework where the framework controls the flow but users fill in the details.

**Don't use when:** The variation points are numerous or complex — Strategy (with composition) is more flexible and avoids inheritance coupling.

**Python notes:** Abstract steps use `@abstractmethod`. Optional "hook" steps use empty default implementations. Keep the template method non-overridable by convention (no `@abstractmethod` on it).

```python
from __future__ import annotations
from abc import ABC, abstractmethod


class DataMigrator(ABC):
    """Template Method: migrate() orchestrates the steps."""

    def migrate(self) -> None:  # the template method — don't override
        data = self.extract()
        transformed = self.transform(data)
        self.load(transformed)
        self.on_success()       # hook — optional override

    @abstractmethod
    def extract(self) -> list[dict]: ...

    @abstractmethod
    def transform(self, data: list[dict]) -> list[dict]: ...

    @abstractmethod
    def load(self, data: list[dict]) -> None: ...

    def on_success(self) -> None:  # hook with empty default
        pass


class CsvToPostgresMigrator(DataMigrator):
    def extract(self) -> list[dict]:
        print("Reading CSV...")
        return [{"id": 1, "name": "Alice"}]

    def transform(self, data: list[dict]) -> list[dict]:
        return [{k: str(v).upper() for k, v in row.items()} for row in data]

    def load(self, data: list[dict]) -> None:
        print(f"Inserting {len(data)} rows into Postgres")

    def on_success(self) -> None:
        print("Migration complete. Sending notification.")
```

---

## State

**Intent:** Allow an object to alter its behaviour when its internal state changes. The object will appear to change its class.

**Use when:**
- An object's behaviour depends heavily on its state and must change at runtime.
- You have many conditionals that switch on an internal state enum/flag.
- State-specific behaviour is complex enough to warrant its own class.

**Don't use when:** There are only two or three states with simple transitions — an `Enum` + `match` is clearer.

**Python notes:** Each state is a class; the context delegates to the current state object. The state objects hold a reference back to the context so they can trigger transitions.

```python
from __future__ import annotations
from abc import ABC, abstractmethod


class OrderState(ABC):
    def __init__(self, order: Order) -> None:
        self._order = order

    @abstractmethod
    def pay(self) -> None: ...
    @abstractmethod
    def ship(self) -> None: ...
    @abstractmethod
    def cancel(self) -> None: ...


class PendingState(OrderState):
    def pay(self) -> None:
        print("Payment received.")
        self._order.set_state(PaidState(self._order))

    def ship(self) -> None:
        print("Cannot ship: order not paid.")

    def cancel(self) -> None:
        print("Order cancelled.")
        self._order.set_state(CancelledState(self._order))


class PaidState(OrderState):
    def pay(self) -> None:
        print("Already paid.")

    def ship(self) -> None:
        print("Shipped!")
        self._order.set_state(ShippedState(self._order))

    def cancel(self) -> None:
        print("Refunding and cancelling.")
        self._order.set_state(CancelledState(self._order))


class ShippedState(OrderState):
    def pay(self) -> None: print("Already paid.")
    def ship(self) -> None: print("Already shipped.")
    def cancel(self) -> None: print("Cannot cancel: already shipped.")


class CancelledState(OrderState):
    def pay(self) -> None: print("Order cancelled.")
    def ship(self) -> None: print("Order cancelled.")
    def cancel(self) -> None: print("Already cancelled.")


class Order:
    def __init__(self) -> None:
        self._state: OrderState = PendingState(self)

    def set_state(self, state: OrderState) -> None:
        self._state = state

    def pay(self) -> None: self._state.pay()
    def ship(self) -> None: self._state.ship()
    def cancel(self) -> None: self._state.cancel()


order = Order()
order.pay()    # Payment received.
order.ship()   # Shipped!
order.cancel() # Cannot cancel: already shipped.
```

---

## Chain of Responsibility

**Intent:** Pass a request along a chain of handlers. Each handler decides either to process the request or to pass it to the next handler in the chain.

**Use when:**
- More than one object may handle a request, and the handler isn't known a priori.
- You want to issue a request without specifying the handler explicitly.
- You're implementing middleware pipelines, event handling, or permission checks.

**Don't use when:** The processing logic is simple and the handler is always known — just call it directly.

**Python notes:** Chains are often built as a linked list of handlers or as a list iterated in a loop. The functional style (a list of handler functions) is often cleaner than a class-based chain.

```python
from __future__ import annotations
from abc import ABC, abstractmethod


class Handler(ABC):
    def __init__(self) -> None:
        self._next: Handler | None = None

    def set_next(self, handler: Handler) -> Handler:
        self._next = handler
        return handler   # allows chaining: h1.set_next(h2).set_next(h3)

    @abstractmethod
    def handle(self, request: int) -> str | None: ...

    def _pass_on(self, request: int) -> str | None:
        if self._next:
            return self._next.handle(request)
        return None


class SmallRequestHandler(Handler):
    def handle(self, request: int) -> str | None:
        if request < 10:
            return f"SmallHandler handled {request}"
        return self._pass_on(request)


class MediumRequestHandler(Handler):
    def handle(self, request: int) -> str | None:
        if request < 100:
            return f"MediumHandler handled {request}"
        return self._pass_on(request)


class LargeRequestHandler(Handler):
    def handle(self, request: int) -> str | None:
        return f"LargeHandler handled {request}"


# Build the chain
small = SmallRequestHandler()
small.set_next(MediumRequestHandler()).set_next(LargeRequestHandler())

for n in [5, 50, 500]:
    print(small.handle(n))


# --- Pythonic list-based pipeline (often simpler) ---
from collections.abc import Callable

type MiddlewareFn = Callable[[dict], dict | None]

def auth_check(req: dict) -> dict | None:
    if not req.get("token"):
        print("Rejected: no token")
        return None
    return req

def rate_limit(req: dict) -> dict | None:
    if req.get("count", 0) > 100:
        print("Rejected: rate limit")
        return None
    return req

def process(req: dict) -> dict | None:
    print(f"Processing: {req}")
    return req

def pipeline(request: dict, handlers: list[MiddlewareFn]) -> dict | None:
    result = request
    for handler in handlers:
        result = handler(result)
        if result is None:
            return None
    return result
```
