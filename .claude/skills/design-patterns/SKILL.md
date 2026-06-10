---
name: design-patterns
description: GoF design patterns in Python — when to use each, concise Python examples, and Pythonic trade-offs. Use when the user asks "which pattern fits here?", "how to implement Factory/Observer/Strategy/etc.", mentions design pattern names, or when code has a recognisable smell (big switch on type → Strategy; tight coupling between publisher and subscribers → Observer; complex object construction → Builder; wrapping a third-party API → Adapter). Reference: refactoring.guru/design-patterns/python
disable-model-invocation: false
---

# Design Patterns in Python

Based on refactoring.guru/design-patterns/python. Three reference files — read the relevant one:

- [Creational](references/creational.md) — Factory Method, Abstract Factory, Builder, Prototype, Singleton
- [Structural](references/structural.md) — Adapter, Bridge, Composite, Decorator, Facade, Proxy
- [Behavioral](references/behavioral.md) — Strategy, Observer, Command, Template Method, State, Chain of Responsibility

## Quick decision table

| Smell / Need | Pattern | File |
|---|---|---|
| Create objects without knowing exact class at compile time | Factory Method | creational |
| Create families of related objects, swap whole family at once | Abstract Factory | creational |
| Construct complex objects step-by-step; same process, different representations | Builder | creational |
| Clone existing objects cheaply; avoid re-running expensive init | Prototype | creational |
| One global instance, controlled access (e.g. config, connection pool) | Singleton | creational |
| Wrap incompatible third-party / legacy interface to match your contract | Adapter | structural |
| Decouple abstraction from implementation so both vary independently | Bridge | structural |
| Tree of uniform leaf/composite objects treated identically | Composite | structural |
| Add behaviour to objects at runtime without subclassing | Decorator (GoF) | structural |
| Simplify a complex subsystem behind one clean entry point | Facade | structural |
| Lazy init / access control / logging / caching around a real object | Proxy | structural |
| Swap algorithms at runtime; eliminate big if/switch on type | Strategy | behavioral |
| Notify many objects when one object's state changes | Observer | behavioral |
| Encapsulate a request as an object; enables undo/redo, queuing | Command | behavioral |
| Skeleton algorithm in base class; subclasses fill in the steps | Template Method | behavioral |
| Object behaviour changes with internal state; eliminate state flags | State | behavioral |
| Pass request along a chain; each handler decides to process or forward | Chain of Responsibility | behavioral |

## Python-specific notes (applies to all patterns)

- **First-class functions** often replace single-method Strategy/Command classes — a plain `Callable` is idiomatic when no state is needed.
- **ABCs** (`abc.ABC` + `@abstractmethod`) define interfaces; they play the role of Java interfaces.
- **`copy.copy` / `copy.deepcopy`** implement Prototype; override `__copy__` / `__deepcopy__` to customise cloning.
- **Metaclasses** can implement thread-safe Singleton; a module-level instance is usually simpler.
- Prefer **composition over inheritance** — use wrapping (Adapter/Decorator) before reaching for subclassing.
- Python's `@property` and `__getattr__` are often the right tool before reaching for a full Proxy.
