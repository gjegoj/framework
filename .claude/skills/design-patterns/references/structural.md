# Structural Patterns

Structural patterns deal with how classes and objects are composed to form larger structures while keeping those structures flexible and efficient.

---

## Adapter

**Intent:** Allow objects with incompatible interfaces to collaborate. Wraps an object to translate one interface into another.

**Use when:**
- You want to use an existing class but its interface doesn't match what your code expects.
- You're integrating a third-party library or legacy code without modifying it.
- You need to make several incompatible classes work through a common interface.

**Don't use when:** You control both sides — just align the interfaces directly.

**Python notes:** Python's duck typing means you sometimes don't need an explicit Adapter — if the wrapped object already has the right method names, pass it directly. An Adapter is most valuable when the method names or signatures genuinely differ, or when you want a typed contract.

```python
from __future__ import annotations


# Your system's expected interface
class DataSource:
    def read(self) -> list[dict]:
        raise NotImplementedError


# Third-party library with a different interface
class LegacyCSVReader:
    def load_file(self, path: str) -> str:
        with open(path) as f:
            return f.read()

    def parse(self, raw: str) -> list[list[str]]:
        return [line.split(",") for line in raw.splitlines()]


# Adapter: wraps LegacyCSVReader, exposes DataSource interface
class CSVAdapter(DataSource):
    def __init__(self, path: str) -> None:
        self._reader = LegacyCSVReader()
        self._path = path

    def read(self) -> list[dict]:
        raw = self._reader.load_file(self._path)
        rows = self._reader.parse(raw)
        if not rows:
            return []
        headers, *data = rows
        return [dict(zip(headers, row)) for row in data]


# Client works only with DataSource — unaware of LegacyCSVReader
def process(source: DataSource) -> None:
    for record in source.read():
        print(record)
```

---

## Bridge

**Intent:** Decouple an abstraction from its implementation so that both can vary independently. The abstraction holds a reference to the implementation object rather than inheriting from it.

**Use when:**
- You have two orthogonal dimensions of variation (e.g. shape × drawing backend; notification × delivery channel).
- You want to avoid a class explosion from inheritance (ShapeCircleVector, ShapeCircleRaster, ShapeSquareVector…).
- You need to switch implementations at runtime.

**Don't use when:** Only one dimension varies — a simple ABC with subclasses is enough.

**Python notes:** Bridge is naturally expressed with composition + dependency injection. The "implementor" is injected, often as an ABC. In this project it appears as `Backbone × Task` — the topology/objective axes vary independently of the backbone.

```python
from __future__ import annotations
from abc import ABC, abstractmethod


# Implementation axis: rendering backends
class Renderer(ABC):
    @abstractmethod
    def render_circle(self, x: float, y: float, radius: float) -> None: ...


class VectorRenderer(Renderer):
    def render_circle(self, x: float, y: float, radius: float) -> None:
        print(f"Drawing vector circle at ({x},{y}) r={radius}")


class RasterRenderer(Renderer):
    def render_circle(self, x: float, y: float, radius: float) -> None:
        print(f"Drawing raster circle at ({x},{y}) r={radius}")


# Abstraction axis: shapes
class Shape(ABC):
    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer   # bridge to the implementation

    @abstractmethod
    def draw(self) -> None: ...
    @abstractmethod
    def resize(self, factor: float) -> None: ...


class Circle(Shape):
    def __init__(self, renderer: Renderer, x: float, y: float, radius: float) -> None:
        super().__init__(renderer)
        self.x, self.y, self.radius = x, y, radius

    def draw(self) -> None:
        self._renderer.render_circle(self.x, self.y, self.radius)

    def resize(self, factor: float) -> None:
        self.radius *= factor
```

---

## Composite

**Intent:** Compose objects into tree structures to represent part-whole hierarchies. Clients treat individual objects and compositions uniformly.

**Use when:**
- You have a tree structure (file system, UI widget hierarchy, DOM, expression tree).
- Client code should be able to treat leaves and containers the same way.

**Don't use when:** Your structure is flat — a plain list is clearer.

**Python notes:** Make leaf and composite share the same ABC. Python's `__iter__` makes tree traversal natural. For read-only trees, `@dataclass(frozen=True)` on nodes is idiomatic.

```python
from __future__ import annotations
from abc import ABC, abstractmethod


class FileSystemNode(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def display(self, indent: int = 0) -> None: ...


# Leaf
class File(FileSystemNode):
    def __init__(self, name: str, size_bytes: int) -> None:
        super().__init__(name)
        self._size = size_bytes

    def size(self) -> int:
        return self._size

    def display(self, indent: int = 0) -> None:
        print(" " * indent + f"- {self.name} ({self._size}B)")


# Composite
class Directory(FileSystemNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._children: list[FileSystemNode] = []

    def add(self, node: FileSystemNode) -> None:
        self._children.append(node)

    def size(self) -> int:
        return sum(child.size() for child in self._children)

    def display(self, indent: int = 0) -> None:
        print(" " * indent + f"+ {self.name}/")
        for child in self._children:
            child.display(indent + 2)


# Client treats File and Directory identically
root = Directory("root")
src = Directory("src")
src.add(File("main.py", 1200))
src.add(File("utils.py", 800))
root.add(src)
root.add(File("README.md", 300))
root.display()
print(f"Total: {root.size()}B")
```

---

## Decorator (GoF)

**Intent:** Attach additional responsibilities to an object dynamically at runtime by wrapping it in a decorator object that has the same interface.

**Use when:**
- You need to add behaviours (logging, caching, validation, compression) that can be combined in any order.
- Subclassing would lead to a combinatorial explosion of subclasses.
- You want to add/remove responsibilities without touching the original class.

**Don't use when:** Python's `@decorator` syntax (function wrappers) already solves the problem — the GoF Decorator pattern applies to objects, not functions. Don't confuse them.

**Python notes:** Both the wrapper and the wrapped object implement the same ABC. This pattern composes well; multiple wrappers can be stacked. Python's `functools.wraps` solves the simpler case of wrapping functions.

```python
from __future__ import annotations
from abc import ABC, abstractmethod


class DataStream(ABC):
    @abstractmethod
    def write(self, data: str) -> None: ...
    @abstractmethod
    def read(self) -> str: ...


class FileStream(DataStream):
    def __init__(self) -> None:
        self._content = ""

    def write(self, data: str) -> None:
        self._content = data

    def read(self) -> str:
        return self._content


# Base Decorator — forwards all calls to wrapped object
class StreamDecorator(DataStream):
    def __init__(self, wrapped: DataStream) -> None:
        self._wrapped = wrapped

    def write(self, data: str) -> None:
        self._wrapped.write(data)

    def read(self) -> str:
        return self._wrapped.read()


class EncryptedStream(StreamDecorator):
    def write(self, data: str) -> None:
        self._wrapped.write(data[::-1])   # toy "encryption"

    def read(self) -> str:
        return self._wrapped.read()[::-1]


class CompressedStream(StreamDecorator):
    def write(self, data: str) -> None:
        self._wrapped.write(data.replace(" ", "_"))  # toy compression

    def read(self) -> str:
        return self._wrapped.read().replace("_", " ")


# Stack decorators in any order
stream = CompressedStream(EncryptedStream(FileStream()))
stream.write("hello world")
print(stream.read())  # hello world
```

---

## Facade

**Intent:** Provide a simplified interface to a complex subsystem. The facade doesn't prevent direct access to the subsystem — it just makes the common cases easy.

**Use when:**
- A subsystem is complex and most callers need only a small slice of it.
- You want to layer a subsystem: a simple public API over a complex internal one.
- You're wrapping a third-party library to isolate your code from its API changes.

**Don't use when:** The subsystem is already simple or callers legitimately need full access.

**Python notes:** A Facade is often just a module with high-level functions — no class needed. It's the pattern behind "service" classes in web frameworks: `UserService.register()` hides the DB + email + logging calls.

```python
from __future__ import annotations


# Complex subsystem classes
class VideoDecoder:
    def decode(self, path: str) -> bytes:
        print(f"Decoding {path}")
        return b"<raw frames>"

class AudioDecoder:
    def decode(self, path: str) -> bytes:
        print(f"Decoding audio {path}")
        return b"<raw audio>"

class BitrateConverter:
    def convert(self, data: bytes, bitrate: int) -> bytes:
        print(f"Converting to {bitrate}kbps")
        return data

class VideoEncoder:
    def encode(self, video: bytes, audio: bytes) -> bytes:
        print("Encoding final video")
        return video + audio


# Facade — simple API for the common case
class VideoConverter:
    def __init__(self) -> None:
        self._video = VideoDecoder()
        self._audio = AudioDecoder()
        self._converter = BitrateConverter()
        self._encoder = VideoEncoder()

    def convert(self, input_path: str, bitrate: int = 1000) -> bytes:
        raw_video = self._video.decode(input_path)
        raw_audio = self._audio.decode(input_path)
        converted = self._converter.convert(raw_video, bitrate)
        return self._encoder.encode(converted, raw_audio)


# Client uses only the facade
result = VideoConverter().convert("movie.avi", bitrate=720)
```

---

## Proxy

**Intent:** Provide a substitute or placeholder for another object. The proxy controls access to the original, performing work before/after forwarding the request.

**Use when:**
- **Virtual proxy:** Lazy initialisation — delay creating a heavy object until it's actually needed.
- **Protection proxy:** Access control — check permissions before forwarding.
- **Caching proxy:** Cache results of expensive operations.
- **Logging proxy:** Record calls transparently.
- **Remote proxy:** Represent an object in a different process or machine.

**Don't use when:** You simply want to add behaviour before/after a method — a Decorator is more explicit.

**Python notes:** Python's `__getattr__` lets you implement a dynamic proxy that forwards all attribute access. For a single method, a closure or `functools.wraps` is simpler. The distinction from Decorator: a Proxy usually manages the lifecycle of the real object (creates it lazily, controls whether it's accessible).

```python
from __future__ import annotations
from abc import ABC, abstractmethod


class ImageLoader(ABC):
    @abstractmethod
    def display(self) -> None: ...


class RealImageLoader(ImageLoader):
    def __init__(self, path: str) -> None:
        self._path = path
        self._data = self._load()   # expensive

    def _load(self) -> bytes:
        print(f"Loading heavy image: {self._path}")
        return b"<image data>"

    def display(self) -> None:
        print(f"Displaying {self._path}")


# Virtual proxy — delays loading until display() is called
class LazyImageLoader(ImageLoader):
    def __init__(self, path: str) -> None:
        self._path = path
        self._real: RealImageLoader | None = None

    def display(self) -> None:
        if self._real is None:
            self._real = RealImageLoader(self._path)  # created on first use
        self._real.display()


# Caching proxy example
class CachingProxy(ImageLoader):
    _cache: dict[str, RealImageLoader] = {}

    def __init__(self, path: str) -> None:
        self._path = path

    def display(self) -> None:
        if self._path not in self._cache:
            self._cache[self._path] = RealImageLoader(self._path)
        self._cache[self._path].display()
```
