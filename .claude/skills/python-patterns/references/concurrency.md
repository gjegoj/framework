# Concurrency

Choosing and using the three concurrency models in Python — threads,
processes, and asyncio. The single most important decision is **I/O-bound vs
CPU-bound**, because the GIL makes them behave very differently. Examples target
Linux (CPython 3.12).

## Contents
- [The decision: I/O-bound vs CPU-bound, and the GIL](#the-decision-io-bound-vs-cpu-bound-and-the-gil)
- [`concurrent.futures` — the high-level default](#concurrentfutures--the-high-level-default)
- [Threading](#threading)
- [Multiprocessing](#multiprocessing)
- [asyncio](#asyncio)
- [Pitfalls](#pitfalls)
- [Rule of thumb](#rule-of-thumb)

## The decision: I/O-bound vs CPU-bound, and the GIL

CPython has a **Global Interpreter Lock (GIL)**: only one thread executes Python
bytecode at a time. The consequences drive every choice here:

- **CPU-bound** pure-Python work (number crunching in Python loops) does **not**
  get faster with threads — they take turns on the one GIL. Use **processes**
  (each has its own interpreter and GIL) to use multiple cores.
- **I/O-bound** work (network, disk, subprocess) spends most of its time
  *waiting*, and the GIL is released during blocking I/O — so **threads** (or
  **asyncio**) overlap that waiting and give a big speedup.
- Heavy C extensions (NumPy, compression, hashing) often release the GIL during
  their compute, so threads can parallelize *those* too.

| Workload | Best tool |
|---|---|
| Waiting on network/disk, dozens of tasks | `ThreadPoolExecutor` |
| Waiting on network, thousands of tasks | `asyncio` |
| CPU-heavy Python across cores | `ProcessPoolExecutor` |
| CPU-heavy inside C libs that release the GIL | threads can work; measure |

(Free-threaded "no-GIL" CPython exists experimentally from 3.13, but default
builds still have the GIL — design as above unless you've explicitly opted in.)

## `concurrent.futures` — the high-level default

`ThreadPoolExecutor` and `ProcessPoolExecutor` share one API, so you can switch
models by changing one line. Reach for this before the lower-level `threading`
or `multiprocessing` modules.

```python
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def fetch(url: str) -> str:
    ...  # I/O-bound


def crunch(chunk: list[int]) -> int:
    return sum(value * value for value in chunk)  # CPU-bound


# I/O-bound: many waits overlap on threads.
def fetch_all(urls: list[str]) -> dict[str, str]:
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(fetch, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                results[url] = future.result()
            except Exception as error:          # one task failing shouldn't kill the batch
                results[url] = f"error: {error}"
    return results


# CPU-bound: real parallelism across cores.
def crunch_all(chunks: list[list[int]]) -> list[int]:
    with ProcessPoolExecutor() as pool:         # defaults to os.cpu_count() workers
        return list(pool.map(crunch, chunks))   # map preserves input order
```

- `submit` returns a `Future`; `as_completed` yields them as they finish (good
  when you want results ASAP and order doesn't matter).
- `map` preserves input order and is convenient for a simple fan-out.
- A `with` block shuts the pool down cleanly. `future.result()` re-raises any
  exception from the worker — handle it where you read results.

## Threading

For I/O concurrency when you want explicit control. Threads share memory, so any
shared mutable state needs a lock to avoid races:

```python
from __future__ import annotations

import threading

counter = 0
lock = threading.Lock()


def worker() -> None:
    global counter
    with lock:                 # only one thread mutates at a time
        counter += 1


threads = [threading.Thread(target=worker) for _ in range(100)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
```

- `threading.Lock` for mutual exclusion; `threading.Event` to signal between
  threads; `queue.Queue` is the thread-safe way to hand work/results between
  threads (no manual locking).
- A daemon thread (`Thread(..., daemon=True)`) won't block interpreter exit — use
  for background loops you don't need to join.
- Prefer message passing over a `queue.Queue` to sharing mutable objects under
  locks; it's far easier to reason about.

## Multiprocessing

Each process has its own memory and its own GIL — real parallelism for CPU-bound
work, at the cost of having to **pickle** arguments, return values, and the
target function across the process boundary.

### Start methods (Linux)

How a worker process is created matters:

- **`fork`** — the historical Linux default (≤ 3.13). Cheapest (copies the parent
  via copy-on-write), but **unsafe when the parent has threads**: `fork` copies
  only the calling thread, so a lock held by another thread stays locked forever
  in the child → deadlocks. Many libraries (loggers, some C libs) start threads.
- **`forkserver`** — forks workers from a small, clean server process started
  early. Avoids the fork-with-threads hazard while staying cheap. **Recommended
  default on Linux.** (CPython 3.14 makes this the Linux default for this reason.)
- **`spawn`** — starts a fresh interpreter and re-imports your module. Safest and
  most portable (it's the Windows/macOS default), but slowest and it re-runs
  module top-level code, so guard your entrypoint with `if __name__ == "__main__":`.

```python
from __future__ import annotations

import multiprocessing as mp


def crunch(chunk: list[int]) -> int:
    return sum(value * value for value in chunk)


def main() -> None:
    context = mp.get_context("forkserver")      # don't rely on the platform default
    with context.Pool(processes=4) as pool:
        print(pool.map(crunch, [[1, 2], [3, 4]]))


if __name__ == "__main__":                      # required for spawn/forkserver
    main()
```

Set it per-call with `get_context(...)` rather than the global
`set_start_method` so libraries don't fight over it.

### Sharing data between processes

Processes don't share memory by default. Options, cheapest-correct first:

- **Pass as arguments / return values** — simplest; everything is pickled. Keep
  payloads modest (pickling large objects is the usual bottleneck).
- **`multiprocessing.Queue` / `Pipe`** — stream messages between processes.
- **`multiprocessing.shared_memory.SharedMemory`** — a raw shared buffer for
  large arrays (e.g. back a NumPy array with it) to avoid copying.
- **`Manager().dict()` / `.list()`** — shared mutable containers, but every
  access is a proxied IPC round-trip — convenient, not fast.

## asyncio

Single-threaded **cooperative** concurrency: one event loop interleaves many
tasks that voluntarily yield at `await`. Ideal for thousands of concurrent I/O
operations (sockets, HTTP) where a thread-per-task would be too heavy.

```python
from __future__ import annotations

import asyncio


async def fetch(client, url: str) -> str:
    async with client.get(url) as response:     # await yields control while waiting
        return await response.text()


async def fetch_all(client, urls: list[str]) -> list[str]:
    async with asyncio.TaskGroup() as group:     # 3.11+: structured, all-or-error
        tasks = [group.create_task(fetch(client, url)) for url in urls]
    return [task.result() for task in tasks]


asyncio.run(fetch_all(client, urls))             # one entrypoint owns the loop
```

- `asyncio.run(...)` is the single entrypoint; don't create loops by hand.
- `TaskGroup` (3.11+) is the modern way to run tasks concurrently — it waits for
  all of them and, on failure, cancels the rest and raises an `ExceptionGroup`
  (handle with `except*`, see `error-handling.md`). `asyncio.gather` is the older
  equivalent.
- The cardinal rule: **never call blocking code directly in a coroutine** — it
  freezes the whole loop. Offload it:

```python
result = await asyncio.to_thread(blocking_io_call, arg)   # run in a thread

loop = asyncio.get_running_loop()
result = await loop.run_in_executor(process_pool, cpu_heavy, arg)  # in a process
```

## Pitfalls

- **Threads won't speed up CPU-bound Python** — the GIL serializes them. Reach
  for processes (or release the GIL in a C extension).
- **`fork` + threads → deadlock.** Use `forkserver`/`spawn` when the parent may
  hold locks (most real programs do).
- **`ProcessPool` can't pickle lambdas, closures, or local functions** — the
  target must be importable (a top-level `def`). Same for its arguments.
- **Reseed RNGs per worker.** Forked workers inherit the parent's random state
  and produce *identical* streams; spawned workers re-seed from entropy but won't
  be reproducible. Seed deterministically per worker (e.g. from the worker index)
  when you need both parallelism and reproducibility.
- **Shared mutable state across threads → races.** Guard with a `Lock`, or
  better, pass messages through a `Queue`.
- **Don't block the event loop.** A sync DB driver or `time.sleep` in a coroutine
  stalls every task — use `await asyncio.to_thread(...)` / an async library.
- **Oversubscription.** A process pool whose workers each run multi-threaded BLAS
  (NumPy/PyTorch) spawns cores² threads and thrashes — cap pool size and/or pin
  intra-op threads (e.g. `OMP_NUM_THREADS=1`).

## Rule of thumb

1. Default to **`concurrent.futures`** — `ThreadPoolExecutor` for I/O,
   `ProcessPoolExecutor` for CPU. One-line switch between them.
2. Drop to **`threading`**/**`multiprocessing`** only when you need primitives the
   executors don't expose (custom synchronization, long-lived workers, shared
   memory).
3. Choose **`asyncio`** when concurrency is in the thousands and the libraries you
   call are async-native.
4. On Linux, pick **`forkserver`** (or `spawn`) explicitly — don't depend on the
   platform default.
5. Always **measure** — concurrency adds overhead (IPC, context switches); for
   small inputs the serial version often wins.
