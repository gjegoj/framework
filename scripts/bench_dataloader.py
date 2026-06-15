"""Benchmark train-dataloader throughput with the data cache on vs off (no model).

Isolates data loading from compute so the cache's effect is visible even when
training is GPU/compute-bound. Builds the DataModule twice from the same config —
once with the configured ``data.cache`` disabled and once enabled — and times
iterating the train DataLoader (no forward/backward).

Usage:
    uv run python scripts/bench_dataloader.py                       # default experiment
    uv run python scripts/bench_dataloader.py experiment=other_exp
    uv run python scripts/bench_dataloader.py data.cache.ram_fraction=0.5
    uv run python scripts/bench_dataloader.py +bench.num_batches=100 +bench.warmup=10

The "cache ON" run uses ``data.cache`` from the config (e.g.
``{ram_fraction: 0.5, max_gb: 4}``); the "cache OFF" run forces it to ``None``.
The OFF case runs first so both iterate files already in the OS page cache — the
remaining difference is the decode (and disk, on a cold cache) the cache removes.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Repo root on sys.path so `import src...` works when run as `python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.models  # noqa: E402,F401 — registers backbones/heads (parity with main.py)
import src.tasks  # noqa: E402,F401 — registers task presets
from src.composition.wiring import build_bindings, build_data_module  # noqa: E402
from src.config import ExperimentConfig, load_config  # noqa: E402
from src.core.runtime import RuntimeContext  # noqa: E402

log = logging.getLogger(__name__)


def _time_loader(loader: DataLoader, num_batches: int, warmup: int) -> float:
    """Return milliseconds per batch over a timed window after ``warmup`` batches."""
    iterator = iter(loader)
    for _ in range(warmup):
        next(iterator)
    start = time.perf_counter()
    for _ in range(num_batches):
        next(iterator)
    return (time.perf_counter() - start) / num_batches * 1000.0


def _benchmark(config: ExperimentConfig, num_batches: int, warmup: int) -> float:
    """Build + warm a DataModule from ``config`` and time its train loader."""
    dm = build_data_module(config, build_bindings(config), RuntimeContext())
    dm.setup()
    loader = dm.train_dataloader()
    available = len(loader)
    warmup = min(warmup, max(0, available - 1))
    batches = min(num_batches, available - warmup)
    if batches < 1:
        raise ValueError(f"Train loader has only {available} batches; need more than warmup={warmup}.")
    return _time_loader(loader, batches, warmup)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(raw, dict)
    bench: dict[str, Any] = raw.pop("bench", {}) or {}  # 'bench' is not part of ExperimentConfig (extra=forbid)
    num_batches = int(bench.get("num_batches", 50))
    warmup = int(bench.get("warmup", 5))

    config = load_config(raw)
    batch_size = config.batch_size

    if config.data.cache is None or config.data.cache.ram_fraction <= 0:
        log.warning(
            "data.cache is disabled in the config — the 'cache ON' run will also be uncached. "
            "Set e.g. 'data.cache.ram_fraction=0.5' to benchmark the cache."
        )

    no_cache = config.model_copy(deep=True)
    no_cache.data.cache = None

    log.info("Measuring cache OFF (also warms the OS page cache)...")
    ms_off = _benchmark(no_cache, num_batches, warmup)
    log.info("Measuring cache ON...")
    ms_on = _benchmark(config, num_batches, warmup)

    log.info("Train dataloader throughput (num_workers=%d, batch_size=%d):", config.dataloader.num_workers, batch_size)
    log.info("  cache OFF: %7.2f ms/batch · %6.0f img/s", ms_off, batch_size / ms_off * 1000.0)
    log.info("  cache ON : %7.2f ms/batch · %6.0f img/s", ms_on, batch_size / ms_on * 1000.0)
    if ms_on < ms_off:
        log.info("  → cache saves %.2f ms/batch (%.2fx faster data loading)", ms_off - ms_on, ms_off / ms_on)
    else:
        log.info("  → no data-loading gain here (not data-bound, or files already in the OS page cache)")


if __name__ == "__main__":
    main()
