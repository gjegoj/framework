"""Progress reporting: one look for every long operation the framework runs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from src.console import console


def track[T](items: Iterable[T], description: str, total: int | None = None) -> Iterator[T]:
    """Yield ``items``, showing a progress bar while a terminal is watching.

    Silent when stdout is not a terminal, so test output and CI logs stay clean
    without a flag anyone has to remember to set.

    Parameters:
        items (Iterable[T]): What to iterate; yielded unchanged.
        description (str): Label shown to the left of the bar.
        total (int | None): Expected count, when it is known in advance.
    """
    display = console()
    if not display.is_terminal:
        yield from items
        return
    columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=display) as progress:
        task = progress.add_task(description, total=total)
        for item in items:
            yield item
            progress.advance(task)
