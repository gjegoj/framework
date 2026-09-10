"""A progress bar over any iterable, silent when the output is not a terminal."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from src.console import console


def track[T](
    items: Iterable[T], description: str, total: int | None = None, status: Callable[[], str] | None = None
) -> Iterator[T]:
    """Yield ``items`` behind a bar; ``status`` is re-read after every item for a live figure beside it."""
    display = console()
    if not display.is_terminal:
        yield from items
        return
    columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        *(() if status is None else (TextColumn("{task.fields[status]}"),)),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=display) as progress:
        task = progress.add_task(description, total=total, status="" if status is None else status())
        for item in items:
            yield item
            if status is None:
                progress.advance(task)
            else:
                progress.update(task, advance=1, status=status())
