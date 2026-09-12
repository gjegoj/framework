"""The terminal: the one console everything prints through, and the one bar anything is watched by."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from rich import get_console
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn


def console() -> Console:
    """Rich's own console, which is what the libraries a run is built on print through.

    Not one of ours: two consoles on one terminal fight over the cursor, and a live progress bar is
    exactly the thing that loses such a fight. Looked up each time rather than held, so a run that
    reconfigures rich — as Lightning's bar does with ``console_kwargs`` — is followed here too.
    """
    return get_console()


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
