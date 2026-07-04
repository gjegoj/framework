"""Console output helpers: config pretty-print and third-party warning suppression."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any

from rich import print as rprint
from rich.syntax import Syntax
from rich.tree import Tree

_NOT_EXPAND = frozenset({"mean", "std"})
_SKIP_KEYS = frozenset({"warnings"})

# Repetitive third-party notices we can't act on per-run; matched (regex) against the
# warning text. Project warnings and anything new still surface.
_NOISY_WARNINGS = (
    r".*LeafSpec.* is deprecated",  # torch pytree deprecation fired inside Lightning
    r".*does not have many workers.*bottleneck",  # DataLoader num_workers tip
    r".*infer the .batch_size. from an ambiguous",  # multi-head batch_size guess
    r".*Precision bf16-mixed is not supported by the model summary.*",  # model summary precision warning
)


def silence_known_warnings() -> None:
    """Filter the repetitive Lightning/torch warnings that clutter the run log.

    Each is an internal notice we can't address per-run (a torch deprecation Lightning
    triggers, the ``num_workers`` tip, the multi-target ``batch_size`` inference). Call
    once at startup; unrelated and new warnings are unaffected.
    """
    for message in _NOISY_WARNINGS:
        warnings.filterwarnings("ignore", message=message)


def print_config(
    config: dict[str, Any],
    not_expand: frozenset[str] | set[str] = _NOT_EXPAND,
    skip_keys: frozenset[str] | set[str] = _SKIP_KEYS,
    lexer: str = "python",
    theme: str = "dracula",
) -> None:
    """Print a resolved config dict as a syntax-highlighted Rich tree.

    Parameters:
        config: Already-resolved plain dict (pass OmegaConf.to_container result).
        not_expand: Keys whose values are always shown inline, even if they are lists.
        skip_keys: Keys excluded entirely from the output.
        lexer: Pygments lexer for syntax highlighting.
        theme: Pygments theme for syntax highlighting.
    """
    syntax = partial(Syntax, lexer=lexer, theme=theme, line_numbers=False, word_wrap=False)

    def _build(tree: Tree, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in skip_keys:
                    continue
                if isinstance(nested, (str, int, float, bool, type(None))) or key in not_expand:
                    tree.add(Tree(syntax(f"'{key}': {nested}"), style=""))
                else:
                    branch = Tree(syntax(f"'{key}'"), style="")
                    _build(branch, nested)
                    tree.add(branch)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                branch = Tree(syntax(str(i)), style="")
                _build(branch, item)
                tree.add(branch)
        else:
            tree.add(Tree(syntax(repr(value)), style=""))

    tree = Tree("[bold blue]Configuration")
    _build(tree, config)
    rprint(tree)
