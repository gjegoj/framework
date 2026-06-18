"""Rich-based console utilities."""

from __future__ import annotations

from functools import partial
from typing import Any

from rich import print as rprint
from rich.syntax import Syntax
from rich.tree import Tree

_NOT_EXPAND = frozenset({"mean", "std"})
_SKIP_KEYS = frozenset({"warnings"})


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
