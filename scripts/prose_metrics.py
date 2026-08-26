"""How much of ``src`` is prose, and where the long docstrings are.

A review aid, not a gate: ``uv run python scripts/prose_metrics.py`` prints the prose
share per package, the docstring-length buckets, the longest docstrings, and the
words the readability pass hunts (history narration, metaphors).
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections import Counter, defaultdict
from pathlib import Path

CAP = 10
"""Longest a docstring may be, unless it is a config-facing class with a Parameters block."""

CAP_WITH_PARAMETERS = 20

HISTORY = r"used to|the previous|withdrawn|was written as|first version|before this change"
METAPHORS = {
    "brick": r"\bbricks?\b",
    "canon": r"\bcanon\b",
    "carrier": r"\bcarriers?\b",
    "rides": r"\brid(?:e|es|ing|er|ers)\b",
    "judgment": r"\bjudg\w*",
    "spoken": r"\bspoken\b",
    "bargain": r"\bbargain\b",
    "voice": r"\bvoices?\b",
}


def main() -> None:
    code, prose, files = defaultdict(int), defaultdict(int), defaultdict(int)
    docstrings: list[tuple[int, str]] = []
    everything = []
    for path in sorted(Path("src").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        everything.append(source)
        tree = ast.parse(source)
        doc_lines = 0
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if ast.get_docstring(node) is None:
                continue
            first = node.body[0]
            length = first.end_lineno - first.lineno + 1
            doc_lines += length
            name = getattr(node, "name", "<module>")
            docstrings.append((length, f"{path}:{first.lineno} {name}"))
        comments = sum(
            1 for tok in tokenize.generate_tokens(io.StringIO(source).readline) if tok.type == tokenize.COMMENT
        )
        nonblank = sum(1 for line in source.splitlines() if line.strip())
        package = path.parts[1] if len(path.parts) > 2 else "(root)"
        code[package] += nonblank - doc_lines - comments
        prose[package] += doc_lines + comments
        files[package] += 1

    total_code, total_prose = sum(code.values()), sum(prose.values())
    print(
        f"src: {sum(files.values())} files, code {total_code}, prose {total_prose}, prose share {total_prose / (total_code + total_prose):.0%}"
    )
    print("\npackage        files  code  prose  share")
    for package in sorted(code, key=lambda key: -code[key]):
        share = prose[package] / (code[package] + prose[package])
        print(f"{package:14} {files[package]:5} {code[package]:5} {prose[package]:6}  {share:.0%}")

    buckets = Counter(
        "1-3" if n <= 3 else "4-10" if n <= CAP else "11-20" if n <= CAP_WITH_PARAMETERS else ">20"
        for n, _ in docstrings
    )
    print(f"\ndocstrings: {len(docstrings)} {dict(sorted(buckets.items()))}")
    print(
        f"over {CAP}: {sum(1 for n, _ in docstrings if n > CAP)}, over {CAP_WITH_PARAMETERS}: {sum(1 for n, _ in docstrings if n > CAP_WITH_PARAMETERS)}"
    )
    print("\nlongest 20:")
    for length, where in sorted(docstrings, reverse=True)[:20]:
        print(f"  {length:3}  {where}")

    joined = "\n".join(everything)
    print(f"\nhistory narration: {len(re.findall(HISTORY, joined))}")
    print(
        "metaphors: " + ", ".join(f"{name} {len(re.findall(pattern, joined))}" for name, pattern in METAPHORS.items())
    )


if __name__ == "__main__":
    main()
