# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install                       # uv sync
uv run pytest                      # full suite
uv run pytest tests/unit/data/test_cache.py -q            # one file
uv run pytest tests/unit/data/test_cache.py::test_name    # one test
make test-unit                     # tests/unit -m "not slow" — the pre-commit gate
make typecheck                     # mypy --strict over src and tests
uv run pre-commit run --files <changed files>             # runs mypy + test-unit + linters
make test-run                      # fetches Oxford-IIIT Pet once, trains the multitask example

uv run main.py experiment=examples/classification lr=3e-4 epochs=50
```

Overrides: a key the composed config declares is overridden by name (`lr=3e-4`);
a key it does not declare is *added* with `+` (`+run.checkpoint_path=...`).

## How to work in this repo

- **Analyze first, implement after approval.** For any non-trivial change: read the
  relevant code, present what is wrong and why, propose a plan with trade-offs named
  honestly, and wait for the owner's approval. Then implement in TODO format with
  self-validation as you go. Questions get assessments, not unsolicited patches.
- **Look for an existing shape before inventing one.** Before writing anything new,
  check whether the repo already has the concept — a seam, a registry, a naming
  pattern, a class that solves the analogous problem — and either reuse it or build
  the new thing *in its shape*, so the codebase keeps one way of doing each kind of
  thing rather than a second near-identical class per feature. The precedent is a
  default, not a law: existing code can itself be the mistake (one whole module has
  been deleted for that), so when honest analysis says the established shape is
  wrong or missing, say so and build the right new one.
- **The repo owner drives git.** Never commit, branch, stash — and never revert a
  file with `git checkout`/`git restore`: the working tree holds uncommitted work.
  Revert experimental edits by replacing the exact lines back.
- **Never edit `pyproject.toml`.**
- **Every new guarantee gets a mutation check.** Break the code the way a regression
  would, watch the *specific* test go red, restore, and confirm green. A test that
  cannot fail is not evidence.
- **Measure before asserting.** Claims about behavior (a library's semantics, a
  performance effect, an edge case) are verified with a small experiment first;
  docstrings record the measurement ("measured, ...") next to the decision it
  justified. Deliberate trade-offs are written down where they were made.
- **Gate before "done":** full `uv run pytest`, then `pre-commit` on the touched
  files (which runs strict mypy over the whole tree and the unit suite). Report
  failures verbatim; never claim green without the run.
- **Match the house style.** Docstrings explain *why* and record constraints, not
  what the next line does. Comments only for what the code cannot say. Test names
  are full sentences about observable behavior (`test_the_label_tells_the_truth_
  when_the_swing_is_clipped`), asserted through the public API.

## The concepts the code is written in

The mechanics live in `README.md` and `docs/`; read them when a detail matters.
What follows is the *design stance* every change is measured against.

- **Arrows point down only.** Thin `core/` (torch and stdlib), capability packages
  around it, one composition root (`cli.py + assembly/`) that alone reads config.
  Third-party stacks are quarantined — Lightning in `training/`, pydantic in
  `config/`, Hydra in `cli.py`, albumentations behind a seam. New code first
  answers: *which layer does this knowledge belong to?* A fact about processes
  belongs where processes are made; presentation belongs to the driver, not the
  component that knows the numbers.
- **Declare once, derive everywhere.** A fact is stated in one place and offered to
  whoever names it; nothing user-visible is written twice, and whatever can be
  computed from data or schema is never asked for in config (`num_classes` does not
  exist there). If a change makes two declarations that can disagree, it is wrong.
- **Config declares *what*, never *how*.** A knob exists only where a real choice
  does — a parameter with one correct value is not a parameter, it is code. Every
  component speaks one grammar (`name` from a registry or `_target_`, the rest
  constructor arguments), so extension is one new class plus one config line, with
  no edit to existing code. Vendor models come through adapters to narrow ports;
  the ports never bend toward a vendor's signatures.
- **A task is a composition, not a type** — `topology × objective × modality`, with
  presets as thin names over familiar combinations. New capability usually means a
  new point in that space, not a new subsystem.
- **Fail at construction, by name.** A bad declaration dies while the experiment is
  built, with a message that names the declaration and the fix — never mid-epoch,
  never by one value silently winning over another. Silent fallback is a defect;
  when the framework substitutes something (val for a missing test split), it says
  so out loud.
- **Reported values mean what their names say.** A label is read back from what was
  actually done, not from what was intended; a metric under an honest name must not
  carry an optimistic number. When behavior and its report can drift apart, the
  report follows the behavior.
- **The reader is a data scientist without a glossary.** Names reveal intention,
  docstrings carry the *why* and the measured facts behind decisions, and any
  mechanism a DS cannot extend by reading one class is too clever to keep.
