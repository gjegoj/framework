# CLAUDE.md

Guidance for Claude Code working in this repository. What follows is the working
agreement, not a description of the code: the mechanics live in `README.md`, and the
design stance below is what every change is measured against.

## Commands

```bash
make install      # sync dependencies — the only command that may touch them
make test         # the whole suite
make test-unit    # one package at a time, nothing assembled
make test-e2e     # assembled runs, end to end
make test-gate    # the suite minus the tests that need a model hub
make typecheck    # strict type checking over source and tests
make check        # typecheck + test
make pre-commit   # every hook, the two above included
make test-run     # fetch the dataset once and train an example on a slice
```

Checks run through these targets and no other way: they carry the flags the project is
judged by, so a bare `pytest`/`mypy`/`ruff` invocation is judging something else. Never
install a tool to make a check pass — everything needed is already here.

**`pre-commit` only sees files git knows about.** An untracked file is skipped in
silence, so add new files to the index before the gate, or it passes on nothing.

## How to work here

- **Understand before changing.** Read the code that exists, say what is wrong and why,
  propose the change with its trade-offs named honestly, and wait for the owner's
  go-ahead. A question is answered with an assessment, not an unsolicited patch.
- **Work in reviewable steps.** Each one ends with a green gate and a short report of
  what changed, what was measured, and what was deliberately left out. The next step
  starts on the owner's word.
- **Rethink; never transcribe.** A reference implementation — an older branch, a paper's
  repository, a library's example — states the *problem*, not the solution. Port the
  intent and re-derive the design in this codebase's terms. Copying a shape imports its
  accidents, and those accidents are usually why it is being replaced; every carried-over
  line justifies itself on its own or does not come.
- **Look for the existing shape first.** Before writing a new class, seam, registry, or
  naming pattern, find the one this codebase already uses for the analogous problem and
  build inside it: one way of doing each kind of thing beats a second near-identical
  mechanism per feature. The precedent is a default, not a law — when honest analysis
  says the established shape is itself the mistake, say so and replace it everywhere,
  not just at the new call site.
- **Nothing without a consumer.** A class, method, constant, enum member, parameter or
  configuration group arrives together with the code that uses it and the test that pins
  it. "We will need it later" is how a codebase fills with things nobody dares delete.
- **One home per invariant.** A fact is stated once and read wherever it is needed, and a
  change that creates two statements free to disagree is wrong. A component checks only
  *what the declaration cannot see*: a typed declaration owns its own constraint and is
  trusted; untyped passthrough has no owner, so whoever consumes it validates; a fact that
  exists only at runtime has no declaration to own it; and agreement between separately
  built parts is visible only to whoever assembles them.
- **Finish the whole thing.** Scope is never quietly narrowed. A part that turns out to
  be blocked is carried as far as it goes and named as unfinished.

## Tests are the specification

- **Write the failing test first, and watch it fail for the right reason.** A test that
  errors on a missing import proves nothing; it has to fail because the behaviour is
  absent. A test written after the code passes immediately, which is not evidence.
- **Then break the implementation.** Every guarantee gets a mutation check: change the
  code the way a regression would, watch that *specific* test go red, restore, confirm
  green. A test that cannot fail is not evidence, and one that passes for a reason other
  than its name is worse than none.
- **Test names are sentences about observable behaviour**, asserted through the public
  interface. The docstring says what would be untrue if the test failed.
- **Fixtures are factories with defaults**, so a test names only what it is about; shared
  declarations live in one support module, and refusal tables are parametrised rather than
  copied. One test per invariant, not per trivial case.
- **Measure; do not assume.** Any claim about a library's semantics, a performance
  effect, or an edge case is settled by a small experiment before it is written down or
  relied on. Record the measurement beside the decision it justified.

## The stance the code is written in

- **Arrows point down only.** A thin core, capability packages around it, one composition
  root that alone sees the whole declaration. Every third-party stack lives behind a seam
  in one package and never leaks its vocabulary upward. New code answers first: *which
  layer does this knowledge belong to?*
- **Declare once, derive everywhere.** Whatever can be computed from the data or the
  schema is never asked for again in configuration.
- **Configuration declares *what*, never *how*.** A knob exists only where a real choice
  does; a parameter with one correct value is code. One grammar for every component, so
  extending the framework is a new class plus a line of declaration, with no edit to
  anything that already works.
- **Fail at construction, by name.** A bad declaration dies while the run is assembled,
  with a message naming the declaration and the fix — never mid-epoch, never by one value
  silently winning over another. Silent fallback is a defect; a substitution says so.
- **Reported values mean what their names say.** A number is read back from what was
  actually done, not from what was intended. Where behaviour and its report can drift
  apart, the report follows the behaviour.
- **Never restate what the language already gives.** A constant duplicating an enum
  member, a conversion the type already performs, a wrapper whose body is a single call —
  each gives one idea a second name, free to drift from the first. Before adding a helper,
  check whether the enum, the dataclass, or the standard library already answers.
- **Abstractions earn their place.** A base class for shared implementation with a
  sensible default; a protocol for an optional capability that has none. A check that can
  never fail is dead code, and a registry for something nobody declares is a closed set
  written the long way.

## Names are the interface

- **The reader is a domain expert without a glossary.** Use the word the field uses, and
  never rename a standard concept for variety. Any mechanism that cannot be extended by
  reading one class is too clever to keep.
- **One concept, one word, everywhere.** The same thing under two names in two packages
  is a defect even when both names are good — and so is one word meaning two things.
- **Names reveal intention, not implementation.** A helper named after the sentence it
  lets its caller read beats one named after what it does inside.
- **Docstrings carry the *why***: the constraint, the measurement, the trade-off, and the
  reason the obvious alternative was rejected. Comments only for what the code cannot
  say; nothing restates the next line. Prose that describes the code is written once the
  code has stopped moving, not alongside it.

## The owner drives

- **Git is the owner's.** Never commit, branch, or stash unasked. Never revert with
  `git checkout` or `git restore`: the working tree holds uncommitted work, and restoring
  from the index destroys it without a word. Undo an experiment with the inverse edit.
- **Dependencies are the owner's.** Never add, remove, or upgrade one.
- **A revert has to be observed.** Apply and undo experimental edits with a script that
  asserts the text actually changed. An edit whose pattern matched nothing reports
  success, and the mutation check then "passes" having tested nothing at all.
- **Report what happened.** Failures verbatim, skipped steps named, and "green" only
  after the run that says so. Work is done when it is verified, not when it is written.
