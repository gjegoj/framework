---
status: accepted
date: 2026-09-06
---

# Derived facts are passed explicitly, never injected by signature

A component that needed a derived fact — a class count, a vocabulary, the transform
geometry, the run's tasks, the model's architecture — used to say so through `named_by`:
`instantiate` matched the fact into any constructor parameter of that name, with
precedence rules for a value written in config as well, and compensating refusals for
the cases the matching got wrong. A constructor's parameter names had become a hidden
protocol; reading a class did not tell what it would receive. We drop the injection.
A fact is passed as an ordinary argument by the caller that knows it, and a class says
what it takes in a way a reader can see.

## Decisions

1. **`instantiate(component, registry)` builds from the declaration alone.** Nothing
   travels beside it; `refuse_a_declared_fact` is where "declare once" is enforced — a
   config carrying a fact the caller derives dies naming the declaration and the fact.
2. **A class says what it takes.** Encoders by marker base (`VocabularyTargetEncoder` takes
   `classes`, `FileTargetEncoder` takes a cache through `use_cache`), passed by
   `issubclass`; criteria of ours through a `sized(facts, ...)` classmethod where the class
   declares it, the bare constructor otherwise; transforms through `with_geometry` on the
   `GeometryAware` port; a logger through `TagsRuns.tag_run(architecture)` after
   construction.
3. **Callbacks read the module.** `TrainingModule.tasks` is public; a callback that needs
   the run's tasks reads them in `setup`, Lightning's own mechanism, so there is one
   declaration of the tasks and no copy to disagree with it.
4. **One bounded exception, for constructors we do not own.** `fill_signature(factory,
   **facts)` hands a torchmetrics metric or a torch scheduler only the facts its signature
   names, and never reaches a constructor that forwards `**kwargs`. The rule "the
   exception is for constructors we do not own" does not survive a third place.

## Consequences

- Deleted: `named_by`, `instantiate`'s `**derived`, the precedence docstrings and the two
  compensating refusals.
- A model that arrives whole declares its own sizes (`num_classes: 3`); no fact is
  matched into its signature (ADR-0002).
- A custom callback that relied on being handed the run's facts reads them from the
  module instead.
