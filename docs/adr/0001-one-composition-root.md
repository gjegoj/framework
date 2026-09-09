---
status: accepted
date: 2026-09-07
---

# One composition root; a package builds its own section

`src/assembly/` had grown into twelve modules (~1 470 lines) holding wiring and domain
rules alike — which encoder a task gets, how losses add up, how a scalar `max_lr` spreads
over parameter groups, how a checkpoint is unwrapped, how export examples are shaped — because
the rule "only the composition root reads config" had drifted into "everything that touches
config lives in assembly". We replace the package with one module, `src/build.py`, that
only wires, and we let each capability package read *its own* section of the config to
build its own part. The rules move to the packages that own the knowledge.

## Decisions

1. **`src/build.py` is the composition root.** `build(config) -> Experiment` and
   `run(experiment, config)`, readable top to bottom; `Experiment` lives there. It is the
   only reader of a whole `ExperimentConfig`. The `assembly/` package is deleted.

2. **`config/` is a leaf; a package reads only its own section.** `config/` depends on
   `core/` alone, so any package may import it without an arrow pointing up. A capability
   package exposes one `build_*` function that takes its section and returns its part;
   it never reads another package's section or the root. Where the knowledge lives:

   | Rule | Home |
   |---|---|
   | sources, schema, split, per-stage transforms, table formats | `data/build.py`, `data/sources.py` |
   | the declared target encoder, or the kind's default, handed what its base takes | `data/build.py` |
   | one criterion, or several added with their weights; `sized` dispatch | `losses/build.py` |
   | one `MetricSet` per stage from declarations and the kind's `metric_kwargs` | `metrics/build.py` |
   | optimizer and scheduler factories, fit-time facts, per-group rates | `training/build.py` |
   | putting weights into a model, grafting beneath adapters | `models/checkpoints.py` |
   | export → verify → report → refuse drift | `export/ship.py` |
   | model composition, adapters, teachers, tasks and heads, trainer, callbacks, checkpoint prefixes, export examples | `build.py` |

3. **`Registry` stays as the extension mechanism.** The audit's P1.1 (registries → dicts)
   is withdrawn: a class that names itself beside its definition, a lookup that lists what
   is registered and a duplicate refused at import are worth their 89 lines; a dict would
   trade a small readable mechanism for one fewer concept. `instantiate(component, registry)`
   keeps its registry argument.

4. **`_target_` is resolved by `hydra.utils.get_object`**, deliberately — no resolver of
   our own. `instantiate`, `resolve_target`, `resolve_params`, `refuse_a_declared_fact` and
   `fill_signature` move to `src/config/instantiate.py`, beside the grammar they interpret.
   Hydra's *composition* stays confined to `cli.py`; its import utility is used in that one
   module.

5. **`classes` has one spelling: on the task.** A target encoder whose base is
   `VocabularyTargetEncoder` receives the task's `classes` as an explicit constructor argument;
   any other encoder refuses a declared vocabulary by name; `classes` written inside the
   `target_encoder` declaration is refused by name, as a head's sizes are. The second
   spelling, and the branch that reconciled the two, go.

6. **Checkpoints split by knowledge.** `models/checkpoints.py` puts weights into a model
   and grafts beneath adapters; which prefixes a checkpoint file carries
   (`TrainingModule.MODEL`, `DistilledModel.STUDENT`) is wiring and stays in `build.py`.
   `training/` gains no arrow to `models/`.

7. **The vendor path rides along until P0.3a**: one fenced block at the bottom of
   `build.py`, marked as going with that stage, rather than a second composition module.

## Considered options

- *Only `build.py` reads config, packages take plain arguments.* Honest cost: the ~250
  lines of column, source and split wiring stay in the root, which becomes a ~650-line
  module — the god package turned into a god module. A hybrid (strict everywhere but
  `data`) is a rule with an exception, which is not a rule.
- *`TaskKind.encoder(...)` builds the encoder* (the audit's sketch). It would move the
  vocabulary rules into `tasks/`, where `VocabularyTargetEncoder` does not live. The kind keeps
  saying *which* encoding its loss needs; `data/` keeps saying what an encoder takes.
- *`training/checkpoints.py`.* Would add the arrow `training → models`; splitting by
  knowledge needs none.

## Consequences

- `build.py` is ~650 lines with the vendor block (measured after the move, docstrings included), ~500
  once P0.3a removes it — more than the ~300 first estimated, because the model, task, trainer and
  export wiring is genuinely wiring and keeps its docstrings.
- About 200 tests under `tests/unit/assembly/` move to their subjects (audit §24), with
  plain `mv` — the owner's commit records the renames. The root's own tests live in
  `tests/unit/test_build/`, not `tests/unit/build/`: pytest's default `norecursedirs`
  skips a directory named `build` without a word (measured: 0 of 95 collected). Two tests pinning the second
  `classes` spelling become one that pins its refusal.
- CLAUDE.md's "one composition root (`cli.py + assembly/`) that alone reads config" is
  reworded to this ADR's rule 2; `docs/concepts.md`, `docs/guides/extending.md` and
  README get the same minimal, truthful edit.
