---
status: accepted
date: 2026-09-06
---

# A task is a kind, not a point in an axis space

A task used to be declared as a *composition* — `output topology × input topology ×
objective × modality`, with `preset` names over the familiar combinations — and served
by an algebra spread over `tasks/objectives.py`, `tasks/topologies.py`, `tasks/builder.py`,
`config/presets.py`, two registries and a legality matrix saying which points of the
space could be built. Adding a capability meant touching every axis, and no single class
told a reader what a segmentation task needs. We replace the algebra with **kinds**: one
class per familiar kind of task in `tasks/kinds.py`, registered on `task_kind_registry`
under the name config spells (`kind: segmentation`), stating in plain methods what the
task needs — encoder, head, loss, activation, target adapter, default metrics, output
shape, whether a batch transform may mix it, and how it is drawn.

## Decisions

1. **`kind` replaces `preset` and the explicit axes.** A declaration names a kind, or
   reaches one by `_target_` (`kind: {_target_: my_pkg.Depth}`); kinds take no constructor
   arguments. `Modality` stays as a fact of the kind; `OutputTopology` survives only as
   `kind.shape`, what readers that branch on the structure of one prediction ask.
2. **Sharing is inheritance, written in the class.** Multilabel segmentation is dense
   segmentation with another loss, and says so by subclassing — not by a matrix row.
3. **A kind builds its parts from the task's facts, never from config.** Sizes come from
   what the data revealed (`TaskFacts`); the declaration may put *overrides* on the kind
   (a head, streams, a loss), which the kind resolves.
4. **A kind declares its own drawing** and imports `visualization/` entities to do it;
   `visualization/` stays a leaf.

## Consequences

- Eleven shipped kinds, one file; extension is a subclass plus a config line.
- `Task` is a plain value — name, kind, facts, weight, learning rate — and the training
  module, the batch transforms and the sample grid ask the kind rather than re-deriving
  legality.
- Detection is a kind with an encoder, default metrics and `mixable = False`, and refuses
  to build a composed model by name until its criterion and decoder land (P0.3b).
- The audit's P0.1 records the migration; the concepts guide and `CONTEXT.md` carry the
  vocabulary.
