# Internals

The framework assembles an experiment in a fixed order — the same flat `build_*` sequence
that `main.py` runs. The diagrams below show the top-level blocks and how they connect; each
diagram after the first zooms into one block.

## The assembly pipeline

```mermaid
flowchart TB
    yaml["configs/ — Hydra groups"] -->|"compose + validate"| cfg["ExperimentConfig"]
    cfg --> data["① Data<br/>build_bindings → build_data_module → setup()"]
    data -->|"setup() fits the target encoders"| rt[("RuntimeContext<br/>num_classes")]
    rt --> tasks["② Tasks<br/>build_tasks()"]
    cfg --> tasks
    tasks --> model["③ Model<br/>build_backbone → build_composite_model"]
    cfg --> model
    model --> wrap["④ Lightning wrappers + optimizer / scheduler<br/>logger / callbacks / trainer"]
    tasks --> wrap
    cfg --> wrap
    wrap --> run["⑤ run_experiment<br/>fit → test → export  (each gated by a run_* flag)"]
```

`setup()` is the hinge: it fits the target encoders and writes `num_classes` into the
`RuntimeContext`, so tasks and model heads — built *next* — always receive concrete output
sizes. Nothing is hardcoded or shared by reference across config, data, model, and module.

## ① Building the data

`build_data_module` then `setup()`: read the annotation table, fit the encoders (which
populate `num_classes`), and split into one `Dataset` per stage.

```mermaid
flowchart LR
    cfg["config.data"] --> enc["TargetEncoder per task<br/>(build_bindings)"]
    cfg --> read["DataSource.read()<br/>CSV / JSON → DataFrame"]
    read --> fit["fit encoders on the column values"]
    enc --> fit
    fit -->|"num_classes"| rt[("RuntimeContext")]
    read --> split["split → Dataset per stage<br/>train / val / test"]
```

## ② Composing a task — the three-axis Bridge

A preset fixes a `(topology, default objective)` point; `TaskBuilder` validates the pair
(e.g. `metric` never pairs with DENSE — on GLOBAL it works via proxy classification
(`arcface_embedding`)) and assembles the bricks into one `Task`.

```mermaid
flowchart TB
    preset["Preset<br/>classification · segmentation · triplet · contrastive · …"] --> topo["Topology<br/>GLOBAL · DENSE · MULTIVIEW · MULTISTREAM"]
    preset --> obj["Objective<br/>multiclass · binary · multilabel · continuous · metric"]
    topo --> tb["TaskBuilder — Bridge<br/>validates the topology × objective pair"]
    obj --> tb
    rt[("RuntimeContext<br/>num_classes")] --> tb
    tb --> task["Task<br/>head_spec · adapter · criterion · activation · metrics"]
```

## ③ Assembling the model

One shared backbone plus one head per task; each head is sized from the backbone's feature
dimension for the stream it reads — so output sizes follow from data, never from config.

```mermaid
flowchart LR
    bb["build_backbone<br/>timm · smp · embedding · multi"] --> cm["build_composite_model"]
    spec["HeadSpec per task<br/>(Task.head_spec)"] --> cm
    cm -->|"each head sized from<br/>backbone.feature_dim(feature_key)"| model["CompositeModel<br/>shared backbone + one head per task"]
```

## ④ A training step

`forward` runs the backbone once and routes its features to each head; then, per task, the
criterion takes logits while the activation feeds metrics; the aggregator sums the weighted
task losses into the scalar that is back-propagated.

```mermaid
flowchart TB
    batch["Batch — inputs + targets"] --> fwd["CompositeModel.forward<br/>backbone → per-task heads"]
    fwd --> logits["task_logits (per task)"]
    logits --> crit["Criterion(logits, target)<br/>→ LossResult"]
    logits --> met["Activation → preds<br/>→ MetricSet.update"]
    crit --> agg["WeightedSumAggregator<br/>Σ weightᵢ · lossᵢ"]
    agg --> back["total loss → backward()"]
```

At epoch end each `MetricSet` is computed, logged as `task/metric/stage` (typed handlers
route scalars / per-class vectors / confusion matrices / curves), then reset.
