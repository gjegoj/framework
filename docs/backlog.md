# Backlog

Decisions deliberately deferred, with enough of the reasoning to pick them up
cold. An entry here is not a bug report: it is a design question whose answer was
out of scope when it surfaced, kept so it does not have to be rediscovered.

## Dot-paths into the model should be relative to the model

**Surfaced:** 2026-08-05, designing distillation.

A `freeze` callback names the modules it holds by their dot-path *in the training
module* — `model.backbone`. The leading `model.` is `TrainingModule.model`, which
is Lightning's business rather than the user's, and every path carries it as
noise.

Worse, the path is knowledge about the module tree, so anything that changes that
tree changes every path. Distillation does: it nests the student, and the backbone
becomes `model.student.backbone`. That was handled by deriving the path
(`backbone_path(config)` in `assembly/models.py`) rather than fixing it as a
constant — the guard that refuses `adapters` plus a `freeze` on the same backbone
had silently stopped matching, and a silent guard is worse than none, because the
failure it guards against is a loss that never moves.

Deriving the path works, but it tracks the problem instead of removing it. The
removal is to declare paths **relative to the model that ships**:

```yaml
callbacks:
  - name: freeze
    modules: [backbone]      # not model.backbone
```

`Freeze` would then resolve against `without_teachers(pl_module.model)`, and no
present or future decorator over `Model` could move a path again — the same
answer that made a checkpoint's keys independent of the wrapper
(`shipped_weights`).

**Why it was not done then:** it breaks every config that names a module, and
distillation was not the change that should carry that. It wants its own pass:
the resolution root, the guard, the guides, and a refusal that recognises an old
`model.`-prefixed path and says what to write instead.

**Narrowed, 2026-08-09.** There were two readers walking the module tree, and the
second has been removed rather than taught: `AnnealCriterion` reached
`pl_module.model.criteria`, which only the composite family has, so a run declaring
both `anneal` and `distillation` died at `on_fit_start`. It now asks the model
through `Model.criterion_of(task)` and knows nothing about the tree at all — the
same answer this entry proposes, applied where it cost no config change. `Freeze`
is the one reader left, and it is the one whose paths a user writes.

**Where to look:** `src/callbacks/freeze.py` (`_resolve`),
`src/assembly/models.py` (`backbone_path`,
`_refuse_a_second_owner_of_the_backbone`), `docs/guides/callbacks.md`.

## `logger: none` does not turn the logger off

**Surfaced:** 2026-08-06, porting the reference's console module.

`configs/config.yaml` reads `- logger: none  # swap to clearml`, and the word says
what a reader expects. What happens is different: `build_trainer` only sets a
logger when `config.logger is not None`, so `None` passes nothing to `Trainer`,
Lightning's own default `logger=True` applies, it looks for `tensorboard` and
`tensorboardX`, finds neither, and falls back to `CSVLogger` — announcing the
substitution in a warning nobody reads as being about them.

Measured: a plain CLI run raises

```
Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the
`lightning.pytorch` package ... `logger=True` will use `CSVLogger` as the default
```

and writes `lightning_logs/version_0/metrics.csv` beside the run.

Two ways out, and choosing between them is the deferred part. Passing
`logger=False` makes the word true and stops writing a file nobody asked for.
Keeping the CSV but *saying so* — a `csv` entry in the logger group — makes the
default honest instead. The second is probably right: a run that records nothing
anywhere is a poor default, and a nameless fallback is what makes it feel like
one.

**Where to look:** `src/assembly/training.py` (`build_trainer`),
`configs/logger/none.yaml`, `docs/guides/logging.md`.

## `num_workers: 0` guarantees three warnings on every run

**Surfaced:** 2026-08-06, porting the reference's console module.

`LoaderConfig.num_workers` defaults to `0`, so Lightning's "does not have many
workers which may be a bottleneck" fires once per stage — three lines of the
eight a plain run prints, provoked by our own default rather than by anything
the user did.

The reference's answer was to silence the tip. The honest choices are to keep the
default and keep the tip (a fair trade: `0` is the value that works everywhere —
no spawn cost on a small dataset, no shared-memory limits in a container, no
start-method surprises on macOS), or to pick a default from the machine and let
the tip disappear because it no longer applies.

There are two filter lists in the repo and they differ on purpose. `pyproject.toml`
silences this tip **for tests**, where CPU-only tiny loaders are the deliberate
setup and the tip answers a question nobody asked. `silence_third_party_notices`
in `cli.py` does not, because in a real run the tip describes a default this
framework chose and hiding it would hide the choice. Whoever settles this entry
should keep that difference stated rather than merge the lists.

**Where to look:** `src/config/training.py` (`LoaderConfig`),
`pyproject.toml` (`[tool.pytest.ini_options] filterwarnings`),
`src/cli.py` (`silence_third_party_notices`).

## `rich` is imported but not declared

**Surfaced:** 2026-08-06, porting the reference's console module.

Five modules import `rich` — `progress.py`, `cli.py`, `callbacks/progress.py`,
`export/verification.py`, `data/datamodules/yolo.py`. It is not in `dependencies`.

Measured, it arrives twice over as somebody else's transitive dependency:

```
rich v15.0.0
├── onnxsim v0.7.0
└── typer v0.27.0
    └── transformers v5.14.1
```

Both paths are incidental. Dropping `onnxsim` and a `transformers` release that
stops using `typer` would take the console with them, and the failure would be an
`ImportError` at startup rather than anything a test caught. `pyproject.toml` is
the user's file, so this is recorded rather than fixed.

`cli.py` also imports `yaml` directly. That one is a milder case of the same
thing: PyYAML is required by `hydra-core` through `omegaconf`, and by `peft`,
`clearml`, `timm` and `albumentationsx` besides — declared dependencies every
one, so it cannot quietly disappear the way `rich` can.

## A regression head with several outputs

**Surfaced:** 2026-08-07, designing sample visualization.

`Regression` in the visualization IR holds one number, matching FiftyOne's
`Regression(value, confidence)`. That is not a guess about the future — it is
what this framework can currently produce: `ContinuousObjective` builds either
`squeeze_single_output` or `expectation_over(class_values)`, and both collapse a
head's output to one value per sample before any consumer sees it.

A head predicting several quantities at once (height *and* weight, or a bounding
box's four numbers) would change that. The shape to reach for is already named by
the vocabulary this IR borrows: FiftyOne pairs a singular label with a plural
container — `Classification`/`Classifications`, `Detection`/`Detections`,
`Keypoint`/`Keypoints`. So the extension is `Regressions(regressions: tuple[Regression, ...])`,
not a `RegressionComponent` field inside `Regression`. The reference took the
second road and had to invent `dim0`/`dim1` naming for components; the plural
container needs none, because each `Regression` can carry its own name.

Three places move together when it lands: `ContinuousObjective.out_features` and
its activation (the model has to emit and keep N numbers), the metric side (a
vector metric per component), and `RegressionAnnotator` plus one
`render_label` registration.

**Where to look:** `src/tasks/objectives.py` (`ContinuousObjective`),
`src/visualization/entities.py`, `src/visualization/annotators.py`.

## No text input loader ships, though text inputs are supported

**Surfaced:** 2026-08-07, designing sample visualization.

`HFTextBackbone` consumes `input_ids` and an optional attention mask, and
`MultiEncoderBackbone` runs an image encoder beside a text one — so a
CLIP-style multimodal run is a supported model shape. But
`input_loader_registry` holds exactly one entry, `image`. A text run therefore
needs a custom loader supplied through `_target_`, which works (it is the
documented escape hatch) but means the framework ships half of a supported
combination.

A `text` loader would tokenize a cell into `input_ids` and a mask — which raises
the question the registry cannot answer alone: a tokenizer is a *fitted* thing
tied to the backbone's hub id, so a loader declared independently of the model
can silently disagree with it. That coupling is the design problem, and it is
why this is a backlog entry rather than a small addition.

The visualization side does not wait for it: the grid reads the raw table cell,
so whatever a text loader eventually produces, the human-readable text is
already what gets drawn.

**Where to look:** `src/data/loaders.py`,
`src/data/registry.py`, `src/models/backbones/hf.py`.

## A dense task with continuous targets has no label to draw

**Surfaced:** 2026-08-07, splitting annotation on the task axes.

`(OutputTopology.DENSE, Objective.CONTINUOUS)` is a pairing the framework supports —
`DenseTopology.supports` admits every objective but `METRIC`, and depth
estimation, heatmap regression and density maps all land there. The
visualization IR has no `Label` for it: `Segmentation` holds boolean masks per
class, and a field of real numbers is neither.

`DenseAnnotation.draws(objective)` therefore refuses the pairing at build time
and the task is skipped with its reason, rather than drawn as something it is
not. That refusal is the whole of the deferral: nothing is half-rendered, and
the log names the task.

What it wants is a `Heatmap(values: np.ndarray, low: float, high: float)` label
beside `Segmentation`, a colormap to encode it (the golden-angle palette is for
identity, and a continuous field wants a perceptually ordered ramp — viridis or
magma, sampled into a lookup table, no matplotlib dependency), and one
`render_label` registration reusing `png_data_uri`. The per-sample range matters:
normalising each sample to its own min/max makes cells incomparable, so the range
should come from the batch, and that is the decision worth thinking about rather
than guessing now.

**Where to look:** `src/visualization/entities.py`,
`src/visualization/annotators.py` (`DenseAnnotation.draws`),
`src/visualization/html.py` (`render_label`),
`src/tasks/topologies.py` (`DenseTopology.supports`).

## Who owns what a sample looks like

**Surfaced:** 2026-08-07, reviewing the samples grid. **Re-measured and
narrowed:** 2026-08-08, after reading the transform it accused.

Drawing a sample needs two facts: which inputs are pictures, and how to give
their pixels back their original look. The grid answers both itself —

```python
def _is_picture(tensor): return tensor.ndim == 4 and tensor.is_floating_point()
images = (images * std + mean).clamp(0.0, 1.0)     # mean/std declared on the callback
```

— and both answers already exist elsewhere.

**"Which inputs are pictures" is declared twice.** `AlbumentationsTransform`
takes `inputs` — each name beside its `Geometry`, derived from the loaders — and
registers exactly those as albumentations targets of that kind. The grid ignores
that mapping and sniffs the tensor
instead, so the two can disagree in both directions: a precomputed 4-D float
feature map is drawn as a photograph, and an image input that stops being
`[B, C, H, W]` float after the pipeline is not drawn at all.

**Normalisation is declared twice.** `albumentations.Normalize(mean: ${mean})` in
the transforms section, and `mean: ${mean}` on the callback. They agree today only
because both point at one root key. Literal numbers in `Normalize`, a second
`Normalize`, or a different transform library breaks the tie silently: the page
mis-colours and nothing says so.

**A claim from the first version of this entry was wrong and is withdrawn.** It
said a CLIP-style run with per-input normalisation already draws its second
picture in the wrong colours. It does not: `AlbumentationsTransform` runs *one*
pipeline over every `inputs` entry with shared sampled parameters, so one
`Normalize` covers them all and one mean/std pair is correct. Reaching the
divergence needs a custom `SampleTransform` supplied through `_target_`. The
callback's warning about several picture inputs stays useful, but the defect is
reachable, not active.

**What is active:** only a *linear* normalisation is invertible by that formula.
Any other pixel-changing operation — CLAHE, posterize, a channel reorder — is not
undone, and the page shows something an eye cannot map back to the source.
Silently.

### Options, with the cost measured

| | Shape | Cost | What it leaves broken |
|---|---|---|---|
| 1 | The grid keeps guessing (today) | none | silently wrong outside the shipped pipeline; every new kind of input widens the guess |
| 2 | The sample carries display pixels beside the model tensor | **+25% per batch** — 19.3 MB → 24.1 MB for 32 images at 224px — paid on every batch of every epoch, through the worker boundary, whether or not the grid is enabled | pays always for a page looked at every N epochs |
| 3 | A transform can be asked how to make an input showable again | a port member on transforms; only the step that normalised can answer | a five-operation pipeline with one invertible step has to say "I cannot", which is at least honest |

Option 2 also changes *what is shown*: the augmented-but-not-normalised image
rather than the normalised one turned back. For a linear normalise those are the
same up to clamping; for a non-linear pipeline, 2 is right and today is wrong.

### Why it is parked

The only active defect needs a configuration this repository does not yet
contain. Option 2 asks 25% of every batch for a page read every N epochs; option 3
asks for a decision about what a mixed pipeline answers. Neither is worth
committing to against a hypothesis.

**The cheap half, if it is ever wanted on its own:** have the grid read the
`Geometry.IMAGE` entries a transform was handed, instead of sniffing shapes. That
removes one of the two double declarations, costs nothing and touches no data-layer
code.

**Pick it up when** a run ships a non-linear preprocessing step, or a custom
`SampleTransform` handles inputs separately. Then there is something to measure
against instead of a guess.

**Where to look:** `src/callbacks/samples.py` (`_is_picture`,
`_to_uint8`, `_warn_once_about_shared_normalisation`),
`src/transforms/albumentations.py` (`inputs`, `Geometry`),
`src/data/schema.py`, `src/data/loaders.py`.

## `HtmlLogger` will want to be an artifact port when there is a second format

**Surfaced:** 2026-08-07, reviewing the samples grid.

`core/ports.py` now has six artifact ports — `MatrixLogger`, `CurveLogger`,
`BarsLogger`, `BoxPlotLogger`, `SingleValueLogger`, `HtmlLogger` — and each names
one thing a tracker can be asked to show. The last two of those were added
deliberately rather than by drift: a class balance and a box plot are each another
*kind of picture*, carrying a typed entity the backend draws, so they belong with
the first two and keep their payloads typed — which a media-typed port could not.
The reference tried the other road here and it is instructive: its `PlotLogger`
took a growing union (`type Plot = BoxPlot`), so a new plot type changed the
port's own type and every backend's translation table. That is fine while each has one caller, and a MIME-agnostic
`ArtifactLogger(name, payload, media_type, iteration)` today would be a general
mechanism built for one case.

The moment it stops being fine is the second page-shaped artifact: a PNG export
of a grid, a JSON dump for an external viewer, a static report. At that point the
ports stop naming *kinds of picture* and start naming *file formats*, which is
the signal to collapse them into one port that takes a media type — and to give
ClearML's side one `report_media` call instead of a method per format.

### Resolved, 2026-08-09: the ports stay, and the consumer was the duplication

Everything above stands, and the entry's own last paragraph turned out to name the
whole of the problem: the six ports were fine, and it was the six *narrowings* that
were written out by hand, with three different answers to "and if no backend can?"
— a debug line, a silent skip, and a warn-once. Two of them also narrowed
`trainer.logger`, which is the *first* configured backend rather than all of them,
so a run with two trackers filled one and silently left the other empty.

All six now read alike — `for drawer in (one for one in loggers if isinstance(one,
CurveLogger)):` — and every one of them reads `trainer.loggers`. That is one line
where the branch-and-else was four, so no helper was needed; one was written and
then cut, because under `mypy --strict` a runtime-checkable Protocol cannot be
passed where `type[T]` is expected and it would have cost six `# type: ignore`
comments at exactly the sites being cleaned.

**The ports themselves are not to be collapsed on the count alone.** They are role
interfaces, each carrying the typed entity a backend draws, and a media-typed
`ArtifactLogger` loses that check. The trigger stated above is still the right one:
the *second page-shaped artifact*, where the ports would start naming file formats
rather than kinds of picture. Until then, nothing to do.

**Where to look:** `src/core/ports.py`, `src/core/reporting.py`,
`src/loggers/clearml.py`.

## `ultralytics` is a hard dependency, and it is AGPL-3.0

**Surfaced:** 2026-08-08, designing detection.

`ultralytics>=8.4.115` sits in `dependencies`, not in an optional extra. Every
install of this framework therefore takes an AGPL-3.0 dependency, including the
installs that only ever train a classifier — and the AGPL's obligations attach to
distribution and to network use, which is exactly what a served model is.

Nothing in the code is wrong. This is a licensing decision that has so far been
made by omission, and it has two candidate answers. Either detection is declared
an optional extra (`pip install ml-framework[detection]`), which keeps the default
install permissive and makes the vendor family's import failure a clear message
rather than a missing name; or the project accepts AGPL for everything, which is a
legitimate choice and should be written down as one.

The design that surfaced it is neutral: the vendor family is recognised by name in
two functions and imports ultralytics lazily either way, so moving it to an extra
later costs a dependency-group edit and one import guard.

**Where to look:** `pyproject.toml` (`dependencies`),
`src/data/datamodules/yolo.py`, and the model module the detection
design adds beside it.

## A vendor family builds `-seg` and `-pose` networks that nothing downstream can read

**Surfaced:** 2026-08-09, writing the detection guide.

`YoloModel` never branches on what kind of network it is building: `YOLO(name)`
picks `DetectionModel`, `SegmentationModel` or `PoseModel` from the file, and the
head is rebuilt at the dataset's class count either way. So a `-seg` or `-pose`
architecture already *trains* — its loss parts arrive scoped by the task and the
run reports them.

What it cannot do is say anything about the result. Three pieces are missing, and
they are the same three for both kinds:

- the currency. `Instances` carries `boxes`, `labels`, `sample_index` and
  `scores`. Masks and keypoints are a fourth and fifth column, and whether they
  belong on the same entity or on a sibling is the design question — a mask per
  instance is large, and a keypoint set has its own arity.
- the metric. `map` compares boxes. torchmetrics computes a mask-IoU mAP from the
  same class, and pose has no equivalent in the registry at all.
- the annotator. `Topology.INSTANCES` has no `AnnotationTopology`, so the samples
  grid names a detection task as undrawable today, whichever kind it is.

**Why it was not done then:** detection was the scope, and each of the three is a
decision about a shape rather than a line of plumbing. Guessing them from the
vendor's side would fix the shape before a second consumer exists to argue with it.

**Where to look:** `src/core/entities.py` (`Instances`),
`src/metrics/detection.py`,
`src/visualization/annotators.py`, `docs/guides/detection.md`.

## `configs/` lives outside the package, so only an editable install can find it

**Surfaced:** 2026-08-09, fixing the `ml-train` entry point.

Hydra resolves a *relative* `config_path` against `task_function.__module__`,
which the `ml-train` console script leaves as `src.cli` — so
`../../configs` became the import path `configs` and every console invocation died
with "Primary config module 'configs' not found", while `python -m
src.cli` worked. That is fixed: `cli.py` now computes an absolute path
from `__file__`, which `compute_search_path_dir` returns verbatim.

The absolute path is the repository's `configs/`, two levels above the package.
That is right for the editable install `make install` produces and wrong for a
wheel, where the directory is not shipped at all — a `pip install ml-framework`
would resolve to a path that does not exist on the target machine.

Two candidate answers. Either `configs/` moves inside the package
(`src/configs/`) and is declared as package data, which makes the
shipped groups importable anywhere and turns a user's own `configs/` into a
Hydra search-path addition; or the CLI grows a `--config-dir` of its own and the
shipped groups stay a repository convenience. The first is the usual answer for a
framework, and it is the larger change: every `defaults:` path, the tests that
compose configs, and the guides all name the directory.

**Why it was not done then:** the work was a documentation pass, and packaging is
`pyproject.toml`'s business, which this project keeps in the user's hands.

**Where to look:** `src/cli.py` (`CONFIG_DIRECTORY`),
`pyproject.toml` (`[tool.hatch.build]`), `configs/`.

## Export assumes every model input is an image of one shape

**Surfaced:** 2026-08-09, auditing the export phase.

`example_inputs` builds one `torch.randn(batch, len(config.mean), *config.image_size)`
per entry in `data.inputs`. For a vision run that is exactly right, and deliberately so:
the shape comes from the two config fields the transform pipeline hands to `Resize` and
`Normalize`, so it is what the model receives by construction rather than by a guess —
which is what lets export run from a checkpoint with no dataset at all.

It is wrong for a run with more than one *kind* of input. `MultiEncoderBackbone` runs an
image encoder beside a text one, and `HFTextBackbone` consumes `input_ids`; a CLIP-style
run is a supported model shape whose export cannot work, because the second input would
be handed a picture-shaped float tensor.

Nothing is silent about it: `_prove_the_example_fits` runs the graph once before any
exporter touches it and turns the resulting torch error into a sentence naming
`image_size`, the channel count, and the fact that a non-image input cannot be exported
yet. So this is a documented limit rather than a defect — but the backlog recorded the
missing *text loader* and not this, and the two are the same gap seen from opposite ends.

What it wants is for an input to be able to say what shape it takes, which is the same
question `data.inputs` already half answers by naming a loader. The natural home is
beside the loader: something that can produce one example value of the right shape and
dtype without reading a file. That also makes the tokenizer coupling in the text-loader
entry unavoidable rather than deferrable — an `input_ids` example needs a vocabulary size
— so the two entries should be picked up together.

**Where to look:** `src/assembly/export.py` (`example_inputs`,
`_prove_the_example_fits`), `src/data/loaders.py`, `src/models/backbones/hf.py`.

## Splitting the model section into `backbone:` and `model:`

**Surfaced:** 2026-08-09, reviewing the vendor-family seam. **Considered and declined.**

One key, `model:`, chooses between two registries: a backbone this framework composes
heads onto, or a family that arrives whole. Which one it is decides the model, the data
pipeline, and a whole table of refusals — and it is decided by a runtime lookup rather
than by the schema. The proposal was to split it: `backbone:` and `model:`, exactly one
of the two, so `is_vendor_family` stops being a function and becomes `config.model is not
None`, checked by pydantic at load.

It is a real improvement to the YAML and it was declined for four reasons.

**It reverses a stated decision.** `docs/concepts.md` says the model section is *"a plain
component: one shape for every model family, with the family following from the name
rather than a switch field."* That is argued, not accidental.

**"Exactly one of two keys" is a switch field wearing different clothes.** It needs a
`model_validator` to enforce the exclusivity, which is the construct the design rejected.

**The Hydra group does not split with it.** `configs/model/` holds `resnet18`, `unet`,
`dpt_dinov3` *and* `yolov8n`, and `override /model: unet` is a public interface. Keeping
one group whose files write into two different schema keys buys the split at the cost of
a new indirection between the group's name and the key it fills.

**And most of the benefit was available without it.** What actually reached a user was
the error on a misspelling: `name: yolov8` fell through to the backbone registry and was
answered with a list of backbones, from a guide that had just taught `model: {name:
yolo}`. That is now `_refuse_a_name_from_neither_registry` in `assembly/models.py` —
fifteen lines, naming both groups and what distinguishes them, and no schema change. The
remaining benefit of the split is that the YAML is self-describing *before* you get it
wrong, which is worth less than it sounds when getting it wrong is answered well.

**Pick it up when** a second vendor family exists **and** the ambiguity is reported by
someone who hit it despite the refusal. Until both hold, the cost is a breaking change to
the most-used key in every config for a confusion nobody is stuck in.

**Where to look:** `src/config/experiment.py` (`model`), `src/assembly/vendor.py`
(`is_vendor_family`), `src/assembly/models.py` (`_refuse_a_name_from_neither_registry`),
`configs/model/`.

## Visualization modules that will want splitting, and when

**Surfaced:** 2026-08-14, during the renderer rework
(docs/superpowers/specs/2026-08-14-visualization-rework-design.md).

Two splits were designed and deliberately not made, because a module boundary
costs reader attention today and the growth that would repay it has not
happened:

- `annotators.py` splits along its own two axes — objectives (how an
  `Objective` reads tensors) and topologies (how an `OutputTopology` draws readings) —
  **when detection annotations land** and push the file past comfortable
  reading. The seam is already clean: the two ABCs, two registries, and their
  implementations interleave nothing.
- Page chrome (sidebar tree, filters, sliders) leaves `html.py` for a
  `controls.py` **if the chrome itself grows**; new label kinds do not touch
  it, so no current roadmap item triggers this.

## The distribution-reporter registry's home

**Surfaced:** 2026-08-14, while retiring `singledispatch` from
`callbacks/dataset_summary.py`
(docs/superpowers/specs/2026-08-14-dataset-summary-reporters-design.md).

`distribution_reporter_registry` is declared in `dataset_summary.py` itself:
`callbacks/registry.py` imports that module to catalogue the callback for config,
so the shared home would be an import cycle — and the registry's producer and
consumer are both that one module. **A third distribution kind, or a second
module consuming reporters, moves the ABC, the registry, and the reporter
classes to `src/callbacks/reporters.py` unchanged.**

## The first future cell of the task grid

**Surfaced:** 2026-08-16, while splitting the task axes
(docs/superpowers/specs/2026-08-16-input-output-topology-design.md).

Supervised multiview (GLOBAL × MULTIVIEW × MULTICLASS — N photos of one item,
one label) is the first currently-refused cell worth serving. It lands as one
`GlobalTopology.supports` change plus a criterion that reads the stacked
carrier; no axis reform.

## A depth encoder makes the dense default objective-aware

**Surfaced:** 2026-08-17, while giving detection annotations a place in the
table grammar (docs/superpowers/specs/2026-08-17-detection-data-design.md).

`DenseTopology.default_target_encoder` is a flat `"mask"`, which is right for
every dense cell that exists today — a mask file, whatever its objective. The
dense × continuous cell (depth) has no built-in encoder, so a task that omits
the declaration there reaches the mask encoder's refusal rather than one written
for it. **When a depth encoder exists**, the dense default becomes a joint
decision of both axes; the seam is `default_target_encoder(output_topology,
objective)` in `src/tasks/builder.py`, which already composes the two voices.

## A second boxes target in one pipeline

**Surfaced:** 2026-08-17, measuring albumentationsx 2.3.7 for the same spec.

Two `Geometry.BOXES` targets are refused at construction, naming both: measured,
the library does not plumb `label_fields` through `additional_targets`, so a
second boxes field's class names would not be filtered with its boxes — a crop
would desynchronise them silently. **If two detection tasks over one image ever
become real**, the seam grows per-target label plumbing (a label field per boxes
target, and the pairs repacked by name) in `AlbumentationsTransform`; the
refusal marks the spot.
