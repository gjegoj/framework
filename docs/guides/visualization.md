# The samples grid

Every N epochs one batch becomes a self-contained HTML page in the tracker:
each sample drawn with what was true and what was predicted over it, a tree of
switches for every field, and the filters that answer *show me the mistakes*.

It draws the forward that already happened, so the page shows the exact batch that
trained at the exact weights that trained — no second inference to pay for, and no
way to draw something the run never computed.

Nothing is kept to make that work. A step **returns** what it produced —
`{"loss": ..., "preview": StepPreview(outputs, targets)}` — and Lightning hands a
step's return value to every `on_*_batch_end` hook, per batch. The callback reads
its own argument: no state on the module, nothing to invalidate, nothing that can
go stale. The preview is detached, so no autograd graph, no loss and none of the
backbone's feature streams come along — returning the `StepResult` itself would
carry all three, and 352 MB of outputs alone on a `[16, 21, 512, 512]`
segmentation batch.

**A preview is built only when something asked for it.** Detaching shares storage
with the activated outputs, and Lightning keeps a step's return value alive through
the optimizer step — so a preview nobody reads still pins those outputs across
`backward()` and into the moment the optimizer allocates its own state. Lightning
runs a callback's `on_*_batch_start` before the step, which is early enough for the
grid to say whether the batch about to run is one it will draw:

```python
class AwaitsPreview(Protocol):     # core/ports.py
    @property
    def awaiting_preview(self) -> bool: ...
```

So a run with no such callback builds no previews at all, and a run with one builds
them on the batches that become pages — with the shipped defaults, one batch every
fifth epoch instead of every step of every stage. Writing your own consumer means
implementing that property and answering it from a batch-start hook; a module
called directly, outside a `Trainer`, is handed everything, because there is nobody
to ask and nothing to save.

A module of your own that does not return a preview draws nothing, and says so
once, naming what it returned instead. That is the one thing no amount of
assembly-time checking can catch: only a step can show what a step returns.

Nothing is drawn during Lightning's sanity check. It runs a validation batch
before a single optimizer step, and its page would land under the same title and
iteration as the first real epoch's — two artifacts, and no way to tell which one
a tracker shows.

## Turning it on

```bash
uv run main.py callbacks=samples logger=clearml
```

That group is the shipped defaults written out:

```yaml
callbacks:
  - name: samples
    every_n_epochs: 5
    stages: [train, val, test]   # every stage; narrow it to skip the ones you don't read
    num_images: 8
    mean: "${mean}"              # the run's own normalisation, to undo it
    std: "${std}"
```

`mean` and `std` default to ImageNet's statistics — the same ones the root config
normalises by, named once in `core/normalisation.py` so the pair cannot drift
apart. Pass them anyway, as the group does: the default is a starting point, not
an assumption, and `"${mean}"` is the same interpolation
`configs/transforms/*.yaml` uses for the very same value, so a run that
normalises differently changes one number at the root and both sides follow. A
grid denormalising by numbers the transforms did not use draws a picture that is
wrong in a way that looks like a model problem.

Every stage draws by default. Val and test are the sample-stable ones — no
augmentation, so the same pictures reappear each epoch and what changes is the
model. Train shows the pixels the model actually trained on, which is the only
way to see a transform that is wrong; note that after a batch transform (MixUp,
Mosaic) the pixels are a blend while the source pill still names the original
file, because the batch's metadata is not rewritten along with them.

One caveat, and the grid says it out loud when it applies: one `mean`/`std` pair
cannot undo two different normalisations. A run with per-input transforms and
more than one image input gets a warning naming them, because wrong colours look
like a model problem.

Values that could only ever draw nothing are refused when the callback is built,
not an epoch later: `num_images` and `every_n_epochs` below 1, a negative
`batch_index`, a `threshold` outside `[0, 1]`, a `std` of a different length than
`mean`.

`stages` defaults to `[val]`: validation batches carry no augmentation, so the
same samples come back epoch after epoch and drift is visible. `batch_index`
defaults to `0` for the same reason.

## What a cell shows

| Part | What it is |
|---|---|
| the picture | every image input of the sample, denormalised |
| a caption strip | every readable text input, taken from the table row |
| filled chips | ground truth |
| outlined chips | predictions, their fill proportional to confidence |
| masks | segmentation classes, ground truth and prediction in one colour per class |
| a pill, top right | the row's own cell — a URL opens, a local path copies |
| a note, top left | the numbers the sample scored (`iou 0.62`, `mae 1.2`) |
| a badge, bottom right | `✓ correct`, `✗ wrong`, or `✗ 1/2 matched` where only one of the sample's tasks missed |

Chips sit on two named rows — `GT` above, `PRED` below — so the two are told
apart by more than filled-versus-outlined, which is easy to lose at 11px on the
same class in the same colour.

A cell that missed is outlined in red and says so in words; a cell where every
task matched is outlined in green and says that. Beside the page title,
`10 samples · 3 with mistakes · 2 correct` says what is there before any filter
narrows it, and each filter option carries the count it would show.

Pictures and masks are downscaled to `max_side` (256 by default) before being
inlined. That bounds what the page weighs rather than what the tensor holds:
measured, eight cells of a ten-class segmentation inlined whole at 512px make a
**49 MB page in 13 seconds** — every N epochs, into a tracker that then has to
embed it. Cells display at about 230px, so the default costs nothing there and
only softens the lightbox. Raise it and pay for it, or set it to `null` to inline
whole.

Where a predicted mask and a true mask of the same class overlap, the two
translucent fills stack and darken. That is IoU by eye, before any number is read.

Click a cell to open it large; arrows and `Esc` work there, and chips show their
full text.

## Finding the bad samples

Two controls, and they combine:

- **`samples: all / correct / mistakes`** — a sample counts as correct only when
  *every* task it was judged on matched. One wrong task makes the whole sample a
  mistake. A sample judged on nothing — one carrying only a mIoU — is neither, so
  it stays out of both narrowed views.
- **a range per measured score** — two handles, because which end is *bad* depends
  on the score: a low IoU and a high error are the same complaint. Each slider
  spans the values the page actually holds, so its travel has resolution where the
  samples are. Untouched, it filters nothing.

  One slider per task **and metric**: a task that measures itself two ways gets
  two. The numbers are named the way the framework names them — `iou`, `mae`, the
  keys `metric_registry` holds and the presets declare — so the page and the
  progress table stop calling one quantity two things.

`n / m shown` under them; and a combination that hides everything replaces the
grid with a line saying so and a **Reset filters** button, because an empty grid
reads as a broken page.

The tree collapses by default and every branch is a button, so a keyboard reaches
what a mouse does. Below 820px the sidebar becomes a top bar — a tracker's
embedded panel is not 1200px wide.

Which task went wrong is read off the cell's own chips, or by switching that
task's fields off in the tree above.

## Which tasks are drawn

An annotator is composed from the task's two axes, the same way
`build_task_components` composes a task's components: an **objective** reads
predictions off the class axis, a **topology** turns that reading into labels and
a verdict.

| | multiclass | binary | multilabel | continuous | metric |
|---|---|---|---|---|---|
| global | chips | chips | chips | chips | *nothing to show* |
| dense | masks | masks | masks | *not yet* | unsupported |
| multistream / multiview | unsupported | unsupported | unsupported | unsupported | *nothing to show* |

A task that draws nothing is skipped with one log line naming it **and the
reason** — metric learning has no per-sample label, a dense regression is a
heatmap and the IR has no label kind for one yet. New task types arrive before
their annotators do, and a grid that silently omitted them would look complete.

## Adding an annotator

A new `Objective` member is one class in `annotation_objective_registry`; a new
`Topology` member is one class in `annotation_topology_registry`. Neither has to
know about the other.

```python
@annotation_objective_registry.register(Objective.MY_OBJECTIVE)
class MyAnnotation(AnnotationObjective):
    """Reads the class axis; the trailing shape is the topology's business."""

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def read_output(self, scores: np.ndarray) -> Reading: ...

    def read_target(self, target: np.ndarray) -> Reading: ...
```

`threshold` and `ignore_index` are offered to every constructor and reach the
ones that name them, so a reader declares the knobs it wants and the rest see
nothing. Both are set on the callback:

```yaml
  - name: samples
    threshold: 0.3
    ignore_index: 255
    mean: ${mean}
    std: ${std}
```

A topology overrides the labeller for each kind of reading it can draw, and says
nothing about the rest:

```python
@annotation_topology_registry.register(OutputTopology.MY_TOPOLOGY)
class MyAnnotation(AnnotationTopology):
    def label_classes(self, view, task, truth, predicted) -> None: ...
    # no label_values: this topology has no label for a field of numbers
```

`draws` is derived from those overrides, so nothing has to be kept in step: a
pairing a topology has not written a labeller for is refused when the callback is
built, with the task and the reason named.

A **new kind of reading** — boxes for detection, say — is one dataclass, one
member of the `Reading` union, one arm in `annotate`'s `match`, and one defaulted
method on `AnnotationTopology`. No existing topology changes, and a topology that
does not draw boxes needs no line about them.

## Adding a kind of label

A reading becomes a *label* the page can draw, and a new kind of label (a
heatmap, detection boxes) touches four named places:

1. the entity joins the `Label` union in `entities.py`;
2. a `LabelRenderer` subclass in `renderers.py`, registered under the entity's
   type — `leaves` names what the palette colours, `render` draws it;
3. an annotation objective/topology in `annotators.py` produces it;
4. the exhaustiveness pin in `test_renderers.py` goes green again.

The renderer registries are keyed by the entity type itself rather than by a
config name: which renderer runs is decided by what the annotator produced,
never by a declaration. Same `Registry` mechanism, minus the `{name: ...}`
sugar.

The tasks a page draws reach the callback from the composition root, as a derived
value — `build_callbacks` already offers them to every entry. The module is asked
only for the thing it alone owns, the step it just ran.

## Where the page goes

Through the `HtmlLogger` port: a tracker that can carry a page gets one. ClearML
reports it as media with an `html` extension, which is what its Debug Samples
panel embeds in place; the page carries its own CSS and JS, so nothing is fetched
when it opens.

A run whose tracker has no `log_html` — `logger: none`, or a CSV run — prints one
warning at setup naming what would provide it, and then trains normally. See
[logging.md](logging.md).

## Where the source pills come from

`TableDataset` carries every **string** cell of a row on the sample, under the key
`Sample.CELLS`; a consumer reads them back through `Batch.cells`, which names the
key, types it and supplies its default in one place. `Batch.meta` itself stays a
loose mapping on purpose — a datamodule wrapping a third-party collate passes that
library's own keys through it — so what is typed is the reading, not the container.

Only input columns. A target's own file would need a key of its own: task names
and input aliases are separate namespaces, and one dictionary holding both would
collide the day a task is named after an input. A string names its content (a path, a URL) or is its content (a
caption); an array is neither, and the tensor built from it already went to the
model. The grid decides what to do with each: an image input's cell becomes a
pill, a text input's cell becomes the caption strip.

A run over in-memory arrays has no cells, and its samples simply draw no pills.
