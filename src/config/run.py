"""What this run does, and where it writes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunConfig(BaseModel):
    """Which stages to run, from which checkpoint, into which directory.

    ``checkpoint_path`` takes only the weights, ``resume_path`` the whole training state;
    neither reaches ``test``, which evaluates whatever the module holds. ``directory`` is
    the single root everything a run produces lives under (``${hydra:run.dir}``).
    ``project``/``name`` are the run's identity, reached everywhere else by interpolation.
    """

    model_config = ConfigDict(extra="forbid")

    project: str | None = Field(
        None,
        description=("Project this run reports under; reach it with ${run.project}. None keeps backend defaults."),
    )
    name: str | None = Field(
        None,
        description=("This run's own name, typically ${now:%Y-%m-%d}/${now:%H-%M-%S}; reach it with ${run.name}."),
    )
    train: bool = Field(True, description="Run the training loop; False evaluates without fitting.")
    test: bool = Field(True, description="Run the test stage once training finishes.")
    checkpoint_path: str | None = Field(
        None,
        description=(
            "Weights this run starts from — a checkpoint of a previous run, loaded into the module "
            "before anything else. Only the weights: the optimizer, the schedules and the epoch "
            "counter start fresh, which is what fine-tuning wants. With 'train: false' the same field "
            "is the evaluate-or-export-a-checkpoint workflow. Named like 'model.checkpoint_path' "
            "because it is the same act at another scale: that one starts a backbone, this one starts "
            "the whole module."
        ),
    )
    resume_path: str | None = Field(
        None,
        description=(
            "The run to continue, when one was interrupted. Unlike 'checkpoint_path' this carries "
            "everything the checkpoint holds — weights, optimizer state, schedules and the epoch "
            "counter — so training picks up where it stopped rather than starting over."
        ),
    )
    directory: str | None = Field(
        None,
        description=(
            "Root directory everything this run produces lives under, so no block invents its own "
            "path. Hydra's run directory is the natural value: ${hydra:run.dir}."
        ),
    )

    @model_validator(mode="after")
    def _one_checkpoint_for_one_act(self) -> RunConfig:
        """Loading weights and resuming an interrupted run are different acts on different files."""
        if self.checkpoint_path is not None and self.resume_path is not None:
            raise ValueError(
                "Set 'checkpoint_path' or 'resume_path', not both: a resumed run already carries the "
                "weights it left off with, so the other file would be loaded and then overwritten."
            )
        if self.resume_path is not None and not self.train:
            raise ValueError(
                "'resume_path' continues an interrupted training run, and this one declares "
                "'train: false'. To evaluate or export a checkpoint, name it in 'checkpoint_path'."
            )
        return self
