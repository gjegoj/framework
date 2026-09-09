# How the tests are written

- **Layout.** `tests/unit/<package>/` mirrors `src/`; the composition root's tests sit in `tests/unit/test_build/` (pytest skips a folder named `build`); `tests/e2e/` holds the acceptance runs, one YAML experiment each; `tests/support/` is the only shared code.
- **Two config styles.** `paper_config(...)` validates without a dataset, for a test about wiring; `disk_config(root, ...)` builds and fits over what `write_dataset` wrote; an e2e run is written as the YAML a user writes and read by `experiment_from_yaml`.
- **The helpers.** `a_task`, `dataset_facts`, `a_composite`, `in_memory_pipeline`, `quiet_trainer`, `clearml_stub`; a test about the thing a helper builds still builds its own.
- **Markers.** `e2e` is applied by `tests/conftest.py` to everything under `tests/e2e/`, never by hand; `slow` marks only the tests that need a model hub.
- **The gate.** `make test-gate` runs the whole suite minus the hub tests and is what pre-commit runs; `make test` runs everything.
- **Style.** A test name is a sentence about observable behaviour, asserted through the public API; the docstring is the one-line *why* and at most four lines; a refusal is matched on the name it must carry, not on the sentence; no fixtures beyond `conftest.py` unless forty tests name one.
- **Guarantees.** Every new guarantee gets a mutation check: break the code, watch the specific test go red, restore, confirm green.
