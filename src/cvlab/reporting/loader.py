"""Loads and validates the Stage 1 run matrix from the results store.

Hard constraint (reporting spec decision 4): reporting reads only the results
store. This module reuses `runner.sweep.enumerate_stage1_runs` as the single
source of truth for which 48 runs are expected, rather than re-deriving that
list -- it never runs an experiment itself."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cvlab.results.schema import RunRecord
from cvlab.results.store import exists, load as load_record
from cvlab.runner.experiment import RunSpec
from cvlab.runner.sweep import enumerate_stage1_runs

EXPECTED_RUN_COUNT = 48


@dataclass(frozen=True)
class RunCell:
    dataset: str
    encoder: str
    method: str
    k: int | str
    records: tuple[RunRecord, ...]  # ordered by seed


def load_run_matrix(
    runs_root: str | Path, specs: list[RunSpec] | None = None
) -> dict[tuple[str, str, str, int | str], RunCell]:
    """Load every run in the Stage 1 grid, grouped into (dataset, encoder, method, K)
    cells. Fails loudly, naming exactly which run_ids are missing, rather than
    silently plotting a gap -- per the reporting spec's explicit instruction."""
    specs = specs if specs is not None else enumerate_stage1_runs()

    missing = [spec.run_id for spec in specs if not exists(runs_root, spec.run_id)]
    if missing:
        raise RuntimeError(
            f"results store at {runs_root!r} is missing {len(missing)} expected "
            f"run(s), cannot build the report:\n  " + "\n  ".join(sorted(missing))
        )

    records_by_id = {spec.run_id: load_record(runs_root, spec.run_id) for spec in specs}
    if len(records_by_id) != EXPECTED_RUN_COUNT:
        raise RuntimeError(
            f"expected exactly {EXPECTED_RUN_COUNT} Stage 1 runs, found "
            f"{len(records_by_id)} (duplicate run_ids in the spec list?)"
        )

    cells: dict[tuple[str, str, str, int | str], list[RunRecord]] = {}
    for spec in specs:
        key = (spec.dataset, spec.encoder, spec.method, spec.k)
        cells.setdefault(key, []).append(records_by_id[spec.run_id])

    return {key: RunCell(*key, tuple(records)) for key, records in cells.items()}
