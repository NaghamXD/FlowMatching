"""M9 test: the loader finds exactly 48 run records and fails loudly if any
expected cell is missing."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cvlab.reporting.loader import EXPECTED_RUN_COUNT, load_run_matrix
from cvlab.runner.sweep import enumerate_stage1_runs


def test_loader_finds_exactly_48_records(synthetic_runs_root: Path) -> None:
    cells = load_run_matrix(synthetic_runs_root)
    total = sum(len(cell.records) for cell in cells.values())
    assert total == EXPECTED_RUN_COUNT == 48


def test_loader_composition_matches_spec(synthetic_runs_root: Path) -> None:
    cells = load_run_matrix(synthetic_runs_root)
    linear_probe_total = sum(len(c.records) for c in cells.values() if c.method == "linear_probe")
    image_prototype_total = sum(len(c.records) for c in cells.values() if c.method == "image_prototype")
    assert linear_probe_total == 27
    assert image_prototype_total == 21


def test_loader_fails_loudly_naming_missing_run(synthetic_runs_root: Path) -> None:
    victim = enumerate_stage1_runs()[0]
    shutil.rmtree(Path(synthetic_runs_root) / victim.run_id)

    with pytest.raises(RuntimeError, match=victim.run_id):
        load_run_matrix(synthetic_runs_root)


def test_loader_fails_on_completely_empty_store(tmp_path: Path) -> None:
    empty_runs_root = tmp_path / "empty_runs"
    empty_runs_root.mkdir()
    with pytest.raises(RuntimeError):
        load_run_matrix(empty_runs_root)
