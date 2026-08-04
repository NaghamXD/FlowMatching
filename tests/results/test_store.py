from __future__ import annotations

from pathlib import Path

from cvlab.results.schema import RunRecord
from cvlab.results.store import exists, list_run_ids, load, save


def _record(run_id: str) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        config={"k": 5, "seed": 0},
        git_commit="unknown",
        feature_cache_manifest_hashes={"train": "abc"},
        test_top1=0.5,
        mean_per_class_accuracy=0.5,
        train_size=100,
        num_classes=5,
        best_epoch=None,
        epoch_curves=[],
        predictions_path=f"runs/{run_id}/predictions.pt",
    )


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    record = _record("run_a")
    save(tmp_path, record)
    loaded = load(tmp_path, "run_a")
    assert loaded == record


def test_exists_false_before_save_true_after(tmp_path: Path) -> None:
    assert not exists(tmp_path, "run_b")
    save(tmp_path, _record("run_b"))
    assert exists(tmp_path, "run_b")


def test_list_run_ids_empty_when_no_runs(tmp_path: Path) -> None:
    assert list_run_ids(tmp_path) == []


def test_list_run_ids_returns_all_saved_runs_sorted(tmp_path: Path) -> None:
    for run_id in ("run_c", "run_a", "run_b"):
        save(tmp_path, _record(run_id))
    assert list_run_ids(tmp_path) == ["run_a", "run_b", "run_c"]
