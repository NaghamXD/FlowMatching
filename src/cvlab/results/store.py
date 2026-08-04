"""JSON run-record store: one directory per run under `runs_root`
(`runs/{run_id}/record.json`, plus predictions saved alongside). Backs M8's
resumable sweep -- `exists()` is how a sweep decides a run is already done."""

from __future__ import annotations

from pathlib import Path

from cvlab.results.schema import RunRecord
from cvlab.utils.io import ensure_dir, read_json, write_json


def run_dir(runs_root: str | Path, run_id: str) -> Path:
    return Path(runs_root) / run_id


def record_path(runs_root: str | Path, run_id: str) -> Path:
    return run_dir(runs_root, run_id) / "record.json"


def exists(runs_root: str | Path, run_id: str) -> bool:
    return record_path(runs_root, run_id).exists()


def save(runs_root: str | Path, record: RunRecord) -> None:
    ensure_dir(run_dir(runs_root, record.run_id))
    write_json(record_path(runs_root, record.run_id), record.to_dict())


def load(runs_root: str | Path, run_id: str) -> RunRecord:
    return RunRecord.from_dict(read_json(record_path(runs_root, run_id)))


def list_run_ids(runs_root: str | Path) -> list[str]:
    root = Path(runs_root)
    if not root.exists():
        return []
    return sorted(p.parent.name for p in root.glob("*/record.json"))
