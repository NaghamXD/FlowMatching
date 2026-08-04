"""Run-record schema and store. One JSON record per run; reporting reads only this store,
never re-runs experiments."""

from cvlab.results.schema import RunRecord
from cvlab.results.store import exists, list_run_ids, load, run_dir, save

__all__ = ["RunRecord", "exists", "list_run_ids", "load", "run_dir", "save"]
