"""Decision 16: "representative run" selection. For a given (dataset, encoder, K)
cell, the representative run is the one whose test top-1 is the *median* across
seeds -- not the best one. Ties at the median value break to the lowest seed
number. Never select by "looks cleanest": a cherry-picked curve is not evidence
of stability."""

from __future__ import annotations

from cvlab.results.schema import RunRecord


def select_representative_run(records: tuple[RunRecord, ...] | list[RunRecord]) -> RunRecord:
    if len(records) % 2 == 0:
        raise ValueError(
            f"representative-run selection requires an odd number of records "
            f"(so the median corresponds to an actual run), got {len(records)}"
        )
    sorted_by_top1 = sorted(records, key=lambda r: r.test_top1)
    median_value = sorted_by_top1[len(sorted_by_top1) // 2].test_top1
    tied_at_median = [r for r in records if r.test_top1 == median_value]
    return min(tied_at_median, key=lambda r: r.config["seed"])
