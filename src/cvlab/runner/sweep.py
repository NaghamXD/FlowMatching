"""Enumerates and runs the full Stage 1 grid: 3 (dataset, encoder) combinations x
2 methods x {K=5, K=10, full} x seeds -- 48 runs total (27 linear_probe + 21
image_prototype; see the spec's Run matrix). Also renders the final
dataset x encoder x method x K accuracy table (mean +/- std over seeds)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from cvlab.config.loader import load_dataclass, load_yaml
from cvlab.config.schema import DatasetConfig, EncoderConfig, MethodConfig
from cvlab.evaluation.aggregate import aggregate
from cvlab.results.schema import RunRecord
from cvlab.results.store import list_run_ids
from cvlab.results.store import load as load_record
from cvlab.runner.experiment import RunSpec, run_single
from cvlab.utils.logging import get_logger

logger = get_logger("runner.sweep")

COMBINATIONS: tuple[tuple[str, str], ...] = (
    ("dtd", "resnet18"),  # C1
    ("dtd", "dinov2_vits14"),  # C2
    ("flowers102", "resnet18"),  # C3
)
METHODS: tuple[str, ...] = ("linear_probe", "image_prototype")
K_SETTINGS: tuple[int | str, ...] = (5, 10, "full")
SEEDS: tuple[int, ...] = (0, 1, 2)


def enumerate_stage1_runs() -> list[RunSpec]:
    """The 48-run Stage 1 grid. Decision: the full-split image-prototype "run" is
    deterministic (no init, no stochasticity), so it's enumerated once, not 3x --
    everything else loops over all 3 seeds."""
    specs: list[RunSpec] = []
    for dataset, encoder in COMBINATIONS:
        for method in METHODS:
            for k in K_SETTINGS:
                seeds = (0,) if (method == "image_prototype" and k == "full") else SEEDS
                for seed in seeds:
                    specs.append(
                        RunSpec(dataset=dataset, encoder=encoder, method=method, k=k, seed=seed)
                    )
    return specs


def _load_dataset_config(name: str, configs_dir: Path) -> DatasetConfig:
    return load_dataclass(DatasetConfig, load_yaml(configs_dir / "datasets" / f"{name}.yaml"))


def _load_encoder_config(name: str, configs_dir: Path) -> EncoderConfig:
    return load_dataclass(EncoderConfig, load_yaml(configs_dir / "encoders" / f"{name}.yaml"))


def _load_method_config(name: str, configs_dir: Path) -> MethodConfig:
    return load_dataclass(MethodConfig, load_yaml(configs_dir / "methods" / f"{name}.yaml"))


def run_sweep(
    configs_dir: str | Path, cache_root: str | Path, runs_root: str | Path
) -> list[RunRecord]:
    configs_dir = Path(configs_dir)

    dataset_cfgs = {name: _load_dataset_config(name, configs_dir) for name, _ in COMBINATIONS}
    encoder_cfgs = {
        encoder: _load_encoder_config(encoder, configs_dir) for _, encoder in COMBINATIONS
    }
    method_cfgs = {name: _load_method_config(name, configs_dir) for name in METHODS}

    specs = enumerate_stage1_runs()
    records = []
    for i, spec in enumerate(specs, 1):
        logger.info("[%d/%d] %s", i, len(specs), spec.run_id)
        record = run_single(
            spec,
            dataset_cfgs[spec.dataset],
            encoder_cfgs[spec.encoder],
            method_cfgs[spec.method],
            cache_root=cache_root,
            runs_root=runs_root,
        )
        records.append(record)
    return records


@dataclasses.dataclass(frozen=True)
class SummaryRow:
    dataset: str
    encoder: str
    method: str
    k: int | str
    mean: float
    std: float
    n: int


def build_summary(runs_root: str | Path) -> list[SummaryRow]:
    groups: dict[tuple[str, str, str, int | str], list[float]] = {}
    for run_id in list_run_ids(runs_root):
        record = load_record(runs_root, run_id)
        cfg = record.config
        key = (cfg["dataset"]["name"], cfg["encoder"]["name"], cfg["method"]["name"], cfg["k"])
        groups.setdefault(key, []).append(record.test_top1)

    rows = [
        SummaryRow(dataset, encoder, method, k, agg.mean, agg.std, agg.n)
        for (dataset, encoder, method, k), values in groups.items()
        for agg in [aggregate(values)]
    ]

    k_order = {5: 0, 10: 1, "full": 2}
    method_order = {"linear_probe": 0, "image_prototype": 1}
    rows.sort(key=lambda r: (r.dataset, r.encoder, method_order[r.method], k_order[r.k]))
    return rows


def render_summary_markdown(rows: list[SummaryRow]) -> str:
    lines = [
        "# Stage 1 Summary",
        "",
        "| Dataset | Encoder | Method | K | Test top-1 (mean ± std, n) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r.dataset} | {r.encoder} | {r.method} | {r.k} | {r.mean:.4f} ± {r.std:.4f} (n={r.n}) |")
    return "\n".join(lines) + "\n"
