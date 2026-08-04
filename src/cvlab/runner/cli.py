"""CLI entry point. Subcommands: `extract-features`, `run-experiment`, `run-sweep`,
`list-runs`. Argument parsing and dispatch only -- all logic lives in
`features.extract`, `runner.experiment`, and `runner.sweep`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cvlab.config.loader import load_dataclass, load_yaml
from cvlab.config.schema import DatasetConfig, EncoderConfig, MethodConfig
from cvlab.features.extract import DEFAULT_BATCH_SIZE, extract_split
from cvlab.results.store import list_run_ids
from cvlab.results.store import load as load_record
from cvlab.runner.experiment import RunSpec, run_single
from cvlab.runner.sweep import build_summary, enumerate_stage1_runs, render_summary_markdown, run_sweep

_DEFAULT_CONFIGS_DIR = "configs"
_SPLITS = ("train", "val", "test")
_METHODS = ("linear_probe", "image_prototype")


def _load_dataset_config(name: str, configs_dir: Path) -> DatasetConfig:
    return load_dataclass(DatasetConfig, load_yaml(configs_dir / "datasets" / f"{name}.yaml"))


def _load_encoder_config(name: str, configs_dir: Path) -> EncoderConfig:
    return load_dataclass(EncoderConfig, load_yaml(configs_dir / "encoders" / f"{name}.yaml"))


def _load_method_config(name: str, configs_dir: Path) -> MethodConfig:
    return load_dataclass(MethodConfig, load_yaml(configs_dir / "methods" / f"{name}.yaml"))


def _parse_k(value: str) -> int | str:
    return value if value == "full" else int(value)


def _cmd_extract_features(args: argparse.Namespace) -> int:
    configs_dir = Path(args.configs_dir)
    dataset_cfg = _load_dataset_config(args.dataset, configs_dir)
    encoder_cfg = _load_encoder_config(args.encoder, configs_dir)
    splits = _SPLITS if args.split == "all" else (args.split,)
    for split in splits:
        extract_split(
            dataset_cfg,
            encoder_cfg,
            split,
            cache_root=args.cache_root,
            device=args.device,
            batch_size=args.batch_size,
        )
    return 0


def _cmd_run_experiment(args: argparse.Namespace) -> int:
    configs_dir = Path(args.configs_dir)
    spec = RunSpec(
        dataset=args.dataset,
        encoder=args.encoder,
        method=args.method,
        k=_parse_k(args.k),
        seed=args.seed,
    )
    record = run_single(
        spec,
        _load_dataset_config(args.dataset, configs_dir),
        _load_encoder_config(args.encoder, configs_dir),
        _load_method_config(args.method, configs_dir),
        cache_root=args.cache_root,
        runs_root=args.runs_root,
    )
    print(f"{record.run_id}: test_top1={record.test_top1:.4f}")
    return 0


def _cmd_run_sweep(args: argparse.Namespace) -> int:
    if args.dry_run:
        specs = enumerate_stage1_runs()
        for spec in specs:
            print(spec.run_id)
        print(f"# {len(specs)} runs total", file=sys.stderr)
        return 0

    run_sweep(args.configs_dir, args.cache_root, args.runs_root)

    rows = build_summary(args.runs_root)
    table = render_summary_markdown(rows)
    print(table)
    (Path(args.runs_root) / "stage1_summary.md").write_text(table)
    return 0


def _cmd_list_runs(args: argparse.Namespace) -> int:
    run_ids = list_run_ids(args.runs_root)
    if not run_ids:
        print("no runs recorded yet", file=sys.stderr)
        return 0
    for run_id in run_ids:
        record = load_record(args.runs_root, run_id)
        print(f"{run_id}\ttest_top1={record.test_top1:.4f}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cvlab")
    subparsers = parser.add_subparsers(dest="command")

    extract = subparsers.add_parser(
        "extract-features",
        help="Extract and cache frozen-encoder features for one dataset split (or all).",
    )
    extract.add_argument("--dataset", required=True, choices=["dtd", "flowers102"])
    extract.add_argument("--encoder", required=True, choices=["resnet18", "dinov2_vits14"])
    extract.add_argument("--split", default="all", choices=[*_SPLITS, "all"])
    extract.add_argument("--cache-root", default="cache/features")
    extract.add_argument("--configs-dir", default=_DEFAULT_CONFIGS_DIR)
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    extract.set_defaults(func=_cmd_extract_features)

    run_experiment = subparsers.add_parser(
        "run-experiment", help="Run one (dataset, encoder, method, K, seed) combination."
    )
    run_experiment.add_argument("--dataset", required=True, choices=["dtd", "flowers102"])
    run_experiment.add_argument("--encoder", required=True, choices=["resnet18", "dinov2_vits14"])
    run_experiment.add_argument("--method", required=True, choices=_METHODS)
    run_experiment.add_argument("--k", required=True, help="5, 10, or 'full'")
    run_experiment.add_argument("--seed", type=int, default=0)
    run_experiment.add_argument("--cache-root", default="cache/features")
    run_experiment.add_argument("--runs-root", default="runs")
    run_experiment.add_argument("--configs-dir", default=_DEFAULT_CONFIGS_DIR)
    run_experiment.set_defaults(func=_cmd_run_experiment)

    sweep = subparsers.add_parser("run-sweep", help="Run the full 48-run Stage 1 grid.")
    sweep.add_argument("--dry-run", action="store_true")
    sweep.add_argument("--cache-root", default="cache/features")
    sweep.add_argument("--runs-root", default="runs")
    sweep.add_argument("--configs-dir", default=_DEFAULT_CONFIGS_DIR)
    sweep.set_defaults(func=_cmd_run_sweep)

    list_runs = subparsers.add_parser("list-runs", help="List recorded runs and their test top-1.")
    list_runs.add_argument("--runs-root", default="runs")
    list_runs.set_defaults(func=_cmd_list_runs)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        print(
            "cvlab CLI: no subcommand given. Try `cvlab extract-features --help`, "
            "`cvlab run-experiment --help`, `cvlab run-sweep --help`, or `cvlab list-runs --help`.",
            file=sys.stderr,
        )
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
