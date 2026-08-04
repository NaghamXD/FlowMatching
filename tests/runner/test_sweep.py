"""M8 test list (verbatim from the spec):
- --dry-run enumerates exactly 48 runs with the expected identifiers
- an interrupted sweep resumes without duplicating records
- re-running a completed sweep is a no-op
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import cvlab.runner.cli as cli_mod
from cvlab.features.cache import save as cache_save
from cvlab.features.manifest import FeatureManifest
from cvlab.results.store import list_run_ids
from cvlab.results.store import load as load_record
from cvlab.runner.sweep import COMBINATIONS, enumerate_stage1_runs, run_sweep

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIGS_DIR = REPO_ROOT / "configs"

# The real feature_dim pinned per encoder in configs/encoders/*.yaml -- matched here
# so synthetic cached tensors are shape-compatible with the real configs used below.
_FEATURE_DIMS = {"resnet18": 512, "dinov2_vits14": 384}


def test_enumerate_stage1_runs_has_exactly_48_with_expected_structure() -> None:
    specs = enumerate_stage1_runs()
    assert len(specs) == 48

    run_ids = [s.run_id for s in specs]
    assert len(run_ids) == len(set(run_ids)), "run_ids must be unique"

    linear_probe = [s for s in specs if s.method == "linear_probe"]
    image_prototype = [s for s in specs if s.method == "image_prototype"]
    assert len(linear_probe) == 27
    assert len(image_prototype) == 21

    # linear_probe: every (combination, K) has all 3 seeds
    for dataset, encoder in COMBINATIONS:
        for k in (5, 10, "full"):
            matching = [s for s in linear_probe if s.dataset == dataset and s.encoder == encoder and s.k == k]
            assert {s.seed for s in matching} == {0, 1, 2}, (dataset, encoder, k)

    # image_prototype: K=5 and K=10 have all 3 seeds; full has exactly 1 (seed 0)
    for dataset, encoder in COMBINATIONS:
        for k in (5, 10):
            matching = [s for s in image_prototype if s.dataset == dataset and s.encoder == encoder and s.k == k]
            assert {s.seed for s in matching} == {0, 1, 2}, (dataset, encoder, k)
        full_matching = [s for s in image_prototype if s.dataset == dataset and s.encoder == encoder and s.k == "full"]
        assert [s.seed for s in full_matching] == [0], (dataset, encoder, "full")

    expected_ids = {
        "dtd_resnet18_linear_probe_k5_seed0",
        "dtd_dinov2_vits14_image_prototype_kfull_seed0",
        "flowers102_resnet18_linear_probe_kfull_seed2",
    }
    assert expected_ids <= set(run_ids)


def test_dry_run_cli_enumerates_exactly_48_expected_identifiers(capsys: pytest.CaptureFixture) -> None:
    rc = cli_mod.main(["run-sweep", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = [line for line in out.strip().splitlines() if line]
    assert len(lines) == 48
    assert "dtd_resnet18_image_prototype_kfull_seed0" in lines
    assert len(set(lines)) == 48


def _synthetic_features(num_classes: int, per_class: int, dim: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, dim, generator=generator) * 5.0
    Z, y = [], []
    for cls in range(num_classes):
        Z.append(centers[cls] + 0.1 * torch.randn(per_class, dim, generator=generator))
        y.extend([cls] * per_class)
    return torch.cat(Z), torch.tensor(y, dtype=torch.long)


def _populate_all_synthetic_caches(cache_root: Path) -> None:
    num_classes = 6
    for dataset, encoder in COMBINATIONS:
        dim = _FEATURE_DIMS[encoder]
        for split, per_class in (("train", 12), ("val", 4), ("test", 4)):  # 12 >= K=10
            seed = abs(hash((dataset, encoder, split))) % 10_000
            Z, y = _synthetic_features(num_classes, per_class, dim, seed)
            manifest = FeatureManifest(
                dataset=dataset,
                split=split,
                encoder=encoder,
                checkpoint_id="fake",
                transform_description="fake",
                dtype="float32",
                shape=tuple(Z.shape),
                code_version="unknown",
            )
            cache_save(cache_root, dataset, split, encoder, Z, y, manifest)


def test_sweep_resumes_after_interruption_and_rerun_is_noop(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    runs_root = tmp_path / "runs"
    _populate_all_synthetic_caches(cache_root)

    specs = enumerate_stage1_runs()
    assert len(specs) == 48

    # Simulate an interrupted sweep: run only a prefix directly via run_single,
    # bypassing run_sweep's own iteration.
    from cvlab.runner.experiment import run_single
    from cvlab.runner.sweep import _load_dataset_config, _load_encoder_config, _load_method_config

    partial = specs[:10]
    for spec in partial:
        run_single(
            spec,
            _load_dataset_config(spec.dataset, REAL_CONFIGS_DIR),
            _load_encoder_config(spec.encoder, REAL_CONFIGS_DIR),
            _load_method_config(spec.method, REAL_CONFIGS_DIR),
            cache_root=cache_root,
            runs_root=runs_root,
        )
    assert set(list_run_ids(runs_root)) == {s.run_id for s in partial}
    sentinel = load_record(runs_root, partial[0].run_id)

    # "Resume": running the full sweep must complete the missing 38 without
    # touching the 10 already recorded.
    records = run_sweep(REAL_CONFIGS_DIR, cache_root, runs_root)
    assert len(records) == 48
    assert set(list_run_ids(runs_root)) == {s.run_id for s in specs}
    assert load_record(runs_root, partial[0].run_id) == sentinel

    # Re-running a completed sweep is a no-op: identical records, nothing recomputed.
    records_again = run_sweep(REAL_CONFIGS_DIR, cache_root, runs_root)
    assert {r.run_id: r for r in records_again} == {r.run_id: r for r in records}
