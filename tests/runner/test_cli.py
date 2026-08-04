from __future__ import annotations

from pathlib import Path

import pytest

import cvlab.runner.cli as cli_mod


def test_no_subcommand_prints_usage_and_exits_zero(capsys: pytest.CaptureFixture) -> None:
    assert cli_mod.main([]) == 0
    assert "extract-features" in capsys.readouterr().err


def test_extract_features_dispatches_with_parsed_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_extract_split(dataset_cfg, encoder_cfg, split, cache_root, device, batch_size):
        calls.append((dataset_cfg.name, encoder_cfg.name, split, cache_root, device, batch_size))

    monkeypatch.setattr(cli_mod, "extract_split", fake_extract_split)

    rc = cli_mod.main(
        [
            "extract-features",
            "--dataset",
            "dtd",
            "--encoder",
            "resnet18",
            "--split",
            "val",
            "--cache-root",
            "/tmp/whatever-cache",
            "--device",
            "cpu",
        ]
    )

    assert rc == 0
    assert len(calls) == 1
    dataset_name, encoder_name, split, cache_root, device, batch_size = calls[0]
    assert dataset_name == "dtd"
    assert encoder_name == "resnet18"
    assert split == "val"
    assert cache_root == "/tmp/whatever-cache"
    assert device == "cpu"
    assert batch_size == cli_mod.DEFAULT_BATCH_SIZE


def test_extract_features_split_all_runs_every_split(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_splits = []

    def fake_extract_split(dataset_cfg, encoder_cfg, split, cache_root, device, batch_size):
        seen_splits.append(split)

    monkeypatch.setattr(cli_mod, "extract_split", fake_extract_split)

    rc = cli_mod.main(
        ["extract-features", "--dataset", "flowers102", "--encoder", "dinov2_vits14", "--split", "all"]
    )

    assert rc == 0
    assert seen_splits == ["train", "val", "test"]


def test_rejects_unknown_dataset_choice() -> None:
    with pytest.raises(SystemExit):
        cli_mod.main(["extract-features", "--dataset", "not-a-dataset", "--encoder", "resnet18"])
