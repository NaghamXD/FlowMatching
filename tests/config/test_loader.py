from __future__ import annotations

from pathlib import Path

from cvlab.config.loader import load_dataclass, load_run_config, load_yaml
from cvlab.config.schema import LinearProbeConfig, MethodConfig


def test_load_yaml_reads_mapping(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("name: dtd\nroot: data/dtd\n")
    assert load_yaml(p) == {"name": "dtd", "root": "data/dtd"}


def test_load_dataclass_recurses_into_nested_dataclass_fields() -> None:
    cfg = load_dataclass(
        MethodConfig,
        {"name": "linear_probe", "linear_probe": {"lr": 5e-4, "max_epochs": 50}},
    )
    assert isinstance(cfg.linear_probe, LinearProbeConfig)
    assert cfg.linear_probe.lr == 5e-4
    assert cfg.linear_probe.max_epochs == 50
    # Fields absent from the YAML fall back to the dataclass default.
    assert cfg.linear_probe.weight_decay == 1e-4


def test_load_dataclass_ignores_unknown_keys() -> None:
    cfg = load_dataclass(LinearProbeConfig, {"lr": 1e-2, "totally_unknown_field": 123})
    assert cfg.lr == 1e-2


def test_load_run_config_from_merged_yaml(tmp_path: Path) -> None:
    p = tmp_path / "run.yaml"
    p.write_text(
        """
dataset:
  name: dtd
  root: data/dtd
encoder:
  name: resnet18
  checkpoint_id: IMAGENET1K_V1
  feature_dim: 512
  input_resolution: 224
method:
  name: linear_probe
  linear_probe:
    lr: 0.001
k: 5
seed: 0
"""
    )
    run = load_run_config(p)
    assert run.dataset.name == "dtd"
    assert run.encoder.feature_dim == 512
    assert run.method.linear_probe.lr == 0.001
    assert run.k == 5
    assert run.seed == 0
