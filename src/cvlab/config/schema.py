"""Typed configuration dataclasses for datasets, encoders, methods, and runs.

Every field that governs a Stage 1 experiment lives here. Downstream modules
receive these dataclasses rather than reading files, env vars, or CLI args
directly, so they stay testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

# K is a positive shot count or the literal "full" (the entire official train split).
KSetting = Union[int, Literal["full"]]


def parse_k_setting(value: KSetting) -> KSetting:
    """Validate a K setting, accepting the literal 'full' or a positive int."""
    if isinstance(value, str):
        if value != "full":
            raise ValueError(f"K string values must be 'full', got {value!r}")
        return value
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"K must be an int or 'full', got {value!r}")
    if value <= 0:
        raise ValueError(f"K must be positive, got {value}")
    return value


@dataclass(frozen=True)
class DatasetConfig:
    """Identifies a dataset and where its raw files live on disk.

    `partition` only applies to DTD (official train/val/test splits are
    numbered 1-10); Stage 1 pins it to 1 and Flowers-102 ignores the field.
    """

    name: Literal["dtd", "flowers102"]
    root: str
    partition: int = 1

    def __post_init__(self) -> None:
        if self.name == "dtd" and self.partition != 1:
            raise ValueError("Stage 1 uses DTD official partition 1 only")


@dataclass(frozen=True)
class EncoderConfig:
    """Identifies a frozen backbone, its checkpoint, and its expected output shape."""

    name: Literal["resnet18", "dinov2_vits14"]
    checkpoint_id: str
    feature_dim: int
    input_resolution: int
    uses_registers: bool = False


@dataclass(frozen=True)
class LinearProbeConfig:
    """Hyperparameters for the trained linear-probe head. See DECISIONS.md decision 13
    for why `patience` defaults to (and stays) disabled in Stage 1."""

    optimizer: Literal["adamw"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    patience: int | None = None


@dataclass(frozen=True)
class ImagePrototypeConfig:
    """Hyperparameters for the training-free class-prototype head.

    `temperature` is a Stage 2 seam only; see DECISIONS.md decision 11 for why
    it must not be tuned in Stage 1.
    """

    temperature: float = 1.0


@dataclass(frozen=True)
class MethodConfig:
    """Selects a classification head and carries hyperparameters for both,
    so the unused head's config is still present and typed rather than absent."""

    name: Literal["linear_probe", "image_prototype"]
    linear_probe: LinearProbeConfig = field(default_factory=LinearProbeConfig)
    image_prototype: ImagePrototypeConfig = field(default_factory=ImagePrototypeConfig)


@dataclass(frozen=True)
class RunConfig:
    """Full specification of a single Stage 1 run: one (dataset, encoder, method, K, seed)
    combination. Decision 4: this single `seed` controls every stochastic element of the
    run (subset sampling, weight init, batch shuffling order)."""

    dataset: DatasetConfig
    encoder: EncoderConfig
    method: MethodConfig
    k: KSetting
    seed: int
    cache_root: str = "cache/features"
    runs_root: str = "runs"

    def __post_init__(self) -> None:
        parse_k_setting(self.k)
