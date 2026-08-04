"""Runs one official dataset split through a frozen encoder exactly once, caching
the resulting [N, D] float32 feature tensor + integer labels + manifest. K-shot
subsetting never re-extracts — it only ever indexes the cached tensor (see `sampling`)."""

from __future__ import annotations

from pathlib import Path

import torch

from cvlab.config.schema import DatasetConfig, EncoderConfig
from cvlab.data.registry import load_dataset
from cvlab.encoders.registry import load_encoder
from cvlab.features.cache import is_cache_valid, save
from cvlab.features.manifest import FeatureManifest, get_code_version
from cvlab.utils.logging import get_logger

logger = get_logger("features.extract")

DEFAULT_BATCH_SIZE = 64


def extract_split(
    dataset_cfg: DatasetConfig,
    encoder_cfg: EncoderConfig,
    split: str,
    cache_root: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Extract and cache features for one (dataset, split, encoder), unless a valid
    cache already exists (decision: cache raw, unnormalized float32 features)."""
    dataset = load_dataset(dataset_cfg, split)
    encoder = load_encoder(encoder_cfg, device=device)

    expected_manifest = FeatureManifest(
        dataset=dataset_cfg.name,
        split=split,
        encoder=encoder_cfg.name,
        checkpoint_id=encoder_cfg.checkpoint_id,
        transform_description=str(encoder.transform),
        dtype="float32",
        shape=(len(dataset), encoder_cfg.feature_dim),
        code_version=get_code_version(),
    )

    if is_cache_valid(cache_root, dataset_cfg.name, split, encoder_cfg.name, expected_manifest):
        logger.info(
            "cache hit for (%s, %s, %s) — skipping extraction",
            dataset_cfg.name,
            split,
            encoder_cfg.name,
        )
        return

    logger.info(
        "extracting (%s, %s, %s): %d images", dataset_cfg.name, split, encoder_cfg.name, len(dataset)
    )

    feature_batches: list[torch.Tensor] = []
    labels: list[int] = []
    image_batch: list[torch.Tensor] = []
    label_batch: list[int] = []

    def flush() -> None:
        if not image_batch:
            return
        batch = torch.stack(image_batch)
        feature_batches.append(encoder.encode_image(batch).to(torch.float32))
        labels.extend(label_batch)
        image_batch.clear()
        label_batch.clear()

    for idx in range(len(dataset)):
        image, label = dataset[idx]
        image_batch.append(encoder.transform(image))
        label_batch.append(label)
        if len(image_batch) == batch_size:
            flush()
    flush()

    Z = torch.cat(feature_batches, dim=0)
    y = torch.tensor(labels, dtype=torch.long)

    save(cache_root, dataset_cfg.name, split, encoder_cfg.name, Z, y, expected_manifest)
    logger.info(
        "cached (%s, %s, %s) -> Z%s y%s",
        dataset_cfg.name,
        split,
        encoder_cfg.name,
        tuple(Z.shape),
        tuple(y.shape),
    )
