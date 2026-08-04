"""Optimization loop for parametric heads (currently: LinearProbe). Prototype
heads bypass this entirely.

Decision 13: always runs the full `max_epochs` budget -- best-val checkpoint
selection already prevents keeping an overfit model, so early stopping would
only change compute, not results, and would truncate the loss curves the spec
wants as overfitting evidence. `patience` must stay unset for every Stage 1 run.

Decision 17: `train_loss`/`train_acc` are evaluated once per epoch, under
`no_grad`, over the *full* training subset with the epoch's final weights --
not accumulated as a running average across mini-batches computed under
different (stale) weights during the epoch. A running average is systematically
pessimistic early in training and understates the true train/val gap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvlab.config.schema import LinearProbeConfig
from cvlab.training.curves import EpochRecord, TrainingResult, tie_break_key
from cvlab.utils.logging import get_logger
from cvlab.utils.seeding import set_seed

logger = get_logger("training.trainer")


def train_linear_probe(
    Z_train: torch.Tensor,
    y_train: torch.Tensor,
    Z_val: torch.Tensor,
    y_val: torch.Tensor,
    num_classes: int,
    feature_dim: int,
    config: LinearProbeConfig,
    seed: int,
) -> tuple[nn.Linear, TrainingResult]:
    """Train `s = W z + b` with AdamW + softmax cross-entropy. Decision 4: `seed`
    controls weight init and batch shuffling order (the only stochastic elements
    of this loop). Returns the model loaded with the best-val checkpoint's
    weights, plus the full per-epoch history."""
    if config.patience is not None:
        raise ValueError(
            "Stage 1 linear-probe runs must not set patience (decision 13); "
            f"got patience={config.patience}"
        )

    set_seed(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)

    model = nn.Linear(feature_dim, num_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    n = Z_train.shape[0]
    history: list[EpochRecord] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, float, int] | None = None
    best_epoch = 0
    step = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=shuffle_generator)
        for start in range(0, n, config.batch_size):
            batch_idx = perm[start : start + config.batch_size]
            zb, yb = Z_train[batch_idx], y_train[batch_idx]

            optimizer.zero_grad()
            logits = model(zb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            step += 1

        # Decision 17: evaluate train/val metrics once, under the epoch's final
        # weights, over the full subsets -- not a mini-batch running average.
        model.eval()
        with torch.no_grad():
            train_logits = model(Z_train)
            train_loss = F.cross_entropy(train_logits, y_train).item()
            train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item()

            val_logits = model(Z_val)
            val_loss = F.cross_entropy(val_logits, y_val).item()
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()

        record = EpochRecord(
            epoch=epoch,
            step=step,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        history.append(record)

        key = tie_break_key(record)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    assert best_state is not None  # max_epochs >= 1 guarantees at least one record
    model.load_state_dict(best_state)
    logger.info(
        "trained linear probe: %d epochs, %d steps, best_epoch=%d, best val_acc=%.4f",
        config.max_epochs,
        step,
        best_epoch,
        best_key[0],
    )
    return model, TrainingResult(history=history, best_epoch=best_epoch)
