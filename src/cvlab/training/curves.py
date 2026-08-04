"""Per-epoch training/validation curve records, and the decision-5 checkpoint
tie-break: highest val accuracy; ties broken by lowest val loss; further ties
broken by earliest epoch."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    step: int  # cumulative optimizer steps completed through this epoch (decision 10)
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


@dataclass
class TrainingResult:
    history: list[EpochRecord]
    best_epoch: int


def tie_break_key(record: EpochRecord) -> tuple[float, float, int]:
    """Larger is better: highest val_acc, then lowest val_loss, then earliest
    epoch. Embedding `-epoch` (rather than breaking ties externally) makes the
    earliest-epoch rule hold under a plain `max()`/`>` regardless of iteration
    order, since a later epoch can never out-rank an equally-good earlier one."""
    return (record.val_acc, -record.val_loss, -record.epoch)
