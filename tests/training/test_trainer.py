"""M6 test list (verbatim from the spec):
- overfits a 20-sample subset to ~100% train accuracy within a few hundred steps
- average loss decreases
- the returned checkpoint's val accuracy equals the maximum in the recorded history
- curve history has exactly max_epochs entries for every run (guards decision 13)
- best_epoch is recorded and indexes the argmax of the val-accuracy history under
  the decision-5 tie-break
- patience=None is the default and Stage 1 configs never override it
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cvlab.config.loader import load_dataclass, load_yaml
from cvlab.config.schema import LinearProbeConfig, MethodConfig
from cvlab.training.curves import tie_break_key
from cvlab.training.trainer import train_linear_probe

REPO_ROOT = Path(__file__).resolve().parents[2]


def _separable_data(
    num_classes: int, per_class: int, dim: int, seed: int, separation: float = 8.0
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, dim, generator=generator) * separation
    Z, y = [], []
    for cls in range(num_classes):
        Z.append(centers[cls] + 0.1 * torch.randn(per_class, dim, generator=generator))
        y.extend([cls] * per_class)
    return torch.cat(Z), torch.tensor(y, dtype=torch.long)


def _separable_train_val_split(
    num_classes: int,
    per_class_train: int,
    per_class_val: int,
    dim: int,
    seed: int,
    separation: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like `_separable_data`, but train and val share the same class centers
    (only the noise differs) -- otherwise "val accuracy" would be scored
    against a different, unrelated set of class boundaries."""
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, dim, generator=generator) * separation
    Z_train, y_train, Z_val, y_val = [], [], [], []
    for cls in range(num_classes):
        Z_train.append(centers[cls] + 0.1 * torch.randn(per_class_train, dim, generator=generator))
        y_train.extend([cls] * per_class_train)
        Z_val.append(centers[cls] + 0.1 * torch.randn(per_class_val, dim, generator=generator))
        y_val.extend([cls] * per_class_val)
    return (
        torch.cat(Z_train),
        torch.tensor(y_train, dtype=torch.long),
        torch.cat(Z_val),
        torch.tensor(y_val, dtype=torch.long),
    )


def test_overfits_small_subset_near_perfect_train_accuracy() -> None:
    Z, y = _separable_data(num_classes=4, per_class=5, dim=8, seed=0)  # 20 samples
    config = LinearProbeConfig(lr=1e-2, weight_decay=0.0, batch_size=64, max_epochs=300)
    _, result = train_linear_probe(Z, y, Z, y, num_classes=4, feature_dim=8, config=config, seed=0)
    assert result.history[-1].train_acc >= 0.95
    assert result.history[-1].step <= 300  # 20 samples / batch_size=64 -> 1 step/epoch


def test_average_loss_decreases() -> None:
    Z, y = _separable_data(num_classes=5, per_class=10, dim=16, seed=1)
    config = LinearProbeConfig(max_epochs=50)
    _, result = train_linear_probe(Z, y, Z, y, num_classes=5, feature_dim=16, config=config, seed=1)
    first_10 = sum(r.train_loss for r in result.history[:10]) / 10
    last_10 = sum(r.train_loss for r in result.history[-10:]) / 10
    assert last_10 < first_10


def test_train_loss_is_full_batch_final_weights_not_running_average() -> None:
    """Decision 17: train_loss/train_acc must reflect one full-batch, no_grad pass
    under the epoch's *final* weights -- not a running average across mini-batches
    computed under earlier (stale) weights. With max_epochs=1, the returned model
    IS the epoch's final weights (it's the only, hence best, epoch), so recomputing
    the full-batch loss independently must match the recorded value exactly. This
    would have failed under the old mini-batch running-average implementation.
    """
    Z, y = _separable_data(num_classes=4, per_class=10, dim=8, seed=0)  # 40 samples
    config = LinearProbeConfig(max_epochs=1, batch_size=8)  # 5 mini-batches in the one epoch
    model, result = train_linear_probe(Z, y, Z, y, num_classes=4, feature_dim=8, config=config, seed=0)

    assert result.best_epoch == 1
    model.eval()
    with torch.no_grad():
        expected_loss = F.cross_entropy(model(Z), y).item()
        expected_acc = (model(Z).argmax(dim=1) == y).float().mean().item()
    assert result.history[0].train_loss == pytest.approx(expected_loss, abs=1e-6)
    assert result.history[0].train_acc == pytest.approx(expected_acc, abs=1e-6)


def test_checkpoint_val_accuracy_equals_max_in_history() -> None:
    Z_train, y_train, Z_val, y_val = _separable_train_val_split(
        num_classes=5, per_class_train=10, per_class_val=4, dim=16, seed=2
    )
    config = LinearProbeConfig(max_epochs=40)
    model, result = train_linear_probe(
        Z_train, y_train, Z_val, y_val, num_classes=5, feature_dim=16, config=config, seed=2
    )
    model.eval()
    with torch.no_grad():
        val_logits = model(Z_val)
        returned_val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
    max_history_val_acc = max(r.val_acc for r in result.history)
    assert returned_val_acc == pytest.approx(max_history_val_acc)


def test_history_has_exactly_max_epochs_entries() -> None:
    Z, y = _separable_data(num_classes=3, per_class=6, dim=8, seed=4)
    for max_epochs in (1, 17, 40):
        config = LinearProbeConfig(max_epochs=max_epochs)
        _, result = train_linear_probe(Z, y, Z, y, num_classes=3, feature_dim=8, config=config, seed=4)
        assert len(result.history) == max_epochs


def test_best_epoch_indexes_tie_break_argmax() -> None:
    Z_train, y_train, Z_val, y_val = _separable_train_val_split(
        num_classes=5, per_class_train=10, per_class_val=4, dim=16, seed=5
    )
    config = LinearProbeConfig(max_epochs=40)
    _, result = train_linear_probe(
        Z_train, y_train, Z_val, y_val, num_classes=5, feature_dim=16, config=config, seed=5
    )
    expected_best = max(result.history, key=tie_break_key)
    assert result.best_epoch == expected_best.epoch


def test_tie_break_prefers_lower_loss_then_earlier_epoch() -> None:
    from cvlab.training.curves import EpochRecord

    a = EpochRecord(epoch=5, step=5, train_loss=0.1, train_acc=1.0, val_loss=0.5, val_acc=0.8)
    b = EpochRecord(epoch=10, step=10, train_loss=0.1, train_acc=1.0, val_loss=0.5, val_acc=0.8)
    # Same val_acc and val_loss -> earliest epoch (a) wins.
    assert max([a, b], key=tie_break_key) is a

    c = EpochRecord(epoch=1, step=1, train_loss=0.1, train_acc=1.0, val_loss=0.9, val_acc=0.8)
    d = EpochRecord(epoch=2, step=2, train_loss=0.1, train_acc=1.0, val_loss=0.2, val_acc=0.8)
    # Same val_acc -> lower val_loss (d) wins, even though it's a later epoch.
    assert max([c, d], key=tie_break_key) is d


def test_patience_none_by_default_and_rejected_if_set() -> None:
    assert LinearProbeConfig().patience is None
    Z, y = _separable_data(num_classes=3, per_class=5, dim=8, seed=7)
    bad_config = LinearProbeConfig(max_epochs=5, patience=3)
    with pytest.raises(ValueError):
        train_linear_probe(Z, y, Z, y, num_classes=3, feature_dim=8, config=bad_config, seed=7)


def test_shipped_linear_probe_config_never_sets_patience() -> None:
    cfg = load_dataclass(
        MethodConfig, load_yaml(REPO_ROOT / "configs" / "methods" / "linear_probe.yaml")
    )
    assert cfg.linear_probe.patience is None


def test_same_seed_is_bit_identical() -> None:
    Z, y = _separable_data(num_classes=4, per_class=8, dim=12, seed=8)
    config = LinearProbeConfig(max_epochs=15)
    model1, result1 = train_linear_probe(Z, y, Z, y, num_classes=4, feature_dim=12, config=config, seed=99)
    model2, result2 = train_linear_probe(Z, y, Z, y, num_classes=4, feature_dim=12, config=config, seed=99)
    assert torch.equal(model1.weight, model2.weight)
    assert torch.equal(model1.bias, model2.bias)
    assert [r.train_loss for r in result1.history] == [r.train_loss for r in result2.history]
