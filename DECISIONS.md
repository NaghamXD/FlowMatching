# Decisions

This log records the pinned design decisions from the Stage 1 spec, plus any
choices made during implementation. Entries are added as each milestone lands.

## Pinned decisions from the spec

1. **No data augmentation anywhere**, including the train split. Features are
   cached once through a deterministic eval-style transform (resize -> center
   crop -> normalize) for every split. This makes the full-data linear probe
   more prone to overfitting than a typical fine-tuning setup would be, since
   it never sees augmented views of a training image.
2. **Feature cache stores raw, unnormalized features.** L2 normalization
   happens inside `ImagePrototype` only. This lets the linear probe and the
   prototype head share one cache per (dataset, split, encoder).
3. **Linear-probe inputs are not L2-normalized**, for both encoders equally.
   If ResNet-18 and DINOv2 feature scales cause optimization instability at
   `lr=1e-3`, that will be reported (M6/M8) rather than silently normalizing
   one encoder and not the other.
4. **One seed controls every stochastic element of a run**: subset sampling,
   weight init, and batch shuffling order. Without this, the "full" K-setting
   would have near-zero, meaningless error bars across its 3 seeds.
5. **Checkpoint-selection tie-break**: highest val accuracy; ties broken by
   lowest val loss; further ties broken by earliest epoch. Implemented
   explicitly in the trainer (M6), not left to argmax's default first-index
   behavior on accuracy alone.
6. **Headline metric is overall top-1 on the full test split.** Mean-per-class
   accuracy is also computed and stored (Flowers-102's test split is
   class-imbalanced) but is not the reported headline number.
7. **Aggregation over the 3 seeds per (combination, K, method)** uses mean and
   *sample* standard deviation (`ddof=1`).
8. **Flowers-102 class names** ship as a static 102-name list in `data/`,
   asserted to have length 102 and to match torchvision's label ordering.
   Not needed for Stage 1 accuracy numbers, but needed for confusion matrices
   and for Stage 2.
9. **DINOv2 checkpoint**: `dinov2_vits14` (ViT-S/14, **no register tokens**)
   loaded via `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`.
   Input resolution pinned to 224x224 (16x16 = 256 patch tokens at patch size
   14). The no-register variant was chosen over `dinov2_vits14_reg` for
   simplicity: registers are an optional architectural addition and the base
   checkpoint is the one most directly comparable to "the DINOv2 ViT-S/14
   representation" as commonly cited. `uses_registers: false` is recorded in
   `configs/encoders/dinov2_vits14.yaml` and in the feature-cache manifest.
10. **Both epochs and optimizer steps are logged** per training run (200
    epochs is ~800 steps at DTD 5-shot but ~12k steps at full DTD).
11. **`ImagePrototype.temperature` defaults to 1.0 and is never tuned in
    Stage 1.** softmax(tau * cos) is standard elsewhere (CLIP, ArcFace/CosFace,
    Prototypical Networks) but tau is learned or tuned there; here the head is
    training-free by construction, so there is no fitting procedure to learn
    it in, and tuning it on val would be an unnecessary hyperparameter search.
    Since softmax and temperature scaling are monotonic, argmax (and therefore
    top-1 accuracy) is invariant to tau, so this cannot affect any Stage 1
    number. The parameter exists purely as a Stage 2 seam. Stage 1 code must
    not report probabilities from this head.
12. **Linear-probe scores and prototype-head scores are not on a comparable
    scale** (roughly +-20 for a trained probe's logits vs. [-1, 1] for cosine
    similarity). Top-1 accuracy is unaffected because each head is argmax'd
    independently, but anything that treats scores from both heads as
    comparable — ensembling, shared confidence/entropy thresholds, calibration
    comparison, cross-method softmax probabilities — is silently wrong. No
    such cross-head comparison is implemented in Stage 1. **Stage 2 must not
    assume the two heads' scores share a scale either**, without an explicit
    calibration step.
13. **All 27 linear-probe runs execute the full 200-epoch budget.** No active
    early stopping. Best-val checkpoint selection (decision 5) already returns
    the best epoch's weights regardless of how long the loop kept running, so
    early stopping would only change compute, not results — and the compute
    here is negligible (a few minutes across all 27 runs on cached tensors).
    Cutting the loop short would also truncate the loss curves the spec asks
    for as overfitting evidence. `best_epoch` is recorded per run instead, as
    the diagnostic:
    - consistently near 200 -> the model was still improving; the epoch
      budget may be too small.
    - consistently under ~20, with val loss rising after -> fast convergence
      and overfitting, expected at K=5 and worth discussing.
    - scattered with no pattern -> val accuracy is noisy at that sample size,
      and the decision-5 tie-break is doing real work.

    The trainer exposes `patience: int | None = None` as a hook for Stage 2
    heads that may be expensive to train, but it stays `None` for every
    Stage 1 config — enforced by never setting it in `configs/methods/*.yaml`
    and by M6's test suite.

## Caveats and flagged ambiguities

- **Spec step-count figures (decisions 10 vs. 13) don't agree with each other,
  and I went with decision 13's number.** Decision 10 says "200 epochs is ~800
  steps at DTD 5-shot but ~12k steps at full DTD"; decision 13 says "~6k steps
  for full DTD and ~800 for 5-shot" for the same setting. With the pinned
  `batch_size=64` (Starting configuration) and full DTD train = 1880 examples:
  `ceil(1880/64) = 30` steps/epoch x 200 = 6000 -- matching decision 13, not
  decision 10 (12k would require `batch_size=32`, i.e. `ceil(1880/32) = 59` x
  200 = 11800). Since `batch_size: 64` is stated explicitly and unambiguously
  in the Starting configuration while the step counts are parenthetical
  asides, I implemented with `batch_size=64` and the resulting ~6k-step figure
  for full DTD, consistent with decision 13. Flagging this rather than
  silently picking one, per "flag, don't silently resolve."

- **Val-split protocol at low K.** The spec mandates using the *full* official
  val split for model selection even at K=5 — for DTD that means selecting a
  checkpoint for a 235-image training set using all 1,880 val images. This is
  implemented as specified (M6/M8), but is flagged here because it is a real
  asymmetry worth being aware of when reading the 5-shot results: the model
  selection signal is far more stable than the training signal at that K.

## Empirical findings worth flagging

- **Flowers-102's official train split has exactly 10 images per class, for
  every one of the 102 classes** (verified directly from the cached labels:
  `bincount` min=max=10). Consequence: for this dataset, `K=10` balanced
  sampling has no actual choice to make -- it selects all 10 available
  examples regardless of seed -- so `K=10` and `K=full` become mathematically
  identical (same training data, same result), and `K=10`'s three seeds
  produce byte-identical results (std=0.0000 in the summary table) rather than
  the small spread seen at `K=10` for DTD (which has 40 images/class, so `K=10`
  is a genuine subset there). This is a real property of the dataset's
  official split, not a sampling or training bug -- confirmed by checking
  `balanced_kshot`'s behavior is correct (M4) and that DTD's `K=10` does show
  seed-to-seed variance as expected.

## Implementation notes not covered above

- **M2 (encoders).** DINOv2 is loaded via `torch.hub.load("facebookresearch/dinov2",
  "dinov2_vits14")`, which pulls the repo code and the `dinov2_vits14_pretrain.pth`
  checkpoint into `~/.cache/torch/hub/` (outside the repo, not gitignored-and-tracked
  since it's never inside the working tree). xFormers is not installed, so DINOv2 runs
  its reference (non-fused) attention/SwiGLU path rather than the fused kernels —
  this only affects speed, not the CLS-token values (M2's bit-identical-forward-pass
  test passed under this reference path on this machine). `forward_features(x)`
  returns a dict; `"x_norm_clstoken"` is the final, post-LayerNorm CLS token used
  here, and `"x_norm_regtokens"` has shape `[B, 0, D]`, confirming this checkpoint
  carries no register tokens as pinned in decision 9. Both encoders' preprocessing
  transforms follow the same Resize(256) -> CenterCrop(224) -> Normalize convention
  (ResNet-18 via its bundled `ResNet18_Weights.IMAGENET1K_V1.transforms()`; DINOv2
  built explicitly to match, since torch.hub only ships the model, not a transform)
  so both satisfy decision 1's "resize -> center crop -> normalize" wording identically.
