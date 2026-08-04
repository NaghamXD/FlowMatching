# Claude Code Prompt — CVLAB Summer Project, Stage 1

> Paste everything below the line into Claude Code as the opening instruction.

---

## Context

You are bootstrapping a research codebase for a computer-vision course project. This is **Stage 1 of 3**. Stage 2 will add a flow-matching (FM) layer operating on frozen features conditioned on class prototypes; Stage 3 will reuse the linear-probe setting. **Design every module so that Stages 2 and 3 can be added without modifying Stage 1 code.** Modularity and reusability matter more than shortest-path implementation.

Do not implement Stage 2 or Stage 3. Do not implement CLIP or zero-shot classification — that branch was not selected.

## Hard scope for this task

### Datasets (exactly two)
1. **DTD** — 47 classes. Use **official partition 1** only.
2. **Oxford Flowers-102** — 102 classes.

Use `torchvision.datasets.DTD(partition=1, ...)` and `torchvision.datasets.Flowers102(...)` with `download=True`. Use the **official train / val / test splits**.

- **Never merge train and val.**
- Val split is used **in full**, for model selection only.
- Test split is used **in full**, and **only** for final evaluation. Test features must not be touched by any `fit` path — enforce this structurally (test features are loaded inside the evaluation function, not the training function).

### Frozen encoders (exactly two)
1. **ImageNet-1K-pretrained ResNet-18** — `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`. Extract the **512-d penultimate representation** (output of global average pooling, i.e. replace `fc` with `nn.Identity()`).
2. **DINOv2 ViT-S/14** — publicly available pretrained checkpoint. Extract the **final CLS-token representation** (384-d). CLS only — do **not** concatenate mean-patch tokens.

All encoder parameters frozen (`requires_grad_(False)`, `.eval()`, forward under `torch.no_grad()`). Use each checkpoint's own associated preprocessing transform.

### The three dataset × encoder combinations
| ID | Encoder | Dataset |
|----|---------|---------|
| C1 | ResNet-18 (IN1K) | DTD |
| C2 | DINOv2 ViT-S/14 | DTD |
| C3 | ResNet-18 (IN1K) | Flowers-102 |

### The two methods

**A. Linear probe.** Multiclass linear classifier `s = W z + b` on the frozen feature `z`, softmax cross-entropy. Only `W, b` trained.

Starting configuration (adjust only if validation results demand it, and log any change):
- Optimizer: AdamW
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Batch size: 64
- Max epochs: 200
- Checkpoint selection: highest validation accuracy

**B. Image-derived class prototypes.** Note: despite being informally called "k-means" in discussion, this is **not** k-means clustering — it is a deterministic class-mean prototype:

```
mu_c = normalize( mean_{i in S_c} ( normalize(z_i) ) )
y_hat = argmax_c cos(z, mu_c)
```

`S_c` = the selected training subset for class `c`. No training, no optimizer, no gradients.

### Run matrix

Training-set sizes: `K ∈ {5, 10, full}`. For K = 5 and K = 10, sample **balanced** subsets from the official train split using subset seeds `{0, 1, 2}`.

**Linear probe** — 3 runs per (combination, K):
- K=5: subset seeds {0,1,2}
- K=10: subset seeds {0,1,2}
- full: 3 classifier-initialization seeds {0,1,2}

→ 3 combinations × 3 K-settings × 3 seeds = **27 runs**

**Image prototypes** — deterministic given the subset:
- K=5: subset seeds {0,1,2} → 3 runs
- K=10: subset seeds {0,1,2} → 3 runs
- full: **1 run**

→ 3 combinations × 7 = **21 runs**

**Note on the full-split prototype:** prototype computation has no initialization and no stochasticity, so running it three times on the full training split would produce three byte-identical results. Record it as a single run. Keep the runner general enough that a seed loop *could* be applied, but do not fabricate variance. If you disagree after implementing it, raise the point rather than silently running it three times.

**Total: 48 runs.** All 48 must be reproducible from cached features by a single sweep command.

---

## Architecture

The organizing invariant: **everything downstream of the encoder sees only cached feature tensors and integer labels.** Pixels are touched in exactly one module.

Build these as separate modules with clean interfaces:

- **`config`** — typed dataclasses (+ YAML loading) for dataset, encoder, method, K, seed, hyperparameters. No other module reads env state or hardcodes paths.
- **`data`** — uniform wrapper over DTD and Flowers-102 returning `(PIL image, label)` plus metadata (`num_classes`, `class_names`, split names). Owns the partition-1 and no-merge constraints. Knows nothing about encoders.
- **`encoders`** — registry of frozen backbones. Each exposes its own preprocessing transform, `encode_image(batch) -> [B, D]`, and `feature_dim`. Only module that sees pixels.
- **`features`** — extraction + caching. Runs a split through an encoder once; writes tensor file + **manifest** (dataset, split, encoder, checkpoint id, transform description, dtype, shape, code version). `load(dataset, split, encoder) -> (Z, y)`. Cache invalidates on manifest mismatch; never trust a bare filename.
- **`sampling`** — deterministic balanced K-shot selection returning **indices**, not data. Signature `balanced_kshot(labels, K, seed) -> LongTensor`. The linear probe and the prototype head must call this same function so both see identical subsets.
- **`heads`** — all classifiers behind one interface: `fit(Z_train, y_train, Z_val=None, y_val=None)` and `logits(Z) -> [N, C]`. Implement `LinearProbe` and `ImagePrototype`. Keep the base interface wide enough that a Stage-2 flow-matching head can satisfy it unchanged. `ImagePrototype` takes an optional `temperature: float = 1.0` and returns `temperature * cosine_similarity`; also expose `prototypes -> [C, D]` so later stages and the feature visualizations can consume `μ_c` directly.
- **`training`** — optimization loop for parametric heads only. Owns AdamW, batching over cached tensors, per-epoch train/val loss and accuracy history, best-val checkpoint selection. Always runs the full epoch budget (see decision 13) and records the best-epoch index. Prototype heads bypass this entirely.
- **`evaluation`** — head-agnostic metrics from logits + labels: top-1, per-class accuracy, row-normalized confusion matrix; plus aggregation over seeds (mean, sample std with `ddof=1`).
- **`results`** — schema + store. One JSON record per run: full config, git commit, feature-cache manifest hashes, test top-1, per-epoch curves, path to saved predictions. Reporting reads only this store, never re-runs experiments.
- **`runner`** — CLI wiring only, no logic: `extract-features`, `run-experiment`, `run-sweep`, `list-runs`.

### Directory structure

```
cvlab-project/
├── configs/
│   ├── datasets/{dtd,flowers102}.yaml
│   ├── encoders/{resnet18,dinov2_vits14}.yaml
│   ├── methods/{linear_probe,image_prototype}.yaml
│   └── sweeps/stage1.yaml
├── src/cvlab/
│   ├── config/
│   ├── data/          # registry.py, dtd.py, flowers102.py, metadata.py
│   ├── encoders/      # base.py, registry.py, resnet.py, dinov2.py
│   ├── features/      # extract.py, cache.py, manifest.py
│   ├── sampling/      # fewshot.py
│   ├── heads/         # base.py, linear.py, image_prototype.py
│   ├── training/      # trainer.py, curves.py
│   ├── evaluation/    # metrics.py, aggregate.py
│   ├── results/       # schema.py, store.py
│   ├── runner/        # cli.py
│   └── utils/         # seeding.py, io.py, logging.py
├── tests/             # mirrors src/
├── cache/features/    # gitignored
├── runs/              # gitignored, one directory per run
├── DECISIONS.md
└── README.md
```

Reporting and plotting are **out of scope for this task** — but leave `reporting/` as an empty package with a docstring stating what will live there, and make sure the results store already contains everything a later reporting module would need (curves, predictions, confusion-matrix inputs).

### Data flow

```
raw images
  → data: (PIL, label) for (dataset, split)
  → encoder's own deterministic transform (NO augmentation, any split)
  → frozen encoder, no_grad, batched
  → features cache: Z [N, D] float32 + y [N] + manifest     ← ONE TIME per (dataset, split, encoder)
  → sampling: indices = balanced_kshot(y_train, K, seed)
  → head.fit(Z_train[idx], y_train[idx], Z_val, y_val)
  → head.logits(Z_test) → evaluation → results store
```

The encoder runs **once per (dataset, split, encoder)** over the entire official train split. K-shot is index selection on the cached tensor, never a re-extraction.

---

## Pinned decisions (implement these; record them in `DECISIONS.md`)

1. **No data augmentation anywhere**, including the train split. "Cache features once" implies a deterministic eval-style transform (resize → center crop → normalize) for all splits. Note in `DECISIONS.md` that this makes the full-data linear probe more prone to overfitting than a typical fine-tuning setup.
2. **Cache raw, unnormalized features.** L2 normalization happens *inside* `ImagePrototype`. This lets the linear probe and the prototype head share one cache.
3. **Linear-probe input normalization:** do not L2-normalize. Apply the same convention to both encoders and state it. If ResNet-18 and DINOv2 feature scales cause instability at `lr=1e-3`, report it rather than silently changing one encoder.
4. **One seed controls everything stochastic in a run** — subset sampling, weight init, and batch shuffling order. Otherwise the "full" setting's error bars will be near-zero and meaningless.
5. **Checkpoint-selection tie-break:** highest val accuracy; ties broken by lowest val loss; further ties broken by earliest epoch. Implement explicitly.
6. **Metric:** overall top-1 on the full test split. Also compute and store mean-per-class accuracy — Flowers-102's test split is class-imbalanced — but report top-1 as the headline number.
7. **Aggregation:** mean and **sample** standard deviation (`ddof=1`) over the 3 runs.
8. **Flowers-102 class names** are not provided by torchvision. Ship the standard 102-name list as a data file in `data/`, cite its source in a comment, and assert `len(names) == 102` and that ordering matches the torchvision label indices. (Not needed for Stage 1 classification, but needed for confusion matrices and Stage 2 — get it right now.)
9. **DINOv2 checkpoint and input resolution:** pick one (patch size 14 ⇒ use a resolution divisible by 14, e.g. 224 → 16×16 patches), pin it in the encoder config, and document the exact checkpoint identifier and whether it uses registers.
10. Log both epochs and optimizer **steps** — 200 epochs is ~800 steps at DTD 5-shot but ~12k steps at full DTD.
11. **`ImagePrototype` temperature defaults to 1.0 and is not tuned in Stage 1.** Softmax over `τ·cos` is a standard construction (CLIP, ArcFace/CosFace, cosine classifiers, Prototypical Networks), but in all of those τ is learned or tuned, and this head is training-free by construction — there is no fitting procedure to learn it in, and tuning it on the val split is a hyperparameter search the spec says is unnecessary. Since softmax and temperature scaling are both monotonic, `argmax` and therefore top-1 accuracy are invariant to τ, so this cannot affect any reported Stage 1 number. The parameter exists purely as a seam for Stage 2. Do not tune it; do not report probabilities from this head.
12. **Scores from the two heads are not on a comparable scale.** A trained linear probe's outputs can span roughly ±20; cosine similarities are bounded in [-1, 1]. Top-1 accuracy is unaffected because each head is argmax'd independently, but anything that consumes scores from both heads as if they shared a scale — ensembling, confidence or entropy thresholds, calibration comparison, softmax probabilities compared across methods — is silently wrong. Do not implement any such cross-head comparison in Stage 1, and state this constraint in `DECISIONS.md` so Stage 2 does not assume otherwise.
13. **Run the full 200 epochs on every linear-probe run. Do not implement early stopping as an active mechanism.** Best-val checkpoint selection already prevents keeping an overfit model — if the val peak is at epoch 37, the returned weights are epoch 37's regardless of whether the loop ran to 57 or 200 — so early stopping would change compute, not results. The compute is negligible here anyway: training a linear layer on cached feature tensors is ~6k steps for full DTD and ~800 for 5-shot, a few minutes across all 27 runs. More importantly, the spec requires loss curves that *"indicate whether substantial overfitting occurs"*, and stopping shortly after the val turn truncates the curve exactly where that evidence lives. A fixed 200 epochs also keeps every history the same length, which keeps plotting and aggregation simple.

    Instead, record **`best_epoch`** in each run record as the diagnostic. Interpretation to note in `DECISIONS.md`: consistently near 200 means the model was still improving and the budget may be too small; consistently under ~20 with val loss rising after means fast convergence and overfitting, which is expected at K=5 and worth discussing; scattered randomly means val accuracy is noisy and the tie-break rule in decision 5 is doing real work.

    Expose `patience: int | None = None` in the trainer config, defaulting to disabled, as a hook for Stage 2 heads that may be genuinely expensive to train. It must remain `None` for every Stage 1 run.

## Flag, don't silently resolve

If you hit an ambiguity not covered above, stop and ask rather than guessing. In particular I expect a question about the val-split protocol: the spec mandates using the *full* official val split for model selection even in the 5-shot setting (for DTD that means selecting a checkpoint for a 235-image training set using 1,880 val images). Implement it as specified, but note the caveat in `DECISIONS.md`.

---

## Build order

Implement in this order. **Each milestone must pass its tests before you start the next one.** Show me the test output at each milestone and pause for review.

**M0 — Skeleton.** Package layout, config dataclasses, global seeding utility, structured logging, `pytest` wired up, `pyproject.toml`, `.gitignore`.
*Test:* seeding utility produces identical tensors across two separate processes.

**M1 — Data layer.** Both datasets behind the uniform interface, download scripted and idempotent.
*Test:* split sizes and class counts match published numbers (DTD partition 1: 1880/1880/1880 over 47 classes; Flowers-102: 1020/1020/6149 over 102 classes); `len(class_names) == num_classes`; no index appears in two splits.

**M2 — Encoder registry.** ResNet-18 first, then DINOv2.
*Test:* output shape equals declared `feature_dim`; every parameter has `requires_grad == False`; two forward passes on the same batch are bit-identical.

**M3 — Feature cache.** Extraction with manifest; resumable; CLI `extract-features`.
*Test:* extract → reload → tensors identical; re-running an unchanged config is a no-op; changing the transform in the config invalidates the cache.

**M4 — Few-shot sampler.**
*Test:* exactly K per class; same seed → same indices; different seeds → different indices; all indices valid positions in the train split; works when a class has fewer than K available (should raise a clear error, not silently undersample).

**M5 — Image-prototype head.** No optimizer needed, so this produces the **first end-to-end accuracy number** and validates M1–M4 at once.
*Test:* prototypes are unit-norm; shape `[C, D]`; classifying the training features used to build a full-split prototype is clearly above chance; a synthetic dataset of well-separated Gaussian clusters is classified at 100%; **top-1 accuracy is identical for `temperature` in {1.0, 10.0, 100.0}** (guards the monotonicity assumption in decision 11); two `fit` calls on the same subset produce bit-identical prototypes.

**M6 — Linear-probe trainer.**
*Test:* overfits a 20-sample subset to ~100% train accuracy within a few hundred steps; average loss decreases; the returned checkpoint's val accuracy equals the maximum in the recorded history; curve history has **exactly `max_epochs` entries** for every run (guards decision 13); `best_epoch` is recorded and indexes the argmax of the val-accuracy history under the decision-5 tie-break; `patience=None` is the default and Stage 1 configs never override it.

**M7 — Evaluation and run records.**
*Test:* top-1 matches a hand-computed value on a toy logits tensor; confusion-matrix rows sum to 1; aggregation over 3 synthetic runs reproduces a known mean and `ddof=1` std; a run record round-trips through JSON without loss.

**M8 — Sweep runner.** The full 48-run Stage 1 grid via one command.
*Test:* `--dry-run` enumerates exactly 48 runs with the expected identifiers; an interrupted sweep resumes without duplicating records; re-running a completed sweep is a no-op.

Finish by printing the accuracy table (dataset × encoder × method × K, mean ± std) to stdout and writing it to `runs/stage1_summary.md`.

## Constraints

- Python, PyTorch, torchvision. Keep dependencies minimal and pinned.
- Type hints throughout; docstrings on every public function.
- No notebooks in the import path.
- No hardcoded absolute paths — everything through config.
- Do not commit downloaded data, cached features, or run outputs.
- Prefer small, testable functions over long scripts.
