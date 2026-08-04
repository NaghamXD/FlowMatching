# CVLAB Summer Project — Stage 1

Frozen-feature classification on DTD and Oxford Flowers-102, using two frozen
encoders (ResNet-18/IN1K, DINOv2 ViT-S/14) and two classification heads
(trained linear probe, training-free image-derived class prototypes).

Stage 1 covers linear probing and image prototypes only. See
[`claude_code_stage1_prompt.md`](claude_code_stage1_prompt.md) for the full
spec and [`DECISIONS.md`](DECISIONS.md) for pinned design decisions. The
codebase is structured so that Stage 2 (a flow-matching head) and Stage 3
(reusing the linear-probe setting) can be added without modifying Stage 1
code — see the module layout below.

## Setup

```bash
conda create -n cvlab python=3.11 -y
conda activate cvlab
pip install -e ".[dev]"
```

## Layout

```
configs/           # YAML fragments for datasets, encoders, methods, sweeps
src/cvlab/
  config/           # typed dataclasses + YAML loading
  data/             # uniform (PIL image, label) interface over DTD / Flowers-102
  encoders/         # frozen backbone registry — the only package that sees pixels
  features/         # extraction + manifest-checked caching
  sampling/         # deterministic balanced K-shot index selection
  heads/            # LinearProbe, ImagePrototype behind one fit/logits interface
  training/         # optimization loop for parametric heads
  evaluation/        # metrics + seed aggregation
  results/          # run-record schema + JSON store
  runner/           # CLI: extract-features, run-experiment, run-sweep, list-runs
  reporting/        # empty in Stage 1 — reads only the results store
tests/              # mirrors src/cvlab/
cache/features/     # gitignored — cached feature tensors + manifests
runs/               # gitignored — one run record (JSON) + artifacts per run
```

## Data flow

```
raw images -> data -> encoder's own eval transform -> frozen encoder (no_grad)
  -> feature cache [N, D] + labels + manifest   (once per dataset x split x encoder)
  -> sampling.balanced_kshot(labels, K, seed) -> indices
  -> head.fit(Z_train[idx], y_train[idx], Z_val, y_val)
  -> head.logits(Z_test) -> evaluation -> results store
```

## Status

Implementing per the milestone order in the spec (M0-M8). See git history /
`runs/stage1_summary.md` (once M8 lands) for progress.

## Tests

```bash
pytest
```
