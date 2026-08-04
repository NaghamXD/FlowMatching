"""Reporting and plotting — out of scope for Stage 1.

Will read exclusively from the `results` store (never re-run experiments) to produce:
tables of test top-1 / mean-per-class accuracy by (dataset, encoder, method, K),
train/val loss and accuracy curves per run, and row-normalized confusion matrices.
Everything this package will need (curves, predictions, confusion-matrix inputs) is
already recorded by `results` as of M7.
"""
