"""Deliverable 1 (M9): the Stage 1 accuracy table, in Markdown/CSV/LaTeX.
Decision 14: deterministic and regenerable -- row order is a fixed sort key,
never dict or filesystem iteration order, so two runs produce byte-identical
output."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from cvlab.evaluation.aggregate import aggregate
from cvlab.reporting.common import (
    ENCODER_LABEL,
    K_ORDER,
    METHOD_LABEL,
    METHOD_ORDER,
    images_per_class,
)
from cvlab.reporting.loader import RunCell


@dataclass(frozen=True)
class AccuracyTableRow:
    dataset: str
    encoder: str
    method: str
    k: int | str
    images_per_class: int
    n_runs: int
    top1_mean: float
    top1_std: float
    mpca_mean: float
    mpca_std: float


def build_accuracy_table(
    cells: dict[tuple[str, str, str, int | str], RunCell],
) -> list[AccuracyTableRow]:
    rows = []
    for (dataset, encoder, method, k), cell in cells.items():
        top1_agg = aggregate([r.test_top1 for r in cell.records])
        mpca_agg = aggregate([r.mean_per_class_accuracy for r in cell.records])
        rows.append(
            AccuracyTableRow(
                dataset=dataset,
                encoder=encoder,
                method=method,
                k=k,
                images_per_class=images_per_class(cell),
                n_runs=top1_agg.n,
                top1_mean=top1_agg.mean,
                top1_std=top1_agg.std,
                mpca_mean=mpca_agg.mean,
                mpca_std=mpca_agg.std,
            )
        )
    rows.sort(key=lambda r: (r.dataset, r.encoder, METHOD_ORDER[r.method], K_ORDER.index(r.k)))
    return rows


_FOOTNOTE = (
    "Single deterministic run (no stochasticity to average over -- not the "
    "same as ± 0.00, which would falsely imply 3 runs agreed exactly)."
)


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}"


def _top1_cell_markdown(row: AccuracyTableRow) -> str:
    if row.n_runs == 1:
        return f"{_fmt_pct(row.top1_mean)}%¹"
    return f"{_fmt_pct(row.top1_mean)}% ± {_fmt_pct(row.top1_std)}%"


def render_markdown(rows: list[AccuracyTableRow]) -> str:
    lines = [
        "# Stage 1 Accuracy Table",
        "",
        "| Dataset | Encoder | Method | K | Images/class | n_runs | Top-1 |",
        "|---|---|---|---|---|---|---|",
    ]
    any_single_run = False
    for r in rows:
        if r.n_runs == 1:
            any_single_run = True
        lines.append(
            f"| {r.dataset} | {ENCODER_LABEL[r.encoder]} | {METHOD_LABEL[r.method]} | "
            f"{r.k} | {r.images_per_class} | {r.n_runs} | {_top1_cell_markdown(r)} |"
        )
    if any_single_run:
        lines += ["", f"¹ {_FOOTNOTE}"]
    return "\n".join(lines) + "\n"


def render_markdown_with_mpca(rows: list[AccuracyTableRow]) -> str:
    lines = [
        "# Stage 1 Accuracy Table (top-1 and mean-per-class)",
        "",
        "| Dataset | Encoder | Method | K | Images/class | n_runs | Top-1 | Mean-per-class |",
        "|---|---|---|---|---|---|---|---|",
    ]
    any_single_run = False
    for r in rows:
        if r.n_runs == 1:
            any_single_run = True
            mpca_cell = f"{_fmt_pct(r.mpca_mean)}%¹"
        else:
            mpca_cell = f"{_fmt_pct(r.mpca_mean)}% ± {_fmt_pct(r.mpca_std)}%"
        lines.append(
            f"| {r.dataset} | {ENCODER_LABEL[r.encoder]} | {METHOD_LABEL[r.method]} | "
            f"{r.k} | {r.images_per_class} | {r.n_runs} | {_top1_cell_markdown(r)} | {mpca_cell} |"
        )
    if any_single_run:
        lines += ["", f"¹ {_FOOTNOTE}"]
    return "\n".join(lines) + "\n"


def _write_csv(rows: list[AccuracyTableRow], columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for r in rows:
        line = [r.dataset, r.encoder, r.method, r.k, r.images_per_class, r.n_runs, f"{r.top1_mean:.4f}", f"{r.top1_std:.4f}"]
        if "mpca_mean" in columns:
            line += [f"{r.mpca_mean:.4f}", f"{r.mpca_std:.4f}"]
        writer.writerow(line)
    return buf.getvalue()


def render_csv(rows: list[AccuracyTableRow]) -> str:
    return _write_csv(
        rows, ["dataset", "encoder", "method", "k", "images_per_class", "n_runs", "top1_mean", "top1_std"]
    )


def render_csv_with_mpca(rows: list[AccuracyTableRow]) -> str:
    return _write_csv(
        rows,
        [
            "dataset", "encoder", "method", "k", "images_per_class", "n_runs",
            "top1_mean", "top1_std", "mpca_mean", "mpca_std",
        ],
    )


def _top1_cell_latex(row: AccuracyTableRow) -> str:
    if row.n_runs == 1:
        return f"{_fmt_pct(row.top1_mean)}$^\\dagger$"
    return f"{_fmt_pct(row.top1_mean)} $\\pm$ {_fmt_pct(row.top1_std)}"


def render_latex(rows: list[AccuracyTableRow]) -> str:
    lines = [
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"Dataset & Encoder & Method & $K$ & Images/class & $n$ & Top-1 (\%) \\",
        r"\midrule",
    ]
    any_single_run = False
    for r in rows:
        if r.n_runs == 1:
            any_single_run = True
        lines.append(
            f"{r.dataset} & {ENCODER_LABEL[r.encoder]} & {METHOD_LABEL[r.method]} & "
            f"{r.k} & {r.images_per_class} & {r.n_runs} & {_top1_cell_latex(r)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if any_single_run:
        lines.append(f"% $\\dagger$ {_FOOTNOTE}")
    return "\n".join(lines) + "\n"


def render_latex_with_mpca(rows: list[AccuracyTableRow]) -> str:
    lines = [
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Dataset & Encoder & Method & $K$ & Images/class & $n$ & Top-1 (\%) & Mean-per-class (\%) \\",
        r"\midrule",
    ]
    any_single_run = False
    for r in rows:
        if r.n_runs == 1:
            any_single_run = True
            mpca_cell = f"{_fmt_pct(r.mpca_mean)}$^\\dagger$"
        else:
            mpca_cell = f"{_fmt_pct(r.mpca_mean)} $\\pm$ {_fmt_pct(r.mpca_std)}"
        lines.append(
            f"{r.dataset} & {ENCODER_LABEL[r.encoder]} & {METHOD_LABEL[r.method]} & "
            f"{r.k} & {r.images_per_class} & {r.n_runs} & {_top1_cell_latex(r)} & {mpca_cell} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if any_single_run:
        lines.append(f"% $\\dagger$ {_FOOTNOTE}")
    return "\n".join(lines) + "\n"
