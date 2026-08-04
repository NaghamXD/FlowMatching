"""Shared figure-saving helper. Every deliverable emits PNG (300 dpi, for slides)
and PDF (vector, for the report); the PDF's embedded creation-date is suppressed
so two `make-report` runs produce byte-identical files (decision 14) -- without
this, matplotlib stamps the current time into every PDF and no two runs match."""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure


def save_figure(fig: matplotlib.figure.Figure, out_dir: str | Path, name: str) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, metadata={"CreationDate": None})
    return png_path, pdf_path
