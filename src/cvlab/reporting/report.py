"""`make-report`: regenerates every Stage 1 report artifact from the results
store. Every function here is pure -- results store in, file paths out; no
global state, no `plt.show()` (constraint from the reporting spec)."""

from __future__ import annotations

from pathlib import Path

from cvlab.reporting.confusion_plots import build_confusion_artifacts
from cvlab.reporting.feature_viz_plots import plot_figure_5a, plot_figure_5b, plot_figure_5c
from cvlab.reporting.loader import load_run_matrix
from cvlab.reporting.plots import plot_accuracy_vs_k
from cvlab.reporting.tables import (
    build_accuracy_table,
    render_csv,
    render_csv_with_mpca,
    render_latex,
    render_latex_with_mpca,
    render_markdown,
    render_markdown_with_mpca,
)
from cvlab.reporting.training_curves import (
    plot_training_curves,
    render_training_curves_captions,
    select_all_representatives,
)
from cvlab.utils.io import ensure_dir
from cvlab.utils.logging import get_logger

logger = get_logger("reporting.report")


def make_report(
    runs_root: str | Path,
    reports_root: str | Path,
    configs_dir: str | Path = "configs",
    cache_root: str | Path = "cache/features",
) -> list[Path]:
    """Regenerate all Stage 1 report artifacts. Returns the list of files written."""
    reports_dir = ensure_dir(reports_root)
    cells = load_run_matrix(runs_root)
    written: list[Path] = []

    rows = build_accuracy_table(cells)
    for name, content in (
        ("accuracy_table.md", render_markdown(rows)),
        ("accuracy_table.csv", render_csv(rows)),
        ("accuracy_table.tex", render_latex(rows)),
        ("accuracy_table_mean_per_class.md", render_markdown_with_mpca(rows)),
        ("accuracy_table_mean_per_class.csv", render_csv_with_mpca(rows)),
        ("accuracy_table_mean_per_class.tex", render_latex_with_mpca(rows)),
    ):
        path = reports_dir / name
        path.write_text(content)
        written.append(path)

    written.extend(plot_accuracy_vs_k(cells, reports_dir))

    written.extend(plot_training_curves(cells, reports_dir))
    representatives = select_all_representatives(cells)
    caption_path = reports_dir / "deliverable3_training_curves_captions.txt"
    caption_path.write_text(render_training_curves_captions(representatives))
    written.append(caption_path)

    written.extend(build_confusion_artifacts(cells, reports_dir, configs_dir))

    for plot_fn, caption_name in (
        (plot_figure_5a, "deliverable5a_dtd_feature_viz_caption.txt"),
        (plot_figure_5b, "deliverable5b_flowers102_feature_viz_caption.txt"),
        (plot_figure_5c, "deliverable5c_dtd_kshot_vs_full_prototypes_caption.txt"),
    ):
        paths, caption = plot_fn(cells, cache_root, reports_dir, configs_dir)
        written.extend(paths)
        caption_path = reports_dir / caption_name
        caption_path.write_text(caption)
        written.append(caption_path)
    written.append(reports_dir / "deliverable5_class_selection.json")

    logger.info("wrote %d report artifact(s) to %s", len(written), reports_dir)
    return written
