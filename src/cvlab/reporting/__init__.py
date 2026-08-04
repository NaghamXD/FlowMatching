"""Reporting and visualization. Reads exclusively from the `results` store --
never re-runs an experiment, never re-extracts a feature, never touches the test
split directly (the one exception is cached features for M12's visualizations).
Also reads static class-name metadata (see `labels.py`), which is neither of
those things but is flagged there as a second data source."""

from cvlab.reporting.confusion import grow_confusion_cluster, hierarchical_leaf_order, load_confusion_matrix
from cvlab.reporting.confusion_plots import build_confusion_artifacts, get_full_split_seed0_confusion
from cvlab.reporting.feature_viz import (
    compute_and_save_class_selection,
    compute_prototypes,
    load_class_selection,
    select_test_indices,
    select_visualization_classes,
)
from cvlab.reporting.feature_viz_plots import plot_figure_5a, plot_figure_5b, plot_figure_5c
from cvlab.reporting.loader import RunCell, load_run_matrix
from cvlab.reporting.projection import joint_pca_projection
from cvlab.reporting.report import make_report
from cvlab.reporting.representative import select_representative_run
from cvlab.reporting.tables import AccuracyTableRow, build_accuracy_table

__all__ = [
    "RunCell",
    "load_run_matrix",
    "make_report",
    "select_representative_run",
    "AccuracyTableRow",
    "build_accuracy_table",
    "grow_confusion_cluster",
    "hierarchical_leaf_order",
    "load_confusion_matrix",
    "build_confusion_artifacts",
    "get_full_split_seed0_confusion",
    "compute_and_save_class_selection",
    "compute_prototypes",
    "load_class_selection",
    "select_test_indices",
    "select_visualization_classes",
    "plot_figure_5a",
    "plot_figure_5b",
    "plot_figure_5c",
    "joint_pca_projection",
]
