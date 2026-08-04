"""Structured logging setup, shared across the CLI and all library modules."""

from __future__ import annotations

import logging
import sys

_ROOT_NAME = "cvlab"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger(_ROOT_NAME)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the shared `cvlab` namespace, e.g. `get_logger("features.cache")`
    yields a logger named `cvlab.features.cache`."""
    _configure_root()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def set_level(level: int) -> None:
    """Set the log level for the entire `cvlab` logger tree, e.g. `set_level(logging.DEBUG)`."""
    _configure_root()
    logging.getLogger(_ROOT_NAME).setLevel(level)
