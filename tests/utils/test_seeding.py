"""M0 test: the seeding utility must produce identical tensors across two
*separate* processes, not just two calls within one process — a weaker,
in-process check could pass while still leaving state leaking between runs
(e.g. a global RNG not actually reset)."""

from __future__ import annotations

import subprocess
import sys

_SCRIPT = """
import json
import random

import numpy as np
import torch

from cvlab.utils.seeding import set_seed

set_seed(1234)
result = {
    "python_random": [random.random() for _ in range(5)],
    "numpy": np.random.rand(5).tolist(),
    "torch": torch.rand(5).tolist(),
}
print(json.dumps(result))
"""


def _run_in_subprocess() -> str:
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def test_set_seed_is_reproducible_across_processes() -> None:
    first = _run_in_subprocess()
    second = _run_in_subprocess()
    assert first == second
    assert first != ""


def test_set_seed_changes_output_for_different_seeds() -> None:
    script_a = _SCRIPT.replace("set_seed(1234)", "set_seed(1)")
    script_b = _SCRIPT.replace("set_seed(1234)", "set_seed(2)")
    out_a = subprocess.run(
        [sys.executable, "-c", script_a], capture_output=True, text=True, check=True
    ).stdout.strip()
    out_b = subprocess.run(
        [sys.executable, "-c", script_b], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out_a != out_b
