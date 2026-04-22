"""
Autoresearch notebook contract test.

Asserts the target notebook:
  1. Executes without error under papermill
  2. Emits a line matching `METRIC: <name>=<value>` in cell output
  3. <value> parses as a finite float

Wire into each repo's CI so the keep/revert loop always has a valid signal.

Usage in a repo:
    pytest tests/test_autoresearch_notebook.py

Override notebook path + metric name via env:
    AUTORESEARCH_NOTEBOOK=notebooks/autoresearch.ipynb
    AUTORESEARCH_METRIC=<metric_name>
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / os.environ.get(
    "AUTORESEARCH_NOTEBOOK", "notebooks/autoresearch.ipynb"
)
METRIC_NAME = os.environ.get("AUTORESEARCH_METRIC", "")


def _papermill_available() -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "papermill", "--version"],
            capture_output=True, check=True, timeout=10,
        )
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _papermill_available(), reason="papermill not installed")
def test_notebook_emits_metric():
    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
    assert METRIC_NAME, "Set AUTORESEARCH_METRIC env var or hardcode in test"

    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tmp:
        out_path = tmp.name

    result = subprocess.run(
        [
            sys.executable, "-m", "papermill",
            str(NOTEBOOK_PATH), out_path,
            "--no-progress-bar",
        ],
        capture_output=True, text=True, timeout=600,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"papermill failed: {result.stderr[-1000:]}"
    )

    import nbformat
    nb = nbformat.read(out_path, as_version=4)
    pattern = re.compile(
        rf"METRIC:\s*{re.escape(METRIC_NAME)}\s*=\s*([0-9eE.+\-]+)"
    )
    found = None
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            text = ""
            if output.get("output_type") == "stream":
                text = output.get("text", "")
            elif output.get("output_type") in ("execute_result", "display_data"):
                text = "".join(output.get("data", {}).get("text/plain", []))
            m = pattern.search(text)
            if m:
                found = m.group(1)
                break
        if found:
            break

    assert found is not None, (
        f"No `METRIC: {METRIC_NAME}=...` line found in notebook output"
    )
    value = float(found)
    assert math.isfinite(value), f"Metric value not finite: {value}"
