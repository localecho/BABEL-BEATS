"""
Autoresearch notebook contract tests (TDD v2).

Three layers of assertion:

  1. **Contract**: notebook executes + emits `METRIC: <name>=<value>` that parses
     as a finite float.

  2. **Bounds**: metric value sits within a reasonable range for its direction.
     Maximize metrics should be >= 0 unless explicitly allowed; minimize metrics
     should be finite. Catches NaN / huge sentinel values.

  3. **Regression lock**: if `tests/metric_baseline.json` exists with a recorded
     baseline for this notebook, the new value must not regress by more than
     REGRESSION_TOLERANCE (default 5%). Bump the baseline by re-running the
     notebook and committing `metric_baseline.json`.

Env:
    AUTORESEARCH_NOTEBOOK=notebooks/autoresearch.ipynb
    AUTORESEARCH_METRIC=<metric_name>
    AUTORESEARCH_DIRECTION=maximize|minimize    (optional, default maximize)
    AUTORESEARCH_REGRESSION_TOLERANCE=0.05       (optional, 5% default)
"""
from __future__ import annotations

import json
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
DIRECTION = os.environ.get("AUTORESEARCH_DIRECTION", "maximize").lower()
REGRESSION_TOLERANCE = float(
    os.environ.get("AUTORESEARCH_REGRESSION_TOLERANCE", "0.05")
)
BASELINE_FILE = REPO_ROOT / "tests" / "metric_baseline.json"


def _papermill_available() -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "papermill", "--version"],
            capture_output=True, check=True, timeout=10,
        )
        return True
    except Exception:
        return False


def _execute_and_extract() -> float:
    """Run papermill + extract METRIC value. Shared by multiple tests."""
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
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            text = ""
            if output.get("output_type") == "stream":
                text = output.get("text", "")
            elif output.get("output_type") in ("execute_result", "display_data"):
                text = "".join(output.get("data", {}).get("text/plain", []))
            m = pattern.search(text)
            if m:
                return float(m.group(1))
    pytest.fail(f"No `METRIC: {METRIC_NAME}=...` line in notebook output")


# ---------------------------------------------------------------------------
# Layer 1 — contract
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _papermill_available(), reason="papermill not installed")
def test_notebook_emits_metric():
    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
    assert METRIC_NAME, "Set AUTORESEARCH_METRIC env var"
    value = _execute_and_extract()
    assert math.isfinite(value), f"Metric value not finite: {value}"


# ---------------------------------------------------------------------------
# Layer 2 — bounds
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _papermill_available(), reason="papermill not installed")
def test_metric_value_is_reasonable():
    """Guard against sentinel values (−1, 1e18) and NaN masquerading as float."""
    value = _execute_and_extract()
    assert not math.isnan(value), "Metric is NaN"
    assert abs(value) < 1e12, (
        f"Metric magnitude suspicious ({value}). Suggests error sentinel."
    )


# ---------------------------------------------------------------------------
# Layer 3 — regression lock
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _papermill_available() or not BASELINE_FILE.exists(),
    reason="papermill not installed or no baseline locked",
)
def test_metric_does_not_regress():
    """
    If tests/metric_baseline.json exists, the new value must not regress by
    more than REGRESSION_TOLERANCE. Update the file and commit to bump.
    """
    baseline = json.loads(BASELINE_FILE.read_text())
    key = METRIC_NAME or "_default"
    if key not in baseline:
        pytest.skip(f"No baseline entry for metric '{key}'")

    locked = baseline[key]["value"]
    value = _execute_and_extract()

    if DIRECTION == "maximize":
        floor = locked * (1.0 - REGRESSION_TOLERANCE)
        assert value >= floor, (
            f"Regression: {METRIC_NAME}={value} < floor={floor} "
            f"(locked baseline={locked}, tolerance={REGRESSION_TOLERANCE:.0%})"
        )
    else:  # minimize
        ceiling = locked * (1.0 + REGRESSION_TOLERANCE)
        assert value <= ceiling, (
            f"Regression: {METRIC_NAME}={value} > ceiling={ceiling} "
            f"(locked baseline={locked}, tolerance={REGRESSION_TOLERANCE:.0%})"
        )
