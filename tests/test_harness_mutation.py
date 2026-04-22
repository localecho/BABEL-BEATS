"""
Mutation test for the autoresearch contract.

Proves the harness catches broken notebooks: we copy the real notebook,
delete the METRIC line, run the contract test against the mutant, and
expect it to FAIL. If it passes, the harness is asleep.

Run alongside `test_autoresearch_notebook.py`:
    pytest tests/

Skipped if papermill is unavailable.
"""
from __future__ import annotations

import json
import os
import shutil
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
HARNESS = REPO_ROOT / "tests" / "test_autoresearch_notebook.py"


def _papermill_available() -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "papermill", "--version"],
            capture_output=True, check=True, timeout=10,
        )
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _papermill_available() or not NOTEBOOK_PATH.exists(),
    reason="papermill not installed or notebook missing",
)
def test_harness_catches_missing_metric_line():
    """
    Mutate a copy of the notebook to remove the METRIC: print, then run
    the harness against it — it MUST fail (return non-zero). If it passes,
    the contract test is not actually enforcing anything.
    """
    with tempfile.TemporaryDirectory() as td:
        # Set up a faux repo with mutated notebook + harness, preserve
        # the repo structure so relative paths still resolve.
        fake_repo = Path(td) / "fake_repo"
        (fake_repo / "notebooks").mkdir(parents=True)
        (fake_repo / "tests").mkdir()

        nb = json.loads(NOTEBOOK_PATH.read_text())
        # Delete any line matching 'METRIC:' in any cell
        for cell in nb["cells"]:
            src = cell.get("source", [])
            if isinstance(src, str):
                src = [src]
            new_src = [line for line in src if "METRIC:" not in line]
            cell["source"] = new_src
        (fake_repo / "notebooks" / "autoresearch.ipynb").write_text(
            json.dumps(nb)
        )
        shutil.copy(HARNESS, fake_repo / "tests" / "test_autoresearch_notebook.py")

        env = dict(os.environ)
        env["AUTORESEARCH_METRIC"] = METRIC_NAME
        env["AUTORESEARCH_NOTEBOOK"] = "notebooks/autoresearch.ipynb"

        # Prefer running only the contract test (not mutation, not regression)
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/test_autoresearch_notebook.py::test_notebook_emits_metric",
                "-q",
            ],
            capture_output=True, text=True, cwd=str(fake_repo), env=env,
            timeout=600,
        )
        assert result.returncode != 0, (
            "Harness failed to catch missing METRIC line — mutation survived. "
            f"stdout: {result.stdout[-500:]}"
        )
