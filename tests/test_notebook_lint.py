"""
Structural lint for autoresearch notebooks.

Cheap, no-execution checks that catch config-cell mistakes before
papermill ever runs. Pairs with test_autoresearch_notebook.py (which
runs the notebook and checks output).

Enforced rules:

  1. Notebook is valid JSON + nbformat v4.
  2. At least one cell declares `METRIC_NAME`, `METRIC_TARGET`, `METRIC_DIRECTION`.
  3. Exactly one cell contains a `print(f"METRIC: {METRIC_NAME}={...}")` idiom.
  4. METRIC_DIRECTION is either 'maximize' or 'minimize' (no typos like "max"/"min"/"maxmize").
  5. METRIC_TARGET parses as a numeric literal.

Fails fast on config drift — the kind of mistake where the harness
might technically pass but the autoresearch loop can't make decisions.
"""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / os.environ.get(
    "AUTORESEARCH_NOTEBOOK", "notebooks/autoresearch.ipynb"
)

REQUIRED_VARS = {"METRIC_NAME", "METRIC_TARGET", "METRIC_DIRECTION"}
VALID_DIRECTIONS = {"maximize", "minimize"}


@pytest.fixture(scope="module")
def notebook():
    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
    return json.loads(NOTEBOOK_PATH.read_text())


@pytest.fixture(scope="module")
def all_code(notebook):
    """Flattened string of every code cell source."""
    parts = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        parts.append(src if isinstance(src, str) else "".join(src))
    return "\n\n".join(parts)


def test_notebook_is_valid_nbformat(notebook):
    assert notebook.get("nbformat") == 4, "Expected nbformat 4"
    assert isinstance(notebook.get("cells"), list), "Missing cells"
    assert len(notebook["cells"]) >= 2, "Need at least config + emit cells"


def test_required_config_vars_present(all_code):
    missing = []
    for var in REQUIRED_VARS:
        if not re.search(rf"\b{var}\s*=", all_code):
            missing.append(var)
    assert not missing, (
        f"Missing required config vars: {missing}. "
        "Add a top cell with METRIC_NAME, METRIC_TARGET, METRIC_DIRECTION."
    )


def test_metric_direction_value_is_valid(all_code):
    """Catches typos like 'max' or 'maxmize' at lint time, not runtime."""
    m = re.search(
        r"METRIC_DIRECTION\s*=\s*['\"]([^'\"]+)['\"]", all_code
    )
    assert m is not None, "METRIC_DIRECTION not assigned a string literal"
    value = m.group(1)
    assert value in VALID_DIRECTIONS, (
        f"METRIC_DIRECTION='{value}' invalid. Must be one of {VALID_DIRECTIONS}."
    )


def test_metric_target_is_numeric(all_code):
    m = re.search(r"METRIC_TARGET\s*=\s*(.+)", all_code)
    assert m is not None, "METRIC_TARGET not assigned"
    expr = m.group(1).split("#", 1)[0].strip()
    try:
        parsed = ast.literal_eval(expr)
    except (ValueError, SyntaxError):
        pytest.fail(
            f"METRIC_TARGET='{expr}' must be a numeric literal "
            "(int or float), not an expression."
        )
    assert isinstance(parsed, (int, float)), (
        f"METRIC_TARGET must be numeric; got {type(parsed).__name__}"
    )


def test_exactly_one_metric_print(all_code):
    """
    Catches the class of bug where someone writes `print('METRIC: foo')`
    without the value, or prints it twice (double emission can confuse
    regex extractors that take the first match).
    """
    # Match the canonical idiom: print(...METRIC: ...=...)
    # Looking for METRIC: in a print statement with an assignment `=`.
    pattern = re.compile(
        r"print\s*\([^)]*METRIC:\s*[^=)]*=[^)]*\)",
        re.DOTALL,
    )
    matches = pattern.findall(all_code)
    assert len(matches) >= 1, (
        "No `print(f'METRIC: {METRIC_NAME}={value}')` line found."
    )
    assert len(matches) == 1, (
        f"Expected exactly one METRIC print, found {len(matches)}. "
        "Multiple prints confuse downstream parsers. "
        "Matches: " + "; ".join(m[:80] for m in matches)
    )
