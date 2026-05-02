"""Insert plotting cells into notebooks/psychbench_colab.ipynb.

Idempotent: looks for a sentinel in an existing cell before inserting.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "psychbench_colab.ipynb"

SENTINEL_P1 = "psychbench-plot-phase1-v1"
SENTINEL_P2 = "psychbench-plot-phase2-v1"

PHASE1_MD = """\
### Phase 1 — visualize the results

Two panels:

- **Left:** headline conformity rate, experimental (public) vs control (private).
  Δ is the social-pressure effect — a big positive Δ means the model caves to
  the majority only when it hears them. If Δ ≈ 0, the model's behavior does
  not depend on whether peers are visible.
- **Right:** one marker per critical trial showing the naive's answer. Green
  circle = correct (independent), red X = conformed with the wrong majority,
  grey square = some *other* wrong answer (the model was confused, not
  conforming).
"""

PHASE1_CODE = f"""\
# {SENTINEL_P1}
%matplotlib inline
import glob
from psychbench.analysis.plots import plot_phase1

exp_summary  = sorted(glob.glob("results/asch_experimental_*.summary.json"))[-1]
ctrl_summary = sorted(glob.glob("results/asch_control_*.summary.json"))[-1]
plot_phase1(exp_summary, ctrl_summary, save="results/phase1_plot.png");
"""

PHASE2_MD = """\
### Phase 2 — visualize the results

Two panels:

- **Left:** per sweep cell, three conformity rates side-by-side —
  **Full** (strict substring match of the wrong answer), **Partial (heuristic)**
  (cheap rules-based check), **Partial (LLM judge)** (another model reads the
  naive's reply and decides if it endorses the wrong claim). When these three
  bars disagree, the finding is definition-sensitive: the naive is hedging
  rather than flatly accepting or rejecting.
- **Right:** per-trial heatmap. Each column is one of the 12 fictional-fact
  trials. Rows show whether the confederates were unanimous, and which of the
  three conformity definitions fired. Use this to spot individual trials that
  drove the aggregate.
"""

PHASE2_CODE = f"""\
# {SENTINEL_P2}
%matplotlib inline
import glob
from psychbench.analysis.plots import plot_phase2

run_dir = sorted(glob.glob("results/asch_documents_*"))[-1]
plot_phase2(run_dir, save="results/phase2_plot.png");
"""


def _contains_sentinel(cells, sentinel: str) -> bool:
    return any(sentinel in c.source for c in cells if c.cell_type == "code")


def _find_cell_starting_with(cells, prefix: str) -> int | None:
    for i, c in enumerate(cells):
        if c.source.lstrip().startswith(prefix):
            return i
    return None


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)

    if _contains_sentinel(nb.cells, SENTINEL_P1) and _contains_sentinel(
        nb.cells, SENTINEL_P2
    ):
        print("Plot cells already present — nothing to do.")
        return

    # Insert after the Phase-2 spot-check cell (starts with `import json, glob`
    # followed by `run_dir = Path(...)`), which is the last cell before
    # "## 7. Scaling up". We locate it by finding the "## 7." markdown header
    # and inserting immediately before it.
    scaling_idx = None
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and c.source.lstrip().startswith(
            "## 7. Scaling up"
        ):
            scaling_idx = i
            break
    if scaling_idx is None:
        raise RuntimeError("Could not locate '## 7. Scaling up' header")

    if not _contains_sentinel(nb.cells, SENTINEL_P2):
        nb.cells.insert(scaling_idx, nbformat.v4.new_code_cell(PHASE2_CODE))
        nb.cells.insert(scaling_idx, nbformat.v4.new_markdown_cell(PHASE2_MD))

    # Insert after the Phase-1 spot-check cell. That cell starts with
    # "# Spot-check one critical trial."; we insert right after it.
    spot_idx = None
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and c.source.lstrip().startswith(
            "# Spot-check one critical trial."
        ):
            spot_idx = i
            break
    if spot_idx is None:
        raise RuntimeError("Could not locate Phase-1 spot-check cell")

    if not _contains_sentinel(nb.cells, SENTINEL_P1):
        nb.cells.insert(spot_idx + 1, nbformat.v4.new_code_cell(PHASE1_CODE))
        nb.cells.insert(spot_idx + 1, nbformat.v4.new_markdown_cell(PHASE1_MD))

    nbformat.write(nb, NB_PATH)
    print(f"Updated {NB_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
