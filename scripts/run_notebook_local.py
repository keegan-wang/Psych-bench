"""Run the Colab notebook locally with minimal patches and report per-cell status.

Patches:
- Cell 2: use the existing local repo instead of cloning to /content/Psych-bench
- Cell 4: RUNTIME = "api"  (no GPU / torch on this Mac)
- Cell 5: skip the `!pip install` lines (already installed)
- Cell 12: force naive backend to "echo" so we don't need API keys
- Cell 18: force Phase 2 backend + judge to "echo" so we don't need API keys
All other cells are executed verbatim, in order, against the real psychbench CLI.

Usage:  python scripts/run_notebook_local.py
"""

from __future__ import annotations
import copy
import json
import os
import sys
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SRC = REPO_ROOT / "notebooks" / "psychbench_colab.ipynb"
NOTEBOOK_OUT = REPO_ROOT / "notebooks" / "psychbench_colab_local_executed.ipynb"


def _strip_shell_pip(src: str) -> str:
    """Replace `!pip install ...` lines with `pass` (preserves indentation)."""
    patched = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!pip "):
            indent = line[: len(line) - len(stripped)]
            patched.append(f"{indent}pass  # [local-run] skipped: {stripped}")
        else:
            patched.append(line)
    return "\n".join(patched)


def _patch(nb):
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source

        # Cell 2: clone/cd -> use local repo
        if 'REPO_DIR = "/content/Psych-bench"' in src:
            cell.source = (
                f'import os\n'
                f'os.chdir({str(REPO_ROOT)!r})\n'
                f'print("cwd:", os.getcwd())\n'
            )
            continue

        # Cell 4: RUNTIME default
        if src.strip().startswith('RUNTIME = "hf"'):
            cell.source = 'RUNTIME = "api"   # local-run override\n'
            continue

        # Cell 5: skip pip installs
        if "!pip install" in src:
            cell.source = _strip_shell_pip(src)
            continue

        # Cell 12: force NAIVE_BACKEND to echo (no API keys locally)
        if 'API_NAIVE_BACKEND = "openai"' in src:
            cell.source = src.replace(
                'API_NAIVE_BACKEND = "openai"',
                'API_NAIVE_BACKEND = "echo"',
            ).replace(
                'API_NAIVE_MODEL   = "gpt-4o-mini"',
                'API_NAIVE_MODEL   = "echo-test"',
            )
            continue

        # Cell 18: force Phase 2 agent + judge backends to echo
        if 'API_P2_BACKEND       = "openai"' in src:
            cell.source = (
                src
                .replace('API_P2_BACKEND       = "openai"',
                         'API_P2_BACKEND       = "echo"')
                .replace('API_P2_MODEL         = "gpt-4o-mini"',
                         'API_P2_MODEL         = "echo-test"')
                .replace('P2_JUDGE_BACKEND     = "anthropic"',
                         'P2_JUDGE_BACKEND     = "echo"')
                .replace('P2_JUDGE_MODEL       = "claude-haiku-4-5-20251001"',
                         'P2_JUDGE_MODEL       = "echo-test"')
            )
            continue

    return nb


def main() -> int:
    nb = nbformat.read(NOTEBOOK_SRC, as_version=4)
    nb = _patch(copy.deepcopy(nb))

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
        allow_errors=True,
    )

    print(f"Executing {NOTEBOOK_SRC.name} with local patches...\n")
    try:
        client.execute()
    except CellExecutionError as e:
        print("Fatal execution error:", e)

    nbformat.write(nb, NOTEBOOK_OUT)

    print("\n=== Per-cell summary ===")
    ok, fail = 0, 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        had_err = any(
            out.get("output_type") == "error" for out in cell.get("outputs", [])
        )
        label = "FAIL" if had_err else "OK"
        first_line = cell.source.splitlines()[0] if cell.source.strip() else "(empty)"
        print(f"  cell[{i:02d}] {label:4s} | {first_line[:70]}")
        if had_err:
            fail += 1
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    print(f"      -> {out['ename']}: {out['evalue']}")
        else:
            ok += 1
    print(f"\nTotals: {ok} ok, {fail} failed")
    print(f"Executed notebook saved to: {NOTEBOOK_OUT.relative_to(REPO_ROOT)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
