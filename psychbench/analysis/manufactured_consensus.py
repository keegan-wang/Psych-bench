"""Aggregate asch_documents sweep outputs into comparison tables + tidy CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_sweep_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _ok_cells(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in manifest["cells"] if c.get("status") == "ok"]


def _match(cell: dict[str, Any], **filters: Any) -> bool:
    sv = cell.get("sweep_values", {})
    for k, v in filters.items():
        found = None
        for sv_key, sv_val in sv.items():
            if sv_key == k or sv_key.rsplit(".", 1)[-1] == k:
                found = sv_val
                break
        if found != v:
            return False
    return True


def _pick(cell: dict[str, Any], basename: str) -> Any:
    for k, v in cell.get("sweep_values", {}).items():
        if k.rsplit(".", 1)[-1] == basename:
            return v
    return None


def count_curve(
    manifest: dict[str, Any],
    document_type: str,
    dissenter: bool = False,
    rate: str = "conformity_rate_unanimous_only",
) -> list[dict[str, Any]]:
    rows = []
    for cell in _ok_cells(manifest):
        if not _match(cell, document_type=document_type, dissenter=dissenter):
            continue
        n = _pick(cell, "n_confederates")
        if n is None:
            continue
        rows.append({
            "n_confederates": n,
            "rate": cell["headline"].get(rate, 0.0),
            "cell_id": cell["cell_id"],
        })
    rows.sort(key=lambda r: r["n_confederates"])
    return rows


def dissenter_effect(
    manifest: dict[str, Any],
    n_confederates: int,
    document_type: str,
    rate: str = "conformity_rate_unanimous_only",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "dissenter_false": None, "dissenter_true": None,
    }
    for cell in _ok_cells(manifest):
        if not _match(cell, n_confederates=n_confederates,
                       document_type=document_type):
            continue
        diss = _pick(cell, "dissenter")
        if diss is None:
            continue
        key = "dissenter_true" if diss else "dissenter_false"
        out[key] = cell["headline"].get(rate, 0.0)
    if out["dissenter_false"] is not None and out["dissenter_true"] is not None:
        out["delta"] = out["dissenter_false"] - out["dissenter_true"]
    else:
        out["delta"] = None
    return out


def authority_ordering(
    manifest: dict[str, Any],
    n_confederates: int,
    dissenter: bool = False,
    rate: str = "conformity_rate_unanimous_only",
) -> list[dict[str, Any]]:
    rows = []
    for cell in _ok_cells(manifest):
        if not _match(cell, n_confederates=n_confederates,
                       dissenter=dissenter):
            continue
        dt = _pick(cell, "document_type")
        if dt is None:
            continue
        rows.append({
            "document_type": dt,
            "rate": cell["headline"].get(rate, 0.0),
            "cell_id": cell["cell_id"],
        })
    rows.sort(key=lambda r: -r["rate"])
    return rows


def write_tidy_csv(manifest: dict[str, Any], path: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    for cell in _ok_cells(manifest):
        base = {
            "cell_id": cell["cell_id"],
            **{k.rsplit(".", 1)[-1]: v
               for k, v in cell.get("sweep_values", {}).items()},
        }
        for rate_type, rate in cell["headline"].items():
            rows.append({**base, "rate_type": rate_type, "rate": rate})
    if not rows:
        Path(path).write_text("")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_report(manifest: dict[str, Any]) -> str:
    ok = sum(1 for c in manifest["cells"] if c.get("status") == "ok")
    failed = sum(1 for c in manifest["cells"] if c.get("status") == "failed")
    lines = [
        "=== asch_documents sweep summary ===",
        f"Cells: {len(manifest['cells'])}",
        f"OK: {ok}",
        f"Failed: {failed}",
    ]
    return "\n".join(lines)
