"""Enumerate sweep cells and run each as a Session with failure isolation."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from psychbench.framework.sweep import SweepCell, expand_sweep

from .experiment import AschDocumentsExperiment


def run_sweep(
    config: dict[str, Any], output_root: str | Path,
) -> dict[str, Any]:
    timestamp = int(time.time())
    run_dir = Path(output_root) / f"asch_documents_{timestamp}"
    cells_dir = run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    sweep_fields = list(config.get("sweep", {}).get("fields", []))
    cells: list[SweepCell] = expand_sweep(config, sweep_fields=sweep_fields)

    cell_records: list[dict[str, Any]] = []
    for cell in cells:
        record = _run_one_cell(cell, cells_dir)
        cell_records.append(record)

    manifest = {
        "run_dir": str(run_dir),
        "timestamp": timestamp,
        "config": config,
        "sweep_fields": sweep_fields,
        "n_cells": len(cells),
        "cells": cell_records,
    }
    (run_dir / "sweep.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _run_one_cell(
    cell: SweepCell, cells_dir: Path,
) -> dict[str, Any]:
    try:
        exp = AschDocumentsExperiment(cell.config)
        out = exp.run(output_dir=cells_dir, session_label=cell.cell_id)
        summary = json.loads(Path(out["summary_path"]).read_text())
        return {
            "cell_id": cell.cell_id,
            "sweep_values": cell.sweep_values,
            "status": "ok",
            "log_path": out["log_path"],
            "summary_path": out["summary_path"],
            "headline": {
                "conformity_rate_unconditional":
                    summary.get("conformity_rate_unconditional"),
                "conformity_rate_unanimous_only":
                    summary.get("conformity_rate_unanimous_only"),
                "confederate_reliability":
                    summary.get("confederate_reliability"),
            },
        }
    except Exception as e:  # noqa: BLE001
        err_path = cells_dir / f"{cell.cell_id}.error.json"
        err_path.write_text(json.dumps({
            "cell_id": cell.cell_id,
            "sweep_values": cell.sweep_values,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, indent=2))
        return {
            "cell_id": cell.cell_id,
            "sweep_values": cell.sweep_values,
            "status": "failed",
            "error_path": str(err_path),
        }
