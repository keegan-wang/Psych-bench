"""Sweep-level aggregation, comparison tables, tidy CSV export."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from psychbench.analysis.manufactured_consensus import (
    authority_ordering, count_curve, dissenter_effect,
    load_sweep_manifest, write_tidy_csv,
)


def _write_manifest(tmp_path: Path) -> Path:
    cells = []
    for n, dt, diss in [
        (1, "wikipedia", False), (3, "wikipedia", False),
        (5, "wikipedia", False), (7, "wikipedia", False),
        (5, "forum", False), (5, "news", False),
        (5, "wikipedia", True),
    ]:
        cells.append({
            "cell_id": (
                f"n_confederates={n}__document_type={dt}__"
                f"dissenter={str(diss).lower()}"
            ),
            "sweep_values": {
                "agents.n_confederates": n,
                "documents.document_type": dt,
                "agents.dissenter": diss,
            },
            "status": "ok",
            "headline": {
                "conformity_rate_unconditional": 0.1 * n,
                "conformity_rate_unanimous_only": 0.12 * n,
                "confederate_reliability": 0.9,
            },
        })
    manifest = {
        "run_dir": str(tmp_path),
        "config": {"experiment": {"type": "asch_documents"}},
        "sweep_fields": [
            "agents.n_confederates",
            "documents.document_type",
            "agents.dissenter",
        ],
        "cells": cells,
    }
    p = tmp_path / "sweep.json"
    p.write_text(json.dumps(manifest))
    return p


def test_load_sweep_manifest(tmp_path):
    p = _write_manifest(tmp_path)
    m = load_sweep_manifest(p)
    assert len(m["cells"]) == 7


def test_count_curve_returns_rates_by_n_confederates(tmp_path):
    p = _write_manifest(tmp_path)
    m = load_sweep_manifest(p)
    curve = count_curve(m, document_type="wikipedia")
    assert [row["n_confederates"] for row in curve] == [1, 3, 5, 7]
    assert [round(row["rate"], 2) for row in curve] == [0.12, 0.36, 0.60, 0.84]


def test_dissenter_effect(tmp_path):
    p = _write_manifest(tmp_path)
    m = load_sweep_manifest(p)
    eff = dissenter_effect(m, n_confederates=5, document_type="wikipedia")
    assert eff["dissenter_false"] == 0.60
    assert eff["dissenter_true"] == 0.60
    assert "delta" in eff


def test_authority_ordering(tmp_path):
    p = _write_manifest(tmp_path)
    m = load_sweep_manifest(p)
    order = authority_ordering(m, n_confederates=5)
    dts = [row["document_type"] for row in order]
    assert set(dts) == {"wikipedia", "forum", "news"}


def test_write_tidy_csv(tmp_path):
    p = _write_manifest(tmp_path)
    m = load_sweep_manifest(p)
    out = tmp_path / "tidy.csv"
    write_tidy_csv(m, out)
    assert out.exists()
    rows = list(csv.DictReader(out.open()))
    assert len(rows) >= len(m["cells"])
    assert "cell_id" in rows[0]
    assert "rate_type" in rows[0]
    assert "rate" in rows[0]
