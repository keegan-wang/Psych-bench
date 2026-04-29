"""Sweep runner: iterate cells, nested output, failure isolation."""
from __future__ import annotations

import json
from pathlib import Path

from psychbench.experiments.asch_documents.sweep_runner import run_sweep


FIXTURE = Path(__file__).parent / "fixtures" / "mini_corpus.yaml"


def _sweep_config():
    return {
        "experiment": {"name": "t", "type": "asch_documents",
                        "trials": 2, "seed": 7, "n_repeats": 1},
        "corpus": {"path": str(FIXTURE)},
        "sweep": {
            "fields": [
                "agents.n_confederates",
                "documents.document_type",
            ],
        },
        "agents": {
            "n_confederates": [1, 2],
            "dissenter": False,
            "confederate": {"type": "model", "backend": "echo",
                             "model": "echo-test", "stateful": False},
            "naive": {"type": "model", "backend": "echo",
                       "model": "echo-test", "stateful": False,
                       "position": "last"},
        },
        "documents": {
            "document_type": ["wikipedia", "forum"],
            "template_strength": "declarative",
            "poisoned_count_per_confederate": 1,
            "shuffle_seed_offset": 0,
        },
        "environment": {"response_visibility": "public",
                         "answer_order": "sequential"},
        "scoring": {
            "full_conformity": "substring",
            "partial_conformity": {
                "method": "llm_judge",
                "judge": {"backend": "echo", "model": "echo-test"},
                "heuristic_sidecar": True,
            },
        },
        "logging": {"output_dir": "results/",
                     "save_context_windows": True,
                     "format": "jsonl"},
    }


def test_run_sweep_creates_nested_directory_with_cells(tmp_path):
    cfg = _sweep_config()
    manifest = run_sweep(cfg, output_root=tmp_path)
    run_dir = Path(manifest["run_dir"])
    assert run_dir.exists()
    cells_dir = run_dir / "cells"
    assert cells_dir.exists()
    assert len(list(cells_dir.glob("*.jsonl"))) == 4
    assert len(list(cells_dir.glob("*.summary.json"))) == 4
    sweep_json = run_dir / "sweep.json"
    assert sweep_json.exists()
    s = json.loads(sweep_json.read_text())
    assert len(s["cells"]) == 4
    assert all(c["status"] == "ok" for c in s["cells"])


def test_run_sweep_isolates_failing_cell(tmp_path):
    cfg = _sweep_config()
    cfg["documents"]["document_type"] = ["wikipedia", "nonexistent_type"]
    manifest = run_sweep(cfg, output_root=tmp_path)
    s = json.loads((Path(manifest["run_dir"]) / "sweep.json").read_text())
    statuses = [c["status"] for c in s["cells"]]
    assert "failed" in statuses
    assert "ok" in statuses
    errors = list((Path(manifest["run_dir"]) / "cells").glob("*.error.json"))
    assert errors


def test_sweep_json_contains_config_and_sweep_values(tmp_path):
    cfg = _sweep_config()
    manifest = run_sweep(cfg, output_root=tmp_path)
    s = json.loads((Path(manifest["run_dir"]) / "sweep.json").read_text())
    assert s["config"]["experiment"]["type"] == "asch_documents"
    for cell in s["cells"]:
        assert "sweep_values" in cell
        assert any("n_confederates" in k for k in cell["sweep_values"].keys())
