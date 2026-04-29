"""End-to-end CLI test for asch_documents (hermetic, echo backend)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_run_smoke_config(tmp_path):
    cfg_path = Path("config/experiments/asch_documents_smoke.yaml")
    assert cfg_path.exists()
    result = subprocess.run(
        [sys.executable, "-m", "psychbench", "run",
         "--config", str(cfg_path),
         "--output-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    run_dirs = list(tmp_path.glob("asch_documents_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "sweep.json").exists()
    assert list((run_dir / "cells").glob("*.jsonl"))


def test_cli_analyze_run_prints_report(tmp_path):
    cfg_path = Path("config/experiments/asch_documents_smoke.yaml")
    subprocess.run(
        [sys.executable, "-m", "psychbench", "run",
         "--config", str(cfg_path),
         "--output-dir", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    run_dir = next(tmp_path.glob("asch_documents_*"))
    analyze = subprocess.run(
        [sys.executable, "-m", "psychbench", "analyze",
         "--run", str(run_dir)],
        capture_output=True, text=True,
    )
    assert analyze.returncode == 0, analyze.stderr
    assert "asch_documents sweep summary" in analyze.stdout
    assert (run_dir / "sweep_tidy.csv").exists()


def test_cli_cost_gate_blocks_huge_sweep(tmp_path):
    big = tmp_path / "big.yaml"
    big.write_text("""
experiment:
  name: big
  type: asch_documents
  trials: 12
  seed: 1
  n_repeats: 3
corpus:
  path: psychbench/experiments/asch_documents/corpus/phase2_fictional.yaml
sweep:
  fields:
    - agents.n_confederates
    - documents.document_type
    - documents.template_strength
    - documents.poisoned_count_per_confederate
agents:
  n_confederates: [1, 3, 5, 7, 9, 11]
  dissenter: false
  confederate:
    type: model
    backend: echo
    model: echo-test
    stateful: false
  naive:
    type: model
    backend: echo
    model: echo-test
    stateful: false
    position: last
documents:
  document_type: [wikipedia, forum, news]
  template_strength: [declarative, hedged, incidental]
  poisoned_count_per_confederate: [1, 3, 5, 7, 9]
  shuffle_seed_offset: 0
environment:
  response_visibility: public
  answer_order: sequential
scoring:
  full_conformity: substring
  partial_conformity:
    method: llm_judge
    judge:
      backend: echo
      model: echo-test
    heuristic_sidecar: true
logging:
  output_dir: results/
  save_context_windows: true
  format: jsonl
""")
    result = subprocess.run(
        [sys.executable, "-m", "psychbench", "run",
         "--config", str(big),
         "--output-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    combined = (result.stderr + result.stdout).lower()
    assert "i-know" in combined
