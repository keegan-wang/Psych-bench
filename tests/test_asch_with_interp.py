"""End-to-end: AschExperiment + interp enabled + tiny pretrained model."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


transformer_lens = pytest.importorskip("transformer_lens")

from psychbench.experiments.asch.experiment import AschExperiment  # noqa: E402
from psychbench.interpretability.storage import (  # noqa: E402
    load_activation_record,
)


def _load_smoke_config(n_trials: int):
    path = Path("config/experiments/asch_phase1_with_interp_smoke.yaml")
    cfg = yaml.safe_load(path.read_text())
    cfg["experiment"]["trials"] = n_trials
    cfg["experiment"]["critical_trials"] = min(
        n_trials, cfg["experiment"]["critical_trials"],
    )
    cfg["experiment"]["critical_trial_indices"] = [
        i for i in cfg["experiment"]["critical_trial_indices"]
        if i < n_trials
    ]
    return cfg


def test_asch_with_interp_end_to_end(tmp_path):
    cfg = _load_smoke_config(n_trials=2)
    exp = AschExperiment(cfg)
    out = exp.run(output_dir=tmp_path)
    exp_result = out["experimental"]
    log_path = Path(exp_result["log_path"])
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert "interpretability" in record
        interp = record["interpretability"]
        assert "activations_path" in interp
        act_path = log_path.parent / interp["activations_path"]
        assert act_path.exists()
        loaded = load_activation_record(act_path)
        assert loaded.n_prompt_tokens >= 1
        assert loaded.layers
        assert all(L in loaded.layer_activations for L in loaded.layers)
        assert all(L in loaded.attention_weights for L in loaded.layers)
