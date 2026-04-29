"""AschDocumentsExperiment per-cell wiring tests."""
from __future__ import annotations

import json
from pathlib import Path

from psychbench.experiments.asch_documents.experiment import (
    AschDocumentsExperiment,
)
from psychbench.framework.types import TrialContext


FIXTURE = Path(__file__).parent / "fixtures" / "mini_corpus.yaml"


def _cell_config(**over):
    cfg = {
        "experiment": {"name": "t", "type": "asch_documents",
                        "trials": 2, "seed": 7, "n_repeats": 1},
        "corpus": {"path": str(FIXTURE)},
        "agents": {
            "n_confederates": 2,
            "dissenter": False,
            "confederate": {"type": "model", "backend": "echo",
                             "model": "echo-test", "stateful": False},
            "naive": {"type": "model", "backend": "echo",
                       "model": "echo-test", "stateful": False,
                       "position": "last"},
        },
        "documents": {
            "document_type": "wikipedia",
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
    for k, v in over.items():
        cfg[k] = v
    return cfg


def test_builds_n_confederates_plus_one_naive_last():
    exp = AschDocumentsExperiment(_cell_config())
    agents = exp.build_agents()
    assert len(agents) == 3
    assert agents[-1].agent_id == "naive"
    assert agents[-1].position == 2


def test_builds_stimuli_one_per_question():
    exp = AschDocumentsExperiment(_cell_config())
    stims = exp.build_stimuli()
    assert len(stims) == 2
    assert all(s.is_critical for s in stims)
    assert "question" in stims[0].metadata


def test_dissenter_flips_one_confederate_to_canonical_docs():
    cfg = _cell_config()
    cfg["agents"]["dissenter"] = True
    exp = AschDocumentsExperiment(cfg)
    agents = exp.build_agents()
    confeds = [a for a in agents if a.agent_id.startswith("confederate_")]
    assert confeds[0].metadata.get("dissenter") is True
    assert confeds[1].metadata.get("dissenter") is False


def test_poisoned_count_per_confederate_respected_in_confederate_prompt():
    cfg = _cell_config()
    cfg["documents"]["poisoned_count_per_confederate"] = 3
    exp = AschDocumentsExperiment(cfg)
    agents = exp.build_agents()
    stim = exp.build_stimuli()[0]
    confed = [a for a in agents if a.agent_id == "confederate_1"][0]
    ctx = TrialContext(stimulus=stim, agent_position=confed.position,
                        agent_id=confed.agent_id, prior_responses=[])
    prompt = confed.prompt_builder(ctx)
    assert prompt.count("[Source ") == 3


def test_full_run_writes_jsonl_and_summary(tmp_path):
    exp = AschDocumentsExperiment(_cell_config())
    out = exp.run(output_dir=tmp_path, session_label="cell_test")
    log = Path(out["log_path"])
    summary = Path(out["summary_path"])
    assert log.exists() and summary.exists()
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert "responses" in rec
    s = json.loads(summary.read_text())
    assert s["n_trials"] == 2
    assert "conformity_rate_unconditional" in s
    assert "conformity_rate_unanimous_only" in s
    assert "partial_conformity_rate_judge" in s
    assert "partial_conformity_rate_heuristic" in s
    assert "confederate_reliability" in s
    assert "per_trial" in s
