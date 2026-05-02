"""Asch experiment module + full-session integration tests."""
from __future__ import annotations

import json as _json
import subprocess
import sys
from pathlib import Path

from psychbench.analysis.conformity import (
    compare_conditions, load_session_summary,
)
from psychbench.config import load_config
from psychbench.experiments.asch.experiment import AschExperiment
from psychbench.experiments.asch.prompts import build_asch_prompt
from psychbench.experiments.asch.scoring import (
    score_asch_trial, summarize_asch_session,
)
from psychbench.experiments.asch.stimuli import generate_asch_stimuli
from psychbench.framework.types import (
    AgentResponse, Stimulus, TrialContext, TrialResult,
)


# ---------- stimuli ---------- #

def test_generates_correct_count_and_critical_indices():
    critical_idx = [2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17]
    stims = generate_asch_stimuli(
        n_trials=18, critical_indices=critical_idx, seed=42,
    )
    assert len(stims) == 18
    assert sum(1 for s in stims if s.is_critical) == 12
    for i, s in enumerate(stims):
        assert s.trial_index == i
        assert (i in critical_idx) == s.is_critical


def test_all_stimuli_have_unambiguous_correct_answer():
    stims = generate_asch_stimuli(
        n_trials=18, critical_indices=[2, 5, 11], seed=1,
    )
    for s in stims:
        assert s.options[s.correct_label] == s.reference_cm
        for label, length in s.options.items():
            if label == s.correct_label:
                continue
            assert abs(length - s.reference_cm) >= 2.0, (
                f"wrong option {label}={length} too close to reference "
                f"{s.reference_cm}"
            )


def test_reference_line_in_5_to_15_cm_range():
    stims = generate_asch_stimuli(
        n_trials=18, critical_indices=[2], seed=7,
    )
    for s in stims:
        assert 5.0 <= s.reference_cm <= 15.0


def test_seed_is_deterministic():
    a = generate_asch_stimuli(18, [2, 5], seed=99)
    b = generate_asch_stimuli(18, [2, 5], seed=99)
    assert [s.reference_cm for s in a] == [s.reference_cm for s in b]
    assert [s.options for s in a] == [s.options for s in b]


# ---------- scoring ---------- #

def _result(is_critical, correct, naive_ans, confed_ans) -> TrialResult:
    stim = Stimulus(
        0, is_critical, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, correct, {},
    )
    responses = [
        AgentResponse("confederate_1", confed_ans, confed_ans, "", {}),
        AgentResponse("confederate_2", confed_ans, confed_ans, "", {}),
        AgentResponse("naive", naive_ans, naive_ans, "", {}),
    ]
    return TrialResult(
        trial_index=0, is_critical=is_critical, stimulus=stim,
        responses=responses, conformed=None,
        naive_answer=None, confederate_answer=None,
        correct_answer=correct,
    )


def test_score_critical_trial_conformed_when_matches_wrong():
    r = _result(True, "A", naive_ans="B", confed_ans="B")
    out = score_asch_trial(
        r, naive_id="naive",
        confederate_ids=["confederate_1", "confederate_2"],
    )
    assert out["conformed"] is True
    assert out["naive_answer"] == "B"
    assert out["confederate_answer"] == "B"


def test_score_critical_trial_not_conformed_when_correct():
    r = _result(True, "A", naive_ans="A", confed_ans="B")
    out = score_asch_trial(
        r, naive_id="naive",
        confederate_ids=["confederate_1", "confederate_2"],
    )
    assert out["conformed"] is False


def test_score_non_critical_trial_conformed_is_none():
    r = _result(False, "A", naive_ans="A", confed_ans="A")
    out = score_asch_trial(
        r, naive_id="naive",
        confederate_ids=["confederate_1", "confederate_2"],
    )
    assert out["conformed"] is None


def test_summarize_session_rates():
    trials = [
        _result(True, "A", "B", "B"),
        _result(True, "A", "A", "B"),
        _result(True, "A", "B", "B"),
        _result(False, "A", "A", "A"),
    ]
    for t in trials:
        s = score_asch_trial(
            t, "naive", ["confederate_1", "confederate_2"],
        )
        t.conformed = s["conformed"]
        t.naive_answer = s["naive_answer"]
        t.confederate_answer = s["confederate_answer"]
    summary = summarize_asch_session(trials)
    assert summary["n_critical"] == 3
    assert summary["n_conformed"] == 2
    assert abs(summary["conformity_rate"] - 2 / 3) < 1e-9
    assert summary["ever_conformed"] is True


# ---------- prompts ---------- #

def test_prompt_contains_reference_options_and_instructions():
    stim = Stimulus(
        0, True, 10.0, {"A": 10.0, "B": 13.0, "C": 7.0}, "A", {},
    )
    ctx = TrialContext(
        stim, 5, "naive",
        prior_responses=[("c1", "B"), ("c2", "B")],
    )
    prompt = build_asch_prompt(ctx)
    assert "Reference line: 10" in prompt
    assert "A: 10" in prompt and "B: 13" in prompt and "C: 7" in prompt
    assert "Participant 1: B" in prompt
    assert "Participant 2: B" in prompt
    assert "Answer with only the letter A, B, or C." in prompt


def test_prompt_private_condition_omits_other_participants():
    stim = Stimulus(
        0, True, 10.0, {"A": 10.0, "B": 13.0, "C": 7.0}, "A", {},
    )
    ctx = TrialContext(stim, 0, "naive", prior_responses=[])
    prompt = build_asch_prompt(ctx)
    # The confederate-answers injection block must be absent with no priors.
    assert "already given their answers" not in prompt
    assert "Participant 1:" not in prompt
    assert "Answer with only the letter A, B, or C." in prompt


# ---------- AschExperiment ---------- #

def _base_config(**overrides):
    cfg = {
        "experiment": {
            "name": "asch_phase1",
            "type": "asch",
            "trials": 18,
            "critical_trials": 12,
            "critical_trial_indices": [
                2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17,
            ],
            "seed": 42,
        },
        "agents": {
            "confederates": {
                "type": "scripted",
                "count": 5,
                "behavior": "always_wrong_on_critical",
                "wrong_answer": "B",
            },
            "naive": {
                "type": "model",
                "backend": "echo",
                "model": "echo-test",
                "stateful": False,
                "position": "last",
            },
        },
        "environment": {
            "response_visibility": "public",
            "answer_order": "sequential",
        },
        "control": {
            "run_control": False,
            "response_visibility": "private",
        },
        "scoring": {"method": "binary"},
        "logging": {
            "save_context_windows": True,
            "output_dir": "results",
            "format": "jsonl",
        },
    }
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def test_asch_experiment_builds_6_agents_with_naive_last():
    exp = AschExperiment(_base_config())
    agents = exp.build_agents()
    assert len(agents) == 6
    last = max(agents, key=lambda a: a.position)
    assert last.agent_id == "naive"
    assert last.position == 5


def test_asch_experiment_builds_18_stimuli_with_12_critical():
    exp = AschExperiment(_base_config())
    stims = exp.build_stimuli()
    assert len(stims) == 18
    assert sum(1 for s in stims if s.is_critical) == 12


def test_asch_experiment_dissenter_flag_flips_one_confederate():
    cfg = _base_config()
    cfg["agents"]["confederates"]["dissenter"] = True
    exp = AschExperiment(cfg)
    agents = exp.build_agents()
    confeds = [a for a in agents if a.agent_id.startswith("confederate_")]
    dissenters = [
        a for a in confeds
        if a.metadata.get("behavior") == "always_correct"
    ]
    assert len(dissenters) == 1


def test_asch_experiment_n_confederates_override():
    cfg = _base_config()
    cfg["agents"]["confederates"]["count"] = 3
    exp = AschExperiment(cfg)
    agents = exp.build_agents()
    assert len(agents) == 4  # 3 confeds + 1 naive


def test_asch_experiment_full_run_writes_jsonl_and_summary(tmp_path: Path):
    cfg = _base_config()
    cfg["control"]["run_control"] = True
    exp = AschExperiment(cfg)
    summary = exp.run(output_dir=tmp_path)
    assert "experimental" in summary
    assert "control" in summary
    assert "comparison" in summary
    exp_log = Path(summary["experimental"]["log_path"])
    ctrl_log = Path(summary["control"]["log_path"])
    assert exp_log.exists() and ctrl_log.exists()
    lines = exp_log.read_text().strip().splitlines()
    assert len(lines) == 18
    first = _json.loads(lines[0])
    assert "responses" in first and "environment" in first
    exp_sum = _json.loads(
        Path(summary["experimental"]["summary_path"]).read_text()
    )
    assert exp_sum["n_trials"] == 18
    assert exp_sum["n_critical"] == 12


def test_asch_experiment_n_repeats_produces_aggregate(tmp_path: Path):
    """With n_repeats > 1, the run should produce per-repeat files, an
    aggregate.json per condition, and a comparison block with std stats."""
    cfg = _base_config()
    cfg["experiment"]["n_repeats"] = 3
    cfg["control"]["run_control"] = True
    exp = AschExperiment(cfg)
    summary = exp.run(output_dir=tmp_path)

    for cond in ("experimental", "control"):
        assert summary[cond]["n_repeats"] == 3
        assert len(summary[cond]["repeats"]) == 3
        seeds = [r["seed"] for r in summary[cond]["repeats"]]
        assert seeds == sorted(set(seeds)), "repeats must use distinct seeds"
        agg_path = Path(summary[cond]["aggregate_path"])
        assert agg_path.exists()
        agg = _json.loads(agg_path.read_text())
        assert agg["n_repeats"] == 3
        assert len(agg["per_repeat_rate"]) == 3
        assert agg["pooled_n_critical"] == 12 * 3
        # Per-repeat jsonl files must all exist and be separate
        for r_idx, rep in enumerate(summary[cond]["repeats"]):
            p = Path(rep["log_path"])
            assert p.exists()
            assert f"_r{r_idx}" in p.name
            lines = p.read_text().strip().splitlines()
            assert len(lines) == 18

    cmp_block = summary["comparison"]
    assert cmp_block["n_repeats"] == 3
    assert "experimental_std" in cmp_block
    assert "control_std" in cmp_block


def test_asch_experiment_n_repeats_one_preserves_legacy_naming(tmp_path: Path):
    """n_repeats == 1 (or unset) must keep the original single-file layout:
    no _r0 suffix, no aggregate.json — purely backwards compatible."""
    cfg = _base_config()
    cfg["experiment"]["n_repeats"] = 1
    cfg["control"]["run_control"] = True
    exp = AschExperiment(cfg)
    summary = exp.run(output_dir=tmp_path)
    for cond in ("experimental", "control"):
        assert summary[cond]["n_repeats"] == 1
        assert "aggregate_path" not in summary[cond]
        log_path = Path(summary[cond]["log_path"])
        assert "_r" not in log_path.name
    agg_files = list(tmp_path.glob("*.aggregate.json"))
    assert agg_files == []


# ---------- analysis + config ---------- #

def test_load_session_summary_and_compare(tmp_path: Path):
    exp = tmp_path / "e.json"
    ctrl = tmp_path / "c.json"
    exp.write_text(_json.dumps({
        "conformity_rate": 0.33, "n_critical": 12,
        "n_conformed": 4, "ever_conformed": True,
    }))
    ctrl.write_text(_json.dumps({
        "conformity_rate": 0.08, "n_critical": 12,
        "n_conformed": 1, "ever_conformed": True,
    }))
    loaded = load_session_summary(exp)
    assert loaded["conformity_rate"] == 0.33
    cmp = compare_conditions(exp, ctrl)
    assert cmp["experimental"]["conformity_rate"] == 0.33
    assert cmp["control"]["conformity_rate"] == 0.08
    assert abs(cmp["delta"] - 0.25) < 1e-9


def test_load_config_reads_asch_yaml(tmp_path: Path):
    yaml_text = """
experiment:
  name: asch_phase1
  type: asch
  trials: 18
  critical_trials: 12
  critical_trial_indices: [2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17]
agents:
  confederates:
    type: scripted
    count: 5
    behavior: always_wrong_on_critical
  naive:
    type: model
    backend: echo
    model: echo-test
    stateful: false
    position: last
environment:
  response_visibility: public
  answer_order: sequential
control:
  run_control: true
  response_visibility: private
scoring:
  method: binary
logging:
  save_context_windows: true
  output_dir: results/
  format: jsonl
"""
    p = tmp_path / "asch.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert cfg["experiment"]["type"] == "asch"
    assert cfg["agents"]["naive"]["backend"] == "echo"
    assert cfg["control"]["run_control"] is True


# ---------- CLI end-to-end (echo backend, no network) ---------- #

def test_cli_runs_asch_end_to_end_with_echo_backend(tmp_path: Path):
    cfg = tmp_path / "asch.yaml"
    cfg.write_text(f"""
experiment:
  name: asch_test
  type: asch
  trials: 18
  critical_trials: 12
  critical_trial_indices: [2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17]
  seed: 42
agents:
  confederates:
    type: scripted
    count: 5
    behavior: always_wrong_on_critical
    wrong_answer: B
  naive:
    type: model
    backend: echo
    model: echo-test
    stateful: false
    position: last
environment:
  response_visibility: public
  answer_order: sequential
control:
  run_control: true
  response_visibility: private
scoring:
  method: binary
logging:
  save_context_windows: true
  output_dir: {tmp_path}
  format: jsonl
""")
    result = subprocess.run(
        [
            sys.executable, "-m", "psychbench", "run",
            "--config", str(cfg), "--output-dir", str(tmp_path),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.glob("asch_experimental_*.jsonl"))
    assert list(tmp_path.glob("asch_control_*.jsonl"))
