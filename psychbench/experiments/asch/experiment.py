"""Asch (1951) line-length conformity experiment."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from psychbench.agents.model_agent import ModelAgent
from psychbench.agents.scripted_agent import ScriptedAgent
from psychbench.framework.agent import BaseAgent
from psychbench.framework.backends import get_backend
from psychbench.framework.environment import Environment
from psychbench.framework.experiment import BaseExperiment, register_experiment
from psychbench.framework.session import Session
from psychbench.framework.types import ResponseVisibility, Stimulus, TrialResult

from .prompts import build_asch_prompt
from .scoring import score_asch_trial, summarize_asch_session
from .stimuli import generate_asch_stimuli


@register_experiment("asch")
class AschExperiment(BaseExperiment):
    def build_stimuli(self) -> list[Stimulus]:
        exp_cfg = self.config["experiment"]
        return generate_asch_stimuli(
            n_trials=exp_cfg["trials"],
            critical_indices=list(exp_cfg["critical_trial_indices"]),
            seed=exp_cfg.get("seed", 0),
        )

    def build_agents(self, *, for_control: bool = False) -> list[BaseAgent]:
        agents_cfg = self.config["agents"]
        confed_cfg = agents_cfg["confederates"]
        naive_cfg = agents_cfg["naive"]
        n_conf = int(confed_cfg.get("count", 5))
        dissenter = bool(confed_cfg.get("dissenter", False))
        behavior = confed_cfg.get("behavior", "always_wrong_on_critical")
        wrong_letter = confed_cfg.get("wrong_answer", "B")

        confederates: list[BaseAgent] = []
        for i in range(n_conf):
            if dissenter and i == 0:
                agent_behavior = "always_correct"
                wrong = None
            else:
                agent_behavior = behavior
                wrong = (
                    wrong_letter
                    if behavior == "always_wrong_on_critical"
                    else None
                )
            confederates.append(ScriptedAgent(
                agent_id=f"confederate_{i+1}",
                position=i,
                behavior=agent_behavior,
                wrong_answer=wrong,
            ))

        from psychbench.interpretability.config import (
            resolve_interpretability,
        )
        interp_cfg = resolve_interpretability(self.config)
        if interp_cfg is not None and not for_control:
            from psychbench.interpretability.backend import (
                TransformerLensBackend,
            )
            from psychbench.interpretability.collector import (
                ActivationCollector,
            )
            naive_backend = TransformerLensBackend(
                model=interp_cfg.model, device=interp_cfg.device,
            )
            collector = ActivationCollector(
                layers=interp_cfg.layers,
                max_new_tokens=interp_cfg.max_new_tokens,
            )
        else:
            naive_backend = get_backend(
                naive_cfg.get("backend", "echo"),
                naive_cfg.get("model", "echo-test"),
            )
            collector = None

        position_cfg = naive_cfg.get("position", "last")
        naive_position = (
            n_conf if position_cfg == "last" else int(position_cfg)
        )
        naive = ModelAgent(
            agent_id="naive",
            position=naive_position,
            backend=naive_backend,
            stateful=bool(naive_cfg.get("stateful", False)),
            prompt_builder=build_asch_prompt,
            activation_collector=collector,
        )

        return [*confederates, naive]

    def _environment(self, *, for_control: bool) -> Environment:
        if for_control:
            vis = self.config.get("control", {}).get(
                "response_visibility", "private"
            )
        else:
            vis = self.config["environment"]["response_visibility"]
        return Environment(visibility=ResponseVisibility(vis))

    def run(self, output_dir: str | Path) -> dict[str, Any]:
        timestamp = int(time.time())
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {}

        for label, for_control in self._conditions():
            agents = self.build_agents(for_control=for_control)
            stimuli = self.build_stimuli()
            env = self._environment(for_control=for_control)
            log_path = out_dir / f"asch_{label}_{timestamp}.jsonl"
            summary_path = (
                out_dir / f"asch_{label}_{timestamp}.summary.json"
            )

            confederate_ids = [
                a.agent_id for a in agents
                if a.agent_id.startswith("confederate_")
            ]

            def _score(
                trial: TrialResult,
                _confederate_ids: list[str] = confederate_ids,
            ) -> dict[str, Any]:
                return score_asch_trial(trial, "naive", _confederate_ids)

            session = Session(
                stimuli=stimuli, agents=agents, environment=env,
                log_path=log_path, summary_path=summary_path,
                config_snapshot=self.config,
                score_trial=_score,
                summarize=summarize_asch_session,
                session_label=label,
            )
            results = session.run()
            summary[label] = {
                "log_path": str(log_path),
                "summary_path": str(summary_path),
                "n_trials": len(results),
            }

        if len(summary) == 2:
            summary["comparison"] = self._load_comparison(summary)
        return summary

    def _conditions(self) -> list[tuple[str, bool]]:
        run_ctrl = bool(
            self.config.get("control", {}).get("run_control", False)
        )
        conds = [("experimental", False)]
        if run_ctrl:
            conds.append(("control", True))
        return conds

    def _load_comparison(self, summary: dict[str, Any]) -> dict[str, Any]:
        exp_s = json.loads(
            Path(summary["experimental"]["summary_path"]).read_text()
        )
        ctrl_s = json.loads(
            Path(summary["control"]["summary_path"]).read_text()
        )
        return {
            "experimental_conformity_rate": exp_s.get("conformity_rate"),
            "control_conformity_rate": ctrl_s.get("conformity_rate"),
            "delta": (
                exp_s.get("conformity_rate", 0.0)
                - ctrl_s.get("conformity_rate", 0.0)
            ),
        }
