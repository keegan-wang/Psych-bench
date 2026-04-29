"""Orchestrate a full session: run all trials, log contexts, write summary."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .agent import BaseAgent
from .environment import Environment
from .logging_utils import JsonlLogger, write_summary
from .trial import run_trial
from .types import Stimulus, TrialResult


class Session:
    def __init__(
        self,
        stimuli: list[Stimulus],
        agents: list[BaseAgent],
        environment: Environment,
        log_path: str | Path,
        summary_path: str | Path,
        config_snapshot: dict[str, Any],
        score_trial: Callable[[TrialResult], dict[str, Any]] | None = None,
        summarize: Callable[[list[TrialResult]], dict[str, Any]] | None = None,
        session_label: str = "session",
    ) -> None:
        self.stimuli = stimuli
        self.agents = agents
        self.environment = environment
        self.log_path = Path(log_path)
        self.summary_path = Path(summary_path)
        self.config_snapshot = config_snapshot
        self.score_trial = score_trial
        self.summarize = summarize
        self.session_label = session_label

    def run(self) -> list[TrialResult]:
        for agent in self.agents:
            agent.reset()
        results: list[TrialResult] = []
        with JsonlLogger(self.log_path) as log:
            for stim in self.stimuli:
                result = run_trial(stim, self.agents, self.environment)
                if self.score_trial is not None:
                    scores = self.score_trial(result)
                    for k, v in scores.items():
                        if hasattr(result, k):
                            setattr(result, k, v)
                    result_extra = scores
                else:
                    result_extra = {}
                log.write(self._serialize_trial(result, result_extra))
                results.append(result)
        summary: dict[str, Any] = {
            "session_label": self.session_label,
            "timestamp": time.time(),
            "n_trials": len(results),
            "config": self.config_snapshot,
        }
        if self.summarize is not None:
            summary.update(self.summarize(results))
        write_summary(self.summary_path, summary)
        return results

    def _serialize_trial(
        self, result: TrialResult, extra: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "trial_index": result.trial_index,
            "is_critical": result.is_critical,
            "stimulus": asdict(result.stimulus),
            "correct_answer": result.correct_answer,
            "responses": [asdict(r) for r in result.responses],
            "scoring": extra,
            "environment": self.environment.snapshot(),
            "session_label": self.session_label,
            "timestamp": time.time(),
        }
