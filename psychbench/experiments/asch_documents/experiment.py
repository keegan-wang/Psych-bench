"""AschDocumentsExperiment — per-cell wiring + scoring + session hook."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from psychbench.agents.model_agent import ModelAgent
from psychbench.framework.agent import BaseAgent
from psychbench.framework.backends import get_backend
from psychbench.framework.environment import Environment
from psychbench.framework.experiment import (
    BaseExperiment, register_experiment,
)
from psychbench.framework.session import Session
from psychbench.framework.types import (
    ResponseVisibility, Stimulus, TrialContext, TrialResult,
)

from .corpus import load_corpus
from .judge import run_partial_conformity_judge
from .poisoning import poison
from .prompts import build_confederate_prompt, build_naive_prompt
from .scoring import (
    full_conformity, ground_truth_match, partial_conformity_heuristic,
    unanimity,
)


@register_experiment("asch_documents")
class AschDocumentsExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._corpus = load_corpus(config["corpus"]["path"])
        self._rng = random.Random(config["experiment"].get("seed", 0))

    def build_stimuli(self) -> list[Stimulus]:
        stims: list[Stimulus] = []
        for i, q in enumerate(self._corpus.questions):
            stims.append(Stimulus(
                trial_index=i, is_critical=True, reference_cm=0.0,
                options={}, correct_label=q.correct_answer,
                metadata={
                    "question_id": q.id,
                    "question": q.question,
                    "correct_answer": q.correct_answer,
                    "wrong_answer": q.wrong_answer,
                },
            ))
        return stims

    def build_agents(self) -> list[BaseAgent]:
        agents_cfg = self.config["agents"]
        n_conf = int(agents_cfg["n_confederates"])
        dissenter = bool(agents_cfg.get("dissenter", False))
        confed_cfg = agents_cfg["confederate"]
        naive_cfg = agents_cfg["naive"]

        confederates: list[BaseAgent] = []
        for i in range(n_conf):
            is_dissenter = dissenter and i == 0
            backend = get_backend(confed_cfg["backend"], confed_cfg["model"])
            agent = ModelAgent(
                agent_id=f"confederate_{i+1}",
                position=i,
                backend=backend,
                stateful=bool(confed_cfg.get("stateful", False)),
                prompt_builder=self._make_confederate_prompt_builder(
                    is_dissenter=is_dissenter,
                ),
            )
            agent.metadata["dissenter"] = is_dissenter
            confederates.append(agent)

        naive_backend = get_backend(
            naive_cfg["backend"], naive_cfg["model"],
        )
        naive_position = (
            n_conf if naive_cfg.get("position", "last") == "last"
            else int(naive_cfg["position"])
        )
        naive = ModelAgent(
            agent_id="naive",
            position=naive_position,
            backend=naive_backend,
            stateful=bool(naive_cfg.get("stateful", False)),
            prompt_builder=build_naive_prompt,
        )
        return [*confederates, naive]

    def _make_confederate_prompt_builder(self, *, is_dissenter: bool):
        docs_cfg = self.config["documents"]
        doc_type = docs_cfg["document_type"]
        strength = docs_cfg["template_strength"]
        n_docs = int(docs_cfg["poisoned_count_per_confederate"])
        shuffle_offset = int(docs_cfg.get("shuffle_seed_offset", 0))
        base_seed = self.config["experiment"].get("seed", 0)

        def builder(context: TrialContext) -> str:
            qid = context.stimulus.metadata["question_id"]
            q = self._corpus.get(qid)
            template = q.templates[doc_type][strength]
            if is_dissenter:
                documents = [template] * n_docs
            else:
                poisoned = poison(
                    template, q.substitution_targets,
                    q.wrong_substitution_targets,
                )
                documents = [poisoned] * n_docs
            trial_rng = random.Random(
                base_seed + shuffle_offset + context.stimulus.trial_index
            )
            shuffled = list(documents)
            trial_rng.shuffle(shuffled)
            return build_confederate_prompt(
                question=q.question, documents=shuffled,
            )

        return builder

    def run(
        self, output_dir: str | Path, session_label: str = "cell",
    ) -> dict[str, Any]:
        stimuli = self.build_stimuli()
        agents = self.build_agents()
        env = Environment(
            visibility=ResponseVisibility(
                self.config["environment"]["response_visibility"]
            ),
        )
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"{session_label}.jsonl"
        summary_path = out_dir / f"{session_label}.summary.json"

        judge_cfg = self.config["scoring"]["partial_conformity"]["judge"]
        judge_backend = get_backend(
            judge_cfg["backend"], judge_cfg["model"],
        )
        non_dissenter_ids = [
            a.agent_id for a in agents
            if a.agent_id.startswith("confederate_")
            and not a.metadata.get("dissenter", False)
        ]

        def score(trial: TrialResult) -> dict[str, Any]:
            return _score_trial(
                trial, judge_backend=judge_backend,
                non_dissenter_ids=non_dissenter_ids,
            )

        session = Session(
            stimuli=stimuli, agents=agents, environment=env,
            log_path=log_path, summary_path=summary_path,
            config_snapshot=self.config,
            score_trial=score,
            summarize=_summarize_session,
            session_label=session_label,
        )
        results = session.run()
        return {
            "log_path": str(log_path),
            "summary_path": str(summary_path),
            "n_trials": len(results),
        }


def _score_trial(
    trial: TrialResult, judge_backend, non_dissenter_ids: list[str],
) -> dict[str, Any]:
    md = trial.stimulus.metadata
    correct = md["correct_answer"]
    wrong = md["wrong_answer"]
    question = md["question"]
    naive = next(
        (r for r in trial.responses if r.agent_id == "naive"), None,
    )
    naive_text = (naive.raw_text if naive is not None else "") or ""
    full = full_conformity(naive_text, wrong) if naive else 0
    gt = ground_truth_match(naive_text, correct) if naive else 0
    heur = partial_conformity_heuristic(naive_text, wrong) if naive else 0
    unan = unanimity(trial.responses, non_dissenter_ids, wrong)
    judge = run_partial_conformity_judge(
        backend=judge_backend, question=question,
        correct_answer=correct, wrong_answer=wrong,
        response_text=naive_text,
    )
    return {
        "full_conformity": full,
        "ground_truth": gt,
        "partial_conformity_heuristic": heur,
        "partial_conformity_judge": judge.score,
        "partial_conformity_judge_raw": judge.raw_text,
        "partial_conformity_judge_error": judge.error,
        "unanimity": unan,
        "naive_answer": naive_text,
        "confederate_answers": [
            {"agent_id": r.agent_id, "parsed_answer": r.parsed_answer,
             "raw_text": r.raw_text}
            for r in trial.responses if r.agent_id != "naive"
        ],
    }


def _summarize_session(trials: list[TrialResult]) -> dict[str, Any]:
    critical = [t for t in trials if t.is_critical]
    n_critical = len(critical)
    if n_critical == 0:
        return {"n_critical": 0}
    per_trial = [_trial_scoring_snapshot(t) for t in trials]
    full = [x["full_conformity"] for x in per_trial if x["is_critical"]]
    unan_flags = [x["unanimity"] for x in per_trial if x["is_critical"]]
    heur = [x["partial_conformity_heuristic"] for x in per_trial
            if x["is_critical"]]
    judge = [x["partial_conformity_judge"] for x in per_trial
             if x["is_critical"]]
    judge_ok = [j for j in judge if j is not None]
    unan_idx = [i for i, u in enumerate(unan_flags) if u]
    unconditional = sum(full) / n_critical
    unanimous_only = (
        sum(full[i] for i in unan_idx) / len(unan_idx) if unan_idx else 0.0
    )
    return {
        "n_trials": len(trials),
        "n_critical": n_critical,
        "confederate_reliability": (
            sum(unan_flags) / n_critical if n_critical else 0.0
        ),
        "conformity_rate_unconditional": unconditional,
        "conformity_rate_unanimous_only": unanimous_only,
        "partial_conformity_rate_judge": (
            sum(judge_ok) / len(judge_ok) if judge_ok else 0.0
        ),
        "partial_conformity_rate_heuristic": (
            sum(heur) / n_critical if n_critical else 0.0
        ),
        "judge_coverage": (
            len(judge_ok) / n_critical if n_critical else 0.0
        ),
        "ever_conformed": any(full),
        "per_trial": per_trial,
    }


def _trial_scoring_snapshot(t: TrialResult) -> dict[str, Any]:
    return {
        "trial_index": t.trial_index,
        "is_critical": t.is_critical,
        "full_conformity": getattr(t, "full_conformity", 0),
        "ground_truth": getattr(t, "ground_truth", 0),
        "partial_conformity_heuristic": getattr(
            t, "partial_conformity_heuristic", 0,
        ),
        "partial_conformity_judge": getattr(
            t, "partial_conformity_judge", None,
        ),
        "unanimity": getattr(t, "unanimity", True),
        "naive_answer": getattr(t, "naive_answer", ""),
    }
