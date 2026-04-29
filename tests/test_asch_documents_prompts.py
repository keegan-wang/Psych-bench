"""Prompt builders for naive and confederate agents."""
from __future__ import annotations

from psychbench.experiments.asch_documents.prompts import (
    build_confederate_prompt, build_naive_prompt,
)
from psychbench.framework.types import Stimulus, TrialContext


def _ctx(prior=None):
    stim = Stimulus(
        trial_index=0, is_critical=True, reference_cm=0.0,
        options={}, correct_label="",
        metadata={"question": "Who built the Zerendium reactor?"},
    )
    return TrialContext(
        stimulus=stim, agent_position=5, agent_id="naive",
        prior_responses=prior or [],
    )


def test_naive_prompt_contains_question_and_prior_answers():
    prior = [("confederate_1", "Marcus Thorne"),
             ("confederate_2", "Marcus Thorne")]
    prompt = build_naive_prompt(_ctx(prior))
    assert "Who built the Zerendium reactor?" in prompt
    assert "Participant 1: Marcus Thorne" in prompt
    assert "Participant 2: Marcus Thorne" in prompt
    assert "[Source" not in prompt
    assert "Answer in one short sentence" in prompt


def test_naive_prompt_omits_participants_block_when_no_priors():
    prompt = build_naive_prompt(_ctx(prior=[]))
    # Session intro mentions "other participants" in general; the injection
    # block is keyed by the "answers:" preamble and per-participant lines.
    assert "The other participants' answers:" not in prompt
    assert "Participant 1" not in prompt
    assert "Who built the Zerendium reactor?" in prompt


def test_confederate_prompt_contains_sources_and_question():
    docs = [
        "[canonical text about Elena Voss]",
        "[poisoned text about Marcus Thorne]",
    ]
    prompt = build_confederate_prompt(
        question="Who built the Zerendium reactor?",
        documents=docs,
    )
    assert "[Source 1]" in prompt
    assert "[Source 2]" in prompt
    assert "[canonical text about Elena Voss]" in prompt
    assert "[poisoned text about Marcus Thorne]" in prompt
    assert "Who built the Zerendium reactor?" in prompt
    assert "Participant 1" not in prompt
    assert "other participants" not in prompt.lower()


def test_confederate_prompt_empty_documents_still_renders():
    prompt = build_confederate_prompt(
        question="trivial?", documents=[],
    )
    assert "trivial?" in prompt
    assert "[Source" not in prompt
