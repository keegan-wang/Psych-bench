"""LLM judge for partial conformity (hermetic via EchoBackend)."""
from __future__ import annotations

from psychbench.experiments.asch_documents.judge import (
    JudgeResult, run_partial_conformity_judge,
)
from psychbench.framework.backends import EchoBackend


def test_judge_returns_zero_with_echo_backend_deterministically():
    backend = EchoBackend(model="echo-test")
    result = run_partial_conformity_judge(
        backend=backend,
        question="Who built it?",
        correct_answer="Elena Voss",
        wrong_answer="Marcus Thorne",
        response_text="It may be Marcus Thorne.",
    )
    assert isinstance(result, JudgeResult)
    # Echo backend picks the last A/B/C, or "A" default; parser sees no 0/1
    # in echo output and returns the conservative 0.
    assert result.score == 0
    assert result.raw_text
    assert result.error is None


def test_judge_logs_raw_response_text():
    backend = EchoBackend(model="echo-test")
    result = run_partial_conformity_judge(
        backend=backend,
        question="Who built it?",
        correct_answer="Elena Voss",
        wrong_answer="Marcus Thorne",
        response_text="Elena Voss.",
    )
    assert result.raw_text is not None


def test_judge_handles_backend_exception_gracefully():
    class FailingBackend:
        model = "x"

        def generate(self, prompt, stateful=False):
            raise RuntimeError("boom")

        def reset(self):
            pass

    result = run_partial_conformity_judge(
        backend=FailingBackend(),
        question="q", correct_answer="c", wrong_answer="w",
        response_text="r",
    )
    assert result.score is None
    assert result.error == "boom"
