"""Non-LLM rule-based agent used for confederates."""
from __future__ import annotations

from typing import Callable

from psychbench.framework.agent import BaseAgent
from psychbench.framework.types import AgentResponse, TrialContext


Behavior = str  # "always_correct" | "always_wrong_on_critical" | "custom_fn"


class ScriptedAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        position: int,
        behavior: Behavior,
        wrong_answer: str | None = None,
        custom_fn: Callable[[TrialContext], str] | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            position=position,
            metadata={"type": "scripted", "behavior": behavior},
        )
        self.behavior = behavior
        self.wrong_answer = wrong_answer
        self.custom_fn = custom_fn
        if behavior == "custom_fn" and custom_fn is None:
            raise ValueError("behavior='custom_fn' requires custom_fn callable")

    def respond(self, context: TrialContext) -> AgentResponse:
        answer = self._pick_answer(context)
        return AgentResponse(
            agent_id=self.agent_id,
            raw_text=answer,
            parsed_answer=answer,
            prompt="",
            metadata={"scripted": True, "behavior": self.behavior},
        )

    def _pick_answer(self, context: TrialContext) -> str:
        if self.behavior == "always_correct":
            return context.stimulus.correct_label
        if self.behavior == "always_wrong_on_critical":
            if not context.stimulus.is_critical:
                return context.stimulus.correct_label
            if self.wrong_answer is not None:
                return self.wrong_answer
            for label in sorted(context.stimulus.options):
                if label != context.stimulus.correct_label:
                    return label
            raise RuntimeError("No wrong options available")
        if self.behavior == "custom_fn":
            assert self.custom_fn is not None
            return self.custom_fn(context)
        raise ValueError(f"Unknown behavior: {self.behavior}")
