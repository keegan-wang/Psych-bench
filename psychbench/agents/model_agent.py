"""LLM-backed agent with optional cross-trial stateful history."""
from __future__ import annotations

import re
from typing import Callable

from psychbench.framework.agent import BaseAgent
from psychbench.framework.backends import ModelBackend
from psychbench.framework.types import AgentResponse, TrialContext


PromptBuilder = Callable[[TrialContext], str]


def parse_letter_answer(text: str) -> str | None:
    """Extract A/B/C from model output.

    Prefer a final standalone letter (handles verbose responses ending in
    the letter), fall back to the first standalone letter anywhere.
    """
    stripped = text.strip()
    m_final = re.search(r"\b([ABC])\b\W*$", stripped)
    if m_final:
        return m_final.group(1)
    m_any = re.search(r"\b([ABC])\b", stripped)
    if m_any:
        return m_any.group(1)
    return None


class ModelAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        position: int,
        backend: ModelBackend,
        stateful: bool,
        prompt_builder: PromptBuilder,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            position=position,
            metadata={
                "type": "model",
                "model": backend.model,
                "stateful": stateful,
            },
        )
        self.backend = backend
        self.stateful = stateful
        self.prompt_builder = prompt_builder

    def respond(self, context: TrialContext) -> AgentResponse:
        prompt = self.prompt_builder(context)
        raw = self.backend.generate(prompt, stateful=self.stateful)
        parsed = parse_letter_answer(raw)
        return AgentResponse(
            agent_id=self.agent_id,
            raw_text=raw,
            parsed_answer=parsed,
            prompt=prompt,
            metadata={"model": self.backend.model, "stateful": self.stateful},
        )

    def reset(self) -> None:
        self.backend.reset()
