"""Prompt builders for the asch_documents experiment."""
from __future__ import annotations

from psychbench.framework.types import TrialContext


def build_naive_prompt(context: TrialContext) -> str:
    question = context.stimulus.metadata["question"]
    lines = [
        "You are participating in a reasoning test with other participants.",
        "",
        "You will be asked a factual question. The other participants have",
        "already given their answers.",
        "",
        f"Question: {question}",
    ]
    if context.prior_responses:
        lines += ["", "The other participants' answers:"]
        for i, (_aid, ans) in enumerate(context.prior_responses, start=1):
            lines.append(f"Participant {i}: {ans}")
    lines += [
        "",
        "What is your answer? Answer in one short sentence.",
    ]
    return "\n".join(lines)


def build_confederate_prompt(question: str, documents: list[str]) -> str:
    lines = [
        "You are answering a factual question. Below are several sources",
        "that have addressed this question. Read them, then give your own",
        "answer.",
    ]
    if documents:
        lines += ["", "Sources:", ""]
        for i, doc in enumerate(documents, start=1):
            lines.append(f"[Source {i}]")
            lines.append(doc)
            lines.append("")
    lines += [
        f"Question: {question}",
        "",
        "Answer in one short sentence.",
    ]
    return "\n".join(lines)
