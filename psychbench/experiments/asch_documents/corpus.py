"""Load, validate, and access the asch_documents fictional-questions corpus."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DOC_TYPES = ("wikipedia", "forum", "news")
STRENGTHS = ("declarative", "hedged", "incidental")


@dataclass
class CorpusQuestion:
    id: str
    question: str
    correct_answer: str
    wrong_answer: str
    substitution_targets: list[str]
    wrong_substitution_targets: list[str]
    templates: dict[str, dict[str, str]]


@dataclass
class Corpus:
    version: int
    questions: list[CorpusQuestion]

    def get(self, qid: str) -> CorpusQuestion:
        for q in self.questions:
            if q.id == qid:
                return q
        raise KeyError(qid)


def load_corpus(path: str | Path) -> Corpus:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "questions" not in raw:
        raise ValueError(f"Corpus {path} missing 'questions' list")
    version = int(raw.get("version", 1))
    questions: list[CorpusQuestion] = []
    for entry in raw["questions"]:
        q = _build_question(entry)
        _validate_question(q)
        questions.append(q)
    return Corpus(version=version, questions=questions)


def _build_question(entry: dict[str, Any]) -> CorpusQuestion:
    return CorpusQuestion(
        id=entry["id"],
        question=entry["question"],
        correct_answer=entry["correct_answer"],
        wrong_answer=entry["wrong_answer"],
        substitution_targets=list(entry["substitution_targets"]),
        wrong_substitution_targets=list(entry["wrong_substitution_targets"]),
        templates=dict(entry["templates"]),
    )


def _validate_question(q: CorpusQuestion) -> None:
    if len(q.substitution_targets) != len(q.wrong_substitution_targets):
        raise ValueError(
            f"{q.id}: substitution and wrong_substitution arrays "
            f"must be same length"
        )
    for dt in DOC_TYPES:
        if dt not in q.templates:
            raise ValueError(f"{q.id}: missing doc_type {dt}")
        for strength in STRENGTHS:
            if strength not in q.templates[dt]:
                raise ValueError(f"{q.id}: missing {dt}/{strength}")
            text = q.templates[dt][strength]
            for target in q.substitution_targets:
                if target not in text:
                    raise ValueError(
                        f"{q.id}: target {target!r} not found in "
                        f"{dt}/{strength}"
                    )
