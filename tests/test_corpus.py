"""Corpus loader and validation tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from psychbench.experiments.asch_documents.corpus import (
    Corpus, CorpusQuestion, load_corpus,
)


FIXTURE = Path(__file__).parent / "fixtures" / "mini_corpus.yaml"


def test_load_corpus_parses_questions():
    corpus = load_corpus(FIXTURE)
    assert isinstance(corpus, Corpus)
    assert len(corpus.questions) == 2
    q = corpus.questions[0]
    assert isinstance(q, CorpusQuestion)
    assert q.id == "q01"
    assert q.correct_answer == "Elena Voss"
    assert q.wrong_answer == "Marcus Thorne"


def test_every_doctype_and_strength_present():
    corpus = load_corpus(FIXTURE)
    for q in corpus.questions:
        for dt in ("wikipedia", "forum", "news"):
            for strength in ("declarative", "hedged", "incidental"):
                assert q.templates[dt][strength], (
                    f"missing {dt}/{strength} for {q.id}"
                )


def test_substitution_arrays_same_length(tmp_path):
    bad = tmp_path / "bad.yaml"
    text = FIXTURE.read_text().replace(
        '      - "Marcus Thorne"\n      - "Dr. Thorne"\n      - "Thorne"\n',
        '      - "Marcus Thorne"\n',
    )
    bad.write_text(text)
    with pytest.raises(ValueError, match="same length"):
        load_corpus(bad)


def test_all_targets_appear_in_every_template(tmp_path):
    bad = tmp_path / "bad.yaml"
    text = FIXTURE.read_text().replace(
        "Elena Voss first described", "Somebody first described",
    )
    bad.write_text(text)
    with pytest.raises(ValueError, match="not found"):
        load_corpus(bad)


def test_corpus_get_question_by_id():
    corpus = load_corpus(FIXTURE)
    q = corpus.get("q02")
    assert q.id == "q02"
    with pytest.raises(KeyError):
        corpus.get("nope")


def test_valid_fixture_loads_without_error():
    corpus = load_corpus(FIXTURE)
    assert corpus is not None
