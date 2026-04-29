"""Targeted-string substitution for poisoned documents."""
from __future__ import annotations

import pytest

from psychbench.experiments.asch_documents.poisoning import poison


def test_single_target_single_occurrence():
    out = poison("Elena Voss wrote it.", ["Elena Voss"], ["Marcus Thorne"])
    assert out == "Marcus Thorne wrote it."


def test_single_target_multiple_occurrences():
    out = poison("Voss. Voss. Voss.", ["Voss"], ["Thorne"])
    assert out == "Thorne. Thorne. Thorne."


def test_longest_first_ordering_avoids_partial_replacement():
    text = "Elena Voss and Dr. Voss and just Voss."
    out = poison(
        text,
        ["Elena Voss", "Dr. Voss", "Voss"],
        ["Marcus Thorne", "Dr. Thorne", "Thorne"],
    )
    assert out == "Marcus Thorne and Dr. Thorne and just Thorne."


def test_target_missing_raises():
    with pytest.raises(ValueError, match="not found"):
        poison("hello world", ["Voss"], ["Thorne"])


def test_empty_targets_returns_text_unchanged():
    out = poison("unchanged", [], [])
    assert out == "unchanged"


def test_target_and_replacement_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        poison("x", ["a", "b"], ["c"])
