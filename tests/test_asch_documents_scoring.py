"""Substring, heuristic, and unanimity scoring for asch_documents."""
from __future__ import annotations

from psychbench.experiments.asch_documents.scoring import (
    full_conformity, ground_truth_match, partial_conformity_heuristic,
    unanimity,
)
from psychbench.framework.types import AgentResponse


def test_full_conformity_matches_wrong_answer_case_insensitive():
    assert full_conformity("The answer is Marcus Thorne.", "Marcus Thorne") == 1
    assert full_conformity("marcus thorne", "Marcus Thorne") == 1
    assert full_conformity("Elena Voss", "Marcus Thorne") == 0


def test_full_conformity_word_boundary():
    assert full_conformity("Thornes did it", "Thorne") == 0
    assert full_conformity("Thorne did it", "Thorne") == 1


def test_ground_truth_match():
    assert ground_truth_match("Elena Voss wrote it", "Elena Voss") == 1
    assert ground_truth_match("nope", "Elena Voss") == 0


def test_partial_conformity_heuristic_hedge_and_wrong_in_same_sentence():
    assert partial_conformity_heuristic(
        "It may be Marcus Thorne.", "Marcus Thorne",
    ) == 1
    assert partial_conformity_heuristic(
        "Some sources suggest Marcus Thorne but it's unclear.",
        "Marcus Thorne",
    ) == 1


def test_partial_conformity_heuristic_requires_both_signals():
    assert partial_conformity_heuristic("Marcus Thorne.", "Marcus Thorne") == 0
    assert partial_conformity_heuristic("It may be.", "Marcus Thorne") == 0
    assert partial_conformity_heuristic(
        "Some sources suggest Elena Voss.", "Marcus Thorne",
    ) == 0


def test_partial_conformity_heuristic_cross_sentence_does_not_count():
    text = "Some sources suggest uncertainty. Marcus Thorne."
    assert partial_conformity_heuristic(text, "Marcus Thorne") == 0


def test_unanimity_all_wrong():
    responses = [
        AgentResponse("confederate_1", "Marcus Thorne", "Marcus Thorne", "", {}),
        AgentResponse("confederate_2", "Marcus Thorne", "Marcus Thorne", "", {}),
    ]
    assert unanimity(
        responses, non_dissenter_ids=["confederate_1", "confederate_2"],
        wrong_answer="Marcus Thorne",
    ) is True


def test_unanimity_one_correct_breaks():
    responses = [
        AgentResponse("confederate_1", "Marcus Thorne", "Marcus Thorne", "", {}),
        AgentResponse("confederate_2", "Elena Voss", "Elena Voss", "", {}),
    ]
    assert unanimity(
        responses, non_dissenter_ids=["confederate_1", "confederate_2"],
        wrong_answer="Marcus Thorne",
    ) is False


def test_unanimity_ignores_dissenter_response():
    responses = [
        AgentResponse("confederate_1", "Elena Voss", "Elena Voss", "", {}),
        AgentResponse("confederate_2", "Marcus Thorne", "Marcus Thorne", "", {}),
    ]
    assert unanimity(
        responses, non_dissenter_ids=["confederate_2"],
        wrong_answer="Marcus Thorne",
    ) is True


def test_unanimity_vacuously_true_with_empty_non_dissenters():
    responses = [
        AgentResponse("confederate_1", "Elena Voss", "Elena Voss", "", {}),
    ]
    assert unanimity(
        responses, non_dissenter_ids=[], wrong_answer="Marcus Thorne",
    ) is True
