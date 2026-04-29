"""Backend factory and EchoBackend (hermetic) tests."""
from __future__ import annotations

import pytest

from psychbench.framework.backends import EchoBackend, get_backend


def test_echo_backend_returns_letter_from_prompt():
    b = EchoBackend(model="echo-test")
    out = b.generate("What is the answer? Say A.", stateful=False)
    assert isinstance(out, str)
    assert out  # non-empty


def test_get_backend_echo_factory():
    b = get_backend("echo", "echo-test")
    assert isinstance(b, EchoBackend)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("not_a_backend", "x")


def test_stateful_echo_backend_tracks_history_and_resets():
    b = EchoBackend(model="echo-test")
    b.generate("first A", stateful=True)
    b.generate("second B", stateful=True)
    assert len(b.history) == 2
    b.reset()
    assert b.history == []
