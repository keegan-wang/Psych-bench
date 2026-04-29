"""TransformerLensBackend smoke tests against the tiny random model."""
from __future__ import annotations

import pytest


transformer_lens = pytest.importorskip("transformer_lens")

from psychbench.interpretability.backend import TransformerLensBackend  # noqa: E402


TINY_MODEL = "roneneldan/TinyStories-1M"


@pytest.fixture(scope="module")
def backend():
    return TransformerLensBackend(model=TINY_MODEL, device="cpu")


def test_backend_loads_tiny_model_and_exposes_hooked_model(backend):
    assert backend.model == TINY_MODEL
    assert backend.hooked_model is not None
    assert hasattr(backend.hooked_model, "cfg")
    assert backend.hooked_model.cfg.n_layers >= 1


def test_backend_generate_returns_non_empty_string(backend):
    out = backend.generate("hello", stateful=False)
    assert isinstance(out, str)


def test_backend_stateful_raises_not_implemented(backend):
    with pytest.raises(NotImplementedError):
        backend.generate("hello", stateful=True)


def test_backend_reset_is_noop(backend):
    backend.reset()


def test_backend_device_explicit_cpu_accepted():
    b = TransformerLensBackend(model=TINY_MODEL, device="cpu")
    assert b.device == "cpu"
