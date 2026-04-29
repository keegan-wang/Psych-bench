"""ActivationCollector and ActivationRecord tests."""
from __future__ import annotations

import numpy as np
import pytest


transformer_lens = pytest.importorskip("transformer_lens")

from psychbench.interpretability.backend import TransformerLensBackend  # noqa: E402
from psychbench.interpretability.collector import (  # noqa: E402
    ActivationCollector, ActivationRecord,
)


TINY_MODEL = "roneneldan/TinyStories-1M"


@pytest.fixture(scope="module")
def backend():
    return TransformerLensBackend(model=TINY_MODEL, device="cpu")


# --- dataclass + constructor ---

def test_activation_record_fields_default_sensibly():
    r = ActivationRecord(
        trial_id=None, trial_type="unknown", outcome=None,
        n_prompt_tokens=3, layers=[0],
        layer_activations={0: np.zeros(4, dtype=np.float32)},
        attention_weights={0: np.zeros((2, 3, 3), dtype=np.float32)},
        token_positions={"full_prompt": [0, 3]},
        generated_text="",
    )
    assert r.n_prompt_tokens == 3
    assert r.layers == [0]
    assert r.generated_text == ""


def test_collector_init_with_all_layers():
    c = ActivationCollector(layers="all", max_new_tokens=8)
    assert c.layers == "all"
    assert c.max_new_tokens == 8


def test_collector_init_with_explicit_layers_normalizes():
    c = ActivationCollector(layers=[1, 0, 1, 2], max_new_tokens=8)
    assert c.layers == [0, 1, 2]


# --- collect() behavior ---

def test_collect_returns_text_and_record(backend):
    collector = ActivationCollector(layers="all", max_new_tokens=4)
    text, record = collector.collect(backend.hooked_model, "hello world")
    assert isinstance(text, str)
    assert isinstance(record, ActivationRecord)
    assert record.n_prompt_tokens >= 1


def test_collect_all_layers_resolved_to_full_range(backend):
    collector = ActivationCollector(layers="all", max_new_tokens=4)
    _, record = collector.collect(backend.hooked_model, "hi")
    assert record.layers == list(range(backend.hooked_model.cfg.n_layers))
    for layer in record.layers:
        assert layer in record.layer_activations
        assert layer in record.attention_weights


def test_collect_layer_activation_shape_is_d_model(backend):
    collector = ActivationCollector(layers=[0], max_new_tokens=4)
    _, record = collector.collect(backend.hooked_model, "hi")
    act = record.layer_activations[0]
    assert act.ndim == 1
    assert act.shape[0] == backend.hooked_model.cfg.d_model
    assert act.dtype == np.float32


def test_collect_attention_shape_is_heads_seq_seq(backend):
    collector = ActivationCollector(layers=[0], max_new_tokens=4)
    _, record = collector.collect(backend.hooked_model, "hi there")
    attn = record.attention_weights[0]
    assert attn.ndim == 3
    n_heads = backend.hooked_model.cfg.n_heads
    assert attn.shape[0] == n_heads
    assert attn.shape[1] == attn.shape[2]
    assert attn.shape[1] == record.n_prompt_tokens
    assert attn.dtype == np.float32


def test_collect_default_token_positions_is_full_prompt(backend):
    collector = ActivationCollector(layers=[0], max_new_tokens=4)
    _, record = collector.collect(backend.hooked_model, "hi")
    assert record.token_positions == {"full_prompt": [0, record.n_prompt_tokens]}


def test_collect_custom_token_labels_preserved(backend):
    collector = ActivationCollector(layers=[0], max_new_tokens=4)
    labels = {"stim": [0, 1], "other": [1, 2]}
    _, record = collector.collect(
        backend.hooked_model, "hi", token_labels=labels,
    )
    assert record.token_positions == labels


def test_collect_out_of_range_layer_raises(backend):
    n = backend.hooked_model.cfg.n_layers
    collector = ActivationCollector(layers=[n + 5], max_new_tokens=4)
    with pytest.raises(ValueError, match="layer"):
        collector.collect(backend.hooked_model, "hi")


def test_collect_twice_does_not_leak_hooks(backend):
    collector = ActivationCollector(layers=[0], max_new_tokens=4)
    collector.collect(backend.hooked_model, "hi")
    collector.collect(backend.hooked_model, "there")
    for hp in backend.hooked_model.hook_dict.values():
        assert len(hp.fwd_hooks) == 0
        assert len(hp.bwd_hooks) == 0


def test_collect_on_generate_error_still_removes_hooks(backend, monkeypatch):
    collector = ActivationCollector(layers=[0], max_new_tokens=4)

    def boom(*a, **kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(backend.hooked_model, "generate", boom)
    with pytest.raises(RuntimeError, match="kaboom"):
        collector.collect(backend.hooked_model, "hi")
    for hp in backend.hooked_model.hook_dict.values():
        assert len(hp.fwd_hooks) == 0
