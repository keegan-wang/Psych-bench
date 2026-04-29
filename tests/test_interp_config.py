"""Validation tests for psychbench.interpretability.config."""
from __future__ import annotations

import pytest

from psychbench.interpretability.config import (
    InterpretabilityConfig, resolve_interpretability,
)


def _base(**over):
    cfg = {
        "agents": {
            "naive": {
                "type": "model",
                "backend": "transformer_lens",
                "model": "roneneldan/TinyStories-1M",
            },
        },
        "interpretability": {
            "enabled": True,
            "backend": "transformer_lens",
            "model": "roneneldan/TinyStories-1M",
            "device": "cpu",
            "layers": "all",
            "max_new_tokens": 64,
        },
    }
    for k, v in over.items():
        cfg[k] = v
    return cfg


def test_returns_none_when_block_absent():
    assert resolve_interpretability({"agents": {"naive": {}}}) is None


def test_returns_none_when_enabled_false():
    cfg = _base()
    cfg["interpretability"]["enabled"] = False
    assert resolve_interpretability(cfg) is None


def test_returns_populated_config_with_defaults_filled():
    cfg = {
        "agents": {"naive": {"backend": "transformer_lens",
                              "model": "roneneldan/TinyStories-1M"}},
        "interpretability": {
            "enabled": True,
            "model": "roneneldan/TinyStories-1M",
        },
    }
    out = resolve_interpretability(cfg)
    assert isinstance(out, InterpretabilityConfig)
    assert out.model == "roneneldan/TinyStories-1M"
    assert out.device is None
    assert out.layers == "all"
    assert out.max_new_tokens == 64


def test_hard_error_when_naive_backend_disagrees():
    cfg = _base()
    cfg["agents"]["naive"]["backend"] = "openai"
    with pytest.raises(ValueError, match="transformer_lens"):
        resolve_interpretability(cfg)


def test_hard_error_when_naive_model_disagrees():
    cfg = _base()
    cfg["agents"]["naive"]["model"] = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    with pytest.raises(ValueError, match="model"):
        resolve_interpretability(cfg)


def test_hard_error_when_backend_not_transformer_lens():
    cfg = _base()
    cfg["interpretability"]["backend"] = "openai"
    with pytest.raises(ValueError, match="transformer_lens"):
        resolve_interpretability(cfg)


def test_rejects_unknown_device():
    cfg = _base()
    cfg["interpretability"]["device"] = "tpu"
    with pytest.raises(ValueError, match="device"):
        resolve_interpretability(cfg)


def test_accepts_device_cpu_cuda_mps_none():
    for dev in ("cpu", "cuda", "mps", None):
        cfg = _base()
        cfg["interpretability"]["device"] = dev
        out = resolve_interpretability(cfg)
        assert out.device == dev


def test_accepts_layers_all_and_int_list():
    cfg = _base()
    cfg["interpretability"]["layers"] = "all"
    assert resolve_interpretability(cfg).layers == "all"
    cfg["interpretability"]["layers"] = [0, 1, 2]
    assert resolve_interpretability(cfg).layers == [0, 1, 2]


def test_rejects_layers_string_other_than_all():
    cfg = _base()
    cfg["interpretability"]["layers"] = "everything"
    with pytest.raises(ValueError, match="layers"):
        resolve_interpretability(cfg)


def test_rejects_layers_nonint_list():
    cfg = _base()
    cfg["interpretability"]["layers"] = [0, "one"]
    with pytest.raises(ValueError, match="layers"):
        resolve_interpretability(cfg)


def test_rejects_nonpositive_max_new_tokens():
    cfg = _base()
    cfg["interpretability"]["max_new_tokens"] = 0
    with pytest.raises(ValueError, match="max_new_tokens"):
        resolve_interpretability(cfg)


def test_accepts_missing_naive_block_when_interp_disabled():
    out = resolve_interpretability({"interpretability": {"enabled": False}})
    assert out is None
