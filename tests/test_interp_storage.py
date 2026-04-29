"""Round-trip save/load for ActivationRecord."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from psychbench.interpretability.collector import ActivationRecord
from psychbench.interpretability.storage import (
    load_activation_record, save_activation_record,
)


def _fake_record():
    return ActivationRecord(
        trial_id=7, trial_type="critical", outcome=None,
        n_prompt_tokens=5, layers=[0, 2],
        layer_activations={
            0: np.arange(4, dtype=np.float32),
            2: np.arange(4, dtype=np.float32) * 2,
        },
        attention_weights={
            0: np.zeros((2, 5, 5), dtype=np.float32),
            2: np.ones((2, 5, 5), dtype=np.float32),
        },
        token_positions={"full_prompt": [0, 5]},
        generated_text="hello",
    )


def test_round_trip_equal_arrays(tmp_path):
    r = _fake_record()
    p = tmp_path / "trial.npz"
    save_activation_record(r, p)
    loaded = load_activation_record(p)
    assert loaded.n_prompt_tokens == 5
    assert loaded.layers == [0, 2]
    assert np.array_equal(loaded.layer_activations[0], r.layer_activations[0])
    assert np.array_equal(loaded.layer_activations[2], r.layer_activations[2])
    assert np.array_equal(loaded.attention_weights[0], r.attention_weights[0])
    assert np.array_equal(loaded.attention_weights[2], r.attention_weights[2])
    assert loaded.token_positions == {"full_prompt": [0, 5]}
    assert loaded.generated_text == "hello"


def test_round_trip_unicode_and_newlines(tmp_path):
    r = _fake_record()
    r.generated_text = "héllo\nworld 😀"
    p = tmp_path / "trial.npz"
    save_activation_record(r, p)
    loaded = load_activation_record(p)
    assert loaded.generated_text == "héllo\nworld 😀"


def test_save_creates_missing_parent_dir(tmp_path):
    r = _fake_record()
    p = tmp_path / "nested" / "sub" / "trial.npz"
    save_activation_record(r, p)
    assert p.exists()


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_activation_record(tmp_path / "nope.npz")
