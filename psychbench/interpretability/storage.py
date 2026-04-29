"""Save/load ActivationRecord to/from .npz sidecar files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .collector import ActivationRecord


def save_activation_record(
    record: ActivationRecord, path: str | Path,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "layers": np.asarray(record.layers, dtype=np.int64),
        "n_prompt_tokens": np.asarray(record.n_prompt_tokens, dtype=np.int64),
        "generated_text": np.asarray([record.generated_text], dtype=object),
        "token_positions": np.asarray(
            [json.dumps(record.token_positions)], dtype=object,
        ),
    }
    for L, arr in record.layer_activations.items():
        arrays[f"resid_{L}"] = arr
    for L, arr in record.attention_weights.items():
        arrays[f"attn_{L}"] = arr
    np.savez_compressed(p, **arrays)


def load_activation_record(path: str | Path) -> ActivationRecord:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with np.load(p, allow_pickle=True) as data:
        layers = [int(x) for x in data["layers"].tolist()]
        n_prompt_tokens = int(data["n_prompt_tokens"])
        generated_text = str(data["generated_text"][0])
        token_positions = json.loads(str(data["token_positions"][0]))
        layer_activations = {
            L: np.asarray(data[f"resid_{L}"]) for L in layers
        }
        attention_weights = {
            L: np.asarray(data[f"attn_{L}"]) for L in layers
        }
    return ActivationRecord(
        trial_id=None,
        trial_type="unknown",
        outcome=None,
        n_prompt_tokens=n_prompt_tokens,
        layers=layers,
        layer_activations=layer_activations,
        attention_weights=attention_weights,
        token_positions=token_positions,
        generated_text=generated_text,
    )
