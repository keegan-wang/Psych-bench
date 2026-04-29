"""ActivationCollector: owns generation, hooks residual/attn, builds records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:  # pragma: no cover
    from transformer_lens import HookedTransformer


@dataclass
class ActivationRecord:
    trial_id: int | None
    trial_type: str
    outcome: str | None
    n_prompt_tokens: int
    layers: list[int]
    layer_activations: dict[int, np.ndarray]
    attention_weights: dict[int, np.ndarray]
    token_positions: dict[str, list[int]]
    generated_text: str


class ActivationCollector:
    """Installs TransformerLens hooks, runs one generation, returns a record.

    Single forward pass: the collector owns generation when attached. Hooks
    capture only during the first (prompt-phase) forward; generation-step
    forwards are skipped by a per-layer capture flag. Hooks are torn down in
    ``finally`` so nothing leaks on exceptions.
    """

    def __init__(
        self,
        layers: list[int] | str = "all",
        max_new_tokens: int = 64,
    ) -> None:
        if isinstance(layers, list):
            self.layers: list[int] | str = sorted(set(layers))
        else:
            self.layers = layers
        self.max_new_tokens = max_new_tokens

    def collect(
        self,
        hooked_model: "HookedTransformer",
        prompt: str,
        token_labels: dict[str, list[int]] | None = None,
    ) -> tuple[str, ActivationRecord]:
        resolved_layers = self._resolve_layers(hooked_model)
        captured_resid: dict[int, np.ndarray] = {}
        captured_attn: dict[int, np.ndarray] = {}

        prompt_tokens = hooked_model.to_tokens(prompt)
        n_prompt_tokens = int(prompt_tokens.shape[-1])

        def make_resid_hook(layer_idx: int):
            def hook(activation, hook):  # noqa: A002
                if layer_idx in captured_resid:
                    return
                arr = activation[0, -1, :].detach().to("cpu").float().numpy()
                captured_resid[layer_idx] = arr.copy()
            return hook

        def make_attn_hook(layer_idx: int):
            def hook(pattern, hook):  # noqa: A002
                if layer_idx in captured_attn:
                    return
                arr = pattern.detach().to("cpu").float().numpy()
                if arr.ndim == 4:
                    arr = arr[0]
                captured_attn[layer_idx] = arr.copy()
            return hook

        hooks: list[tuple[str, Any]] = []
        for layer in resolved_layers:
            hooks.append(
                (f"blocks.{layer}.hook_resid_post", make_resid_hook(layer))
            )
            hooks.append(
                (f"blocks.{layer}.attn.hook_pattern", make_attn_hook(layer))
            )

        try:
            for name, fn in hooks:
                hooked_model.add_hook(name, fn)
            generated = hooked_model.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
                do_sample=False,
                verbose=False,
            )
        finally:
            hooked_model.reset_hooks()

        if isinstance(generated, str) and generated.startswith(prompt):
            generated_text = generated[len(prompt):]
        else:
            generated_text = (
                generated if isinstance(generated, str) else str(generated)
            )

        if token_labels is None:
            token_positions = {"full_prompt": [0, n_prompt_tokens]}
        else:
            token_positions = dict(token_labels)

        return generated_text, ActivationRecord(
            trial_id=None,
            trial_type="unknown",
            outcome=None,
            n_prompt_tokens=n_prompt_tokens,
            layers=list(resolved_layers),
            layer_activations=dict(captured_resid),
            attention_weights=dict(captured_attn),
            token_positions=token_positions,
            generated_text=generated_text,
        )

    def _resolve_layers(
        self, hooked_model: "HookedTransformer",
    ) -> list[int]:
        n_layers = int(hooked_model.cfg.n_layers)
        if isinstance(self.layers, str) and self.layers == "all":
            return list(range(n_layers))
        assert isinstance(self.layers, list)
        for L in self.layers:
            if L < 0 or L >= n_layers:
                raise ValueError(
                    f"layer index {L} out of range for model with "
                    f"{n_layers} layers"
                )
        return list(self.layers)
