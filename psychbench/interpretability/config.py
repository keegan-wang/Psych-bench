"""Validate and normalize the `interpretability:` YAML block."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LEGAL_DEVICES = (None, "cpu", "cuda", "mps")


@dataclass
class InterpretabilityConfig:
    model: str
    device: str | None
    layers: list[int] | str  # "all" sentinel or concrete list
    max_new_tokens: int


def resolve_interpretability(
    config: dict[str, Any],
) -> InterpretabilityConfig | None:
    """Return a validated InterpretabilityConfig, or None if interp is off."""
    interp = config.get("interpretability")
    if not interp or not interp.get("enabled"):
        return None

    backend = interp.get("backend", "transformer_lens")
    if backend != "transformer_lens":
        raise ValueError(
            f"interpretability.backend must be 'transformer_lens', "
            f"got {backend!r}"
        )

    model = interp.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("interpretability.model must be a non-empty string")

    device = interp.get("device", None)
    if device not in LEGAL_DEVICES:
        raise ValueError(
            f"interpretability.device must be one of {LEGAL_DEVICES}, "
            f"got {device!r}"
        )

    layers = interp.get("layers", "all")
    if isinstance(layers, str):
        if layers != "all":
            raise ValueError(
                f"interpretability.layers must be 'all' or a list of ints, "
                f"got {layers!r}"
            )
    elif isinstance(layers, list):
        if not all(isinstance(x, int) and not isinstance(x, bool)
                   for x in layers):
            raise ValueError(
                f"interpretability.layers list must contain only ints, "
                f"got {layers!r}"
            )
    else:
        raise ValueError(
            f"interpretability.layers must be 'all' or a list of ints, "
            f"got {layers!r}"
        )

    max_new_tokens = int(interp.get("max_new_tokens", 64))
    if max_new_tokens <= 0:
        raise ValueError(
            f"interpretability.max_new_tokens must be positive, "
            f"got {max_new_tokens}"
        )

    naive = config.get("agents", {}).get("naive", {})
    naive_backend = naive.get("backend")
    if naive_backend is not None and naive_backend != "transformer_lens":
        raise ValueError(
            "interpretability.enabled=true requires "
            "agents.naive.backend='transformer_lens' (got "
            f"{naive_backend!r}). The interp layer needs direct model "
            "access and cannot use API backends."
        )
    naive_model = naive.get("model")
    if naive_model is not None and naive_model != model:
        raise ValueError(
            f"agents.naive.model ({naive_model!r}) must match "
            f"interpretability.model ({model!r}) when interp is enabled"
        )

    return InterpretabilityConfig(
        model=model, device=device, layers=layers,
        max_new_tokens=max_new_tokens,
    )
