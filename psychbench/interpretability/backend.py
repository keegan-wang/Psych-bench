"""TransformerLens-backed ModelBackend for interpretability runs."""
from __future__ import annotations


class TransformerLensBackend:
    """ModelBackend implementation that owns a HookedTransformer.

    Lazy-imports torch/transformer_lens in __init__ so Phase 1/2 (interp-off)
    runs don't pay the import cost.
    """

    def __init__(self, model: str, device: str | None = None) -> None:
        self.model = model
        try:
            import torch  # type: ignore
            from transformer_lens import HookedTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "TransformerLensBackend requires transformer_lens and torch. "
                "Install with `pip install -r requirements.txt`."
            ) from e
        self._torch = torch
        self.device = device or _auto_device(torch)
        self.hooked_model = HookedTransformer.from_pretrained(
            model, device=self.device,
        )

    def generate(self, prompt: str, stateful: bool = False) -> str:
        if stateful:
            raise NotImplementedError(
                "TransformerLensBackend does not support stateful=True in "
                "Phase 3a. Stateful interp is a future extension."
            )
        out = self.hooked_model.generate(
            prompt,
            max_new_tokens=64,
            temperature=0.0,
            do_sample=False,
            verbose=False,
        )
        if isinstance(out, str) and out.startswith(prompt):
            return out[len(prompt):]
        return out if isinstance(out, str) else str(out)

    def reset(self) -> None:
        return None


def _auto_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
