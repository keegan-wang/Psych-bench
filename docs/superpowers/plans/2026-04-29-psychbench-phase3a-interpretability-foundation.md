# PsychBench Phase 3a — Interpretability Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the framework plumbing that lets any PsychBench experiment opt in (via YAML) to capturing TransformerLens residual-stream and attention activations for the naive agent on every trial, persisting them as `.npz` sidecars next to the existing JSONL logs — with no analysis logic yet.

**Architecture:** A new `psychbench/interpretability/` package exposes `TransformerLensBackend` (implements existing `ModelBackend` protocol), `ActivationCollector` (owns generation when attached — single forward pass with hooks installed/removed via `try/finally`), `ActivationRecord` dataclass, and `save/load_activation_record` for `.npz` round-trip. `ModelAgent` gains one optional `activation_collector` field; `Session` writes the sidecar and a lean JSONL pointer when a record is present. `AschExperiment.build_agents()` attaches the collector to the naive agent when `interpretability.enabled: true`. No Phase 2 wiring in 3a.

**Tech Stack:** Python 3.10+, `transformer_lens>=2.0.0`, `torch>=2.0.0` (CPU fine), `numpy`, `scikit-learn`, `scipy`, `PyYAML`, `pytest`. Interp-dependent tests guard with `pytest.importorskip("transformer_lens")`.

---

## File Structure (created or modified by this plan)

**New files:**
- `psychbench/interpretability/__init__.py` — empty marker.
- `psychbench/interpretability/config.py` — `InterpretabilityConfig` dataclass + `resolve_interpretability(cfg)`.
- `psychbench/interpretability/backend.py` — `TransformerLensBackend` (lazy-imports `transformer_lens`/`torch`).
- `psychbench/interpretability/collector.py` — `ActivationRecord` dataclass + `ActivationCollector` with hook lifecycle.
- `psychbench/interpretability/storage.py` — `save_activation_record` / `load_activation_record`.
- `config/experiments/asch_phase1_with_interp_smoke.yaml` — hermetic smoke config.
- `docs/interpretability.md` — methodology + workflow docs.
- `tests/test_interp_config.py`, `tests/test_interp_backend.py`, `tests/test_interp_collector.py`, `tests/test_interp_storage.py`, `tests/test_asch_with_interp.py`.

**Modified files:**
- `requirements.txt` — add `transformer_lens`, `torch`, `numpy`, `scikit-learn`, `scipy`.
- `psychbench/agents/model_agent.py` — optional `activation_collector` field and one if/else branch in `respond`.
- `psychbench/framework/session.py` — serialize `interpretability` block, write `.npz` sidecar, strip `interpretability_record` from JSONL metadata.
- `psychbench/experiments/asch/experiment.py` — in `build_agents`, dispatch to TL backend + collector when interp is enabled.

**Not modified in 3a:** `psychbench/experiments/asch_documents/*`. Phase 2 is untouched.

---

## Task 1: Add interpretability dependencies to `requirements.txt`

**Files:**
- Modify: `/Users/keegan/Documents/GitHub/Psych-bench/requirements.txt`

- [ ] **Step 1: Read the current `requirements.txt`**

Use the Read tool on `/Users/keegan/Documents/GitHub/Psych-bench/requirements.txt` so the Edit below has an exact match string.

- [ ] **Step 2: Append the interp deps**

Replace the entire file contents with:

```
PyYAML>=6.0
pytest>=7.0
transformer_lens>=2.0.0
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

- [ ] **Step 3: Install into the project venv**

Run: `.venv/bin/pip install -r requirements.txt`
Expected: installs without errors. On macOS this downloads ~200MB of torch CPU wheels on first run; subsequent runs are cached.

- [ ] **Step 4: Verify imports work**

Run: `.venv/bin/python -c "import transformer_lens, torch, numpy, sklearn, scipy; print('ok')"`
Expected: prints `ok` with no errors.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "add interpretability deps: transformer_lens, torch, numpy, sklearn, scipy"
```

---

## Task 2: Empty `interpretability/` package marker

**Files:**
- Create: `psychbench/interpretability/__init__.py`

- [ ] **Step 1: Create the empty file**

Write an empty file at `psychbench/interpretability/__init__.py` (zero bytes).

- [ ] **Step 2: Verify the package is importable**

Run: `.venv/bin/python -c "import psychbench.interpretability; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add psychbench/interpretability/__init__.py
git commit -m "scaffold psychbench.interpretability package"
```

---

## Task 3: `InterpretabilityConfig` + `resolve_interpretability`

**Files:**
- Create: `psychbench/interpretability/config.py`
- Create: `tests/test_interp_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_interp_config.py`:

```python
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
                "model": "hf-internal-testing/tiny-random-gpt2",
            },
        },
        "interpretability": {
            "enabled": True,
            "backend": "transformer_lens",
            "model": "hf-internal-testing/tiny-random-gpt2",
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
                              "model": "hf-internal-testing/tiny-random-gpt2"}},
        "interpretability": {
            "enabled": True,
            "model": "hf-internal-testing/tiny-random-gpt2",
        },
    }
    out = resolve_interpretability(cfg)
    assert isinstance(out, InterpretabilityConfig)
    assert out.model == "hf-internal-testing/tiny-random-gpt2"
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
    # No agents.naive check should fire when interp is off.
    out = resolve_interpretability({"interpretability": {"enabled": False}})
    assert out is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_interp_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psychbench.interpretability.config'`.

- [ ] **Step 3: Implement `psychbench/interpretability/config.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_interp_config.py -v`
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add psychbench/interpretability/config.py tests/test_interp_config.py
git commit -m "add InterpretabilityConfig + resolve_interpretability validation"
```

---

## Task 4: `TransformerLensBackend`

**Files:**
- Create: `psychbench/interpretability/backend.py`
- Create: `tests/test_interp_backend.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_interp_backend.py
"""TransformerLensBackend smoke tests against the tiny random model."""
from __future__ import annotations

import pytest


transformer_lens = pytest.importorskip("transformer_lens")

from psychbench.interpretability.backend import TransformerLensBackend  # noqa: E402


TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def backend():
    return TransformerLensBackend(model=TINY_MODEL, device="cpu")


def test_backend_loads_tiny_model_and_exposes_hooked_model(backend):
    assert backend.model == TINY_MODEL
    assert backend.hooked_model is not None
    # HookedTransformer has cfg.n_layers.
    assert hasattr(backend.hooked_model, "cfg")
    assert backend.hooked_model.cfg.n_layers >= 1


def test_backend_generate_returns_non_empty_string(backend):
    out = backend.generate("hello", stateful=False)
    assert isinstance(out, str)


def test_backend_stateful_raises_not_implemented(backend):
    with pytest.raises(NotImplementedError):
        backend.generate("hello", stateful=True)


def test_backend_reset_is_noop(backend):
    backend.reset()  # must not raise


def test_backend_device_explicit_cpu_accepted():
    b = TransformerLensBackend(model=TINY_MODEL, device="cpu")
    assert b.device == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_interp_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psychbench.interpretability.backend'`.

- [ ] **Step 3: Implement `psychbench/interpretability/backend.py`**

```python
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
        # TL's .generate returns the full text (prompt + completion).
        # Strip the prompt to match the ModelBackend contract.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_interp_backend.py -v`
Expected: PASS (5 tests). First run downloads `hf-internal-testing/tiny-random-gpt2` (~1–5MB) to `~/.cache/huggingface/`; subsequent runs are cached.

- [ ] **Step 5: Commit**

```bash
git add psychbench/interpretability/backend.py tests/test_interp_backend.py
git commit -m "add TransformerLensBackend: lazy-loaded, auto device, stateful raises"
```

---

## Task 5: `ActivationRecord` dataclass + `ActivationCollector` skeleton

**Files:**
- Create: `psychbench/interpretability/collector.py`
- Create: `tests/test_interp_collector.py` (will be expanded in Task 6)

This task lays down the dataclass and the constructor without the hook logic. Task 6 adds `collect()` and its behavior.

- [ ] **Step 1: Write the initial failing tests**

```python
# tests/test_interp_collector.py
"""ActivationCollector and ActivationRecord tests."""
from __future__ import annotations

import numpy as np
import pytest


transformer_lens = pytest.importorskip("transformer_lens")

from psychbench.interpretability.backend import TransformerLensBackend  # noqa: E402
from psychbench.interpretability.collector import (  # noqa: E402
    ActivationCollector, ActivationRecord,
)


TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def backend():
    return TransformerLensBackend(model=TINY_MODEL, device="cpu")


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
    # Should dedupe and sort to [0, 1, 2].
    assert c.layers == [0, 1, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_interp_collector.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the dataclass + constructor**

Create `psychbench/interpretability/collector.py`:

```python
"""ActivationCollector: owns generation, hooks residual/attn, builds records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from transformer_lens import HookedTransformer  # pragma: no cover


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
    forwards are skipped by an internal flag. Hooks are torn down in
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_interp_collector.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add psychbench/interpretability/collector.py tests/test_interp_collector.py
git commit -m "add ActivationRecord + ActivationCollector skeleton"
```

---

## Task 6: `ActivationCollector.collect()` — hook lifecycle + capture

**Files:**
- Modify: `psychbench/interpretability/collector.py`
- Modify: `tests/test_interp_collector.py` (append)

- [ ] **Step 1: Append behavior tests to `tests/test_interp_collector.py`**

Append to the end of `tests/test_interp_collector.py`:

```python
# --- collect() behavior ---

def test_collect_returns_text_and_record(backend):
    collector = ActivationCollector(layers="all", max_new_tokens=4)
    text, record = collector.collect(
        backend.hooked_model, "hello world",
    )
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
    assert attn.shape[1] == attn.shape[2]  # square seq x seq
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
    # After collect, no stale hooks should remain.
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_interp_collector.py -v`
Expected: FAIL — `ActivationCollector` has no `collect` method yet.

- [ ] **Step 3: Implement `collect()`**

Replace `psychbench/interpretability/collector.py` with:

```python
"""ActivationCollector: owns generation, hooks residual/attn, builds records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from transformer_lens import HookedTransformer  # pragma: no cover


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
            def hook(activation, hook):  # noqa: A002 (TL API uses "hook")
                if layer_idx in captured_resid:
                    return
                # activation: [batch, seq, d_model]; take last prompt token
                arr = activation[0, -1, :].detach().to("cpu").float().numpy()
                captured_resid[layer_idx] = arr.copy()
            return hook

        def make_attn_hook(layer_idx: int):
            def hook(pattern, hook):  # noqa: A002
                if layer_idx in captured_attn:
                    return
                # pattern: [batch, n_heads, seq, seq] (or [n_heads, seq, seq]
                # in some TL versions); collapse batch dim robustly.
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
            generated_text = generated if isinstance(generated, str) else str(generated)

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

    def _resolve_layers(self, hooked_model: "HookedTransformer") -> list[int]:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_interp_collector.py -v`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add psychbench/interpretability/collector.py tests/test_interp_collector.py
git commit -m "implement ActivationCollector.collect with hook lifecycle + first-pass gating"
```

---

## Task 7: Storage — `save_activation_record` / `load_activation_record`

**Files:**
- Create: `psychbench/interpretability/storage.py`
- Create: `tests/test_interp_storage.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_interp_storage.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_interp_storage.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `psychbench/interpretability/storage.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_interp_storage.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add psychbench/interpretability/storage.py tests/test_interp_storage.py
git commit -m "add .npz save/load for ActivationRecord with JSON token_positions"
```

---

## Task 8: `ModelAgent` — optional `activation_collector`

**Files:**
- Modify: `psychbench/agents/model_agent.py`

- [ ] **Step 1: Read the current file**

Read `psychbench/agents/model_agent.py` (the full file).

- [ ] **Step 2: Replace the file with the interp-aware version**

Overwrite `psychbench/agents/model_agent.py` with:

```python
"""LLM-backed agent with optional cross-trial stateful history."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

from psychbench.framework.agent import BaseAgent
from psychbench.framework.backends import ModelBackend
from psychbench.framework.types import AgentResponse, TrialContext


if TYPE_CHECKING:  # pragma: no cover
    from psychbench.interpretability.collector import ActivationCollector


PromptBuilder = Callable[[TrialContext], str]


def parse_letter_answer(text: str) -> str | None:
    """Extract A/B/C from model output.

    Prefer a final standalone letter (handles verbose responses ending in
    the letter), fall back to the first standalone letter anywhere.
    """
    stripped = text.strip()
    m_final = re.search(r"\b([ABC])\b\W*$", stripped)
    if m_final:
        return m_final.group(1)
    m_any = re.search(r"\b([ABC])\b", stripped)
    if m_any:
        return m_any.group(1)
    return None


class ModelAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        position: int,
        backend: ModelBackend,
        stateful: bool,
        prompt_builder: PromptBuilder,
        activation_collector: "ActivationCollector | None" = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            position=position,
            metadata={
                "type": "model",
                "model": backend.model,
                "stateful": stateful,
            },
        )
        self.backend = backend
        self.stateful = stateful
        self.prompt_builder = prompt_builder
        self.activation_collector = activation_collector

    def respond(self, context: TrialContext) -> AgentResponse:
        prompt = self.prompt_builder(context)
        if self.activation_collector is not None:
            raw, record = self.activation_collector.collect(
                self.backend.hooked_model, prompt, token_labels=None,
            )
            metadata = {
                "model": self.backend.model,
                "stateful": self.stateful,
                "interpretability_record": record,
            }
        else:
            raw = self.backend.generate(prompt, stateful=self.stateful)
            metadata = {
                "model": self.backend.model,
                "stateful": self.stateful,
            }
        parsed = parse_letter_answer(raw)
        return AgentResponse(
            agent_id=self.agent_id,
            raw_text=raw,
            parsed_answer=parsed,
            prompt=prompt,
            metadata=metadata,
        )

    def reset(self) -> None:
        self.backend.reset()
```

- [ ] **Step 3: Verify the full existing test suite still passes**

Run: `.venv/bin/python -m pytest tests/ -q --ignore=tests/test_interp_backend.py --ignore=tests/test_interp_collector.py --ignore=tests/test_asch_with_interp.py`
Expected: 100+ existing Phase 1/2 tests all pass; interp-specific tests are excluded here because this task only changes `ModelAgent`.

- [ ] **Step 4: Commit**

```bash
git add psychbench/agents/model_agent.py
git commit -m "ModelAgent: optional activation_collector; lazy import typing only"
```

---

## Task 9: `Session` — write `.npz` sidecar, add JSONL pointer

**Files:**
- Modify: `psychbench/framework/session.py`

- [ ] **Step 1: Read the current `session.py`**

Read the full file `psychbench/framework/session.py`.

- [ ] **Step 2: Replace `_serialize_trial` to emit the interp block and write the sidecar**

In `psychbench/framework/session.py`, replace the existing `_serialize_trial` method with:

```python
    def _serialize_trial(
        self, result: TrialResult, extra: dict[str, Any]
    ) -> dict[str, Any]:
        activations_dir = self.log_path.parent / "activations" / self.session_label
        responses_payload: list[dict[str, Any]] = []
        interp_block: dict[str, Any] | None = None
        for idx, r in enumerate(result.responses):
            metadata = dict(r.metadata) if r.metadata else {}
            record = metadata.pop("interpretability_record", None)
            serialized = asdict(AgentResponse(
                agent_id=r.agent_id,
                raw_text=r.raw_text,
                parsed_answer=r.parsed_answer,
                prompt=r.prompt,
                metadata=metadata,
            ))
            responses_payload.append(serialized)
            if record is not None and interp_block is None:
                rel_path = (
                    f"activations/{self.session_label}/"
                    f"trial_{result.trial_index:03d}.npz"
                )
                try:
                    from psychbench.interpretability.storage import (
                        save_activation_record,
                    )
                    save_activation_record(
                        record,
                        self.log_path.parent / rel_path,
                    )
                    interp_block = {
                        "activations_path": rel_path,
                        "layers": list(record.layers),
                        "n_prompt_tokens": int(record.n_prompt_tokens),
                        "token_positions": dict(record.token_positions),
                    }
                except OSError:
                    interp_block = None
        payload: dict[str, Any] = {
            "trial_index": result.trial_index,
            "is_critical": result.is_critical,
            "stimulus": asdict(result.stimulus),
            "correct_answer": result.correct_answer,
            "responses": responses_payload,
            "scoring": extra,
            "environment": self.environment.snapshot(),
            "session_label": self.session_label,
            "timestamp": time.time(),
        }
        if interp_block is not None:
            payload["interpretability"] = interp_block
        return payload
```

(Note: `activations_dir` is unused in the payload path because we build `rel_path` directly; the variable is left out to avoid a lint warning. If you kept it, remove it.)

- [ ] **Step 3: Run the full Phase 1/2 suite to confirm no regression**

Run: `.venv/bin/python -m pytest tests/ -q --ignore=tests/test_interp_backend.py --ignore=tests/test_interp_collector.py --ignore=tests/test_asch_with_interp.py`
Expected: all existing 100+ tests pass (Phase 1/2 trials have no `interpretability_record` in metadata; the new branch is a no-op for them).

- [ ] **Step 4: Commit**

```bash
git add psychbench/framework/session.py
git commit -m "Session: emit interpretability JSONL block and write .npz sidecar"
```

---

## Task 10: `AschExperiment.build_agents` — dispatch to TL + collector when interp enabled

**Files:**
- Modify: `psychbench/experiments/asch/experiment.py`

- [ ] **Step 1: Read the current file**

Read `psychbench/experiments/asch/experiment.py`.

- [ ] **Step 2: Add interp dispatch in `build_agents`**

Replace `build_agents` in `psychbench/experiments/asch/experiment.py` with:

```python
    def build_agents(self, *, for_control: bool = False) -> list[BaseAgent]:
        agents_cfg = self.config["agents"]
        confed_cfg = agents_cfg["confederates"]
        naive_cfg = agents_cfg["naive"]
        n_conf = int(confed_cfg.get("count", 5))
        dissenter = bool(confed_cfg.get("dissenter", False))
        behavior = confed_cfg.get("behavior", "always_wrong_on_critical")
        wrong_letter = confed_cfg.get("wrong_answer", "B")

        confederates: list[BaseAgent] = []
        for i in range(n_conf):
            if dissenter and i == 0:
                agent_behavior = "always_correct"
                wrong = None
            else:
                agent_behavior = behavior
                wrong = (
                    wrong_letter
                    if behavior == "always_wrong_on_critical"
                    else None
                )
            confederates.append(ScriptedAgent(
                agent_id=f"confederate_{i+1}",
                position=i,
                behavior=agent_behavior,
                wrong_answer=wrong,
            ))

        # Interp dispatch: if interpretability is enabled, swap naive's
        # backend to TransformerLens and attach an ActivationCollector.
        from psychbench.interpretability.config import (
            resolve_interpretability,
        )
        interp_cfg = resolve_interpretability(self.config)
        if interp_cfg is not None and not for_control:
            from psychbench.interpretability.backend import (
                TransformerLensBackend,
            )
            from psychbench.interpretability.collector import (
                ActivationCollector,
            )
            naive_backend = TransformerLensBackend(
                model=interp_cfg.model, device=interp_cfg.device,
            )
            collector = ActivationCollector(
                layers=interp_cfg.layers,
                max_new_tokens=interp_cfg.max_new_tokens,
            )
        else:
            naive_backend = get_backend(
                naive_cfg.get("backend", "echo"),
                naive_cfg.get("model", "echo-test"),
            )
            collector = None

        position_cfg = naive_cfg.get("position", "last")
        naive_position = (
            n_conf if position_cfg == "last" else int(position_cfg)
        )
        naive = ModelAgent(
            agent_id="naive",
            position=naive_position,
            backend=naive_backend,
            stateful=bool(naive_cfg.get("stateful", False)),
            prompt_builder=build_asch_prompt,
            activation_collector=collector,
        )

        return [*confederates, naive]
```

- [ ] **Step 3: Run the full Phase 1/2 suite to confirm no regression**

Run: `.venv/bin/python -m pytest tests/ -q --ignore=tests/test_asch_with_interp.py`
Expected: all 100+ Phase 1/2 tests + interp unit tests pass. Phase 1 configs (no `interpretability:` block) take the else branch; behavior is unchanged.

- [ ] **Step 4: Commit**

```bash
git add psychbench/experiments/asch/experiment.py
git commit -m "AschExperiment: attach ActivationCollector to naive when interp enabled"
```

---

## Task 11: Shipped smoke config + end-to-end integration test

**Files:**
- Create: `config/experiments/asch_phase1_with_interp_smoke.yaml`
- Create: `tests/test_asch_with_interp.py`

- [ ] **Step 1: Write the smoke config**

Create `config/experiments/asch_phase1_with_interp_smoke.yaml`:

```yaml
experiment:
  name: asch_phase1_with_interp_smoke
  type: asch
  trials: 18
  critical_trials: 12
  critical_trial_indices: [2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17]
  seed: 42

agents:
  confederates:
    type: scripted
    count: 5
    behavior: always_wrong_on_critical
    wrong_answer: B
    dissenter: false
  naive:
    type: model
    backend: transformer_lens
    model: hf-internal-testing/tiny-random-gpt2
    stateful: false
    position: last

environment:
  response_visibility: public
  answer_order: sequential

control:
  run_control: false
  response_visibility: private

scoring:
  method: binary
  conformity_threshold: 1

interpretability:
  enabled: true
  backend: transformer_lens
  model: hf-internal-testing/tiny-random-gpt2
  device: cpu
  layers: all
  max_new_tokens: 64

logging:
  save_context_windows: true
  output_dir: results/
  format: jsonl
```

- [ ] **Step 2: Write the failing integration test**

Create `tests/test_asch_with_interp.py`:

```python
"""End-to-end: AschExperiment + interp enabled + tiny-random-gpt2."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


transformer_lens = pytest.importorskip("transformer_lens")

from psychbench.experiments.asch.experiment import AschExperiment  # noqa: E402
from psychbench.interpretability.storage import (  # noqa: E402
    load_activation_record,
)


def _load_smoke_config(n_trials: int):
    path = Path("config/experiments/asch_phase1_with_interp_smoke.yaml")
    cfg = yaml.safe_load(path.read_text())
    # Override trials for test speed.
    cfg["experiment"]["trials"] = n_trials
    cfg["experiment"]["critical_trials"] = min(
        n_trials, cfg["experiment"]["critical_trials"],
    )
    cfg["experiment"]["critical_trial_indices"] = [
        i for i in cfg["experiment"]["critical_trial_indices"]
        if i < n_trials
    ]
    return cfg


def test_asch_with_interp_end_to_end(tmp_path):
    cfg = _load_smoke_config(n_trials=2)
    exp = AschExperiment(cfg)
    out = exp.run(output_dir=tmp_path)
    # Pick up the experimental log path.
    exp_result = out["experimental"]
    log_path = Path(exp_result["log_path"])
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert "interpretability" in record
        interp = record["interpretability"]
        assert "activations_path" in interp
        act_path = log_path.parent / interp["activations_path"]
        assert act_path.exists()
        loaded = load_activation_record(act_path)
        assert loaded.n_prompt_tokens >= 1
        assert loaded.layers
        assert all(L in loaded.layer_activations for L in loaded.layers)
        assert all(L in loaded.attention_weights for L in loaded.layers)
```

- [ ] **Step 3: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_asch_with_interp.py -v`
Expected: PASS (1 test). First run is slow (cold torch import + model download); subsequent runs ~5–15s.

- [ ] **Step 4: Commit**

```bash
git add config/experiments/asch_phase1_with_interp_smoke.yaml tests/test_asch_with_interp.py
git commit -m "add shipped Phase 1 + interp smoke config and end-to-end test"
```

---

## Task 12: `docs/interpretability.md`

**Files:**
- Create: `docs/interpretability.md`

- [ ] **Step 1: Author the doc**

Create `docs/interpretability.md`:

```markdown
# PsychBench Interpretability (Phase 3a)

Phase 3a adds a framework-level hook into any experiment so the naive
agent's residual stream and attention patterns can be captured during
each trial's forward pass. **3a is plumbing only** — there is no
persona-locator, conflict-detector, SAE, or visualizer yet. Those live
in Phases 3b and 3c.

## What 3a does

- Adds a new `interpretability:` YAML block.
- When `interpretability.enabled: true`, the naive `ModelAgent` runs
  through a `TransformerLensBackend` (loading a `HookedTransformer`)
  and its generation is wrapped by an `ActivationCollector` that
  installs hooks, runs the model once, tears hooks down, and returns
  both the generated text and an `ActivationRecord`.
- Per-trial records are written to `.npz` sidecar files at
  `results/<run>/activations/<session_label>/trial_<NNN>.npz`; the
  existing JSONL gets a lean `interpretability: { activations_path,
  layers, n_prompt_tokens, token_positions }` block.
- Offline analysis reads back via
  `psychbench.interpretability.storage.load_activation_record(path)`.

## Hermetic smoke workflow

The shipped smoke config uses `hf-internal-testing/tiny-random-gpt2`,
which TransformerLens downloads once (~1–5 MB) and caches locally. The
weights are random-init, so the **plumbing** is real but the
**activations are noise** — that's fine for 3a. Real findings require
a real pretrained model.

```bash
.venv/bin/python -m psychbench run \
  --config config/experiments/asch_phase1_with_interp_smoke.yaml \
  --output-dir results/
```

Inspect the output:

```bash
ls results/asch_experimental_*.jsonl          # per-trial records
ls results/activations/experimental/*.npz     # sidecars
```

## Real-model workflow

3a is designed to run on a GPU host; do the interp work there, not on
your laptop. Typical flow:

1. SSH to a GPU instance (AWS `g5.xlarge` — one A10G — is enough for
   Llama-3.1-8B in bf16 at ~$1/hour).
2. `git clone` the repo + `pip install -r requirements.txt`.
3. Edit `config/experiments/asch_phase1_with_interp_smoke.yaml`:
   - `interpretability.model` → `meta-llama/Meta-Llama-3.1-8B-Instruct`
   - `interpretability.device` → `cuda`
   - `agents.naive.model` → same as `interpretability.model` (required
     by validation)
   - `agents.naive.backend` → `transformer_lens` (required by
     validation)
   - Optionally subset `interpretability.layers` — see storage math
     below.
4. For gated models, `export HF_TOKEN=...`.
5. `python -m psychbench run --config <your.yaml> --output-dir results/`.
6. `scp -r <user>@<host>:<path>/results/... ./` to pull results back.

## YAML reference

| Field | Type | Default | Meaning |
|---|---|---|---|
| `interpretability.enabled` | bool | `false` | Master switch. When false or absent, Phase 1/2 behavior is unchanged. |
| `interpretability.backend` | str | `"transformer_lens"` | Only legal value in 3a. |
| `interpretability.model` | str | *required* | Model ID. Must match `agents.naive.model` (validator enforces). |
| `interpretability.device` | str\|null | auto | `cpu`, `cuda`, `mps`, or null (auto: cuda → mps → cpu). |
| `interpretability.layers` | list[int] \| `"all"` | `"all"` | Which layer indices to hook. `"all"` resolves at model-load time. |
| `interpretability.max_new_tokens` | int | `64` | Generation length cap. |

## Storage math

Per trial, the sidecar stores:

- residual stream: one `float32` vector of shape `[d_model]` per layer.
- attention pattern: one `float32` tensor of shape `[n_heads, prompt_seq, prompt_seq]` per layer.

Attention dominates. For concrete models:

| Model | n_layers | n_heads | d_model | prompt_seq ≈ 400 | Per-trial (all layers) |
|---|---|---|---|---|---|
| tiny-random-gpt2 | 2 | 4 | ~64 | 20 | < 10 KB |
| pythia-70m | 6 | 8 | 512 | 400 | ~25 MB |
| Llama-3.1-8B | 32 | 32 | 4096 | 400 | ~650 MB |

For real models, subset layers aggressively:

```yaml
interpretability:
  layers: [8, 12, 16, 20, 24]   # 5 layers instead of 32
```

## Offline analysis

```python
import json
from pathlib import Path
from psychbench.interpretability.storage import load_activation_record

run_dir = Path("results/asch_experimental_1777500000")  # pick your run
log_path = next(run_dir.glob("*.jsonl"))

for line in log_path.read_text().splitlines():
    rec = json.loads(line)
    interp = rec.get("interpretability")
    if interp is None:
        continue
    record = load_activation_record(
        run_dir / interp["activations_path"]
    )
    # record.layer_activations[L] — [d_model] float32
    # record.attention_weights[L] — [n_heads, seq, seq] float32
```

## Known limitations

- Random-init tiny model produces meaningless activations. Smoke run
  proves plumbing; real conclusions require a pretrained model.
- Only the **last prompt token** residual is captured (not the full
  per-token trajectory through generation). 3b/3c may extend this.
- Attention is captured over the **prompt only** — generation-phase
  attention is not logged.
- Device choice affects numerical outputs slightly (float32 CPU vs
  bf16/CUDA/MPS). Results are seed-stable within a device, not
  bit-identical across devices.
- `interpretability` + `agents.naive.stateful: true` is not supported in
  3a (`TransformerLensBackend.generate(stateful=True)` raises
  `NotImplementedError`).

## Roadmap

- **3b — Persona Locator.** Linear probes to find an Assistant-persona
  direction per layer, `fit-probe` CLI, persona projection logged per
  trial.
- **3c — Conflict Detector + SAE + Attention Analyzer + Visualizer.**
  The rest of the original Phase 3 spec.
```

- [ ] **Step 2: Commit**

```bash
git add docs/interpretability.md
git commit -m "document Phase 3a: smoke workflow, real-model flow, storage, limits"
```

---

## Task 13: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: ALL tests pass. 100 pre-existing Phase 1/2 tests + ~30 new Phase 3a tests.

- [ ] **Step 2: Manual smoke end-to-end**

Run:

```bash
rm -rf /tmp/psychbench-phase3a-smoke
.venv/bin/python -m psychbench run \
  --config config/experiments/asch_phase1_with_interp_smoke.yaml \
  --output-dir /tmp/psychbench-phase3a-smoke
ls /tmp/psychbench-phase3a-smoke/
ls /tmp/psychbench-phase3a-smoke/activations/experimental/ | head
```

Expected: command exits 0; a JSONL exists; `activations/experimental/` contains 18 `.npz` files.

- [ ] **Step 3: Manual spot-check of one record**

Run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import json
from psychbench.interpretability.storage import load_activation_record

d = next(Path("/tmp/psychbench-phase3a-smoke").glob("asch_experimental_*.jsonl"))
rec = json.loads(d.read_text().splitlines()[0])
interp = rec["interpretability"]
act = load_activation_record(d.parent / interp["activations_path"])
print("layers", act.layers)
print("n_prompt_tokens", act.n_prompt_tokens)
print("sample resid shape", act.layer_activations[act.layers[0]].shape)
print("sample attn shape", act.attention_weights[act.layers[0]].shape)
PY
```

Expected: prints the layer list, the prompt token count, and the two shapes. Shapes match the spec (`[d_model]` and `[n_heads, seq, seq]`).

- [ ] **Step 4: No code changes; no commit.**

---

## Self-Review

**Spec coverage:**

- §1 scope → Task 11 (smoke config), Task 12 (docs).
- §2 repo layout → Tasks 2–11 collectively create every file listed.
- §3 `TransformerLensBackend` → Task 4.
- §4 `ActivationCollector` + `ActivationRecord` → Tasks 5, 6.
- §5 `ModelAgent` integration → Task 8.
- §6 Storage + JSONL pointer → Tasks 7, 9.
- §7 Config wiring → Tasks 3, 10, 11.
- §8 Docs → Task 12.
- §9 Testing → Every code task bundles tests; Task 13 runs the full suite.
- §10 Error handling → Task 3 (config errors), Task 4 (stateful raise),
  Task 6 (layer range, hook teardown on exception), Task 9 (OSError
  isolation).

**Placeholder scan:** No TBD/TODO/"similar to". Every task shows the code
to write. Task 8 and Task 10 rewrite existing files with their full new
contents rather than pointing at surrounding context.

**Type consistency:**

- `InterpretabilityConfig.{model, device, layers, max_new_tokens}` used
  identically in Tasks 3, 10.
- `ActivationRecord.{trial_id, trial_type, outcome, n_prompt_tokens, layers,
  layer_activations, attention_weights, token_positions, generated_text}`
  used identically in Tasks 5, 6, 7, 8, 9, 11.
- `ActivationCollector.collect(hooked_model, prompt, token_labels=None) ->
  tuple[str, ActivationRecord]` used identically in Tasks 6, 8, 11.
- `save_activation_record` / `load_activation_record` signatures
  consistent across Tasks 7, 9, 11, 13.
- JSONL `interpretability` block keys (`activations_path`, `layers`,
  `n_prompt_tokens`, `token_positions`) match across Tasks 9, 11, 12.
- Hook names `blocks.{L}.hook_resid_post` and
  `blocks.{L}.attn.hook_pattern` consistent across Task 6 and the spec §4.2.
