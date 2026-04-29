# PsychBench Phase 3a — Interpretability Foundation Design Spec

**Status:** Draft for user review
**Author:** Keegan (brainstormed interactively)
**Date:** 2026-04-29
**Relation to Phase 3:** 3a is the first of three sub-specs (3a foundation → 3b persona locator → 3c conflict + SAE + attention + visualizer). Phases 3b and 3c are out of scope here and will each get their own spec/plan cycles.

---

## 1. Scope

Phase 3a is the **interpretability foundation layer** for PsychBench: framework plumbing only, zero analysis logic. It makes the rest of Phase 3 possible by wiring TransformerLens in as an opt-in naive-agent backend, capturing residual-stream and attention-pattern activations at configured layers every trial, and persisting them to disk alongside the existing JSONL logs.

### 1.1 Deliverables

- `psychbench/interpretability/` — new top-level package.
- `TransformerLensBackend` implementing the existing `ModelBackend` protocol (`generate`, `reset`), owning a `HookedTransformer` instance.
- `ActivationCollector` + `ActivationRecord` — hooks into a `HookedTransformer`, owns generation when attached, captures residual stream at the last prompt token and attention over the prompt at every configured layer.
- `.npz` sidecar persistence: per-trial activations live at `<run_dir>/activations/<session_label>/trial_<nnn>.npz`; the per-trial JSONL gains a lean `interpretability: { activations_path, layers, n_prompt_tokens, token_positions }` block.
- `save_activation_record` / `load_activation_record` — canonical round-trip API for 3b/3c and notebook users.
- YAML config: new top-level `interpretability:` block with `enabled`, `backend`, `model`, `device`, `layers`, `max_new_tokens`.
- `ModelAgent` gains an optional `activation_collector` field; when set, collector owns generation (one forward pass, not two).
- `AschExperiment.build_agents()` wires the collector to the naive agent when interp is enabled. Phase 2 (`asch_documents`) is not modified here.
- `config/experiments/asch_phase1_with_interp_smoke.yaml` — hermetic Phase 1 smoke config using `hf-internal-testing/tiny-random-gpt2`.
- `docs/interpretability.md` — paradigm, config shape, hermetic vs real-model workflow, storage math, known limitations.
- `requirements.txt` gains `transformer_lens`, `torch`, `numpy`, `scikit-learn`, `scipy`.
- Tests: real TransformerLens against the tiny test model, `importorskip`-guarded.

### 1.2 Non-goals (deferred)

- `PersonaLocator`, `ConflictDetector`, `SAEInterface`, `AttentionAnalyzer`, `Visualizer`. (3b/3c.)
- `fit-probe` and `visualize` CLI subcommands. (3b/3c.)
- Structured token labels per experiment (Asch passes `None`; record logs `{"full_prompt": [0, n_prompt_tokens]}`). (3b/3c.)
- `persona_projections`, `conflict_scores`, `top_sae_features`, per-layer derived metrics in JSONL. (3b/3c.)
- Instrumenting confederate agents in `asch_documents`. (3b/3c.)
- `stateful: true` interp runs (raises `NotImplementedError` in 3a).
- Cost/disk gate for large interp sweeps (Phase 2 has trial-count gate; add later if needed).
- Multi-GPU sharding, quantization, batched generation.

### 1.3 Success criterion

Running `python -m psychbench run --config config/experiments/asch_phase1_with_interp_smoke.yaml` on the user's laptop:

1. Completes end-to-end without errors.
2. Produces a results JSONL whose every trial record carries an `interpretability` block with a real `activations_path`.
3. `load_activation_record(path)` returns a populated record with shape-correct arrays.
4. The existing Phase 1/2 test suite (100 tests) still passes with no code touched in Phase 1/2 modules beyond `ModelAgent` and `Session`.

Activation *values* are not part of the success criterion — tiny-random-gpt2 has random weights, so numbers are noise. 3a proves plumbing.

---

## 2. Architectural approach

**Q-by-Q locked choices from brainstorming:**

- **Q2 — agent integration:** B. `ModelAgent` gains an optional `activation_collector` field; no new agent subclass.
- **Q3 — forward-pass architecture:** B. Collector owns generation when attached (one forward pass total, collector installs/removes hooks around `hooked_model.generate`).
- **Q4 — test model:** single `hf-internal-testing/tiny-random-gpt2` used for both tests and shipped smoke config (~1–5MB cached once).
- **Q5 — layers:** D. Default `layers: "all"`, user can override with an int list. Storage math documented.
- **Q6 — token labels:** A with 3a default. Experiments pass `token_labels` into the collector; 3a's Asch passes `None`, record logs `{"full_prompt": [0, n_tokens]}`.
- **Q7 — config wiring:** A. New top-level `interpretability:` block; hard error if `agents.naive.backend/model` disagrees with `interpretability.backend/model`.
- **Q8 — JSONL schema:** B. Bulky arrays to `.npz` sidecar; JSONL gets a lean pointer + metadata.
- **Q9 — testing boundary:** A. Real TransformerLens + real tiny model in tests, `importorskip`-guarded so the suite degrades gracefully if deps aren't installed.

### 2.1 Repo layout

```
psychbench/
├── interpretability/                   # NEW module
│   ├── __init__.py
│   ├── backend.py                      # TransformerLensBackend
│   ├── collector.py                    # ActivationCollector + ActivationRecord
│   ├── storage.py                      # save_activation_record / load_activation_record
│   └── config.py                       # resolve_interpretability + InterpretabilityConfig
├── agents/
│   └── model_agent.py                  # MODIFY: optional activation_collector field
├── framework/
│   └── session.py                      # MODIFY: serialize interpretability block + sidecar .npz
├── experiments/asch/
│   └── experiment.py                   # MODIFY: if interp enabled, attach collector to naive
config/experiments/
└── asch_phase1_with_interp_smoke.yaml  # NEW — hermetic smoke
docs/
└── interpretability.md                 # NEW
requirements.txt                        # MODIFY — add interp deps
tests/
├── test_interp_config.py               # NEW
├── test_interp_backend.py              # NEW — importorskip
├── test_interp_collector.py            # NEW — importorskip
├── test_interp_storage.py              # NEW
└── test_asch_with_interp.py            # NEW — importorskip, end-to-end
```

---

## 3. `TransformerLensBackend`

**File:** `psychbench/interpretability/backend.py`

Implements the existing `ModelBackend` protocol. `transformer_lens` and `torch` are imported lazily inside `__init__` so Phase 1/2 (interp-off) runs never pay the import cost.

### 3.1 Public surface

```python
class TransformerLensBackend:
    model: str                           # protocol: the model string ID
    hooked_model: "HookedTransformer"    # public — collector attaches hooks to this
    device: str                          # "cpu" | "cuda" | "mps"

    def __init__(self, model: str, device: str | None = None) -> None: ...
    def generate(self, prompt: str, stateful: bool = False) -> str: ...
    def reset(self) -> None: ...
```

### 3.2 Behavior

- **Device resolution.** `device=None` → `cuda` if `torch.cuda.is_available()`, else `mps` if Apple Silicon MPS is available, else `cpu`. Explicit `device` always wins.
- **`generate()`.** Calls `hooked_model.generate(prompt, max_new_tokens=64, temperature=0.0, do_sample=False)` and returns only the newly generated suffix (strips the prompt).
- **`generate(stateful=True)`.** Raises `NotImplementedError` — 3a does not support cross-trial conversation state for interp runs.
- **`reset()`.** No-op (nothing to clear since stateful is disallowed).
- **`hooked_model` is public**. The collector needs direct access. Naming: `model` is the string (protocol requirement); `hooked_model` is the instance. No accessor obfuscation.
- **Device fixed at construction.** No `move_to(device)` method. To change device, make a new backend.

---

## 4. `ActivationCollector` and `ActivationRecord`

**File:** `psychbench/interpretability/collector.py`

Per Q3-B, the collector owns generation when attached. It installs hooks, runs the model, tears hooks down, and returns both the generated text and a populated `ActivationRecord`.

### 4.1 `ActivationRecord` schema

```python
@dataclass
class ActivationRecord:
    trial_id: int | None                 # set by session, None at capture time
    trial_type: str                      # "critical" | "non_critical" | "unknown"
    outcome: str | None                  # "conformed" | "resisted" | None (post-hoc)
    n_prompt_tokens: int
    layers: list[int]                    # resolved from "all" or explicit list
    layer_activations: dict[int, np.ndarray]   # layer -> [d_model] float32 (last prompt token)
    attention_weights: dict[int, np.ndarray]    # layer -> [n_heads, prompt_seq, prompt_seq] float32
    token_positions: dict[str, list[int]]       # e.g. {"full_prompt": [0, n_prompt_tokens]}
    generated_text: str
```

### 4.2 What 3a captures

For each configured layer:

- **`blocks.{L}.hook_resid_post` at the last prompt token** → `[d_model]` float32. Not the full sequence; the spec's target is the single last-token residual.
- **`blocks.{L}.attn.hook_pattern` over the prompt** → `[n_heads, prompt_seq, prompt_seq]` float32. Captured during the *first* forward pass only — subsequent generation-step passes are ignored via a "have we captured yet?" flag inside the hook.

### 4.3 Collector API

```python
class ActivationCollector:
    def __init__(
        self,
        layers: list[int] | str = "all",
        max_new_tokens: int = 64,
    ) -> None: ...

    def collect(
        self,
        hooked_model: "HookedTransformer",
        prompt: str,
        token_labels: dict[str, list[int]] | None = None,
    ) -> tuple[str, ActivationRecord]:
        """Install hooks, run generate, tear down, return (generated_text, record)."""
```

### 4.4 Key behaviors

- **Hook lifecycle safety.** `collect` wraps `hooked_model.generate(...)` in `try / finally`; `finally` calls `hooked_model.reset_hooks()` so no hooks leak, even if generation throws.
- **First-forward-only gating.** Hook function checks `if L in self._captured: return`; captures and stores, then is a no-op on subsequent generation steps.
- **`layers="all"` resolution.** On first `collect`, resolve via `list(range(hooked_model.cfg.n_layers))`; cache the resolved list so every subsequent record reports the same `layers`.
- **Duplicate/out-of-range validation.** Explicit lists are sorted and deduped. An index `< 0` or `>= n_layers` raises `ValueError` at first `collect` call with the model's actual layer count.
- **Token labels.** `token_labels=None` → record's `token_positions = {"full_prompt": [0, n_prompt_tokens]}`. A supplied dict is preserved as-is.
- **Return shape.** `(generated_text: str, record: ActivationRecord)` — text is first-class because the agent still needs to return it to the session. Folding it into the record would work too; the tuple is more honest about what's model output vs side-channel.

### 4.5 Invariant documented in the collector docstring

"The last-token axis index `-1` on the cached hook tensor is the final **prompt** token, because the hook only captures during the first forward pass (the prompt-phase pass). Subsequent generation steps are skipped by the capture flag, so `cache[..., -1, :]` is always the end-of-prompt residual, never a mid-generation step."

---

## 5. `ModelAgent` integration

**File (modify):** `psychbench/agents/model_agent.py`

One optional constructor argument, lazy import inside `respond()`:

```python
class ModelAgent(BaseAgent):
    def __init__(
        self,
        ...,
        activation_collector: "ActivationCollector | None" = None,
    ) -> None:
        ...
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
```

The `ActivationRecord` rides along inside `AgentResponse.metadata["interpretability_record"]` as a Python object. Serialization to disk is `Session`'s job, not `ModelAgent`'s — keeps the agent thin.

### 5.1 `reset()` behavior

`ModelAgent.reset()` continues to call `backend.reset()`. When a collector is attached, the collector has no per-session state to clear (it's created fresh per cell/session), so there's nothing to add here.

---

## 6. Storage — `.npz` sidecar + JSONL pointer

**File:** `psychbench/interpretability/storage.py`

### 6.1 Disk layout

```
results/<run_name>/
├── <session_label>.jsonl                  # Phase 1 flat
├── <session_label>.summary.json           # Phase 1 flat
├── cells/<cell_id>.jsonl                  # Phase 2 nested
├── cells/<cell_id>.summary.json           # Phase 2 nested
└── activations/
    └── <session_label_or_cell_id>/
        ├── trial_000.npz
        ├── trial_001.npz
        └── ...
```

`Session` derives the activations directory from the log path's parent plus `activations/<session_label>/`; the collector/storage never see the run-dir structure directly.

### 6.2 `.npz` contents

Built with `np.savez_compressed`. Keys:

- `layers`: `int64` 1D array of captured layer indices.
- For each layer `L`: `resid_{L}` → shape `[d_model]`, `float32`.
- For each layer `L`: `attn_{L}` → shape `[n_heads, prompt_seq, prompt_seq]`, `float32`.
- `n_prompt_tokens`: `int64` scalar.
- `generated_text`: object-dtype length-1 array (numpy's standard lossless string container).
- `token_positions`: object-dtype length-1 array containing a JSON string of `{label: [int, int]}`.
- `trial_id` / `trial_type` / `outcome` are **not** duplicated into `.npz` — they live in JSONL. Offline tools that need them join by `activations_path`.

### 6.3 Storage API

```python
def save_activation_record(record: ActivationRecord, path: str | Path) -> None: ...
def load_activation_record(path: str | Path) -> ActivationRecord: ...
```

Round-trip invariant: `load(save(r)) == r` for every field — tested in `test_interp_storage.py`. `save` creates missing parent directories. `load` raises `FileNotFoundError` naming the path.

### 6.4 JSONL pointer shape

Per-trial record gains a single top-level `interpretability` key:

```json
{
  "trial_index": 5,
  "is_critical": true,
  "stimulus": {"...": "..."},
  "responses": [{"...": "..."}],
  "scoring": {"...": "..."},
  "environment": {"...": "..."},
  "session_label": "experimental",
  "timestamp": 1777500000.0,
  "interpretability": {
    "activations_path": "activations/experimental/trial_005.npz",
    "layers": [0, 1],
    "n_prompt_tokens": 127,
    "token_positions": {"full_prompt": [0, 127]}
  }
}
```

Paths are **relative to the run directory** so a whole run is portable via `scp`. When interpretability is disabled, the `interpretability` key is absent (not `null`).

### 6.5 `Session` change

`psychbench/framework/session.py` gets one small addition in `_serialize_trial`: if the naive-agent response carries `interpretability_record` in its metadata, `Session` writes `<run_dir>/activations/<session_label>/trial_<NNN>.npz` and adds the pointer block to the JSONL record. The record is stripped from the metadata that gets serialized into `responses[*].metadata` (binary arrays don't belong in JSONL via the default serializer).

If `save_activation_record` raises `OSError`, the trial's interp block is *omitted* and a warning is logged; the rest of the trial record still writes. Mirrors Phase 2's per-cell isolation at the trial level.

---

## 7. Config wiring

**File:** `psychbench/interpretability/config.py`

### 7.1 YAML block

```yaml
interpretability:
  enabled: true
  backend: transformer_lens         # only legal value in 3a
  model: hf-internal-testing/tiny-random-gpt2
  device: cpu                        # cpu | cuda | mps | null (auto)
  layers: all                        # "all" or list of ints
  max_new_tokens: 64
```

Defaults when `enabled: true` but a field is omitted:

- `backend` → `"transformer_lens"`
- `device` → `None` (auto-detect)
- `layers` → `"all"`
- `max_new_tokens` → `64`

When `enabled: false` or the whole block is absent, Phase 1/2 behave identically to today.

### 7.2 `resolve_interpretability`

```python
@dataclass
class InterpretabilityConfig:
    model: str
    device: str | None
    layers: list[int] | str    # "all" or resolved list
    max_new_tokens: int


def resolve_interpretability(config: dict) -> InterpretabilityConfig | None:
    """Return a validated config, or None if interp is not enabled."""
```

Validation:

- `backend` must be `"transformer_lens"` (error naming the offending value).
- `agents.naive.backend`, if present, must be `"transformer_lens"` (hard error, per Q7).
- `agents.naive.model`, if present, must equal `interpretability.model` (hard error).
- `device` ∈ `{"cpu", "cuda", "mps", None}`.
- `layers` is `"all"` or a list of `int`; anything else is an error.
- `max_new_tokens` is a positive int.

Errors are raised early — before any model load, before any trial runs — so misconfigured experiments fail in milliseconds.

### 7.3 `AschExperiment.build_agents()` integration

```python
interp_cfg = resolve_interpretability(self.config)
if interp_cfg is not None:
    from psychbench.interpretability.backend import TransformerLensBackend
    from psychbench.interpretability.collector import ActivationCollector
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

naive = ModelAgent(
    agent_id="naive",
    position=naive_position,
    backend=naive_backend,
    stateful=bool(naive_cfg.get("stateful", False)),
    prompt_builder=build_asch_prompt,
    activation_collector=collector,
)
```

Lazy imports inside the `if` branch: Phase 1/2 runs never touch TransformerLens when interp is off.

### 7.4 Shipped smoke config

`config/experiments/asch_phase1_with_interp_smoke.yaml`:

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

- `agents.naive.backend` and `agents.naive.model` match the `interpretability` block (required by validation). They're kept in the YAML for readability.
- `control.run_control: false` — no second session; doubling interp overhead for a smoke is pointless.
- No Phase 2 counterpart in 3a.

---

## 8. Documentation: `docs/interpretability.md`

Sections:

1. **What Phase 3a is** — framework plumbing only; no analysis; proof of wiring.
2. **Hermetic smoke workflow** — one-command run on the user's laptop with the tiny random model; what to expect (random outputs, real JSONL structure, real `.npz` files).
3. **Real-model workflow** — SSH to a GPU host (AWS `g5.xlarge` used as a concrete example), `pip install -r requirements.txt`, edit `interpretability.model` and `agents.naive.model` to match a real model (Llama-3.1-8B is the worked example), set `device: cuda`, run.
4. **YAML reference** — every field in the `interpretability:` block with its type, default, and semantics.
5. **Storage math** — concrete table: tiny model (2 layers, d_model small) vs a real 32-layer model at d_model=4096. Warn that full-layer capture on real models is tens of MB per trial; show how to subset with `layers: [8, 12, 16, 20, 24]`.
6. **Offline analysis recipe** — how to load results from a notebook (`load_activation_record` + joining to JSONL by relative path).
7. **Known limitations** — random tiny-model activations are meaningless; last-prompt-token snapshot only; attention over prompt only; device-dependent numeric differences.
8. **Phase 3b/3c roadmap** — pointer to upcoming persona-locator and full analysis layer specs.

---

## 9. Testing plan

All interp tests guarded with `pytest.importorskip("transformer_lens")` at the top.

| File | Coverage |
|---|---|
| `tests/test_interp_config.py` | `resolve_interpretability` returns None / populated; defaults filled; hard errors on backend/model mismatch; accepts `layers: "all"` and `layers: [0, 1]`; rejects bad devices and unknown backends. No importorskip needed. |
| `tests/test_interp_backend.py` | `TransformerLensBackend(tiny-random-gpt2)` loads; `hooked_model` is a `HookedTransformer`; `generate("hello")` returns a non-empty string; `stateful=True` raises; `reset` is no-op. |
| `tests/test_interp_collector.py` | `collect` returns `(text, record)`; `record.layers` equals full range when `"all"`; `layer_activations[L]` shape `[d_model]`; `attention_weights[L]` shape `[n_heads, prompt_seq, prompt_seq]`; `layers=[0]` subsets; re-calling `collect` leaks no hooks; exception in generate still tears hooks down; `token_labels=None` default vs custom preserved. |
| `tests/test_interp_storage.py` | Round-trip `save`/`load` on a synthetic record; unicode / newlines in `generated_text`; missing parent dir auto-created; `load` on nonexistent path raises. |
| `tests/test_asch_with_interp.py` | End-to-end `AschExperiment` with a 2-trial override of the smoke config; produces JSONL with `interpretability` blocks; every `activations_path` exists; `load_activation_record` returns populated records. |

**Regression check.** The existing 100-test Phase 1/2 suite must still pass after `ModelAgent` and `Session` edits.

**Not tested.** Activation-value correctness (tiny-random-gpt2); real-model runs; stateful interp (raises by design); batching.

---

## 10. Error handling and edge cases

### 10.1 Fail-loud, config/load-time

- `interpretability.backend != "transformer_lens"` → `ValueError`.
- `agents.naive.backend != "transformer_lens"` under `enabled: true` → `ValueError` with the "interp needs direct model access" message.
- `agents.naive.model` disagrees with `interpretability.model` → `ValueError`.
- `interpretability.device` not in `{cpu, cuda, mps, None}` → `ValueError`.
- `layers: N` where `N < 0` or `N >= cfg.n_layers` → `ValueError` at first `collect()` with the real layer count.
- `transformer_lens` / `torch` missing when interp is enabled → `ImportError` with install instructions.

### 10.2 Fail-loud, runtime

- `TransformerLensBackend.generate(stateful=True)` → `NotImplementedError`.
- `generate()` throws (CUDA OOM, torch bug) → hooks removed via `try/finally`; exception propagates; session's existing per-trial error handling logs it.
- `save_activation_record` raises `OSError` → trial's `interpretability` field is omitted (trial still written); a warning is logged. Single-trial isolation.

### 10.3 Edge cases explicitly handled

- Empty prompt — tokenizer provides at least BOS; `n_prompt_tokens >= 1`.
- Prompt exceeds context — TransformerLens truncates; documented, no chunking.
- `layers: [0, 0, 1]` — normalized to sorted unique list.
- Activations dir from prior run with same session label — `save` overwrites; run dirs are timestamped upstream, so collisions are rare.
- Model with 0 layers (nonsense) — `layers="all"` resolves to `[]`; returns empty-dict record. No crash.

### 10.4 Threats to validity (documented in `docs/interpretability.md`)

- Random-weight tiny model: activations are noise. Smoke run proves plumbing, not results.
- Last-prompt-token snapshot only; no per-token trajectory through generation in 3a.
- Attention over prompt only; not extended through generation.
- Device-dependent float precision; seed-stable within a device, not bit-identical across devices.

---

## 11. What the implementation plan will cover

The plan (next step) decomposes this spec into TDD-style tasks:

1. `requirements.txt` update; import-skip wiring.
2. `InterpretabilityConfig` + `resolve_interpretability` + validation tests.
3. `TransformerLensBackend` + backend tests (importorskip).
4. `ActivationRecord` dataclass; `ActivationCollector` skeleton without hook logic + unit tests.
5. `ActivationCollector.collect` with hook install/teardown/first-forward gating + integration tests.
6. `save_activation_record` / `load_activation_record` + round-trip tests.
7. `ModelAgent` modification + guard (no regression in Phase 1/2 suite).
8. `Session` modification to write `.npz` sidecar and interp block on trial serialize + regression.
9. `AschExperiment.build_agents` wiring + shipped smoke YAML.
10. End-to-end integration test (`test_asch_with_interp.py`).
11. `docs/interpretability.md`.
12. Final: full-suite pass, manual smoke run.

---

## 12. Self-review

- **Placeholders:** None. Every section states concrete content or an explicit non-goal.
- **Internal consistency:** ActivationRecord fields in §4.1 match storage keys in §6.2 match JSONL pointer shape in §6.4. Config fields in §7.1 match the smoke YAML in §7.4 and validation in §7.2.
- **Scope:** One plan, one module, two small existing-file edits (`ModelAgent`, `Session`), one experiment edit (`AschExperiment`). Testable independently of Phase 1/2 (which must continue to pass).
- **Ambiguity:** The word "layer" consistently means TL block index throughout (§3.2, §4.1, §4.4, §7.1). The word "last token" consistently means last *prompt* token (§4.2, §4.5).
