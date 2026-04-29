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

The shipped smoke config uses `roneneldan/TinyStories-1M` — a small
pretrained TransformerLens-supported model (~4MB cached once in
`~/.cache/huggingface/`). This runs on your laptop's CPU in seconds.

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

Because TinyStories-1M is a tiny storytelling model trained on
children's fiction, its outputs on Asch-style "which line matches"
prompts are gibberish. **3a's smoke run proves plumbing, not
results** — real findings require a real pretrained instruction model
on a GPU host.

## Real-model workflow

3a is designed to run on a GPU host; do the interp work there, not on
your laptop. Typical flow:

1. SSH to a GPU instance (AWS `g5.xlarge` — one A10G — is enough for
   Llama-3.1-8B in bf16 at roughly $1/hour).
2. `git clone` the repo + `pip install -r requirements.txt`.
3. Edit `config/experiments/asch_phase1_with_interp_smoke.yaml`:
   - `interpretability.model` → `meta-llama/Llama-3.1-8B-Instruct`
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

The TransformerLens allowlist of supported models is printed as part
of any error message when an unsupported model is requested. Common
good choices: `EleutherAI/pythia-70m` (tiny, CPU-fine),
`gpt2` (124M, still small), `meta-llama/Llama-3.1-8B-Instruct` (needs
GPU).

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
- attention pattern: one `float32` tensor of shape
  `[n_heads, prompt_seq, prompt_seq]` per layer.

Attention dominates. Concrete estimates:

| Model | n_layers | n_heads | d_model | prompt_seq ≈ 400 | Per-trial (all layers) |
|---|---|---|---|---|---|
| TinyStories-1M | 8 | 16 | 64 | 20 | < 50 KB |
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

run_dir = Path("results")
log_path = next(run_dir.glob("asch_experimental_*.jsonl"))

for line in log_path.read_text().splitlines():
    rec = json.loads(line)
    interp = rec.get("interpretability")
    if interp is None:
        continue
    record = load_activation_record(
        log_path.parent / interp["activations_path"]
    )
    # record.layer_activations[L] — [d_model] float32
    # record.attention_weights[L] — [n_heads, seq, seq] float32
```

## Known limitations

- The shipped smoke model (TinyStories-1M) is a storytelling LM, not
  an instruction-tuned assistant. The smoke run exercises plumbing,
  not conformity research.
- Only the **last prompt token** residual is captured (not the full
  per-token trajectory through generation). 3b/3c may extend this.
- Attention is captured over the **prompt only** — generation-phase
  attention is not logged.
- Device choice affects numerical outputs slightly (float32 CPU vs
  bf16/CUDA/MPS). Results are seed-stable within a device, not
  bit-identical across devices.
- `interpretability` + `agents.naive.stateful: true` is not supported
  in 3a (`TransformerLensBackend.generate(stateful=True)` raises
  `NotImplementedError`).

## Roadmap

- **3b — Persona Locator.** Linear probes to find an Assistant-persona
  direction per layer, `fit-probe` CLI, persona projection logged per
  trial.
- **3c — Conflict Detector + SAE + Attention Analyzer + Visualizer.**
  The rest of the original Phase 3 spec.
