# PsychBench

Multi-agent social psychology simulation framework for LLMs.

## Phase 1: Asch (1951) Conformity

Run the base experiment:

    python -m psychbench run --config config/experiments/asch_phase1.yaml

Analyze a run:

    python -m psychbench analyze --results results/asch_experimental_<timestamp>.summary.json

Full methodology: [docs/experiments/asch.md](docs/experiments/asch.md).

## Phase 2: Asch-Documents (Manufactured Consensus)

Phase 2 extends Phase 1's multi-agent Asch paradigm. Confederate
`ModelAgent`s are manufactured wrong by **poisoned documents in their own
prompts**; the naive agent still sees only the question and the other
agents' answers. Full methodology:
[docs/experiments/asch_documents.md](docs/experiments/asch_documents.md).

Hermetic smoke test (no API keys):

    python -m psychbench run --config config/experiments/asch_documents_smoke.yaml
    python -m psychbench analyze --run results/asch_documents_<timestamp>/

Full sweep (needs API keys; bypass cost gate with `--i-know`):

    python -m psychbench run --config config/experiments/asch_documents_phase2.yaml --i-know

## Framework

See [docs/architecture.md](docs/architecture.md) for how experiments
compose framework primitives (`BaseAgent`, `BaseExperiment`,
`Environment`, `Session`, `Sweep`, `ModelBackend`).

## Setup

    pip install -r requirements.txt
    # install whichever model backends you need:
    pip install openai anthropic transformers torch

Set API keys via env:

    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...
    export HF_TOKEN=...

## Repo layout

See `psychbench/` for framework and experiments, `config/experiments/` for
YAML configs, `results/` for logs, `tests/` for tests.

## Adding a new experiment

1. Create `psychbench/experiments/<name>/experiment.py` subclassing `BaseExperiment`.
2. Decorate it with `@register_experiment("<name>")`.
3. Add a YAML under `config/experiments/`.
