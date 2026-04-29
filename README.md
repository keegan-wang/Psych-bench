# PsychBench

Multi-agent social psychology simulation framework for LLMs.

## Phase 1: Asch (1951) Conformity

Run the base experiment:

    python -m psychbench run --config config/experiments/asch_phase1.yaml

Analyze a run:

    python -m psychbench analyze --results results/asch_experimental_<timestamp>.summary.json

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
