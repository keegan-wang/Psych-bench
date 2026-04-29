# PsychBench Framework Architecture

PsychBench is built as a small, experiment-agnostic core under
`psychbench/framework/` plus independent experiment modules under
`psychbench/experiments/<name>/`. A new experiment does not require any
changes to framework code.

## Core types (`framework/types.py`)

| Type | Purpose |
| --- | --- |
| `Stimulus` | One trial's stimulus. Carries `correct_label`, `options`, and free-form `metadata` so experiments can attach domain-specific fields (e.g., `question`, `wrong_answer`). |
| `TrialContext` | What an agent receives on its turn: the stimulus, its own position, and any prior-agent responses gated by the `Environment`. |
| `AgentResponse` | Raw + parsed response, the exact prompt the agent saw, and free-form `metadata`. |
| `TrialResult` | One trial's output: stimulus, all agent responses in order, scoring fields set by the experiment. |
| `ResponseVisibility` | `PUBLIC`, `PRIVATE`, or `PARTIAL`; controls what prior responses each agent sees. |

## `BaseAgent` (`framework/agent.py`)

Abstract class. Every agent implements `respond(context) -> AgentResponse`
and `reset()`. Concrete types: `ScriptedAgent`, `ModelAgent`, `HybridAgent`
(under `psychbench/agents/`). Agents have an `agent_id`, a `position`
(response order), and a free-form `metadata` dict.

## `BaseExperiment` + registry (`framework/experiment.py`)

Subclass `BaseExperiment`, decorate with `@register_experiment("<name>")`.
Implement `build_stimuli()` and `build_agents()`. The CLI finds your
experiment by name via `get_experiment_class()`.

## `Environment` (`framework/environment.py`)

Sequential-response environment. For each trial, `begin_trial(stim)`, then
agents respond in position order. `visible_prior_responses(agent_id,
position)` returns the prior responses the current agent is allowed to see,
gated by visibility mode:

- `PUBLIC`: everyone sees all prior responses.
- `PRIVATE`: everyone sees nothing (control condition).
- `PARTIAL`: per-agent visibility via a `{agent_id: "public"|"private"}`
  map.

## `Session` (`framework/session.py`)

Runs all trials for one session, one stimulus at a time. Takes optional
`score_trial` and `summarize` callables so experiments attach their own
scoring without modifying framework code. Logs one JSONL record per trial
plus a session summary JSON.

## `Sweep` (`framework/sweep.py`) — new in Phase 2

Cross-product config enumerator. Call
`expand_sweep(config, sweep_fields=["a.b.c", ...])` to get a list of
`SweepCell`s, each with:

- `.config` — a deep-copied dict with sweep fields collapsed to scalars.
- `.cell_id` — a deterministic `"field1=val1__field2=val2"` identifier.
- `.sweep_values` — `{field_path: value}` dict for that cell.

Only the named fields are expanded — other list-valued config fields pass
through untouched. Phase 2's sweep runner uses this to enumerate IV cells;
Phase 1 could use it to sweep `agents.confederates.count` without code
changes.

## `ModelBackend` (`framework/backends.py`)

Protocol: `generate(prompt, stateful=False) -> str`, `reset()`.
Implementations: `OpenAIBackend`, `AnthropicBackend`, `HuggingFaceBackend`,
and `EchoBackend` (hermetic, deterministic — for tests). Select by kind
via `get_backend(kind, model)`. API keys come from env vars
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`).

## Adding a new experiment

1. Create `psychbench/experiments/<name>/__init__.py` and
   `experiment.py`.
2. Subclass `BaseExperiment`, implement `build_stimuli()` and
   `build_agents()`, decorate with `@register_experiment("<name>")`.
3. Put any experiment-specific helpers (prompts, scoring, corpus) in the
   same module.
4. Add a YAML under `config/experiments/`.
5. Optional: wire a sweep runner if your experiment has IVs.
6. Register the module in `psychbench/cli.py` so its decorator runs on
   import.
7. Run via `python -m psychbench run --config <your.yaml>`.

The Phase 1 `asch` and Phase 2 `asch_documents` modules are the reference
implementations — the former for a single-condition experiment, the
latter for an IV-sweep experiment.
