# PsychBench Phase 1 (Asch 1951) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an extensible Python framework (`psychbench`) for multi-agent social-psych simulations with LLMs, and implement the Asch (1951) line-length conformity experiment as the first module, runnable end-to-end via CLI with YAML configs.

**Architecture:** A small, experiment-agnostic `framework/` core provides base classes (`BaseAgent`, `BaseExperiment`), a sequential `Environment` that controls information flow, and `Trial`/`Session` runners that log full context windows per agent per trial. Experiments live under `psychbench/experiments/<name>/` and register themselves via a lightweight registry. Three agent types — `ScriptedAgent`, `ModelAgent`, `HybridAgent` — share a single `respond(context) -> AgentResponse` interface. Model backends (OpenAI, Anthropic, HuggingFace transformers) are abstracted behind a `ModelBackend` protocol so agents don't depend on any specific SDK. Config drives everything; no experiment logic is baked into the CLI.

**Tech Stack:** Python 3.10+, stdlib-heavy. Dependencies: `PyYAML` (configs), `pytest` (tests). Optional extras loaded lazily on demand: `openai`, `anthropic`, `transformers`+`torch`. All API keys via env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`). JSONL logs via stdlib `json`.

---

## File Structure

**Core framework (`psychbench/framework/`):**
- `types.py` — `Stimulus`, `TrialContext`, `AgentResponse`, `TrialResult` dataclasses + `ResponseVisibility` enum
- `agent.py` — `BaseAgent` abstract class
- `experiment.py` — `BaseExperiment` abstract class + `EXPERIMENT_REGISTRY` dict and `register_experiment` decorator
- `environment.py` — `Environment` class managing per-agent visibility & sequential response order
- `trial.py` — `run_trial()` function
- `session.py` — `Session` class orchestrating all trials + logging
- `backends.py` — `ModelBackend` protocol + `OpenAIBackend`, `AnthropicBackend`, `HuggingFaceBackend` with lazy imports + `get_backend(name, model)` factory
- `logging_utils.py` — JSONL writer + summary JSON writer

**Agents (`psychbench/agents/`):**
- `scripted_agent.py`
- `model_agent.py`
- `hybrid_agent.py`

**Asch experiment (`psychbench/experiments/asch/`):**
- `stimuli.py` — stimulus generation with critical/non-critical trial labels
- `scoring.py` — answer parsing + conformity scoring
- `experiment.py` — `AschExperiment` subclass; builds agents/environment/stimuli from config; runs both experimental and optional control condition
- `prompts.py` — naive-agent prompt template

**Analysis (`psychbench/analysis/`):**
- `conformity.py` — load JSONL, compute per-session & comparative stats, pretty-print

**Top-level package:**
- `psychbench/__init__.py`
- `psychbench/__main__.py` — `python -m psychbench` dispatcher
- `psychbench/cli.py` — argparse CLI (`run`, `analyze` subcommands)
- `psychbench/config.py` — YAML loader + schema validation

**Configs / tests / docs:**
- `config/experiments/asch_phase1.yaml`
- `tests/test_asch.py`
- `tests/test_framework.py`
- `tests/test_agents.py`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `results/.gitkeep`

---

## Task 1: Scaffold repo, requirements, README, gitignore

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`
- Create: `results/.gitkeep`
- Create: `psychbench/__init__.py`, `psychbench/framework/__init__.py`, `psychbench/agents/__init__.py`, `psychbench/experiments/__init__.py`, `psychbench/experiments/asch/__init__.py`, `psychbench/analysis/__init__.py`, `tests/__init__.py`, `config/experiments/.gitkeep`

- [ ] **Step 1: Write `requirements.txt`**

```
PyYAML>=6.0
pytest>=7.0
```

Optional backends installed separately by the user: `openai`, `anthropic`, `transformers`, `torch`.

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.env
results/*.jsonl
results/*.json
!results/.gitkeep
.pytest_cache/
.DS_Store
*.egg-info/
```

- [ ] **Step 3: Write `README.md`**

```markdown
# PsychBench

Multi-agent social psychology simulation framework for LLMs.

## Phase 1: Asch (1951) Conformity

Run the base experiment:

    python -m psychbench run --config config/experiments/asch_phase1.yaml

Analyze a run:

    python -m psychbench analyze --results results/asch_phase1_<timestamp>.jsonl

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
```

- [ ] **Step 4: Create empty package init files**

Create each `__init__.py` as empty files: `psychbench/__init__.py`, `psychbench/framework/__init__.py`, `psychbench/agents/__init__.py`, `psychbench/experiments/__init__.py`, `psychbench/experiments/asch/__init__.py`, `psychbench/analysis/__init__.py`, `tests/__init__.py`. Also create `results/.gitkeep` and `config/experiments/.gitkeep` as empty files.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore README.md results/.gitkeep config/experiments/.gitkeep psychbench tests
git commit -m "scaffold psychbench repo structure"
```

---

## Task 2: Framework core types

**Files:**
- Create: `psychbench/framework/types.py`
- Test: `tests/test_framework.py` (created here, extended later)

- [ ] **Step 1: Write `tests/test_framework.py` (types portion)**

```python
from psychbench.framework.types import (
    Stimulus, TrialContext, AgentResponse, TrialResult, ResponseVisibility,
)


def test_stimulus_has_correct_answer_and_options():
    s = Stimulus(trial_index=0, is_critical=True, reference_cm=10.0,
                 options={"A": 10.0, "B": 12.0, "C": 7.0}, correct_label="A",
                 metadata={})
    assert s.correct_label == "A"
    assert s.options["A"] == 10.0
    assert s.is_critical is True


def test_trial_context_carries_prior_responses():
    s = Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    ctx = TrialContext(stimulus=s, agent_position=5, agent_id="naive",
                       prior_responses=[("confed_1", "A"), ("confed_2", "A")],
                       extra={})
    assert len(ctx.prior_responses) == 2
    assert ctx.agent_position == 5


def test_agent_response_fields():
    r = AgentResponse(agent_id="naive", raw_text="I think A", parsed_answer="A",
                      prompt="...", metadata={"backend": "openai"})
    assert r.parsed_answer == "A"


def test_response_visibility_enum():
    assert ResponseVisibility("public") == ResponseVisibility.PUBLIC
    assert ResponseVisibility("private") == ResponseVisibility.PRIVATE
    assert ResponseVisibility("partial") == ResponseVisibility.PARTIAL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_framework.py -v`
Expected: FAIL (ImportError / ModuleNotFoundError for `psychbench.framework.types`)

- [ ] **Step 3: Implement `psychbench/framework/types.py`**

```python
"""Core dataclasses and enums shared across the framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResponseVisibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    PARTIAL = "partial"


@dataclass
class Stimulus:
    trial_index: int
    is_critical: bool
    reference_cm: float
    options: dict[str, float]
    correct_label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialContext:
    stimulus: Stimulus
    agent_position: int
    agent_id: str
    prior_responses: list[tuple[str, str]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    agent_id: str
    raw_text: str
    parsed_answer: str | None
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    trial_index: int
    is_critical: bool
    stimulus: Stimulus
    responses: list[AgentResponse]
    conformed: bool | None
    naive_answer: str | None
    confederate_answer: str | None
    correct_answer: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_framework.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add psychbench/framework/types.py tests/test_framework.py
git commit -m "add framework core dataclasses and visibility enum"
```

---

## Task 3: BaseAgent abstract class

**Files:**
- Create: `psychbench/framework/agent.py`
- Modify: `tests/test_framework.py` (append)

- [ ] **Step 1: Append tests to `tests/test_framework.py`**

```python
import pytest
from psychbench.framework.agent import BaseAgent


def test_base_agent_is_abstract():
    with pytest.raises(TypeError):
        BaseAgent(agent_id="x", position=0)


def test_base_agent_subclass_implements_respond():
    class DummyAgent(BaseAgent):
        def respond(self, context):
            return AgentResponse(agent_id=self.agent_id, raw_text="A",
                                 parsed_answer="A", prompt="", metadata={})

    a = DummyAgent(agent_id="d", position=0)
    s = Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    ctx = TrialContext(stimulus=s, agent_position=0, agent_id="d")
    r = a.respond(ctx)
    assert r.parsed_answer == "A"
    a.reset()  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_framework.py -v`
Expected: FAIL (ImportError for `psychbench.framework.agent`)

- [ ] **Step 3: Implement `psychbench/framework/agent.py`**

```python
"""Abstract base class for all agent types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import AgentResponse, TrialContext


class BaseAgent(ABC):
    """Shared interface for scripted, model-backed, and hybrid agents."""

    def __init__(self, agent_id: str, position: int,
                 metadata: dict[str, Any] | None = None) -> None:
        self.agent_id = agent_id
        self.position = position
        self.metadata = metadata or {}

    @abstractmethod
    def respond(self, context: TrialContext) -> AgentResponse:
        """Produce a response for the given trial context."""

    def reset(self) -> None:
        """Clear any per-session state. Default no-op."""
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_framework.py -v`
Expected: PASS (6 tests total)

- [ ] **Step 5: Commit**

```bash
git add psychbench/framework/agent.py tests/test_framework.py
git commit -m "add BaseAgent abstract class"
```

---

## Task 4: BaseExperiment + experiment registry

**Files:**
- Create: `psychbench/framework/experiment.py`
- Modify: `tests/test_framework.py` (append)

- [ ] **Step 1: Append tests**

```python
from psychbench.framework.experiment import (
    BaseExperiment, register_experiment, EXPERIMENT_REGISTRY, get_experiment_class,
)


def test_register_and_retrieve_experiment():
    @register_experiment("dummy_exp")
    class DummyExp(BaseExperiment):
        def build_stimuli(self):
            return []
        def build_agents(self):
            return []

    assert "dummy_exp" in EXPERIMENT_REGISTRY
    assert get_experiment_class("dummy_exp") is DummyExp


def test_get_experiment_class_raises_on_unknown():
    with pytest.raises(KeyError):
        get_experiment_class("definitely_not_registered")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_framework.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/framework/experiment.py`**

```python
"""Abstract base for experiments + lightweight string-keyed registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from .agent import BaseAgent
from .types import Stimulus


EXPERIMENT_REGISTRY: dict[str, type["BaseExperiment"]] = {}


def register_experiment(name: str) -> Callable[[type["BaseExperiment"]],
                                                type["BaseExperiment"]]:
    def decorator(cls: type["BaseExperiment"]) -> type["BaseExperiment"]:
        EXPERIMENT_REGISTRY[name] = cls
        cls.experiment_name = name
        return cls
    return decorator


def get_experiment_class(name: str) -> type["BaseExperiment"]:
    if name not in EXPERIMENT_REGISTRY:
        raise KeyError(
            f"Unknown experiment '{name}'. Registered: {list(EXPERIMENT_REGISTRY)}"
        )
    return EXPERIMENT_REGISTRY[name]


class BaseExperiment(ABC):
    """Subclasses define how to build stimuli, agents, and score trials."""

    experiment_name: str = "base"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def build_stimuli(self) -> list[Stimulus]:
        """Return the full ordered list of stimuli for the session."""

    @abstractmethod
    def build_agents(self) -> list[BaseAgent]:
        """Return the ordered list of agents (response order = list order)."""

    def score_trial(self, stimulus: Stimulus,
                     responses: list["AgentResponse"]) -> dict[str, Any]:  # noqa: F821
        """Override to add experiment-specific scoring. Default: empty dict."""
        return {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_framework.py -v`
Expected: PASS (8 tests total)

- [ ] **Step 5: Commit**

```bash
git add psychbench/framework/experiment.py tests/test_framework.py
git commit -m "add BaseExperiment and experiment registry"
```

---

## Task 5: Environment with per-agent visibility

**Files:**
- Create: `psychbench/framework/environment.py`
- Modify: `tests/test_framework.py` (append)

- [ ] **Step 1: Append tests**

```python
from psychbench.framework.environment import Environment


def test_environment_public_visibility_gives_all_prior_responses():
    env = Environment(visibility=ResponseVisibility.PUBLIC)
    stim = Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    env.begin_trial(stim)
    env.record_response("agent_1", 0, "A")
    env.record_response("agent_2", 1, "A")
    visible = env.visible_prior_responses(agent_id="agent_3", position=2)
    assert visible == [("agent_1", "A"), ("agent_2", "A")]


def test_environment_private_visibility_hides_responses():
    env = Environment(visibility=ResponseVisibility.PRIVATE)
    stim = Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    env.begin_trial(stim)
    env.record_response("agent_1", 0, "A")
    visible = env.visible_prior_responses(agent_id="agent_2", position=1)
    assert visible == []


def test_environment_partial_visibility_uses_per_agent_map():
    env = Environment(
        visibility=ResponseVisibility.PARTIAL,
        per_agent_visibility={"naive": "public", "confed_1": "private"},
    )
    stim = Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    env.begin_trial(stim)
    env.record_response("confed_1", 0, "A")
    env.record_response("confed_2", 1, "A")
    assert env.visible_prior_responses("naive", 2) == [
        ("confed_1", "A"), ("confed_2", "A"),
    ]
    assert env.visible_prior_responses("confed_1", 0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_framework.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/framework/environment.py`**

```python
"""Controls what each agent sees of other agents' responses each trial."""
from __future__ import annotations

from typing import Any

from .types import ResponseVisibility, Stimulus


class Environment:
    """Sequential-response environment with configurable visibility.

    Each trial, agents respond in position order. `visible_prior_responses`
    tells each agent what it is allowed to see when its turn comes up.
    """

    def __init__(
        self,
        visibility: ResponseVisibility = ResponseVisibility.PUBLIC,
        per_agent_visibility: dict[str, str] | None = None,
    ) -> None:
        self.visibility = visibility
        self.per_agent_visibility = per_agent_visibility or {}
        self._current_stimulus: Stimulus | None = None
        self._trial_responses: list[tuple[str, int, str]] = []

    def begin_trial(self, stimulus: Stimulus) -> None:
        self._current_stimulus = stimulus
        self._trial_responses = []

    def record_response(self, agent_id: str, position: int, answer: str) -> None:
        self._trial_responses.append((agent_id, position, answer))

    def visible_prior_responses(
        self, agent_id: str, position: int
    ) -> list[tuple[str, str]]:
        mode = self._resolve_mode(agent_id)
        if mode == ResponseVisibility.PRIVATE:
            return []
        return [
            (aid, ans)
            for aid, pos, ans in self._trial_responses
            if pos < position
        ]

    def _resolve_mode(self, agent_id: str) -> ResponseVisibility:
        if self.visibility != ResponseVisibility.PARTIAL:
            return self.visibility
        per = self.per_agent_visibility.get(agent_id, "private")
        return ResponseVisibility(per)

    def snapshot(self) -> dict[str, Any]:
        return {
            "visibility": self.visibility.value,
            "per_agent_visibility": dict(self.per_agent_visibility),
            "current_trial_responses": list(self._trial_responses),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_framework.py -v`
Expected: PASS (11 tests total)

- [ ] **Step 5: Commit**

```bash
git add psychbench/framework/environment.py tests/test_framework.py
git commit -m "add Environment with public/private/partial visibility"
```

---

## Task 6: Trial runner

**Files:**
- Create: `psychbench/framework/trial.py`
- Modify: `tests/test_framework.py` (append)

- [ ] **Step 1: Append tests**

```python
from psychbench.framework.trial import run_trial
from psychbench.framework.agent import BaseAgent


class _FixedAgent(BaseAgent):
    def __init__(self, agent_id, position, answer):
        super().__init__(agent_id, position)
        self._answer = answer
        self.seen = None

    def respond(self, context):
        self.seen = list(context.prior_responses)
        return AgentResponse(agent_id=self.agent_id, raw_text=self._answer,
                             parsed_answer=self._answer,
                             prompt=f"pos={self.position}", metadata={})


def test_run_trial_sequential_visibility_public():
    env = Environment(visibility=ResponseVisibility.PUBLIC)
    stim = Stimulus(0, True, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    a1 = _FixedAgent("a1", 0, "B")
    a2 = _FixedAgent("a2", 1, "B")
    a3 = _FixedAgent("a3", 2, "A")
    result = run_trial(stim, [a1, a2, a3], env)
    assert [r.parsed_answer for r in result.responses] == ["B", "B", "A"]
    assert a3.seen == [("a1", "B"), ("a2", "B")]
    assert a1.seen == []


def test_run_trial_private_visibility_hides_all():
    env = Environment(visibility=ResponseVisibility.PRIVATE)
    stim = Stimulus(0, True, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    a1 = _FixedAgent("a1", 0, "B")
    a2 = _FixedAgent("a2", 1, "A")
    run_trial(stim, [a1, a2], env)
    assert a2.seen == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_framework.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/framework/trial.py`**

```python
"""Run a single trial: agents respond in order, environment gates visibility."""
from __future__ import annotations

from .agent import BaseAgent
from .environment import Environment
from .types import AgentResponse, Stimulus, TrialContext, TrialResult


def run_trial(
    stimulus: Stimulus,
    agents: list[BaseAgent],
    environment: Environment,
) -> TrialResult:
    environment.begin_trial(stimulus)
    ordered = sorted(agents, key=lambda a: a.position)
    responses: list[AgentResponse] = []
    for agent in ordered:
        prior = environment.visible_prior_responses(agent.agent_id, agent.position)
        ctx = TrialContext(
            stimulus=stimulus,
            agent_position=agent.position,
            agent_id=agent.agent_id,
            prior_responses=prior,
        )
        resp = agent.respond(ctx)
        responses.append(resp)
        if resp.parsed_answer is not None:
            environment.record_response(
                agent.agent_id, agent.position, resp.parsed_answer,
            )
    return TrialResult(
        trial_index=stimulus.trial_index,
        is_critical=stimulus.is_critical,
        stimulus=stimulus,
        responses=responses,
        conformed=None,
        naive_answer=None,
        confederate_answer=None,
        correct_answer=stimulus.correct_label,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_framework.py -v`
Expected: PASS (13 tests total)

- [ ] **Step 5: Commit**

```bash
git add psychbench/framework/trial.py tests/test_framework.py
git commit -m "add single-trial runner with sequential response order"
```

---

## Task 7: Logging utilities (JSONL + summary JSON)

**Files:**
- Create: `psychbench/framework/logging_utils.py`
- Modify: `tests/test_framework.py` (append)

- [ ] **Step 1: Append tests**

```python
import json as _json
from pathlib import Path
from psychbench.framework.logging_utils import JsonlLogger, write_summary


def test_jsonl_logger_writes_one_record_per_line(tmp_path):
    p = tmp_path / "out.jsonl"
    with JsonlLogger(p) as log:
        log.write({"a": 1})
        log.write({"b": 2})
    lines = p.read_text().strip().splitlines()
    assert [_json.loads(l) for l in lines] == [{"a": 1}, {"b": 2}]


def test_write_summary_produces_valid_json(tmp_path):
    p = tmp_path / "summary.json"
    write_summary(p, {"conformity_rate": 0.33, "n_critical": 12})
    loaded = _json.loads(p.read_text())
    assert loaded["conformity_rate"] == 0.33
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_framework.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/framework/logging_utils.py`**

```python
"""JSONL logger (one record per line) and summary JSON writer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def __enter__(self) -> "JsonlLogger":
        self._fh = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("JsonlLogger used outside of context manager")
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        self._fh.flush()


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, default=_json_default))


def _json_default(o: Any) -> Any:
    # Support dataclasses and enums transparently.
    from dataclasses import asdict, is_dataclass
    from enum import Enum

    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Enum):
        return o.value
    raise TypeError(f"Not JSON serializable: {type(o).__name__}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_framework.py -v`
Expected: PASS (15 tests total)

- [ ] **Step 5: Commit**

```bash
git add psychbench/framework/logging_utils.py tests/test_framework.py
git commit -m "add JSONL and summary JSON logging utilities"
```

---

## Task 8: Session runner

**Files:**
- Create: `psychbench/framework/session.py`
- Modify: `tests/test_framework.py` (append)

- [ ] **Step 1: Append tests**

```python
from psychbench.framework.session import Session


class _ScriptedForSession(BaseAgent):
    def __init__(self, agent_id, position, critical_answer, normal_answer="A"):
        super().__init__(agent_id, position)
        self._crit = critical_answer
        self._norm = normal_answer

    def respond(self, context):
        ans = self._crit if context.stimulus.is_critical else self._norm
        return AgentResponse(agent_id=self.agent_id, raw_text=ans,
                             parsed_answer=ans,
                             prompt=f"pos={self.position}", metadata={})


def test_session_runs_all_trials_and_logs_jsonl(tmp_path):
    stims = [
        Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {}),
        Stimulus(1, True, 10.0, {"A": 10.0, "B": 13.0, "C": 7.0}, "A", {}),
    ]
    agents = [
        _ScriptedForSession("c1", 0, "B"),
        _ScriptedForSession("naive", 1, "A"),
    ]
    env = Environment(visibility=ResponseVisibility.PUBLIC)
    log_path = tmp_path / "run.jsonl"
    summary_path = tmp_path / "summary.json"
    session = Session(
        stimuli=stims, agents=agents, environment=env,
        log_path=log_path, summary_path=summary_path,
        config_snapshot={"test": True},
    )
    results = session.run()
    assert len(results) == 2
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = _json.loads(lines[1])
    assert rec["trial_index"] == 1
    assert rec["is_critical"] is True
    assert any(r["agent_id"] == "naive" for r in rec["responses"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_framework.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/framework/session.py`**

```python
"""Orchestrate a full session: run all trials, log contexts, write summary."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .agent import BaseAgent
from .environment import Environment
from .logging_utils import JsonlLogger, write_summary
from .trial import run_trial
from .types import Stimulus, TrialResult


class Session:
    def __init__(
        self,
        stimuli: list[Stimulus],
        agents: list[BaseAgent],
        environment: Environment,
        log_path: str | Path,
        summary_path: str | Path,
        config_snapshot: dict[str, Any],
        score_trial: Callable[[TrialResult], dict[str, Any]] | None = None,
        summarize: Callable[[list[TrialResult]], dict[str, Any]] | None = None,
        session_label: str = "session",
    ) -> None:
        self.stimuli = stimuli
        self.agents = agents
        self.environment = environment
        self.log_path = Path(log_path)
        self.summary_path = Path(summary_path)
        self.config_snapshot = config_snapshot
        self.score_trial = score_trial
        self.summarize = summarize
        self.session_label = session_label

    def run(self) -> list[TrialResult]:
        for agent in self.agents:
            agent.reset()
        results: list[TrialResult] = []
        with JsonlLogger(self.log_path) as log:
            for stim in self.stimuli:
                result = run_trial(stim, self.agents, self.environment)
                if self.score_trial is not None:
                    scores = self.score_trial(result)
                    for k, v in scores.items():
                        setattr(result, k, v) if hasattr(result, k) else None
                    result_extra = scores
                else:
                    result_extra = {}
                log.write(self._serialize_trial(result, result_extra))
                results.append(result)
        summary = {
            "session_label": self.session_label,
            "timestamp": time.time(),
            "n_trials": len(results),
            "config": self.config_snapshot,
        }
        if self.summarize is not None:
            summary.update(self.summarize(results))
        write_summary(self.summary_path, summary)
        return results

    def _serialize_trial(
        self, result: TrialResult, extra: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "trial_index": result.trial_index,
            "is_critical": result.is_critical,
            "stimulus": asdict(result.stimulus),
            "correct_answer": result.correct_answer,
            "responses": [asdict(r) for r in result.responses],
            "scoring": extra,
            "environment": self.environment.snapshot(),
            "session_label": self.session_label,
            "timestamp": time.time(),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_framework.py -v`
Expected: PASS (16 tests total)

- [ ] **Step 5: Commit**

```bash
git add psychbench/framework/session.py tests/test_framework.py
git commit -m "add Session runner with per-trial JSONL logging and summary"
```

---

## Task 9: Model backends (OpenAI, Anthropic, HuggingFace)

**Files:**
- Create: `psychbench/framework/backends.py`
- Create: `tests/test_backends.py`

- [ ] **Step 1: Write `tests/test_backends.py`**

```python
import pytest
from psychbench.framework.backends import (
    ModelBackend, get_backend, EchoBackend,
)


def test_echo_backend_returns_prompt_suffix():
    b = EchoBackend(model="echo-test")
    out = b.generate("What is the answer? Say A.", stateful=False)
    assert isinstance(out, str)
    assert out  # non-empty


def test_get_backend_echo_factory():
    b = get_backend("echo", "echo-test")
    assert isinstance(b, EchoBackend)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("not_a_backend", "x")


def test_stateful_echo_backend_tracks_history():
    b = EchoBackend(model="echo-test")
    b.generate("first", stateful=True)
    b.generate("second", stateful=True)
    assert len(b.history) == 2
    b.reset()
    assert b.history == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backends.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/framework/backends.py`**

```python
"""Model backend protocol + concrete implementations (lazy-imported)."""
from __future__ import annotations

import os
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelBackend(Protocol):
    model: str

    def generate(self, prompt: str, stateful: bool = False) -> str: ...
    def reset(self) -> None: ...


class EchoBackend:
    """Deterministic offline backend for tests. Extracts last A/B/C in prompt."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.history: list[tuple[str, str]] = []

    def generate(self, prompt: str, stateful: bool = False) -> str:
        m = re.findall(r"\b([ABC])\b", prompt)
        answer = m[-1] if m else "A"
        if stateful:
            self.history.append((prompt, answer))
        return answer

    def reset(self) -> None:
        self.history = []


class OpenAIBackend:
    def __init__(self, model: str) -> None:
        self.model = model
        self._conversation: list[dict] = []
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OpenAI backend requires `pip install openai`"
            ) from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY env var not set")
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, stateful: bool = False) -> str:
        if stateful:
            self._conversation.append({"role": "user", "content": prompt})
            messages = list(self._conversation)
        else:
            messages = [{"role": "user", "content": prompt}]
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        if stateful:
            self._conversation.append({"role": "assistant", "content": text})
        return text

    def reset(self) -> None:
        self._conversation = []


class AnthropicBackend:
    def __init__(self, model: str) -> None:
        self.model = model
        self._conversation: list[dict] = []
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Anthropic backend requires `pip install anthropic`"
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY env var not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, stateful: bool = False) -> str:
        if stateful:
            self._conversation.append({"role": "user", "content": prompt})
            messages = list(self._conversation)
        else:
            messages = [{"role": "user", "content": prompt}]
        resp = self._client.messages.create(
            model=self.model, max_tokens=256, messages=messages, temperature=0.0,
        )
        text = "".join(block.text for block in resp.content
                       if getattr(block, "type", "") == "text")
        if stateful:
            self._conversation.append({"role": "assistant", "content": text})
        return text

    def reset(self) -> None:
        self._conversation = []


class HuggingFaceBackend:
    def __init__(self, model: str) -> None:
        self.model = model
        self._history: list[dict] = []
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            import torch  # type: ignore
        except ImportError as e:
            raise ImportError(
                "HuggingFace backend requires `pip install transformers torch`"
            ) from e
        token = os.environ.get("HF_TOKEN")
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model, token=token)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, token=token, torch_dtype="auto", device_map="auto",
        )

    def generate(self, prompt: str, stateful: bool = False) -> str:
        torch = self._torch
        if stateful:
            self._history.append({"role": "user", "content": prompt})
            messages = list(self._history)
        else:
            messages = [{"role": "user", "content": prompt}]
        try:
            text_in = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text_in = prompt
        inputs = self._tokenizer(text_in, return_tensors="pt").to(
            self._model.device
        )
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(gen, skip_special_tokens=True).strip()
        if stateful:
            self._history.append({"role": "assistant", "content": text})
        return text

    def reset(self) -> None:
        self._history = []


_BACKENDS = {
    "echo": EchoBackend,
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "huggingface": HuggingFaceBackend,
}


def get_backend(kind: str, model: str) -> ModelBackend:
    if kind not in _BACKENDS:
        raise ValueError(
            f"Unknown backend '{kind}'. Known: {list(_BACKENDS)}"
        )
    return _BACKENDS[kind](model)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backends.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add psychbench/framework/backends.py tests/test_backends.py
git commit -m "add model backends (openai/anthropic/hf) with offline echo backend"
```

---

## Task 10: ScriptedAgent

**Files:**
- Create: `psychbench/agents/scripted_agent.py`
- Create: `tests/test_agents.py`

- [ ] **Step 1: Write `tests/test_agents.py`**

```python
from psychbench.agents.scripted_agent import ScriptedAgent
from psychbench.framework.types import Stimulus, TrialContext


def _ctx(is_critical: bool, correct="A"):
    stim = Stimulus(0, is_critical, 5.0,
                    {"A": 5.0, "B": 8.0, "C": 2.0}, correct, {})
    return TrialContext(stimulus=stim, agent_position=0, agent_id="s")


def test_scripted_always_correct():
    a = ScriptedAgent(agent_id="s", position=0, behavior="always_correct")
    r = a.respond(_ctx(True, correct="A"))
    assert r.parsed_answer == "A"


def test_scripted_always_wrong_on_critical_picks_consistent_wrong():
    a = ScriptedAgent(agent_id="s", position=0,
                       behavior="always_wrong_on_critical")
    c_ctx = _ctx(True, correct="A")
    n_ctx = _ctx(False, correct="A")
    assert a.respond(c_ctx).parsed_answer in {"B", "C"}
    assert a.respond(n_ctx).parsed_answer == "A"


def test_scripted_all_confederates_unanimous_on_critical():
    # Two confederates created with same seed pick the same wrong answer.
    a1 = ScriptedAgent(agent_id="c1", position=0,
                        behavior="always_wrong_on_critical", wrong_answer="B")
    a2 = ScriptedAgent(agent_id="c2", position=1,
                        behavior="always_wrong_on_critical", wrong_answer="B")
    ctx = _ctx(True, correct="A")
    assert a1.respond(ctx).parsed_answer == "B"
    assert a2.respond(ctx).parsed_answer == "B"


def test_scripted_custom_fn():
    def pick(ctx):
        return "C"
    a = ScriptedAgent(agent_id="s", position=0, behavior="custom_fn",
                       custom_fn=pick)
    assert a.respond(_ctx(False, correct="A")).parsed_answer == "C"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/agents/scripted_agent.py`**

```python
"""Non-LLM rule-based agent used for confederates."""
from __future__ import annotations

from typing import Callable

from psychbench.framework.agent import BaseAgent
from psychbench.framework.types import AgentResponse, TrialContext


Behavior = str  # "always_correct" | "always_wrong_on_critical" | "custom_fn"


class ScriptedAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        position: int,
        behavior: Behavior,
        wrong_answer: str | None = None,
        custom_fn: Callable[[TrialContext], str] | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, position=position,
                         metadata={"type": "scripted", "behavior": behavior})
        self.behavior = behavior
        self.wrong_answer = wrong_answer
        self.custom_fn = custom_fn
        if behavior == "custom_fn" and custom_fn is None:
            raise ValueError("behavior='custom_fn' requires custom_fn callable")

    def respond(self, context: TrialContext) -> AgentResponse:
        answer = self._pick_answer(context)
        return AgentResponse(
            agent_id=self.agent_id,
            raw_text=answer,
            parsed_answer=answer,
            prompt="",
            metadata={"scripted": True, "behavior": self.behavior},
        )

    def _pick_answer(self, context: TrialContext) -> str:
        if self.behavior == "always_correct":
            return context.stimulus.correct_label
        if self.behavior == "always_wrong_on_critical":
            if not context.stimulus.is_critical:
                return context.stimulus.correct_label
            if self.wrong_answer is not None:
                return self.wrong_answer
            # Pick the first non-correct label deterministically.
            for label in sorted(context.stimulus.options):
                if label != context.stimulus.correct_label:
                    return label
            raise RuntimeError("No wrong options available")
        if self.behavior == "custom_fn":
            assert self.custom_fn is not None
            return self.custom_fn(context)
        raise ValueError(f"Unknown behavior: {self.behavior}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add psychbench/agents/scripted_agent.py tests/test_agents.py
git commit -m "add ScriptedAgent with always_correct/always_wrong_on_critical/custom_fn"
```

---

## Task 11: ModelAgent

**Files:**
- Create: `psychbench/agents/model_agent.py`
- Modify: `tests/test_agents.py` (append)

- [ ] **Step 1: Append tests**

```python
from psychbench.agents.model_agent import ModelAgent
from psychbench.framework.backends import EchoBackend


def test_model_agent_uses_backend_and_parses_letter():
    backend = EchoBackend(model="echo-test")
    agent = ModelAgent(
        agent_id="naive", position=5, backend=backend, stateful=False,
        prompt_builder=lambda ctx: (
            "Reference 5cm. Options A, B, C. Prior: none. Answer A, B, or C."
        ),
    )
    stim = Stimulus(0, True, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    ctx = TrialContext(stimulus=stim, agent_position=5, agent_id="naive")
    r = agent.respond(ctx)
    assert r.parsed_answer in {"A", "B", "C"}
    assert r.prompt.startswith("Reference 5cm")


def test_model_agent_stateful_retains_history():
    backend = EchoBackend(model="echo-test")
    agent = ModelAgent(
        agent_id="naive", position=5, backend=backend, stateful=True,
        prompt_builder=lambda ctx: f"trial {ctx.stimulus.trial_index} option A",
    )
    stim1 = Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    stim2 = Stimulus(1, True, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    agent.respond(TrialContext(stim1, 5, "naive"))
    agent.respond(TrialContext(stim2, 5, "naive"))
    assert len(backend.history) == 2
    agent.reset()
    assert backend.history == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/agents/model_agent.py`**

```python
"""LLM-backed agent with optional cross-trial stateful history."""
from __future__ import annotations

import re
from typing import Callable

from psychbench.framework.agent import BaseAgent
from psychbench.framework.backends import ModelBackend
from psychbench.framework.types import AgentResponse, TrialContext


PromptBuilder = Callable[[TrialContext], str]


def parse_letter_answer(text: str) -> str | None:
    """Extract A/B/C from model output.

    Strategy: look for a final standalone letter first (most reliable when
    the model gives a verbose response followed by the letter), then fall
    back to the first standalone letter anywhere.
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
    ) -> None:
        super().__init__(agent_id=agent_id, position=position,
                         metadata={"type": "model", "model": backend.model,
                                    "stateful": stateful})
        self.backend = backend
        self.stateful = stateful
        self.prompt_builder = prompt_builder

    def respond(self, context: TrialContext) -> AgentResponse:
        prompt = self.prompt_builder(context)
        raw = self.backend.generate(prompt, stateful=self.stateful)
        parsed = parse_letter_answer(raw)
        return AgentResponse(
            agent_id=self.agent_id,
            raw_text=raw,
            parsed_answer=parsed,
            prompt=prompt,
            metadata={"model": self.backend.model, "stateful": self.stateful},
        )

    def reset(self) -> None:
        self.backend.reset()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents.py -v`
Expected: PASS (6 tests total)

- [ ] **Step 5: Commit**

```bash
git add psychbench/agents/model_agent.py tests/test_agents.py
git commit -m "add ModelAgent with letter-answer parsing and stateful mode"
```

---

## Task 12: HybridAgent

**Files:**
- Create: `psychbench/agents/hybrid_agent.py`
- Modify: `tests/test_agents.py` (append)

- [ ] **Step 1: Append tests**

```python
from psychbench.agents.hybrid_agent import HybridAgent


def test_hybrid_agent_scripted_answer_model_surface():
    backend = EchoBackend(model="echo-test")
    agent = HybridAgent(
        agent_id="confed_natural", position=0,
        answer_fn=lambda ctx: "B",
        backend=backend,
        surface_prompt_builder=lambda ctx, ans: (
            f"Say in natural words that your answer is {ans}."
        ),
    )
    stim = Stimulus(0, True, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    ctx = TrialContext(stimulus=stim, agent_position=0, agent_id="confed_natural")
    r = agent.respond(ctx)
    assert r.parsed_answer == "B"  # scripted, not parsed from surface text
    assert "B" in r.raw_text  # echo backend picks the letter in the prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/agents/hybrid_agent.py`**

```python
"""Scripted answer, model-generated surface text (natural-sounding confederate)."""
from __future__ import annotations

from typing import Callable

from psychbench.framework.agent import BaseAgent
from psychbench.framework.backends import ModelBackend
from psychbench.framework.types import AgentResponse, TrialContext


AnswerFn = Callable[[TrialContext], str]
SurfacePromptBuilder = Callable[[TrialContext, str], str]


class HybridAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        position: int,
        answer_fn: AnswerFn,
        backend: ModelBackend,
        surface_prompt_builder: SurfacePromptBuilder,
    ) -> None:
        super().__init__(agent_id=agent_id, position=position,
                         metadata={"type": "hybrid", "model": backend.model})
        self.answer_fn = answer_fn
        self.backend = backend
        self.surface_prompt_builder = surface_prompt_builder

    def respond(self, context: TrialContext) -> AgentResponse:
        answer = self.answer_fn(context)
        prompt = self.surface_prompt_builder(context, answer)
        raw = self.backend.generate(prompt, stateful=False)
        return AgentResponse(
            agent_id=self.agent_id,
            raw_text=raw,
            parsed_answer=answer,
            prompt=prompt,
            metadata={"model": self.backend.model, "hybrid": True},
        )

    def reset(self) -> None:
        self.backend.reset()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents.py -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```bash
git add psychbench/agents/hybrid_agent.py tests/test_agents.py
git commit -m "add HybridAgent: scripted logic with model-generated surface text"
```

---

## Task 13: Asch stimuli generator

**Files:**
- Create: `psychbench/experiments/asch/stimuli.py`
- Create: `tests/test_asch.py`

- [ ] **Step 1: Write `tests/test_asch.py` (stimuli portion)**

```python
from psychbench.experiments.asch.stimuli import generate_asch_stimuli


def test_generates_correct_count_and_critical_indices():
    critical_idx = [2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17]
    stims = generate_asch_stimuli(n_trials=18, critical_indices=critical_idx,
                                   seed=42)
    assert len(stims) == 18
    assert sum(1 for s in stims if s.is_critical) == 12
    for i, s in enumerate(stims):
        assert s.trial_index == i
        assert (i in critical_idx) == s.is_critical


def test_all_stimuli_have_unambiguous_correct_answer():
    stims = generate_asch_stimuli(n_trials=18, critical_indices=[2, 5, 11],
                                   seed=1)
    for s in stims:
        correct_len = s.options[s.correct_label]
        assert correct_len == s.reference_cm
        for label, length in s.options.items():
            if label == s.correct_label:
                continue
            assert abs(length - s.reference_cm) >= 2.0, (
                f"wrong option {label}={length} too close to reference "
                f"{s.reference_cm}"
            )


def test_reference_line_in_5_to_15_cm_range():
    stims = generate_asch_stimuli(n_trials=18, critical_indices=[2], seed=7)
    for s in stims:
        assert 5.0 <= s.reference_cm <= 15.0


def test_seed_is_deterministic():
    a = generate_asch_stimuli(18, [2, 5], seed=99)
    b = generate_asch_stimuli(18, [2, 5], seed=99)
    assert [s.reference_cm for s in a] == [s.reference_cm for s in b]
    assert [s.options for s in a] == [s.options for s in b]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_asch.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/experiments/asch/stimuli.py`**

```python
"""Generate line-length comparison stimuli for Asch Phase 1."""
from __future__ import annotations

import random

from psychbench.framework.types import Stimulus


def generate_asch_stimuli(
    n_trials: int,
    critical_indices: list[int],
    seed: int = 0,
    reference_min: float = 5.0,
    reference_max: float = 15.0,
    wrong_gap_min: float = 2.0,
) -> list[Stimulus]:
    """Produce ``n_trials`` Asch-style line-length stimuli.

    Reference line length is uniform in [reference_min, reference_max] rounded
    to the nearest 0.5cm. Exactly one option equals the reference; the other
    two differ by at least ``wrong_gap_min`` cm (kept unambiguous).
    """
    rng = random.Random(seed)
    stimuli: list[Stimulus] = []
    critical_set = set(critical_indices)
    labels = ["A", "B", "C"]
    for i in range(n_trials):
        reference = round(rng.uniform(reference_min, reference_max) * 2) / 2
        correct_label = rng.choice(labels)
        wrong_labels = [l for l in labels if l != correct_label]
        wrong_lengths = _pick_unambiguous_wrong_lengths(
            rng, reference, reference_min, reference_max, wrong_gap_min,
        )
        options = {correct_label: reference}
        for label, length in zip(wrong_labels, wrong_lengths):
            options[label] = length
        stimuli.append(Stimulus(
            trial_index=i,
            is_critical=i in critical_set,
            reference_cm=reference,
            options=options,
            correct_label=correct_label,
            metadata={"seed": seed},
        ))
    return stimuli


def _pick_unambiguous_wrong_lengths(
    rng: random.Random,
    reference: float,
    ref_min: float,
    ref_max: float,
    gap_min: float,
) -> list[float]:
    # Expand search window well past [ref_min, ref_max] so every reference has
    # feasible wrong options.
    low = max(1.0, ref_min - 4.0)
    high = ref_max + 4.0
    picks: list[float] = []
    attempts = 0
    while len(picks) < 2 and attempts < 500:
        attempts += 1
        candidate = round(rng.uniform(low, high) * 2) / 2
        if abs(candidate - reference) < gap_min:
            continue
        if any(abs(candidate - p) < gap_min for p in picks):
            continue
        picks.append(candidate)
    if len(picks) < 2:
        # Deterministic fallback: reference +/- 3cm (and -3 or +3 offsets).
        picks = [reference + 3.0, reference - 3.0]
        picks = [p if p > 0 else reference + 4.0 for p in picks]
    return picks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_asch.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add psychbench/experiments/asch/stimuli.py tests/test_asch.py
git commit -m "add Asch stimulus generator with unambiguous wrong options"
```

---

## Task 14: Asch scoring

**Files:**
- Create: `psychbench/experiments/asch/scoring.py`
- Modify: `tests/test_asch.py` (append)

- [ ] **Step 1: Append tests**

```python
from psychbench.experiments.asch.scoring import (
    score_asch_trial, summarize_asch_session,
)
from psychbench.framework.types import (
    AgentResponse, Stimulus, TrialResult,
)


def _result(is_critical, correct, naive_ans, confed_ans):
    stim = Stimulus(0, is_critical, 5.0,
                    {"A": 5.0, "B": 8.0, "C": 2.0}, correct, {})
    responses = [
        AgentResponse("c1", confed_ans, confed_ans, "", {}),
        AgentResponse("c2", confed_ans, confed_ans, "", {}),
        AgentResponse("naive", naive_ans, naive_ans, "", {}),
    ]
    return TrialResult(
        trial_index=0, is_critical=is_critical, stimulus=stim,
        responses=responses, conformed=None,
        naive_answer=None, confederate_answer=None,
        correct_answer=correct,
    )


def test_score_critical_trial_conformed_when_matches_wrong():
    r = _result(True, "A", naive_ans="B", confed_ans="B")
    out = score_asch_trial(r, naive_id="naive",
                            confederate_ids=["c1", "c2"])
    assert out["conformed"] is True
    assert out["naive_answer"] == "B"
    assert out["confederate_answer"] == "B"


def test_score_critical_trial_not_conformed_when_correct():
    r = _result(True, "A", naive_ans="A", confed_ans="B")
    out = score_asch_trial(r, naive_id="naive",
                            confederate_ids=["c1", "c2"])
    assert out["conformed"] is False


def test_score_non_critical_trial_conformed_is_none():
    r = _result(False, "A", naive_ans="A", confed_ans="A")
    out = score_asch_trial(r, naive_id="naive",
                            confederate_ids=["c1", "c2"])
    assert out["conformed"] is None


def test_summarize_session_rates():
    trials = [
        _result(True, "A", "B", "B"),   # conformed
        _result(True, "A", "A", "B"),   # resisted
        _result(True, "A", "B", "B"),   # conformed
        _result(False, "A", "A", "A"),  # non-critical
    ]
    # Attach scoring as the session does.
    scored = []
    for t in trials:
        s = score_asch_trial(t, "naive", ["c1", "c2"])
        t.conformed = s["conformed"]
        t.naive_answer = s["naive_answer"]
        t.confederate_answer = s["confederate_answer"]
        scored.append(t)
    summary = summarize_asch_session(scored)
    assert summary["n_critical"] == 3
    assert summary["n_conformed"] == 2
    assert abs(summary["conformity_rate"] - 2/3) < 1e-9
    assert summary["ever_conformed"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_asch.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/experiments/asch/scoring.py`**

```python
"""Binary conformity scoring for Asch Phase 1."""
from __future__ import annotations

from typing import Any

from psychbench.framework.types import AgentResponse, TrialResult


def _find(responses: list[AgentResponse], agent_id: str) -> AgentResponse | None:
    for r in responses:
        if r.agent_id == agent_id:
            return r
    return None


def score_asch_trial(
    result: TrialResult, naive_id: str, confederate_ids: list[str],
) -> dict[str, Any]:
    naive = _find(result.responses, naive_id)
    naive_ans = naive.parsed_answer if naive is not None else None
    confed_answers = [
        r.parsed_answer for r in result.responses
        if r.agent_id in confederate_ids and r.parsed_answer is not None
    ]
    confed_ans = confed_answers[0] if confed_answers else None
    if not result.is_critical:
        return {
            "conformed": None,
            "naive_answer": naive_ans,
            "confederate_answer": confed_ans,
        }
    conformed = (
        naive_ans is not None
        and confed_ans is not None
        and naive_ans == confed_ans
        and naive_ans != result.correct_answer
    )
    return {
        "conformed": bool(conformed),
        "naive_answer": naive_ans,
        "confederate_answer": confed_ans,
    }


def summarize_asch_session(trials: list[TrialResult]) -> dict[str, Any]:
    critical = [t for t in trials if t.is_critical]
    n_critical = len(critical)
    n_conformed = sum(1 for t in critical if t.conformed is True)
    conformity_rate = n_conformed / n_critical if n_critical else 0.0
    per_trial = [
        {
            "trial_index": t.trial_index,
            "is_critical": t.is_critical,
            "naive_answer": t.naive_answer,
            "confederate_answer": t.confederate_answer,
            "correct_answer": t.correct_answer,
            "conformed": t.conformed,
        }
        for t in trials
    ]
    return {
        "n_trials": len(trials),
        "n_critical": n_critical,
        "n_conformed": n_conformed,
        "conformity_rate": conformity_rate,
        "ever_conformed": n_conformed > 0,
        "per_trial": per_trial,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_asch.py -v`
Expected: PASS (8 tests total)

- [ ] **Step 5: Commit**

```bash
git add psychbench/experiments/asch/scoring.py tests/test_asch.py
git commit -m "add Asch binary conformity scoring + session summary"
```

---

## Task 15: Asch prompt builder

**Files:**
- Create: `psychbench/experiments/asch/prompts.py`
- Modify: `tests/test_asch.py` (append)

- [ ] **Step 1: Append tests**

```python
from psychbench.experiments.asch.prompts import build_asch_prompt


def test_prompt_contains_reference_options_and_instructions():
    stim = Stimulus(0, True, 10.0, {"A": 10.0, "B": 13.0, "C": 7.0}, "A", {})
    ctx = TrialContext(stim, 5, "naive",
                        prior_responses=[("c1", "B"), ("c2", "B")])
    prompt = build_asch_prompt(ctx)
    assert "Reference line: 10" in prompt
    assert "A: 10" in prompt and "B: 13" in prompt and "C: 7" in prompt
    assert "Participant 1: B" in prompt
    assert "Participant 2: B" in prompt
    assert "Answer with only the letter A, B, or C." in prompt


def test_prompt_private_condition_omits_other_participants():
    stim = Stimulus(0, True, 10.0, {"A": 10.0, "B": 13.0, "C": 7.0}, "A", {})
    ctx = TrialContext(stim, 0, "naive", prior_responses=[])
    prompt = build_asch_prompt(ctx)
    assert "other participants" not in prompt.lower()
    assert "Answer with only the letter A, B, or C." in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_asch.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/experiments/asch/prompts.py`**

```python
"""Prompt template for the naive Asch participant."""
from __future__ import annotations

from psychbench.framework.types import TrialContext


def _fmt_cm(value: float) -> str:
    return f"{value:g}"


def build_asch_prompt(context: TrialContext) -> str:
    stim = context.stimulus
    lines = [
        "You are participating in a vision test with other participants.",
        "",
        "You will be shown a reference line and three comparison lines.",
        "Your task is to identify which comparison line (A, B, or C) matches",
        "the reference line in length.",
        "",
        f"Reference line: {_fmt_cm(stim.reference_cm)}cm",
        "",
        "Comparison lines:",
    ]
    for label in ("A", "B", "C"):
        lines.append(f"{label}: {_fmt_cm(stim.options[label])}cm")

    if context.prior_responses:
        lines += ["", "The other participants have already given their answers:"]
        for i, (_aid, ans) in enumerate(context.prior_responses, start=1):
            lines.append(f"Participant {i}: {ans}")

    lines += ["", "Which line matches the reference line? "
               "Answer with only the letter A, B, or C."]
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_asch.py -v`
Expected: PASS (10 tests total)

- [ ] **Step 5: Commit**

```bash
git add psychbench/experiments/asch/prompts.py tests/test_asch.py
git commit -m "add Asch naive-participant prompt template"
```

---

## Task 16: AschExperiment class (config → agents, stimuli, session wiring)

**Files:**
- Create: `psychbench/experiments/asch/experiment.py`
- Modify: `psychbench/experiments/asch/__init__.py`
- Modify: `tests/test_asch.py` (append)

- [ ] **Step 1: Append tests**

```python
from psychbench.experiments.asch.experiment import AschExperiment


def _base_config(**overrides):
    cfg = {
        "experiment": {
            "name": "asch_phase1",
            "type": "asch",
            "trials": 18,
            "critical_trials": 12,
            "critical_trial_indices": [2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17],
            "seed": 42,
        },
        "agents": {
            "confederates": {
                "type": "scripted",
                "count": 5,
                "behavior": "always_wrong_on_critical",
            },
            "naive": {
                "type": "model",
                "backend": "echo",
                "model": "echo-test",
                "stateful": False,
                "position": "last",
            },
        },
        "environment": {
            "response_visibility": "public",
            "answer_order": "sequential",
        },
        "control": {"run_control": False, "response_visibility": "private"},
        "scoring": {"method": "binary"},
        "logging": {"save_context_windows": True, "output_dir": "results",
                     "format": "jsonl"},
    }
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def test_asch_experiment_builds_6_agents_with_naive_last():
    exp = AschExperiment(_base_config())
    agents = exp.build_agents()
    assert len(agents) == 6
    last = max(agents, key=lambda a: a.position)
    assert last.agent_id == "naive"
    assert last.position == 5


def test_asch_experiment_builds_18_stimuli_with_12_critical():
    exp = AschExperiment(_base_config())
    stims = exp.build_stimuli()
    assert len(stims) == 18
    assert sum(1 for s in stims if s.is_critical) == 12


def test_asch_experiment_dissenter_flag_flips_one_confederate(tmp_path):
    cfg = _base_config()
    cfg["agents"]["confederates"]["dissenter"] = True
    exp = AschExperiment(cfg)
    agents = exp.build_agents()
    confeds = [a for a in agents if a.agent_id.startswith("confederate_")]
    dissenters = [a for a in confeds
                   if a.metadata.get("behavior") == "always_correct"]
    assert len(dissenters) == 1


def test_asch_experiment_n_confederates_override():
    cfg = _base_config()
    cfg["agents"]["confederates"]["count"] = 3
    exp = AschExperiment(cfg)
    agents = exp.build_agents()
    assert len(agents) == 4  # 3 confeds + 1 naive
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_asch.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/experiments/asch/experiment.py`**

```python
"""Asch (1951) line-length conformity experiment."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from psychbench.agents.hybrid_agent import HybridAgent
from psychbench.agents.model_agent import ModelAgent
from psychbench.agents.scripted_agent import ScriptedAgent
from psychbench.framework.agent import BaseAgent
from psychbench.framework.backends import get_backend
from psychbench.framework.environment import Environment
from psychbench.framework.experiment import BaseExperiment, register_experiment
from psychbench.framework.session import Session
from psychbench.framework.types import ResponseVisibility, Stimulus, TrialResult

from .prompts import build_asch_prompt
from .scoring import score_asch_trial, summarize_asch_session
from .stimuli import generate_asch_stimuli


@register_experiment("asch")
class AschExperiment(BaseExperiment):
    def build_stimuli(self) -> list[Stimulus]:
        exp_cfg = self.config["experiment"]
        return generate_asch_stimuli(
            n_trials=exp_cfg["trials"],
            critical_indices=list(exp_cfg["critical_trial_indices"]),
            seed=exp_cfg.get("seed", 0),
        )

    def build_agents(self, *, for_control: bool = False) -> list[BaseAgent]:
        agents_cfg = self.config["agents"]
        confed_cfg = agents_cfg["confederates"]
        naive_cfg = agents_cfg["naive"]
        n_conf = int(confed_cfg.get("count", 5))
        dissenter = bool(confed_cfg.get("dissenter", False))
        behavior = confed_cfg.get("behavior", "always_wrong_on_critical")

        # Pick one shared wrong letter so confederates give a unanimous answer.
        wrong_letter = confed_cfg.get("wrong_answer", "B")

        confederates: list[BaseAgent] = []
        for i in range(n_conf):
            if dissenter and i == 0:
                agent_behavior = "always_correct"
                wrong = None
            else:
                agent_behavior = behavior
                wrong = wrong_letter if behavior == "always_wrong_on_critical" else None
            confederates.append(ScriptedAgent(
                agent_id=f"confederate_{i+1}",
                position=i,
                behavior=agent_behavior,
                wrong_answer=wrong,
            ))

        naive_backend = get_backend(
            naive_cfg.get("backend", "echo"),
            naive_cfg.get("model", "echo-test"),
        )
        naive_position = n_conf if naive_cfg.get("position", "last") == "last" \
            else int(naive_cfg["position"])
        naive = ModelAgent(
            agent_id="naive",
            position=naive_position,
            backend=naive_backend,
            stateful=bool(naive_cfg.get("stateful", False)),
            prompt_builder=build_asch_prompt,
        )

        return [*confederates, naive]

    def _environment(self, *, for_control: bool) -> Environment:
        if for_control:
            vis = self.config.get("control", {}).get(
                "response_visibility", "private")
        else:
            vis = self.config["environment"]["response_visibility"]
        return Environment(visibility=ResponseVisibility(vis))

    def run(self, output_dir: str | Path) -> dict[str, Any]:
        timestamp = int(time.time())
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {}

        for label, for_control in self._conditions():
            agents = self.build_agents(for_control=for_control)
            stimuli = self.build_stimuli()
            env = self._environment(for_control=for_control)
            log_path = out_dir / f"asch_{label}_{timestamp}.jsonl"
            summary_path = out_dir / f"asch_{label}_{timestamp}.summary.json"

            confederate_ids = [a.agent_id for a in agents
                                if a.agent_id.startswith("confederate_")]

            def _score(trial: TrialResult) -> dict[str, Any]:
                return score_asch_trial(trial, "naive", confederate_ids)

            session = Session(
                stimuli=stimuli, agents=agents, environment=env,
                log_path=log_path, summary_path=summary_path,
                config_snapshot=self.config,
                score_trial=_score,
                summarize=summarize_asch_session,
                session_label=label,
            )
            results = session.run()
            summary[label] = {
                "log_path": str(log_path),
                "summary_path": str(summary_path),
                "n_trials": len(results),
            }

        if len(summary) == 2:
            summary["comparison"] = self._load_comparison(summary, out_dir)
        return summary

    def _conditions(self) -> list[tuple[str, bool]]:
        run_ctrl = bool(self.config.get("control", {}).get("run_control", False))
        conds = [("experimental", False)]
        if run_ctrl:
            conds.append(("control", True))
        return conds

    def _load_comparison(self, summary: dict[str, Any],
                          out_dir: Path) -> dict[str, Any]:
        import json
        exp_s = json.loads(
            Path(summary["experimental"]["summary_path"]).read_text()
        )
        ctrl_s = json.loads(
            Path(summary["control"]["summary_path"]).read_text()
        )
        return {
            "experimental_conformity_rate": exp_s.get("conformity_rate"),
            "control_conformity_rate": ctrl_s.get("conformity_rate"),
            "delta": (exp_s.get("conformity_rate", 0.0)
                       - ctrl_s.get("conformity_rate", 0.0)),
        }
```

- [ ] **Step 4: Update `psychbench/experiments/asch/__init__.py`**

```python
"""Asch 1951 line-length conformity experiment module."""
from .experiment import AschExperiment  # noqa: F401  (registers experiment)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_asch.py -v`
Expected: PASS (14 tests total)

- [ ] **Step 6: Commit**

```bash
git add psychbench/experiments/asch tests/test_asch.py
git commit -m "add AschExperiment wiring with control condition and dissenter flag"
```

---

## Task 17: Analysis module

**Files:**
- Create: `psychbench/analysis/conformity.py`
- Modify: `tests/test_asch.py` (append)

- [ ] **Step 1: Append tests**

```python
import json as _json
from psychbench.analysis.conformity import load_session_summary, compare_conditions


def test_load_session_summary(tmp_path):
    p = tmp_path / "summary.json"
    p.write_text(_json.dumps({"conformity_rate": 0.33, "n_critical": 12,
                               "n_conformed": 4, "ever_conformed": True}))
    s = load_session_summary(p)
    assert s["conformity_rate"] == 0.33


def test_compare_conditions(tmp_path):
    exp = tmp_path / "e.json"
    ctrl = tmp_path / "c.json"
    exp.write_text(_json.dumps({"conformity_rate": 0.33, "n_critical": 12,
                                 "n_conformed": 4, "ever_conformed": True}))
    ctrl.write_text(_json.dumps({"conformity_rate": 0.08, "n_critical": 12,
                                  "n_conformed": 1, "ever_conformed": True}))
    cmp = compare_conditions(exp, ctrl)
    assert cmp["experimental"]["conformity_rate"] == 0.33
    assert cmp["control"]["conformity_rate"] == 0.08
    assert abs(cmp["delta"] - 0.25) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_asch.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/analysis/conformity.py`**

```python
"""Load session summaries and compute comparative conformity stats."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_session_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def compare_conditions(
    experimental_summary: str | Path,
    control_summary: str | Path,
) -> dict[str, Any]:
    e = load_session_summary(experimental_summary)
    c = load_session_summary(control_summary)
    return {
        "experimental": {
            "conformity_rate": e.get("conformity_rate"),
            "n_conformed": e.get("n_conformed"),
            "n_critical": e.get("n_critical"),
        },
        "control": {
            "conformity_rate": c.get("conformity_rate"),
            "n_conformed": c.get("n_conformed"),
            "n_critical": c.get("n_critical"),
        },
        "delta": (e.get("conformity_rate", 0.0)
                   - c.get("conformity_rate", 0.0)),
    }


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        "=== PsychBench Session Summary ===",
        f"Trials: {summary.get('n_trials')}",
        f"Critical trials: {summary.get('n_critical')}",
        f"Conformed: {summary.get('n_conformed')}",
        f"Conformity rate: {summary.get('conformity_rate'):.3f}"
        if summary.get("conformity_rate") is not None else "Conformity rate: n/a",
        f"Ever conformed: {summary.get('ever_conformed')}",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_asch.py -v`
Expected: PASS (16 tests total)

- [ ] **Step 5: Commit**

```bash
git add psychbench/analysis/conformity.py tests/test_asch.py
git commit -m "add conformity analysis module"
```

---

## Task 18: Config loader

**Files:**
- Create: `psychbench/config.py`
- Create: `config/experiments/asch_phase1.yaml`
- Modify: `tests/test_asch.py` (append)

- [ ] **Step 1: Append tests**

```python
from psychbench.config import load_config


def test_load_config_reads_asch_yaml(tmp_path):
    yaml_text = """
experiment:
  name: asch_phase1
  type: asch
  trials: 18
  critical_trials: 12
  critical_trial_indices: [2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17]
agents:
  confederates:
    type: scripted
    count: 5
    behavior: always_wrong_on_critical
  naive:
    type: model
    backend: echo
    model: echo-test
    stateful: false
    position: last
environment:
  response_visibility: public
  answer_order: sequential
control:
  run_control: true
  response_visibility: private
scoring:
  method: binary
logging:
  save_context_windows: true
  output_dir: results/
  format: jsonl
"""
    p = tmp_path / "asch.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert cfg["experiment"]["type"] == "asch"
    assert cfg["agents"]["naive"]["backend"] == "echo"
    assert cfg["control"]["run_control"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_asch.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `psychbench/config.py`**

```python
"""Load and minimally validate YAML experiment configs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_KEYS = ("experiment", "agents", "environment")


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text()
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    missing = [k for k in REQUIRED_TOP_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    return cfg
```

- [ ] **Step 4: Write `config/experiments/asch_phase1.yaml`**

```yaml
experiment:
  name: asch_phase1
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
    backend: openai            # openai | anthropic | huggingface | echo
    model: gpt-4o-mini
    stateful: false
    position: last

environment:
  response_visibility: public
  answer_order: sequential

control:
  run_control: true
  response_visibility: private

scoring:
  method: binary
  conformity_threshold: 1

logging:
  save_context_windows: true
  output_dir: results/
  format: jsonl
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_asch.py -v`
Expected: PASS (17 tests total)

- [ ] **Step 6: Commit**

```bash
git add psychbench/config.py config/experiments/asch_phase1.yaml tests/test_asch.py
git commit -m "add YAML config loader and asch_phase1 config"
```

---

## Task 19: CLI (`python -m psychbench`)

**Files:**
- Create: `psychbench/cli.py`
- Create: `psychbench/__main__.py`
- Modify: `tests/test_asch.py` (append)

- [ ] **Step 1: Append tests (integration end-to-end using echo backend)**

```python
import subprocess, sys, shutil


def test_cli_runs_asch_end_to_end_with_echo_backend(tmp_path, monkeypatch):
    # Write a minimal config using echo backend (no network, deterministic).
    cfg = tmp_path / "asch.yaml"
    cfg.write_text("""
experiment:
  name: asch_test
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
  naive:
    type: model
    backend: echo
    model: echo-test
    stateful: false
    position: last
environment:
  response_visibility: public
  answer_order: sequential
control:
  run_control: true
  response_visibility: private
scoring:
  method: binary
logging:
  save_context_windows: true
  output_dir: %s
  format: jsonl
""" % tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "psychbench", "run", "--config", str(cfg),
         "--output-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    produced = list(tmp_path.glob("asch_experimental_*.jsonl"))
    assert produced, "experimental jsonl should be written"
    ctrl = list(tmp_path.glob("asch_control_*.jsonl"))
    assert ctrl, "control jsonl should be written"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_asch.py::test_cli_runs_asch_end_to_end_with_echo_backend -v`
Expected: FAIL (module not yet importable as `-m psychbench`)

- [ ] **Step 3: Implement `psychbench/cli.py`**

```python
"""Command-line interface: `python -m psychbench run|analyze ...`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import asch module so its @register_experiment decorator runs.
from psychbench.experiments import asch  # noqa: F401
from psychbench.analysis.conformity import (
    compare_conditions, format_report, load_session_summary,
)
from psychbench.config import load_config
from psychbench.framework.experiment import get_experiment_class


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    exp_type = cfg["experiment"]["type"]
    exp_cls = get_experiment_class(exp_type)
    exp = exp_cls(cfg)
    out_dir = Path(args.output_dir or cfg.get("logging", {}).get(
        "output_dir", "results"))
    summary = exp.run(output_dir=out_dir)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    if args.experimental and args.control:
        cmp = compare_conditions(args.experimental, args.control)
        print(json.dumps(cmp, indent=2))
    elif args.results:
        s = load_session_summary(args.results)
        print(format_report(s))
    else:
        print("analyze requires --results, or both --experimental and --control",
              file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="psychbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run an experiment from a YAML config")
    pr.add_argument("--config", required=True, help="Path to experiment YAML")
    pr.add_argument("--output-dir", default=None,
                     help="Directory for JSONL logs and summary JSON")
    pr.set_defaults(func=_cmd_run)

    pa = sub.add_parser("analyze", help="Summarize/compare session summaries")
    pa.add_argument("--results", default=None,
                     help="Path to a single .summary.json")
    pa.add_argument("--experimental", default=None,
                     help="Path to experimental .summary.json")
    pa.add_argument("--control", default=None,
                     help="Path to control .summary.json")
    pa.set_defaults(func=_cmd_analyze)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement `psychbench/__main__.py`**

```python
from psychbench.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_asch.py -v`
Expected: PASS (18 tests total)

- [ ] **Step 6: Commit**

```bash
git add psychbench/cli.py psychbench/__main__.py tests/test_asch.py
git commit -m "add CLI entrypoint: python -m psychbench run|analyze"
```

---

## Task 20: Full test-suite sweep + README polish

**Files:**
- Modify: `README.md` (add example output snippet + variations docs)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: ALL PASS (~35 tests across framework/agents/asch)

- [ ] **Step 2: Manual end-to-end smoke test using echo backend**

Run:

```bash
python -m psychbench run --config config/experiments/asch_phase1.yaml \
    --output-dir results/
```

(Temporarily edit the config's `naive.backend` to `echo` for offline smoke
test.) Expected: prints JSON summary; JSONL + summary JSON exist under
`results/`.

- [ ] **Step 3: Append a "Variations" section to `README.md`**

```markdown
## Variations (Phase 1)

All implemented via the YAML config — no code changes needed:

| Flag | Location | Effect |
|------|----------|--------|
| `environment.response_visibility: private` | top-level | Naive agent never sees confederate responses (≈ Asch private condition). |
| `agents.confederates.dissenter: true` | agents block | One confederate gives the correct answer. |
| `agents.confederates.count: N` | agents block | Vary the number of confederates (1, 2, 3, 5, 7). |
| `agents.naive.stateful: true` | agents block | Naive agent maintains conversation history across all 18 trials. |
| `control.run_control: true` | top-level | Also runs a matched control session with `response_visibility: private`. |

Outputs:

- `results/asch_experimental_<ts>.jsonl` — per-trial records
- `results/asch_experimental_<ts>.summary.json` — session-level metrics
- If control is enabled, matched `asch_control_<ts>.*` files
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "document phase-1 variations and outputs in README"
```

---

## Self-Review Notes

- Spec coverage: framework + 3 agent types + Asch stimuli/scoring/experiment + control condition + config-driven variations (private / dissenter / n_confederates / stateful) + JSONL+summary logging + CLI + backends for OpenAI/Anthropic/HF — all mapped to tasks above.
- No placeholders: every step shows actual code or exact commands.
- Types are consistent across tasks: `Stimulus`, `TrialContext`, `AgentResponse`, `TrialResult`, `ResponseVisibility`, `BaseAgent`, `BaseExperiment`, `ModelBackend`, `Environment`, `Session` — each defined once and referenced as defined.
- `EchoBackend` lets every test run hermetically without network/API keys.
