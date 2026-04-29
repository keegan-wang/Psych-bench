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
        "delta": (
            e.get("conformity_rate", 0.0) - c.get("conformity_rate", 0.0)
        ),
    }


def format_report(summary: dict[str, Any]) -> str:
    rate = summary.get("conformity_rate")
    rate_line = (
        f"Conformity rate: {rate:.3f}"
        if rate is not None else "Conformity rate: n/a"
    )
    lines = [
        "=== PsychBench Session Summary ===",
        f"Trials: {summary.get('n_trials')}",
        f"Critical trials: {summary.get('n_critical')}",
        f"Conformed: {summary.get('n_conformed')}",
        rate_line,
        f"Ever conformed: {summary.get('ever_conformed')}",
    ]
    return "\n".join(lines)
