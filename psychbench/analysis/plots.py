"""Matplotlib plots for PsychBench Phase 1 (Asch) and Phase 2 (asch_documents).

Intentionally zero pandas dependency — only matplotlib + numpy + stdlib json.

Entry points:
    plot_phase1(exp_summary, ctrl_summary, save=None) -> matplotlib.figure.Figure
    plot_phase2(run_dir, save=None) -> matplotlib.figure.Figure
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

COLOR_EXP = "#1f77b4"
COLOR_CTRL = "#ff7f0e"
COLOR_CONFORM = "#d62728"
COLOR_INDEPENDENT = "#2ca02c"
COLOR_OTHER = "#7f7f7f"


def _load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def plot_phase1(
    exp_summary: str | Path,
    ctrl_summary: str | Path,
    exp_log: str | Path | None = None,
    ctrl_log: str | Path | None = None,
    save: str | Path | None = None,
) -> plt.Figure:
    """Two-panel Phase-1 figure: headline conformity bar + per-critical-trial strip.

    Left panel shows experimental vs control conformity rate — the canonical
    Asch comparison. Right panel shows, for each critical trial, whether the
    naive agent answered correctly (green), conformed with the wrong majority
    (red X), or picked some *other* wrong answer (grey square).
    """
    exp = _load_summary(exp_summary)
    ctrl = _load_summary(ctrl_summary)
    exp_log_p = exp_log or str(exp_summary).replace(".summary.json", ".jsonl")

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 2.2]}
    )

    ax = axes[0]
    rates = [exp["conformity_rate"], ctrl["conformity_rate"]]
    counts = [
        (exp["n_conformed"], exp["n_critical"]),
        (ctrl["n_conformed"], ctrl["n_critical"]),
    ]
    bars = ax.bar(
        ["Experimental\n(public)", "Control\n(private)"],
        rates,
        color=[COLOR_EXP, COLOR_CTRL],
        edgecolor="black",
    )
    for bar, rate, (nc, nt) in zip(bars, rates, counts):
        ax.annotate(
            f"{rate:.1%}\n({nc}/{nt})",
            xy=(bar.get_x() + bar.get_width() / 2, rate),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    ax.set_ylim(0, max(1.0, max(rates) * 1.3 + 0.1))
    ax.set_ylabel("Conformity rate on critical trials")
    delta = exp["conformity_rate"] - ctrl["conformity_rate"]
    ax.set_title(f"Phase 1 conformity  |  Δ = {delta:+.1%}")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    trials = [r for r in _load_jsonl(exp_log_p) if r["is_critical"]]
    xs = list(range(len(trials)))
    marker_color: list[str] = []
    marker_shape: list[str] = []
    for t in trials:
        conf = t["scoring"].get("confederate_answer")
        naive = t["scoring"].get("naive_answer")
        correct = t["correct_answer"]
        if naive == correct:
            marker_color.append(COLOR_INDEPENDENT)
            marker_shape.append("o")
        elif naive == conf:
            marker_color.append(COLOR_CONFORM)
            marker_shape.append("X")
        else:
            marker_color.append(COLOR_OTHER)
            marker_shape.append("s")

    for x, c, m in zip(xs, marker_color, marker_shape):
        ax.scatter([x], [1], c=[c], s=200, marker=m, edgecolor="black", zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(t["trial_index"]) for t in trials])
    ax.set_yticks([])
    ax.set_xlabel("Critical trial index")
    ax.set_title("Experimental: naive answer on each critical trial")
    ax.set_ylim(0.5, 1.5)
    ax.grid(axis="x", alpha=0.3)
    handles = [
        mpatches.Patch(color=COLOR_INDEPENDENT, label="Correct (independent)"),
        mpatches.Patch(color=COLOR_CONFORM, label="Conformed with majority"),
        mpatches.Patch(color=COLOR_OTHER, label="Other wrong answer"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_phase2(
    run_dir: str | Path,
    save: str | Path | None = None,
) -> plt.Figure:
    """Two-panel Phase-2 figure: per-cell conformity bars + per-trial heatmap.

    Left panel: for each sweep cell, three bars side-by-side — full conformity
    (strict substring), heuristic partial conformity, and LLM-judge partial
    conformity. The spread across the three bars tells you how much the
    definition of "conforming" matters.

    Right panel: trial-by-trial heatmap for the first cell, with rows for
    unanimity, full, heuristic, and judge. Lets you eyeball which individual
    trials drove the aggregate.
    """
    run = Path(run_dir)
    cells_dir = run / "cells"
    cell_summaries = sorted(cells_dir.glob("*.summary.json"))
    if not cell_summaries:
        raise FileNotFoundError(f"No cell summaries under {cells_dir}")
    cells = [
        (f.name.removesuffix(".summary.json"), json.loads(f.read_text()))
        for f in cell_summaries
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.0, 1.8]}
    )

    ax = axes[0]
    labels = [lbl for lbl, _ in cells]
    n = len(labels)
    xs = np.arange(n)
    full = [c["conformity_rate_unconditional"] for _, c in cells]
    part_heur = [c["partial_conformity_rate_heuristic"] for _, c in cells]
    part_judge = [c.get("partial_conformity_rate_judge") or 0.0 for _, c in cells]

    w = 0.27
    ax.bar(
        xs - w, full, width=w, label="Full (substring)",
        color="#d62728", edgecolor="black",
    )
    ax.bar(
        xs, part_heur, width=w, label="Partial (heuristic)",
        color="#ff7f0e", edgecolor="black",
    )
    ax.bar(
        xs + w, part_judge, width=w, label="Partial (LLM judge)",
        color="#1f77b4", edgecolor="black",
    )
    for i, (f, h, j) in enumerate(zip(full, part_heur, part_judge)):
        for dx, v in zip((-w, 0.0, w), (f, h, j)):
            if v > 0:
                ax.annotate(
                    f"{v:.0%}",
                    xy=(i + dx, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

    ymax = max(1.0, max(full + part_heur + part_judge + [0.0]) * 1.3 + 0.1)
    ax.set_ylim(0, ymax)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Conformity rate")
    ax.set_title("Phase 2: conformity by scoring definition")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    lbl0, _ = cells[0]
    jsonl = cells_dir / f"{lbl0}.jsonl"
    records = _load_jsonl(jsonl)
    metrics = [
        ("Unanimity", [1 if r["scoring"]["unanimity"] else 0 for r in records]),
        ("Full conformed", [r["scoring"]["full_conformity"] for r in records]),
        ("Partial (heur)", [r["scoring"]["partial_conformity_heuristic"] for r in records]),
        (
            "Partial (judge)",
            [(r["scoring"].get("partial_conformity_judge") or 0) for r in records],
        ),
    ]
    data = np.array([row for _, row in metrics])
    ax = axes[1]
    im = ax.imshow(data, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([name for name, _ in metrics])
    ax.set_xticks(range(len(records)))
    ax.set_xticklabels(
        [str(r["trial_index"]) for r in records], fontsize=8,
    )
    ax.set_xlabel("Trial index")
    ax.set_title(f"Per-trial metrics  ({lbl0})")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j]:
                ax.text(
                    j, i, "1", ha="center", va="center",
                    color="white", fontsize=8,
                )
    fig.colorbar(im, ax=ax, shrink=0.6, label="1 = yes, 0 = no")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig
