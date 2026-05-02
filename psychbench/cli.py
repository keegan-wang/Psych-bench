"""Command-line interface: `python -m psychbench run|analyze ...`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import experiments so their @register_experiment decorators run.
from psychbench.experiments import asch  # noqa: F401
from psychbench.experiments import asch_documents  # noqa: F401
from psychbench.analysis.conformity import (
    compare_conditions, format_report as format_phase1_report,
    load_session_summary,
)
from psychbench.analysis.manufactured_consensus import (
    format_report as format_sweep_report, load_sweep_manifest,
    write_tidy_csv,
)
from psychbench.config import load_config
from psychbench.framework.experiment import get_experiment_class
from psychbench.framework.sweep import expand_sweep


COST_GATE = 5000


def _cost_gate_total(cfg: dict) -> int:
    sweep_fields = list(cfg.get("sweep", {}).get("fields", []))
    if not sweep_fields:
        n_cells = 1
    else:
        cells = expand_sweep(cfg, sweep_fields=sweep_fields)
        n_cells = len(cells)
    n_repeats = int(cfg.get("experiment", {}).get("n_repeats", 1))
    n_trials = int(cfg.get("experiment", {}).get("trials", 1))
    return n_cells * n_repeats * n_trials


def _cmd_run(args: argparse.Namespace) -> int:
    if getattr(args, "verbose", False):
        from psychbench.framework.progress import stderr_printer, subscribe
        subscribe(stderr_printer)

    cfg = load_config(args.config)
    exp_type = cfg["experiment"]["type"]

    if cfg.get("sweep", {}).get("fields"):
        total = _cost_gate_total(cfg)
        if total > COST_GATE and not args.i_know:
            print(
                f"Sweep would run {total} trials (>{COST_GATE}). "
                f"Re-run with --i-know to proceed.",
                file=sys.stderr,
            )
            return 2

    out_dir = Path(
        args.output_dir
        or cfg.get("logging", {}).get("output_dir", "results")
    )
    if exp_type == "asch_documents":
        from psychbench.experiments.asch_documents.sweep_runner import (
            run_sweep,
        )
        manifest = run_sweep(cfg, output_root=out_dir)
        print(json.dumps(
            {"run_dir": manifest["run_dir"],
             "n_cells": manifest["n_cells"]},
            indent=2,
        ))
        return 0

    exp_cls = get_experiment_class(exp_type)
    exp = exp_cls(cfg)
    summary = exp.run(output_dir=out_dir)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    if args.run:
        manifest_path = Path(args.run) / "sweep.json"
        manifest = load_sweep_manifest(manifest_path)
        print(format_sweep_report(manifest))
        write_tidy_csv(manifest, Path(args.run) / "sweep_tidy.csv")
        return 0
    if args.experimental and args.control:
        cmp = compare_conditions(args.experimental, args.control)
        print(json.dumps(cmp, indent=2))
        return 0
    if args.results:
        s = load_session_summary(args.results)
        print(format_phase1_report(s))
        return 0
    print(
        "analyze requires --run <dir>, --results <summary.json>, "
        "or both --experimental and --control",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="psychbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run an experiment from a YAML config")
    pr.add_argument("--config", required=True)
    pr.add_argument("--output-dir", default=None)
    pr.add_argument(
        "--i-know", action="store_true",
        help="Bypass cost gate for large sweeps",
    )
    pr.add_argument(
        "--verbose", "-v", action="store_true",
        help="Stream live trial/agent events to stderr as the run proceeds",
    )
    pr.set_defaults(func=_cmd_run)

    pa = sub.add_parser("analyze", help="Summarize/compare runs")
    pa.add_argument("--run", default=None,
                     help="Path to a sweep run directory")
    pa.add_argument("--results", default=None,
                     help="Path to a single .summary.json (Phase 1)")
    pa.add_argument("--experimental", default=None)
    pa.add_argument("--control", default=None)
    pa.set_defaults(func=_cmd_analyze)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
