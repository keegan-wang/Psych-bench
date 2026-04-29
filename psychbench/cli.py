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
        print(
            "analyze requires --results, or both --experimental and --control",
            file=sys.stderr,
        )
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="psychbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run an experiment from a YAML config")
    pr.add_argument("--config", required=True, help="Path to experiment YAML")
    pr.add_argument(
        "--output-dir", default=None,
        help="Directory for JSONL logs and summary JSON",
    )
    pr.set_defaults(func=_cmd_run)

    pa = sub.add_parser("analyze", help="Summarize/compare session summaries")
    pa.add_argument(
        "--results", default=None,
        help="Path to a single .summary.json",
    )
    pa.add_argument(
        "--experimental", default=None,
        help="Path to experimental .summary.json",
    )
    pa.add_argument(
        "--control", default=None,
        help="Path to control .summary.json",
    )
    pa.set_defaults(func=_cmd_analyze)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
