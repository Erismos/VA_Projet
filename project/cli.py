from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _run_module(module_name: str, args: list[str]) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [module_name, *args]
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = old_argv


def _run_script(script_path: Path, args: list[str]) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *args]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified project launcher for P1-P5")
    parser.add_argument(
        "component",
        choices=["p1", "p2", "p3", "p3-benchmark", "p4", "p4-eval", "p4-visualize", "p5", "e2e-smoke"],
        help="Pipeline component to run",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected component")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    component_args = args.args

    root = Path(__file__).resolve().parents[1]

    if args.component == "p1":
        _run_module("P1.cli", component_args)
        return
    if args.component == "p2":
        _run_module("p2.cli", component_args)
        return
    if args.component == "p3":
        _run_module("vision_transformer_p3.detector", component_args)
        return
    if args.component == "p3-benchmark":
        _run_module("vision_transformer_p3.benchmark", component_args)
        return
    if args.component == "p4":
        _run_script(root / "pipeline-suivi-P4" / "track-bytetrack.py", component_args)
        return
    if args.component == "p4-eval":
        _run_script(root / "pipeline-suivi-P4" / "evaluate.py", component_args)
        return
    if args.component == "p4-visualize":
        _run_script(root / "pipeline-suivi-P4" / "visualize.py", component_args)
        return
    if args.component == "p5":
        _run_module("main_p5", component_args)
        return
    if args.component == "e2e-smoke":
        _run_module("project.e2e", component_args)
        return

    raise ValueError(f"Unsupported component: {args.component}")


if __name__ == "__main__":
    main()
