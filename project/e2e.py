from __future__ import annotations

import argparse
import csv
import json
import runpy
import sys
from pathlib import Path
from typing import Any

from p5.pipeline import run_p5_pipeline


def _run_script(script_path: Path, args: list[str]) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *args]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv


def run_e2e_smoke(
    mot_seq_dir: str,
    gt_json: str,
    p2_preds: str,
    p3_preds: str,
    output_dir: str,
    p5_output_dir: str | None,
    p4_output_dir: str | None,
    output_name: str,
    skip_p4: bool,
    ) -> dict[str, object]:
    shared_output_dir = Path(output_dir)
    shared_output_dir.mkdir(parents=True, exist_ok=True)

    p5_out_dir = p5_output_dir or str(shared_output_dir / "p5")
    p4_out_dir = p4_output_dir or str(shared_output_dir / "p4")

    p5_summary = run_p5_pipeline(
        gt_path=gt_json,
        p2_preds=p2_preds,
        p3_preds=p3_preds,
        output_dir=p5_out_dir,
        skip_prepare_data=True,
        verify_p2_train_cmd=False,
    )

    p4_summary = None
    if not skip_p4:
        root = Path(__file__).resolve().parents[1]
        _run_script(
            root / "pipeline-suivi-P4" / "track-bytetrack.py",
            [
                "--detector-backend",
                "detections-json",
                "--mot-seq-dir",
                mot_seq_dir,
                "--detections-json",
                p2_preds,
                "--output-dir",
                p4_out_dir,
                "--output-name",
                output_name,
            ],
        )
        p4_summary_file = Path(p4_out_dir) / f"{output_name}_summary.json"
        p4_summary = json.loads(p4_summary_file.read_text(encoding="utf-8")) if p4_summary_file.exists() else None

    project_report = _build_project_report(p5_summary, p4_summary)
    _write_project_report(shared_output_dir, project_report)

    return {
        "report_dir": str(shared_output_dir),
        "p5": p5_summary,
        "p4": p4_summary,
        "project_report": project_report,
    }


def _build_project_report(p5_summary: dict[str, Any], p4_summary: dict[str, Any] | None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "p5": p5_summary,
        "p4": p4_summary,
    }
    return report


def _write_project_report(output_dir: Path, report: dict[str, Any]) -> None:
    json_path = output_dir / "project_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_rows: list[dict[str, Any]] = []
    p5_eval = report.get("p5", {}).get("evaluation", {}) if isinstance(report.get("p5"), dict) else {}
    if isinstance(p5_eval, dict):
        csv_rows.extend(
            [
                {"component": "p5", "metric": "p2_map_50_95", "value": p5_eval.get("p2_map_50_95")},
                {"component": "p5", "metric": "p3_map_50_95", "value": p5_eval.get("p3_map_50_95")},
            ]
        )
    p4 = report.get("p4")
    if isinstance(p4, dict):
        csv_rows.extend(
            [
                {"component": "p4", "metric": "frames", "value": p4.get("frames")},
                {"component": "p4", "metric": "tracks", "value": p4.get("tracks")},
            ]
        )

    csv_path = output_dir / "project_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["component", "metric", "value"])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end smoke pipeline")
    parser.add_argument("--output-dir", default="results/project")
    parser.add_argument("--mot-seq-dir", default="data/raw/MOT17/train/MOT17-02-FRCNN")
    parser.add_argument("--gt-json", default="data/processed/val_gt.json")
    parser.add_argument("--p2-preds", default="results/p2/inference/predictions.json")
    parser.add_argument("--p3-preds", default="results/p3/predictions.json")
    parser.add_argument("--p5-output-dir", default=None)
    parser.add_argument("--p4-output-dir", default=None)
    parser.add_argument("--output-name", default="MOT17-02-smoke")
    parser.add_argument("--skip-p4", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_e2e_smoke(
        mot_seq_dir=args.mot_seq_dir,
        gt_json=args.gt_json,
        p2_preds=args.p2_preds,
        p3_preds=args.p3_preds,
        output_dir=args.output_dir,
        p5_output_dir=args.p5_output_dir,
        p4_output_dir=args.p4_output_dir,
        output_name=args.output_name,
        skip_p4=args.skip_p4,
    )
    print(summary)


if __name__ == "__main__":
    main()
