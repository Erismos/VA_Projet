from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from project.e2e import run_e2e_smoke


def _write_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (30, 40), (255, 255, 255), -1)
    cv2.imwrite(str(path), frame)


def _write_gt_and_preds(tmp_path: Path) -> tuple[Path, Path, Path]:
    seq_dir = tmp_path / "MOT17-TEST"
    img_dir = seq_dir / "img1"
    _write_frame(img_dir / "000001.jpg")
    _write_frame(img_dir / "000002.jpg")

    gt = {
        "images": [
            {"id": 1, "file_name": "MOT17-TEST/img1/000001.jpg", "width": 64, "height": 64},
            {"id": 2, "file_name": "MOT17-TEST/img1/000002.jpg", "width": 64, "height": 64},
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}],
    }
    preds = [
        {"frame_id": 0, "bbox": [10, 10, 20, 30], "score": 0.9, "class_name": "person"},
        {"frame_id": 1, "bbox": [12, 11, 20, 30], "score": 0.88, "class_name": "person"},
    ]

    gt_json = tmp_path / "val_gt.json"
    p2_preds = tmp_path / "p2_predictions.json"
    p3_preds = tmp_path / "p3_predictions.json"
    gt_json.write_text(json.dumps(gt), encoding="utf-8")
    p2_preds.write_text(json.dumps(preds), encoding="utf-8")
    p3_preds.write_text(json.dumps(preds), encoding="utf-8")
    return seq_dir, gt_json, p2_preds, p3_preds


def test_run_e2e_smoke_without_p4(tmp_path: Path) -> None:
    seq_dir, gt_json, p2_preds, p3_preds = _write_gt_and_preds(tmp_path)

    summary = run_e2e_smoke(
        mot_seq_dir=str(seq_dir),
        gt_json=str(gt_json),
        p2_preds=str(p2_preds),
        p3_preds=str(p3_preds),
        output_dir=str(tmp_path / "project_results"),
        p5_output_dir=str(tmp_path / "p5_results"),
        p4_output_dir=str(tmp_path / "p4_results"),
        output_name="smoke",
        skip_p4=True,
    )

    report_path = Path(summary["report_dir"]) / "project_report.json"
    assert summary["p4"] is None
    assert report_path.exists()
    assert (tmp_path / "p5_results" / "comparison_report.json").exists()


def test_run_e2e_smoke_with_p4(tmp_path: Path) -> None:
    seq_dir, gt_json, p2_preds, p3_preds = _write_gt_and_preds(tmp_path)

    summary = run_e2e_smoke(
        mot_seq_dir=str(seq_dir),
        gt_json=str(gt_json),
        p2_preds=str(p2_preds),
        p3_preds=str(p3_preds),
        output_dir=str(tmp_path / "project_results"),
        p5_output_dir=str(tmp_path / "p5_results"),
        p4_output_dir=str(tmp_path / "p4_results"),
        output_name="smoke",
        skip_p4=False,
    )

    report_path = Path(summary["report_dir"]) / "project_report.json"
    assert summary["p4"] is not None
    assert report_path.exists()
    assert (tmp_path / "p4_results" / "smoke.txt").exists()
    assert (tmp_path / "p4_results" / "smoke_summary.json").exists()
