from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_p4_tracker_module() -> object:
    module_path = Path("pipeline-suivi-P4") / "track-bytetrack.py"
    spec = importlib.util.spec_from_file_location("p4_track_bytetrack", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load track-bytetrack.py module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (30, 40), (255, 255, 255), -1)
    cv2.imwrite(str(path), frame)


def test_tracking_with_detections_json_backend(tmp_path: Path) -> None:
    module = _load_p4_tracker_module()

    seq_dir = tmp_path / "MOT17-TEST" 
    img_dir = seq_dir / "img1"
    _write_frame(img_dir / "000001.jpg")
    _write_frame(img_dir / "000002.jpg")

    detections = [
        {"frame_id": 0, "bbox": [10, 10, 20, 30], "score": 0.9},
        {"frame_id": 1, "bbox": [12, 11, 20, 30], "score": 0.88},
    ]
    detections_path = tmp_path / "detections.json"
    detections_path.write_text(json.dumps(detections), encoding="utf-8")

    output_dir = tmp_path / "results"
    summary = module.run_tracking(
        mot_seq_dir=str(seq_dir),
        output_dir=str(output_dir),
        output_name="track_test",
        detector_backend="detections-json",
        weights="yolov8n.pt",
        detections_json=str(detections_path),
        conf=0.3,
        iou=0.5,
        device="cpu",
        trail_length=5,
        tracker_iou=0.2,
        tracker_max_age=5,
    )

    mot_file = Path(summary["mot_file"])
    summary_file = output_dir / "track_test_summary.json"

    assert summary["backend"] == "detections-json"
    assert mot_file.exists()
    assert summary_file.exists()

    lines = [line for line in mot_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 2
