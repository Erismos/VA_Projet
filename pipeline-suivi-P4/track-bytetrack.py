from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from p2.detectors import DetectorConfig, create_detector


def get_color(track_id: int) -> tuple[int, int, int]:
    digest = hashlib.md5(str(track_id).encode("utf-8")).hexdigest()
    r = int(digest[0:2], 16)
    g = int(digest[2:4], 16)
    b = int(digest[4:6], 16)
    return b, g, r


@dataclass
class Detection:
    bbox: list[float]  # [x, y, w, h]
    conf: float


class IoUTracker:
    """Simple IoU tracker for detector outputs when ByteTrack is unavailable."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 20) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _iou_xywh(a: list[float], b: list[float]) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, aw) * max(0.0, ah)
        area_b = max(0.0, bw) * max(0.0, bh)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def update(self, detections: list[Detection], frame_idx: int) -> list[tuple[int, Detection]]:
        assigned_tracks: set[int] = set()
        result: list[tuple[int, Detection]] = []

        for det in detections:
            best_track_id = None
            best_iou = 0.0
            for track_id, state in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                iou = self._iou_xywh(det.bbox, state["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= self.iou_threshold:
                self.tracks[best_track_id]["bbox"] = det.bbox
                self.tracks[best_track_id]["last_frame"] = frame_idx
                assigned_tracks.add(best_track_id)
                result.append((best_track_id, det))
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {"bbox": det.bbox, "last_frame": frame_idx}
                assigned_tracks.add(track_id)
                result.append((track_id, det))

        stale_ids = [
            track_id
            for track_id, state in self.tracks.items()
            if frame_idx - int(state["last_frame"]) > self.max_age
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

        return result


def _read_mot_images(mot_seq_dir: str) -> list[str]:
    image_dir = Path(mot_seq_dir) / "img1"
    image_paths = sorted(glob.glob(str(image_dir / "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return image_paths


def _load_json_detections(path: str) -> dict[int, list[Detection]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("Detections JSON must be a list.")

    grouped: dict[int, list[Detection]] = {}
    for det in payload:
        if not isinstance(det, dict):
            continue
        frame_id = det.get("frame_id")
        bbox = det.get("bbox")
        score = det.get("score", det.get("conf", 0.0))
        if frame_id is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            frame_id_i = int(frame_id)
            bbox_f = [float(v) for v in bbox]
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        grouped.setdefault(frame_id_i, []).append(Detection(bbox=bbox_f, conf=score_f))
    return grouped


def _p2_detections_for_frame(detector: Any, frame: Any, frame_idx: int) -> list[Detection]:
    raw = detector.predict_frame(frame, frame_id=frame_idx)
    out: list[Detection] = []
    for det in raw:
        bbox = det.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        out.append(Detection(bbox=[float(v) for v in bbox], conf=float(det.get("score", 0.0))))
    return out


def _draw_track(frame: Any, track_id: int, det: Detection, trail: list[tuple[int, int]]) -> None:
    x1, y1, w, h = [int(v) for v in det.bbox]
    x2, y2 = x1 + w, y1 + h
    color = get_color(track_id)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"ID {track_id}",
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    for i in range(1, len(trail)):
        alpha = i / len(trail)
        thickness = max(1, int(alpha * 3))
        cv2.line(frame, trail[i - 1], trail[i], color, thickness)


def run_tracking(
    mot_seq_dir: str,
    output_dir: str,
    output_name: str,
    detector_backend: str,
    weights: str,
    detections_json: str | None,
    conf: float,
    iou: float,
    device: str,
    trail_length: int,
    tracker_iou: float,
    tracker_max_age: int,
) -> dict[str, Any]:
    image_paths = _read_mot_images(mot_seq_dir)
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        raise RuntimeError(f"Unable to read first frame: {image_paths[0]}")

    height, width = first_frame.shape[:2]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_video = out_dir / f"{output_name}.mp4"
    output_mot = out_dir / f"{output_name}.txt"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (width, height),
    )

    mot_rows: list[list[float]] = []
    history: dict[int, list[tuple[int, int]]] = {}

    # Backend initialization
    yolo_track_model = None
    p2_detector = None
    json_detections: dict[int, list[Detection]] = {}
    tracker = IoUTracker(iou_threshold=tracker_iou, max_age=tracker_max_age)

    if detector_backend == "ultralytics-track":
        yolo_track_model = YOLO(weights)
    elif detector_backend in {"p2-yolo", "p2-fasterrcnn"}:
        model_name = "yolo" if detector_backend == "p2-yolo" else "fasterrcnn"
        cfg = DetectorConfig(model=model_name, weights=weights, conf=conf, iou=iou, device=device)
        p2_detector = create_detector(cfg)
    elif detector_backend == "detections-json":
        if not detections_json:
            raise ValueError("--detections-json is required when detector-backend=detections-json")
        json_detections = _load_json_detections(detections_json)
    else:
        raise ValueError(f"Unsupported detector backend: {detector_backend}")

    print(f"Processing {len(image_paths)} frames from {mot_seq_dir}")

    for frame_idx, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        tracked: list[tuple[int, Detection]] = []
        if detector_backend == "ultralytics-track":
            assert yolo_track_model is not None
            results = yolo_track_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=conf,
                iou=iou,
                verbose=False,
            )
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                for box, track_id, score in zip(boxes, ids, confs):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    tracked.append((int(track_id), Detection(bbox=[x1, y1, x2 - x1, y2 - y1], conf=float(score))))
        else:
            detections: list[Detection]
            if detector_backend in {"p2-yolo", "p2-fasterrcnn"}:
                assert p2_detector is not None
                detections = _p2_detections_for_frame(p2_detector, frame, frame_idx)
            else:
                detections = json_detections.get(frame_idx, [])
            tracked = tracker.update(detections, frame_idx)

        for track_id, det in tracked:
            x1, y1, w_box, h_box = det.bbox
            cx = int(x1 + w_box / 2)
            cy = int(y1 + h_box / 2)

            mot_rows.append(
                [
                    frame_idx + 1,
                    track_id,
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(w_box), 2),
                    round(float(h_box), 2),
                    round(float(det.conf), 4),
                    -1,
                    -1,
                    -1,
                ]
            )

            history.setdefault(track_id, []).append((cx, cy))
            if len(history[track_id]) > trail_length:
                history[track_id].pop(0)

            _draw_track(frame, track_id, det, history[track_id])

        writer.write(frame)

        if (frame_idx + 1) % 50 == 0:
            print(f"  Frame {frame_idx + 1}/{len(image_paths)} processed")

    writer.release()

    mot_df = pd.DataFrame(
        mot_rows,
        columns=["frame", "id", "x", "y", "w", "h", "conf", "x3d", "y3d", "z3d"],
    )
    mot_df.to_csv(output_mot, header=False, index=False)

    summary = {
        "backend": detector_backend,
        "mot_seq_dir": mot_seq_dir,
        "frames": len(image_paths),
        "tracks": int(mot_df["id"].nunique()) if not mot_df.empty else 0,
        "mot_file": str(output_mot),
        "video_file": str(output_video),
    }
    with (out_dir / f"{output_name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P4 tracking pipeline with configurable detector backend")
    parser.add_argument("--mot-seq-dir", default="data/raw/MOT17/train/MOT17-02-FRCNN")
    parser.add_argument("--output-dir", default="results/p4")
    parser.add_argument("--output-name", default="MOT17-02")
    parser.add_argument(
        "--detector-backend",
        choices=["ultralytics-track", "p2-yolo", "p2-fasterrcnn", "detections-json"],
        default="ultralytics-track",
    )
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--detections-json", default=None)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--tracker-iou", type=float, default=0.3)
    parser.add_argument("--tracker-max-age", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_tracking(
        mot_seq_dir=args.mot_seq_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
        detector_backend=args.detector_backend,
        weights=args.weights,
        detections_json=args.detections_json,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        trail_length=args.trail_length,
        tracker_iou=args.tracker_iou,
        tracker_max_age=args.tracker_max_age,
    )
    print(summary)


if __name__ == "__main__":
    main()
