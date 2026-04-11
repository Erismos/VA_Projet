from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2

from p2.detectors import DetectorConfig, create_detector
from p2.export_utils import save_csv, save_json


def _draw_detections(frame: Any, detections: list[dict[str, Any]]) -> Any:
    for det in detections:
        x, y, w, h = det["bbox"]
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        label = f"{det['class_name']} {det['score']:.2f}"
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (p1[0], max(0, p1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def run_inference(
    model: str,
    weights: str,
    source: str,
    output_dir: str,
    conf: float,
    iou: float,
    device: str,
    save_video: bool,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = DetectorConfig(model=model, weights=weights, conf=conf, iou=iou, device=device)
    detector = create_detector(cfg)

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    detections: list[dict[str, Any]] = []
    total_frames = 0
    total_time = 0.0

    if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {source}")
        start = time.perf_counter()
        frame_dets = detector.predict_frame(frame, frame_id=0)
        total_time = time.perf_counter() - start
        total_frames = 1
        detections.extend(frame_dets)
        annotated = _draw_detections(frame, frame_dets)
        cv2.imwrite(str(output_path / "annotated_image.jpg"), annotated)
    else:
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(
                str(output_path / "annotated_video.mp4"), fourcc, fps, (width, height)
            )

        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            frame_dets = detector.predict_frame(frame, frame_id=frame_id)
            total_time += time.perf_counter() - start
            detections.extend(frame_dets)
            total_frames += 1
            if writer is not None:
                writer.write(_draw_detections(frame, frame_dets))
            frame_id += 1

        cap.release()
        if writer is not None:
            writer.release()

    json_path = output_path / "predictions.json"
    csv_path = output_path / "predictions.csv"
    save_json(detections, json_path)
    save_csv(detections, csv_path)

    fps = float(total_frames / total_time) if total_time > 0 else 0.0
    summary = {
        "source": str(source_path),
        "model": model,
        "weights": weights,
        "frames": total_frames,
        "detections": len(detections),
        "inference_fps": fps,
        "json": str(json_path),
        "csv": str(csv_path),
    }
    save_json([summary], output_path / "run_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P2 inference runner")
    parser.add_argument("--model", default="yolo", choices=["yolo", "fasterrcnn"])
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--source", required=True, help="Image or video path")
    parser.add_argument("--output-dir", default="results/p2/inference")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-video", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_inference(
        model=args.model,
        weights=args.weights,
        source=args.source,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_video=args.save_video,
    )
    print(summary)


if __name__ == "__main__":
    main()
