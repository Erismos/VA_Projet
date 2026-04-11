from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import torch

from p2.detectors import DetectorConfig, create_detector
from p2.export_utils import save_json


def benchmark_video(
    model: str,
    weights: str,
    source: str,
    conf: float,
    iou: float,
    device: str,
    warmup_frames: int,
    max_frames: int,
    output_path: str,
) -> dict[str, Any]:
    cfg = DetectorConfig(model=model, weights=weights, conf=conf, iou=iou, device=device)
    detector = create_detector(cfg)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")

    frame_times: list[float] = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        _ = detector.predict_frame(frame, frame_id=frame_index)
        dt = time.perf_counter() - t0

        if frame_index >= warmup_frames:
            frame_times.append(dt)
        frame_index += 1

        if max_frames > 0 and frame_index >= max_frames:
            break

    cap.release()

    measured_frames = len(frame_times)
    total = sum(frame_times)
    fps = float(measured_frames / total) if total > 0 else 0.0
    avg_ms = float((total / measured_frames) * 1000.0) if measured_frames > 0 else 0.0

    peak_gpu_mb = 0.0
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_gpu_mb = float(torch.cuda.max_memory_allocated() / (1024**2))

    report = {
        "model": model,
        "weights": weights,
        "source": source,
        "device": device,
        "warmup_frames": warmup_frames,
        "measured_frames": measured_frames,
        "avg_inference_ms": avg_ms,
        "inference_fps": fps,
        "peak_gpu_memory_mb": peak_gpu_mb,
    }

    save_json([report], Path(output_path))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P2 benchmark runner")
    parser.add_argument("--model", default="yolo", choices=["yolo", "fasterrcnn"])
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--output", default="results/p2/benchmark_report.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    report = benchmark_video(
        model=args.model,
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        warmup_frames=args.warmup_frames,
        max_frames=args.max_frames,
        output_path=args.output,
    )
    print(report)


if __name__ == "__main__":
    main()
