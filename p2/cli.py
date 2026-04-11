from __future__ import annotations

import argparse

from p2.benchmark import benchmark_video
from p2.inference import run_inference
from p2.train import train_fasterrcnn_placeholder, train_yolo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P2 unified CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="Run training")
    tr.add_argument("--model", choices=["yolo", "fasterrcnn"], required=True)
    tr.add_argument("--weights", default="yolo11n.pt")
    tr.add_argument("--dataset-yaml")
    tr.add_argument("--dataset-root")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--imgsz", type=int, default=640)
    tr.add_argument("--batch", type=int, default=16)
    tr.add_argument("--device", default="cpu")
    tr.add_argument("--project", default="models/p2")
    tr.add_argument("--name", default="baseline")
    tr.add_argument("--output-dir", default="models/p2/fasterrcnn_baseline")

    inf = sub.add_parser("infer", help="Run inference and export predictions")
    inf.add_argument("--model", choices=["yolo", "fasterrcnn"], default="yolo")
    inf.add_argument("--weights", default="yolo11n.pt")
    inf.add_argument("--source", required=True)
    inf.add_argument("--output-dir", default="results/p2/inference")
    inf.add_argument("--conf", type=float, default=0.25)
    inf.add_argument("--iou", type=float, default=0.45)
    inf.add_argument("--device", default="cpu")
    inf.add_argument("--save-video", action="store_true")

    bm = sub.add_parser("benchmark", help="Measure fps and gpu memory")
    bm.add_argument("--model", choices=["yolo", "fasterrcnn"], default="yolo")
    bm.add_argument("--weights", default="yolo11n.pt")
    bm.add_argument("--source", required=True)
    bm.add_argument("--conf", type=float, default=0.25)
    bm.add_argument("--iou", type=float, default=0.45)
    bm.add_argument("--device", default="cpu")
    bm.add_argument("--warmup-frames", type=int, default=10)
    bm.add_argument("--max-frames", type=int, default=200)
    bm.add_argument("--output", default="results/p2/benchmark_report.json")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        if args.model == "yolo":
            if not args.dataset_yaml:
                raise ValueError("--dataset-yaml is required for yolo training")
            out = train_yolo(
                weights=args.weights,
                dataset_yaml=args.dataset_yaml,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                project=args.project,
                name=args.name,
            )
        else:
            if not args.dataset_root:
                raise ValueError("--dataset-root is required for fasterrcnn training")
            out = train_fasterrcnn_placeholder(
                dataset_root=args.dataset_root,
                epochs=args.epochs,
                device=args.device,
                output_dir=args.output_dir,
            )
        print(out)
        return

    if args.command == "infer":
        out = run_inference(
            model=args.model,
            weights=args.weights,
            source=args.source,
            output_dir=args.output_dir,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save_video=args.save_video,
        )
        print(out)
        return

    out = benchmark_video(
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
    print(out)


if __name__ == "__main__":
    main()
