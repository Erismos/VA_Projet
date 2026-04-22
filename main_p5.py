import argparse

from p5.pipeline import run_p5_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P5 unified preparation + evaluation pipeline")
    parser.add_argument("--mot-root", default="data/raw/MOT17")
    parser.add_argument("--gt-json", default="data/processed/val_gt.json")
    parser.add_argument("--p2-preds", default="results/p2/inference/predictions.json")
    parser.add_argument("--p3-preds", default="results/p3/predictions.json")
    parser.add_argument("--output-dir", default="results/p5")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run only dataset/data preparation and skip evaluation",
    )
    parser.add_argument(
        "--skip-prepare-data",
        action="store_true",
        help="Skip MOT download/format conversion and only run evaluation",
    )
    parser.add_argument(
        "--verify-p2-train-cmd",
        action="store_true",
        help="Run a dry-run validation that P2 YOLO training inputs are usable",
    )
    return parser

if __name__ == "__main__":
    args = build_parser().parse_args()
    summary = run_p5_pipeline(
        mot_root=args.mot_root,
        gt_path=args.gt_json,
        p2_preds=args.p2_preds,
        p3_preds=args.p3_preds,
        output_dir=args.output_dir,
        skip_prepare_data=args.skip_prepare_data,
        prepare_only=args.prepare_only,
        verify_p2_train_cmd=args.verify_p2_train_cmd,
    )
    print(summary)
