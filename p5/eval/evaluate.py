import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_coco(gt_path, pred_path):
    """
    Evaluates predictions against ground truth using COCO metrics.
    gt_path: Path to ground truth JSON file
    pred_path: Path to predictions JSON file
    """
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats

def compare_models(results_dict):
    """
    Prints a comparison table for different models.
    results_dict: {model_name: stats_array}
    """
    print("\n" + "="*50)
    print(f"{'Model':<15} | {'mAP@.5:.95':<12} | {'mAP@.5':<12} | {'mAP@.75':<12}")
    print("-" * 50)
    for model, stats in results_dict.items():
        # stats[0] is mAP @ .5:.95
        # stats[1] is mAP @ .5
        # stats[2] is mAP @ .75
        print(f"{model:<15} | {stats[0]:.4f}      | {stats[1]:.4f}      | {stats[2]:.4f}")
    print("="*50 + "\n")


def save_comparison_json(results_dict, output_path):
    """
    Saves key detection metrics to JSON for reporting automation.
    results_dict: {model_name: stats_array}
    """
    rows = []
    for model, stats in results_dict.items():
        rows.append(
            {
                "model": model,
                "mAP_50_95": float(stats[0]),
                "mAP_50": float(stats[1]),
                "mAP_75": float(stats[2]),
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

if __name__ == "__main__":
    # Example usage:
    # stats_p2 = evaluate_coco('data/processed/val_gt.json', 'results/p2/preds.json')
    # stats_p3 = evaluate_coco('data/processed/val_gt.json', 'results/p3/preds.json')
    # compare_models({'YOLOv11': stats_p2, 'DETR': stats_p3})
    pass
