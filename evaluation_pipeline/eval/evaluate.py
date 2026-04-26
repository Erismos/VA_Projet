import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
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


def evaluate_coco_subset(gt_path: str, pred_path: str, image_ids: list[int]) -> np.ndarray:
    """Evaluate COCO metrics on a subset of GT image ids."""
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _bbox_iou_xywh(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return float(inter_area / denom)


def compute_operating_metrics(
    gt_path: str | Path,
    pred_path: str | Path,
    *,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    image_ids: set[int] | None = None,
) -> dict[str, float]:
    """Compute Precision/Recall/F1 at a configurable operating score threshold."""
    gt = _load_json(gt_path)
    preds = _load_json(pred_path)

    if not isinstance(gt, dict):
        raise ValueError("GT file must contain a COCO dict payload.")
    if not isinstance(preds, list):
        raise ValueError("Predictions file must contain a list payload.")

    ann_by_key: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    gt_count = 0
    for ann in gt.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        bbox = ann.get("bbox")
        if img_id is None or cat_id is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        img_id = int(img_id)
        if image_ids is not None and img_id not in image_ids:
            continue
        ann_by_key[(img_id, int(cat_id))].append([float(v) for v in bbox])
        gt_count += 1

    pred_by_key: dict[tuple[int, int], list[tuple[float, list[float]]]] = defaultdict(list)
    for pred in preds:
        if not isinstance(pred, dict):
            continue
        img_id = pred.get("image_id")
        cat_id = pred.get("category_id")
        bbox = pred.get("bbox")
        score = float(pred.get("score", 0.0))
        if img_id is None or cat_id is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        img_id = int(img_id)
        if image_ids is not None and img_id not in image_ids:
            continue
        if score < score_threshold:
            continue
        pred_by_key[(img_id, int(cat_id))].append((score, [float(v) for v in bbox]))

    tp = 0
    fp = 0

    all_keys = set(ann_by_key.keys()) | set(pred_by_key.keys())
    for key in all_keys:
        gts = ann_by_key.get(key, [])
        preds_for_key = sorted(pred_by_key.get(key, []), key=lambda row: row[0], reverse=True)
        matched_gt = [False] * len(gts)

        for _, pred_box in preds_for_key:
            best_idx = -1
            best_iou = 0.0
            for idx, gt_box in enumerate(gts):
                if matched_gt[idx]:
                    continue
                iou = _bbox_iou_xywh(pred_box, gt_box)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0:
                matched_gt[best_idx] = True
                tp += 1
            else:
                fp += 1

    fn = max(0, gt_count - tp)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "operating_threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def split_coco_image_ids_by_sequence(gt_path: str | Path) -> dict[str, list[int]]:
    """Group GT COCO image ids by sequence name inferred from file_name."""
    gt = _load_json(gt_path)
    if not isinstance(gt, dict):
        return {}

    by_sequence: dict[str, list[int]] = defaultdict(list)
    for image in gt.get("images", []):
        if not isinstance(image, dict):
            continue
        img_id = image.get("id")
        file_name = str(image.get("file_name", ""))
        if img_id is None:
            continue
        parts = [part for part in file_name.replace("\\", "/").split("/") if part]
        sequence = parts[0] if parts else "unknown"
        by_sequence[sequence].append(int(img_id))

    return dict(by_sequence)

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
