from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms import functional as F


@dataclass
class DetectorConfig:
    model: str = "yolo"
    weights: str = "yolo11n.pt"
    conf: float = 0.25
    iou: float = 0.45
    device: str = "cpu"


class BaseDetector:
    def predict_frame(self, frame_bgr: Any, frame_id: int) -> list[dict[str, Any]]:
        raise NotImplementedError


class YoloDetector(BaseDetector):
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("ultralytics is required for YOLO detector") from exc
        self.model = YOLO(cfg.weights)

    def predict_frame(self, frame_bgr: Any, frame_id: int) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.cfg.device,
            verbose=False,
        )
        detections: list[dict[str, Any]] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        boxes_xywh = result.boxes.xywh.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()

        for i in range(len(boxes_xywh)):
            x_center, y_center, w, h = boxes_xywh[i]
            x = float(x_center - w / 2.0)
            y = float(y_center - h / 2.0)
            class_id = int(cls_ids[i])
            detections.append(
                {
                    "frame_id": frame_id,
                    "class_name": result.names.get(class_id, str(class_id)),
                    "category_id": class_id,
                    "bbox": [x, float(y), float(w), float(h)],
                    "score": float(scores[i]),
                }
            )
        return detections


class FasterRCNNDetector(BaseDetector):
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        custom_weights = Path(cfg.weights)
        if custom_weights.exists() and custom_weights.suffix in {".pt", ".pth"}:
            state = torch.load(custom_weights, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict_frame(self, frame_bgr: Any, frame_id: int) -> list[dict[str, Any]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(frame_rgb).to(self.device)
        with torch.no_grad():
            output = self.model([tensor])[0]

        detections: list[dict[str, Any]] = []
        boxes = output["boxes"].detach().cpu()
        labels = output["labels"].detach().cpu()
        scores = output["scores"].detach().cpu()

        for box, label, score in zip(boxes, labels, scores):
            score_f = float(score.item())
            if score_f < self.cfg.conf:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            detections.append(
                {
                    "frame_id": frame_id,
                    "class_name": f"class_{int(label.item())}",
                    "category_id": int(label.item()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score_f,
                }
            )
        return detections


def create_detector(cfg: DetectorConfig) -> BaseDetector:
    model_name = cfg.model.lower().strip()
    if model_name in {"yolo", "yolov11"}:
        return YoloDetector(cfg)
    if model_name in {"faster-rcnn", "fasterrcnn", "faster_r_cnn"}:
        return FasterRCNNDetector(cfg)
    raise ValueError(f"Unsupported model type: {cfg.model}")
