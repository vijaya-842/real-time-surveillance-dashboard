"""
Object detector wrapping torchvision Faster R-CNN.
Handles model loading, batched inference, and result filtering.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

log = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox:       np.ndarray   # [x1, y1, x2, y2]
    confidence: float
    class_id:   int
    label:      str


class PersonDetector:
    """
    Wraps Faster R-CNN (ResNet-50 FPN v2) for real-time person detection.
    Uses the COCO-pretrained checkpoint from torchvision.
    """

    COCO_PERSON_CLASS = 1

    def __init__(self,
                 confidence_threshold: float = 0.55,
                 nms_iou_threshold:    float = 0.45,
                 device:               str   = "auto") -> None:

        self.conf_thresh = confidence_threshold
        self.nms_thresh  = nms_iou_threshold
        self.device      = self._resolve_device(device)

        log.info("Loading Faster R-CNN on device: %s", self.device)
        weights       = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model    = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.model.to(self.device).eval()
        self.transform = weights.transforms()
        self.labels    = weights.meta["categories"]
        log.info("Detector ready (%d COCO classes)", len(self.labels))

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil   = torchvision.transforms.functional.to_pil_image(rgb)
        return self.transform(pil).to(self.device)

    @torch.inference_mode()
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.
        Returns only person detections above confidence threshold.
        """
        tensor  = self._preprocess(frame)
        outputs = self.model([tensor])[0]

        boxes   = outputs["boxes"].cpu().numpy()
        scores  = outputs["scores"].cpu().numpy()
        labels  = outputs["labels"].cpu().numpy()

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if label != self.COCO_PERSON_CLASS:
                continue
            if score < self.conf_thresh:
                continue
            detections.append(Detection(
                bbox       = box.astype(np.float32),
                confidence = float(score),
                class_id   = int(label),
                label      = self.labels[label],
            ))

        # Apply NMS across all person detections
        if len(detections) > 1:
            detections = self._nms(detections)

        return detections

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        boxes  = torch.tensor([d.bbox for d in detections])
        scores = torch.tensor([d.confidence for d in detections])
        keep   = torchvision.ops.nms(boxes, scores, self.nms_thresh)
        return [detections[i] for i in keep.tolist()]
