from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import cv2

from src.config import BASE_DIR

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(BASE_DIR / "Ultralytics"),
)

from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.45,
        target_labels: Iterable[str] | None = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.target_labels = set(target_labels or ["person"])

    def detect(self, frame) -> list[Detection]:
        # Tối ưu hóa tốc độ: Sử dụng imgsz=416 để tăng tốc độ phát hiện người trên CPU gấp 2-3 lần
        result = self.model(frame, imgsz=416, conf=self.confidence, verbose=False)[0]
        detections: list[Detection] = []
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = names.get(cls_id, str(cls_id))
            score = float(box.conf[0])
            if self.target_labels and label not in self.target_labels:
                continue

            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            height, width = frame.shape[:2]
            detections.append(
                Detection(
                    label=label,
                    confidence=score,
                    x1=max(0, min(x1, width - 1)),
                    y1=max(0, min(y1, height - 1)),
                    x2=max(0, min(x2, width - 1)),
                    y2=max(0, min(y2, height - 1)),
                )
            )

        return detections


def draw_detection(frame, detection: Detection, text: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = detection.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_y = max(22, y1 - 8)
    cv2.putText(
        frame,
        text,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
