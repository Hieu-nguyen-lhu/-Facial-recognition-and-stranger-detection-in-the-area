from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import cv2


class AlertLogger:
    def __init__(
        self,
        captures_dir: Path,
        logs_dir: Path,
        interval_seconds: int = 8,
        stranger_hold_seconds: int = 10,
    ) -> None:
        self.captures_dir = Path(captures_dir)
        self.logs_dir = Path(logs_dir)
        self.interval_seconds = interval_seconds
        self.stranger_hold_seconds = stranger_hold_seconds
        self._last_alert_at = 0.0
        self._stranger_first_seen_at: float | None = None
        self._captured_current_presence = False
        self.log_file = self.logs_dir / "detections.csv"
        self._prepare_log()

    def update_stranger_presence(
        self,
        frame,
        stranger_detected: bool,
        bbox: tuple[int, int, int, int] | None,
        score: float,
        source: str,
    ) -> Path | None:
        now = time.time()
        if not stranger_detected or bbox is None:
            self._stranger_first_seen_at = None
            self._captured_current_presence = False
            return None

        if self._stranger_first_seen_at is None:
            self._stranger_first_seen_at = now
            return None

        visible_seconds = now - self._stranger_first_seen_at
        recently_alerted = now - self._last_alert_at < self.interval_seconds
        if (
            visible_seconds >= self.stranger_hold_seconds
            and not self._captured_current_presence
            and not recently_alerted
        ):
            self._captured_current_presence = True
            return self.save_stranger(frame, bbox, score, source)
        return None

    def stranger_visible_seconds(self) -> float:
        if self._stranger_first_seen_at is None:
            return 0.0
        return time.time() - self._stranger_first_seen_at

    def save_stranger(self, frame, bbox: tuple[int, int, int, int], score: float, source: str) -> Path | None:
        now = time.time()
        if now - self._last_alert_at < self.interval_seconds:
            return None
        self._last_alert_at = now

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = self.captures_dir / f"stranger_{timestamp}.jpg"
        cv2.imwrite(str(image_path), frame)
        self._write_log(timestamp, "stranger", bbox, score, source, image_path)
        return image_path

    def save_snapshot(self, frame) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = self.captures_dir / f"snapshot_{timestamp}.jpg"
        cv2.imwrite(str(image_path), frame)
        return image_path

    def _prepare_log(self) -> None:
        if self.log_file.exists():
            return
        with self.log_file.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "event", "x1", "y1", "x2", "y2", "score", "source", "image"])

    def _write_log(
        self,
        timestamp: str,
        event: str,
        bbox: tuple[int, int, int, int],
        score: float,
        source: str,
        image_path: Path,
    ) -> None:
        x1, y1, x2, y2 = bbox
        with self.log_file.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, event, x1, y1, x2, y2, f"{score:.3f}", source, image_path.name])
