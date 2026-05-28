from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
import sys
import threading
import time

import cv2

from src.alert_logger import AlertLogger
from src.camera_utils import AUTO_CAMERA_SOURCE, open_camera_source
from src.config import (
    CAPTURES_DIR,
    DEFAULT_ALERT_INTERVAL_SECONDS,
    DEFAULT_CONFIDENCE,
    DEFAULT_EMBEDDING_THRESHOLD,
    DEFAULT_FACE_TOLERANCE,
    DEFAULT_FALLBACK_THRESHOLD,
    DEFAULT_LBPH_CANDIDATE_THRESHOLD,
    DEFAULT_LBPH_THRESHOLD,
    DEFAULT_MIN_FACE_SHARPNESS,
    DEFAULT_MIN_FACE_SIZE,
    DEFAULT_RECOGNITION_CANDIDATE_VOTE_MIN,
    DEFAULT_RECOGNITION_VOTE_MIN,
    DEFAULT_RECOGNITION_VOTE_WINDOW,
    DEFAULT_UNKNOWN_MARGIN,
    DEFAULT_OPENFACE_MODEL,
    DEFAULT_PROFILE_MARGIN,
    DEFAULT_PROFILE_MIN_THRESHOLD,
    DEFAULT_PROFILE_STD_MULTIPLIER,
    DEFAULT_EMBEDDING_MIN_SUPPORT,
    DEFAULT_EMBEDDING_TOPK,
    DEFAULT_YOLO_MODEL,
    KNOWN_FACES_DIR,
    LOGS_DIR,
    UNKNOWN_FACES_DIR,
    ensure_project_dirs,
)
from src.face_recognizer import FaceRecognizer, RecognitionResult
from src.perf_monitor import PerfSampler, PerfSnapshot
from src.yolo_detector import Detection, YoloDetector


BLOCKING_FACE_REASONS = {"no_face", "no_face_features", "low_quality", "no_encoding", "no_embedding"}
UNKNOWN_PENDING_REASONS = {"warming_up", "ambiguous_match", *BLOCKING_FACE_REASONS}


@dataclass
class RecognitionTrack:
    bbox: tuple[int, int, int, int]
    history: deque[RecognitionResult] = field(default_factory=deque)
    last_seen: int = 0


@dataclass(frozen=True)
class FaceObservation:
    detection: Detection | None
    face_box: tuple[int, int, int, int]
    recognition_box: tuple[int, int, int, int]
    recognition_crop: object
    source: str


@dataclass(frozen=True)
class DisplayAnnotation:
    bbox: tuple[int, int, int, int]
    text: str
    color: tuple[int, int, int]


@dataclass
class ProcessingState:
    annotations: list[DisplayAnnotation] = field(default_factory=list)
    stranger_detected: bool = False
    stranger_bbox: tuple[int, int, int, int] | None = None
    stranger_score: float = 0.0
    updated_at: float = 0.0
    busy: bool = False


class RecognitionVoter:
    def __init__(
        self,
        window_size: int = DEFAULT_RECOGNITION_VOTE_WINDOW,
        min_votes: int = DEFAULT_RECOGNITION_VOTE_MIN,
        candidate_min_votes: int = DEFAULT_RECOGNITION_CANDIDATE_VOTE_MIN,
        iou_threshold: float = 0.25,
        max_missed_updates: int = 18,
    ) -> None:
        self.window_size = window_size
        self.min_votes = min_votes
        self.candidate_min_votes = candidate_min_votes
        self.iou_threshold = iou_threshold
        self.max_missed_updates = max_missed_updates
        self._tracks: list[RecognitionTrack] = []
        self._update_index = 0

    def update(self, bbox: tuple[int, int, int, int], result: RecognitionResult) -> RecognitionResult:
        self._update_index += 1
        self._tracks = [
            track
            for track in self._tracks
            if self._update_index - track.last_seen <= self.max_missed_updates
        ]

        track = self._find_track(bbox)
        if track is None:
            track = RecognitionTrack(bbox=bbox, history=deque(maxlen=self.window_size), last_seen=self._update_index)
            self._tracks.append(track)

        track.bbox = bbox
        track.last_seen = self._update_index
        track.history.append(result)
        return self._stable_result(track)

    def _find_track(self, bbox: tuple[int, int, int, int]) -> RecognitionTrack | None:
        best_track = None
        best_score = 0.0
        for track in self._tracks:
            score = bbox_iou(bbox, track.bbox)
            if score > best_score:
                best_score = score
                best_track = track
        return best_track if best_score >= self.iou_threshold else None

    def _stable_result(self, track: RecognitionTrack) -> RecognitionResult:
        history = list(track.history)
        latest = history[-1]

        # 1. Ưu tiên kiểm tra biểu quyết lịch sử xem có người quen ổn định không trước.
        # Điều này giúp duy trì nhận diện khi người đó đã được xác nhận ở các frame trước,
        # tránh bị nhấp nháy (flickering) khi Haar Cascade trượt mắt trong 1 vài frame.
        known_names = [result.name for result in history if result.name]
        if known_names:
            name, votes = Counter(known_names).most_common(1)[0]
            if votes >= self.min_votes:
                voted_results = [result for result in history if result.name == name]
                score = sum(result.score for result in voted_results) / len(voted_results)
                distance_values = [result.distance for result in voted_results if result.distance is not None]
                distance = min(distance_values) if distance_values else None
                return RecognitionResult(name, score, history[-1].backend, True, distance, "voted_known")

        # 2. Nếu không đủ phiếu bầu người quen ổn định, lúc này mới áp dụng các rule chặn
        if latest.reason in BLOCKING_FACE_REASONS:
            return RecognitionResult(None, latest.score, latest.backend, False, latest.distance, latest.reason)

        recent_blocked = sum(1 for result in history[-self.min_votes :] if result.reason in BLOCKING_FACE_REASONS)
        if recent_blocked >= self.min_votes:
            return RecognitionResult(None, latest.score, latest.backend, False, latest.distance, "no_face")

        # ĐÃ XÓA: candidate_names voting (voted_borderline)
        # Lý do: cơ chế này cho phép kết quả 'hard_cap_exceeded' tích lũy
        # candidate_name qua nhiều frame và được nâng lên thành người quen.
        # Chỉ có name=not None (nhận diện trực tiếp) mới được bầu.

        unknown_votes = len(history) - len(known_names)
        if unknown_votes >= self.min_votes:
            unknown_results = [result for result in history if not result.name]
            score = sum(result.score for result in unknown_results) / len(unknown_results)
            return RecognitionResult(None, score, history[-1].backend, history[-1].face_found, history[-1].distance, "voted_unknown")

        return RecognitionResult(None, latest.score, latest.backend, latest.face_found, latest.distance, "warming_up")


@dataclass
class LivenessTrack:
    bbox: tuple[int, int, int, int]
    name: str
    last_face_center: tuple[float, float] | None = None
    movement_score: float = 0.0
    verified_until: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)


class MotionLivenessVerifier:
    def __init__(
        self,
        verified_seconds: float = 6.0,
        timeout_seconds: float = 4.0,
        required_movement: float = 0.010,
        iou_threshold: float = 0.25,
    ) -> None:
        self.verified_seconds = verified_seconds
        self.timeout_seconds = timeout_seconds
        self.required_movement = required_movement
        self.iou_threshold = iou_threshold
        self._tracks: list[LivenessTrack] = []

    def update(
        self,
        bbox: tuple[int, int, int, int],
        face_box: tuple[int, int, int, int] | None,
        name: str,
    ) -> str:
        now = time.monotonic()
        self._tracks = [track for track in self._tracks if now - track.last_seen <= self.timeout_seconds * 2]
        track = self._find_track(bbox, name)
        if track is None:
            track = LivenessTrack(bbox=bbox, name=name)
            self._tracks.append(track)

        track.bbox = bbox
        track.last_seen = now
        if now <= track.verified_until:
            return "verified"

        movement = self._face_movement(face_box, track)
        track.movement_score = max(0.0, (track.movement_score * 0.65) + movement)
        if track.movement_score >= self.required_movement:
            track.verified_until = now + self.verified_seconds
            track.movement_score = 0.0
            track.started_at = now
            return "verified"

        if now - track.started_at >= self.timeout_seconds:
            track.movement_score = 0.0
            track.started_at = now
            return "failed"

        return "pending"

    def _find_track(self, bbox: tuple[int, int, int, int], name: str) -> LivenessTrack | None:
        best_track = None
        best_score = 0.0
        for track in self._tracks:
            if track.name != name:
                continue
            score = bbox_iou(bbox, track.bbox)
            if score > best_score:
                best_score = score
                best_track = track
        return best_track if best_score >= self.iou_threshold else None

    def _face_movement(
        self,
        face_box: tuple[int, int, int, int] | None,
        track: LivenessTrack,
    ) -> float:
        if face_box is None:
            return 0.0
        x1, y1, x2, y2 = face_box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if track.last_face_center is None:
            track.last_face_center = center
            return 0.0
        dx = abs(center[0] - track.last_face_center[0]) / width
        dy = abs(center[1] - track.last_face_center[1]) / height
        track.last_face_center = center
        movement = dx + dy
        if movement > 0.18:
            return 0.0
        return movement


def bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def expand_box(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    padding_ratio: float = 0.18,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    pad = int(max(width, height) * padding_ratio)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(frame_shape[1], x2 + pad),
        min(frame_shape[0], y2 + pad),
    )


def upper_face_guess_box(
    person_box: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = person_box
    width = x2 - x1
    height = y2 - y1
    guess_width = int(width * 0.5)
    guess_height = min(int(height * 0.34), int(guess_width * 1.2))
    center_x = x1 + width // 2
    crop_y1 = y1 + int(height * 0.02)
    crop_x1 = center_x - guess_width // 2
    return (
        max(0, crop_x1),
        max(0, crop_y1),
        min(frame_shape[1], crop_x1 + guess_width),
        min(frame_shape[0], crop_y1 + guess_height),
    )


def crop_from_box(frame, box: tuple[int, int, int, int]):
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2]


def draw_box_label(frame, bbox: tuple[int, int, int, int], text: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
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


def draw_annotations(frame, annotations: list[DisplayAnnotation]) -> None:
    for annotation in annotations:
        draw_box_label(frame, annotation.bbox, annotation.text, annotation.color)


def face_xywh_to_xyxy(
    face_box: tuple[int, int, int, int],
    offset_x: int,
    offset_y: int,
) -> tuple[int, int, int, int]:
    fx, fy, fw, fh = face_box
    return offset_x + fx, offset_y + fy, offset_x + fx + fw, offset_y + fy + fh


def face_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_contains_point(box: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = box
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def associate_face_detection(
    face_box: tuple[int, int, int, int],
    detections: list[Detection],
) -> tuple[int | None, Detection | None]:
    if not detections:
        return None, None

    center = face_center(face_box)
    containing = [
        (index, detection)
        for index, detection in enumerate(detections)
        if box_contains_point(detection.bbox, center)
    ]
    if containing:
        return min(containing, key=lambda item: detection_area(item[1]))

    best_index = None
    best_detection = None
    best_score = 0.0
    for index, detection in enumerate(detections):
        score = bbox_iou(face_box, detection.bbox)
        if score > best_score:
            best_index = index
            best_detection = detection
            best_score = score
    if best_score > 0:
        return best_index, best_detection
    return None, None


def detection_area(detection: Detection) -> int:
    x1, y1, x2, y2 = detection.bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def is_background_scale_face(
    face_box: tuple[int, int, int, int],
    person_box: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    min_face_size: int,
    min_face_height_ratio: float,
    min_person_height_ratio: float,
) -> bool:
    frame_height = max(1, frame_shape[0])
    face_height = max(0, face_box[3] - face_box[1])
    person_height = max(0, person_box[3] - person_box[1])
    required_face_height = max(min_face_size, int(frame_height * min_face_height_ratio))
    required_person_height = int(frame_height * min_person_height_ratio)
    return face_height < required_face_height and person_height < required_person_height


def collect_face_observations(
    frame,
    detections: list[Detection],
    recognizer: FaceRecognizer,
    min_face_size: int,
    min_face_height_ratio: float,
    min_person_height_ratio: float,
) -> tuple[list[FaceObservation], set[int]]:
    observations: list[FaceObservation] = []

    for face_xywh in recognizer.find_face_boxes(frame):
        face_abs_box = face_xywh_to_xyxy(face_xywh, 0, 0)
        detection_index, detection = associate_face_detection(face_abs_box, detections)
        person_box = detection.bbox if detection is not None else face_abs_box
        if is_background_scale_face(
            face_abs_box,
            person_box,
            frame.shape,
            min_face_size,
            min_face_height_ratio,
            min_person_height_ratio,
        ):
            continue

        crop_box = expand_box(face_abs_box, frame.shape)
        observations.append(
            FaceObservation(
                detection=detection,
                face_box=face_abs_box,
                recognition_box=crop_box,
                recognition_crop=crop_from_box(frame, crop_box),
                source="full_frame",
            )
        )

    for detection_index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.bbox
        person_crop = frame[y1:y2, x1:x2]
        for face_box in recognizer.find_face_boxes(person_crop):
            face_abs_box = face_xywh_to_xyxy(face_box, x1, y1)
            if is_background_scale_face(
                face_abs_box,
                detection.bbox,
                frame.shape,
                min_face_size,
                min_face_height_ratio,
                min_person_height_ratio,
            ):
                continue

            crop_box = expand_box(face_abs_box, frame.shape)
            recognition_crop = crop_from_box(frame, crop_box)
            observations.append(
                FaceObservation(
                    detection=detection,
                    face_box=face_abs_box,
                    recognition_box=crop_box,
                    recognition_crop=recognition_crop,
                    source="haar_face",
                )
            )

    observations.sort(
        key=lambda item: (
            (item.face_box[2] - item.face_box[0]) * (item.face_box[3] - item.face_box[1]),
            item.detection.confidence if item.detection is not None else 0.0,
        ),
        reverse=True,
    )
    deduped: list[FaceObservation] = []
    for observation in observations:
        if all(bbox_iou(observation.face_box, kept.face_box) < 0.45 for kept in deduped):
            deduped.append(observation)
    detections_with_faces = {
        index
        for index, detection in enumerate(detections)
        if any(observation.detection is detection for observation in deduped)
    }
    return deduped, detections_with_faces


def process_security_frame(
    frame,
    source: str,
    detector: YoloDetector,
    recognizer: FaceRecognizer,
    voter: RecognitionVoter,
    liveness: MotionLivenessVerifier,
    alerts: AlertLogger,
    args: argparse.Namespace,
) -> ProcessingState:
    detections = detector.detect(frame)
    observations, detections_with_faces = collect_face_observations(
        frame,
        detections,
        recognizer,
        args.min_face_size,
        args.min_face_height_ratio,
        args.min_person_height_ratio,
    )
    annotations: list[DisplayAnnotation] = []
    stranger_detected = False
    stranger_bbox = None
    stranger_score = 0.0

    for detection_index, detection in enumerate(detections):
        if detection_index not in detections_with_faces:
            annotations.append(DisplayAnnotation(detection.bbox, "DANG XAC NHAN", (0, 180, 255)))

    for observation in observations:
        result = recognizer.recognize(observation.recognition_crop, assume_face=True)
        result = voter.update(observation.face_box, result)
        liveness_status = ""
        if result.is_known:
            liveness_status = liveness.update(observation.face_box, observation.face_box, result.name or "")
        if args.debug_recognition:
            log_message(
                "recognition "
                + str(
                    {
                        "name": result.name,
                        "score": round(result.score, 3),
                        "distance": None if result.distance is None else round(result.distance, 2),
                        "face_found": result.face_found,
                        "reason": result.reason,
                        "candidate": result.candidate_name,
                        "liveness": liveness_status,
                        "face_box": observation.face_box,
                        "person_box": None if observation.detection is None else observation.detection.bbox,
                        "crop": (observation.source, observation.recognition_crop.shape[:2]),
                    }
                )
            )

        if result.is_known:
            if liveness_status == "verified":
                text = f"{result.name} {result.score:.2f}"
                color = (40, 190, 80)
            elif liveness_status == "failed":
                text = "NGHI GIA MAO"
                color = (20, 20, 230)
                stranger_detected = True
                if result.score >= stranger_score:
                    stranger_bbox = observation.recognition_box
                    stranger_score = result.score
            else:
                text = "GIU MAT TU NHIEN"
                color = (0, 180, 255)
        elif result.reason in UNKNOWN_PENDING_REASONS:
            text = f"{result.candidate_name} DANG XAC NHAN" if result.candidate_name else "DANG XAC NHAN"
            color = (0, 180, 255)
        else:
            text = f"NGUOI LA {result.score:.2f}"
            color = (20, 20, 230)
            stranger_detected = True
            if result.score >= stranger_score:
                stranger_bbox = observation.recognition_box
                stranger_score = result.score

        annotations.append(DisplayAnnotation(observation.face_box, text, color))

    saved_frame = frame.copy()
    draw_annotations(saved_frame, annotations)
    saved_path = alerts.update_stranger_presence(
        saved_frame,
        stranger_detected,
        stranger_bbox,
        stranger_score,
        source,
    )
    if saved_path:
        log_message(f"Canh bao nguoi la qua {args.stranger_hold_seconds}s: {saved_path}")

    return ProcessingState(
        annotations=annotations,
        stranger_detected=stranger_detected,
        stranger_bbox=stranger_bbox,
        stranger_score=stranger_score,
        updated_at=time.monotonic(),
    )


def log_message(message: str) -> None:
    try:
        if sys.stdout is not None:
            print(message)
    except Exception:
        pass
    try:
        ensure_project_dirs()
        with (LOGS_DIR / "app.log").open("a", encoding="utf-8") as file:
            file.write(message + "\n")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face recognition and stranger detection with YOLO.")
    parser.add_argument("--source", default=AUTO_CAMERA_SOURCE, help="Webcam index, video path, or RTSP/IP camera URL.")
    parser.add_argument("--model", default=DEFAULT_YOLO_MODEL, help="YOLO model path/name.")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="YOLO confidence threshold.")
    parser.add_argument("--known-dir", type=Path, default=KNOWN_FACES_DIR, help="Known faces directory.")
    parser.add_argument("--alert-interval", type=int, default=DEFAULT_ALERT_INTERVAL_SECONDS)
    parser.add_argument("--stranger-hold-seconds", type=int, default=10, help="Save stranger only after continuous detection.")
    parser.add_argument("--face-tolerance", type=float, default=DEFAULT_FACE_TOLERANCE)
    parser.add_argument("--fallback-threshold", type=float, default=DEFAULT_FALLBACK_THRESHOLD)
    parser.add_argument("--lbph-threshold", type=float, default=DEFAULT_LBPH_THRESHOLD, help="Lower is stricter for OpenCV LBPH.")
    parser.add_argument("--lbph-candidate-threshold", type=float, default=DEFAULT_LBPH_CANDIDATE_THRESHOLD, help="Borderline LBPH matches must pass repeated-frame voting.")
    parser.add_argument("--min-face-size", type=int, default=DEFAULT_MIN_FACE_SIZE, help="Ignore faces smaller than this many pixels.")
    parser.add_argument("--min-face-height-ratio", type=float, default=0.055, help="Ignore distant background faces below this frame-height ratio when the person box is also small.")
    parser.add_argument("--min-person-height-ratio", type=float, default=0.18, help="Treat small person boxes as background unless the face is still clear enough.")
    parser.add_argument("--min-face-sharpness", type=float, default=DEFAULT_MIN_FACE_SHARPNESS, help="Ignore blurry face crops below this Laplacian variance.")
    parser.add_argument("--unknown-margin", type=float, default=DEFAULT_UNKNOWN_MARGIN, help="Required gap between best and runner-up match.")
    parser.add_argument("--vote-window", type=int, default=DEFAULT_RECOGNITION_VOTE_WINDOW, help="Number of recent frames used for recognition voting.")
    parser.add_argument("--vote-min", type=int, default=DEFAULT_RECOGNITION_VOTE_MIN, help="Minimum votes needed before a known/unknown result is accepted.")
    parser.add_argument("--candidate-vote-min", type=int, default=DEFAULT_RECOGNITION_CANDIDATE_VOTE_MIN, help="Minimum stable borderline votes before accepting a known person.")
    parser.add_argument("--process-every", type=int, default=1, help="Run YOLO/face recognition every N camera frames (dùng khi --process-mode=manual).")
    parser.add_argument("--process-mode", choices=["auto", "manual"], default="auto",
                        help="'auto': tu dong dieu chinh process_every theo CPU/GPU load. 'manual': dung gia tri --process-every co dinh.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Thiet bi xu ly YOLO: 'auto' tu detect GPU, 'cpu', 'cuda'.")
    parser.add_argument("--debug-recognition", action="store_true", help="Print face recognition score details.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Performance HUD
# ---------------------------------------------------------------------------

_PROC_EVERY_MIN = 1
_PROC_EVERY_MAX = 6
# --- Ngưỡng CPU chế độ tự động (chỉ áp dụng khi không có GPU) ---
_CPU_HIGH = 80.0         # % CPU — tăng process_every
_CPU_LOW  = 45.0         # % CPU — giảm process_every
_INFER_HIGH_MS = 100.0   # ms inference trên CPU — tăng process_every
_INFER_LOW_MS  = 50.0    # ms inference trên CPU — giảm process_every
# --- Ngưỡng GPU (inference GPU nhanh hơn nhiều) ---
_GPU_INFER_HIGH_MS = 40.0  # ms inference trên GPU — tăng process_every
_GPU_INFER_LOW_MS  = 20.0  # ms inference trên GPU — giảm process_every
_GPU_LOAD_HIGH = 90.0      # % GPU load — tăng process_every


def _model_short_name(model_path: str) -> str:
    """Rút gọn tên model để hiển thị, ví dụ 'yolov8n' từ đường dẫn dài."""
    stem = Path(model_path).stem  # e.g. 'yolov8n'
    return stem


def draw_perf_hud(
    frame,
    snap: PerfSnapshot,
    detector: YoloDetector,
    fps: float,
    process_every: int,
    process_mode: str,
    model_name: str,
) -> None:
    """Vẽ overlay HUD hiệu suất ở góc trên-phải của frame."""
    h, w = frame.shape[:2]

    # --- Chuẩn bị các dòng text ---
    gpu = snap.primary_gpu
    if gpu is not None:
        vram_used = gpu.mem_used_mb / 1024
        vram_total = gpu.mem_total_mb / 1024
        gpu_line = f"GPU {gpu.load_percent:.0f}%  VRAM {vram_used:.1f}/{vram_total:.1f}GB  {gpu.name}"
    else:
        gpu_line = "GPU: Khong co"

    mode_tag = "auto" if process_mode == "auto" else "manual"
    lines = [
        f"CPU {snap.cpu_percent:.0f}%  RAM {snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f}GB",
        gpu_line,
        f"Model: {model_name}  Dev: {detector.device_name}  Infer: {detector.last_inference_ms:.0f}ms",
        f"FPS: {fps:.1f}  ProcEvery: {process_every} ({mode_tag})",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    thickness = 1
    line_h = 20
    pad_x, pad_y = 8, 6
    max_line_w = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
    box_w = max_line_w + pad_x * 2
    box_h = line_h * len(lines) + pad_y * 2
    x0 = w - box_w - 8
    y0 = 8

    # Nền tối mờ
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Viền mỏng
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (60, 60, 60), 1)

    # Text từng dòng
    colors = [
        (180, 220, 255),   # CPU/RAM — xanh nhạt
        (120, 255, 180) if gpu is not None else (120, 120, 120),  # GPU — xanh lá hoặc xám
        (255, 220, 100),   # Model/Device/Infer — vàng
        (200, 200, 200),   # FPS/ProcEvery — trắng xám
    ]
    for index, (line, color) in enumerate(zip(lines, colors)):
        ty = y0 + pad_y + line_h * index + 14
        cv2.putText(frame, line, (x0 + pad_x, ty), font, font_scale, color, thickness, cv2.LINE_AA)


def _adaptive_process_every(
    current: int,
    snap: PerfSnapshot,
    infer_ms: float,
    using_gpu: bool,
) -> int:
    """Tự động điều chỉnh process_every dựa trên CPU/GPU load và thời gian inference."""
    if using_gpu and snap.primary_gpu is not None:
        # Chế độ GPU: dùng ngưỡng inference GPU và GPU load
        gpu_load = snap.primary_gpu.load_percent
        if infer_ms > _GPU_INFER_HIGH_MS or gpu_load > _GPU_LOAD_HIGH:
            return min(current + 1, _PROC_EVERY_MAX)
        if infer_ms < _GPU_INFER_LOW_MS and gpu_load < 70.0 and current > _PROC_EVERY_MIN:
            return max(current - 1, _PROC_EVERY_MIN)
        return current
    else:
        # Chế độ CPU: dùng ngưỡng CPU%
        cpu = snap.cpu_percent
        if cpu > _CPU_HIGH or infer_ms > _INFER_HIGH_MS:
            return min(current + 1, _PROC_EVERY_MAX)
        if cpu < _CPU_LOW and infer_ms < _INFER_LOW_MS and current > _PROC_EVERY_MIN:
            return max(current - 1, _PROC_EVERY_MIN)
        return current


def show_camera_error(message: str) -> None:
    log_message(message)
    try:
        from tkinter import messagebox

        messagebox.showerror("Khong mo duoc camera", message)
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    ensure_project_dirs()

    recognizer = FaceRecognizer(
        args.known_dir,
        tolerance=args.face_tolerance,
        fallback_threshold=args.fallback_threshold,
        lbph_threshold=args.lbph_threshold,
        lbph_candidate_threshold=args.lbph_candidate_threshold,
        embedding_model_path=DEFAULT_OPENFACE_MODEL,
        embedding_threshold=DEFAULT_EMBEDDING_THRESHOLD,
        unknown_dir=UNKNOWN_FACES_DIR,
        profile_std_multiplier=DEFAULT_PROFILE_STD_MULTIPLIER,
        profile_min_threshold=DEFAULT_PROFILE_MIN_THRESHOLD,
        profile_margin=DEFAULT_PROFILE_MARGIN,
        embedding_min_support=DEFAULT_EMBEDDING_MIN_SUPPORT,
        embedding_topk=DEFAULT_EMBEDDING_TOPK,
        min_face_size=args.min_face_size,
        min_sharpness=args.min_face_sharpness,
        unknown_margin=args.unknown_margin,
    )
    voter = RecognitionVoter(
        window_size=args.vote_window,
        min_votes=args.vote_min,
        candidate_min_votes=args.candidate_vote_min,
    )
    liveness = MotionLivenessVerifier()
    detector = YoloDetector(
        args.model,
        confidence=args.confidence,
        target_labels=["person"],
        device=args.device,
    )
    alerts = AlertLogger(
        CAPTURES_DIR,
        LOGS_DIR,
        interval_seconds=args.alert_interval,
        stranger_hold_seconds=args.stranger_hold_seconds,
    )

    # --- Khởi động Performance Sampler ---
    perf_sampler = PerfSampler(interval=0.5)
    perf_sampler.start()

    camera_result = open_camera_source(args.source)
    camera = camera_result.camera
    if camera is None:
        perf_sampler.stop()
        show_camera_error(camera_result.error or f"Khong mo duoc nguon camera/video: {args.source}")
        return 1

    model_name = _model_short_name(args.model)
    log_message(f"YOLO model: {args.model}  |  Device: {detector.device_name}  |  Process mode: {args.process_mode}")
    log_message(f"Camera source: {camera_result.source} | Backend: {camera_result.backend_name}")
    log_message(f"Face backend: {recognizer.backend} | Known faces: {recognizer.known_count}")
    if recognizer.known_count == 0:
        log_message("Canh bao: Chua load duoc anh nguoi quen nao trong known_faces/. Hay them nguoi bang admin_app.py.")

    using_gpu = "CUDA" in detector.device_name.upper() or "GPU" in detector.device_name.upper()
    process_every = max(1, args.process_every)
    process_mode = args.process_mode
    if using_gpu and process_mode == "auto":
        process_every = 1

    log_message(f"Che do muot: hien thi moi frame, xu ly nhan dien moi {process_every} frame (mode={process_mode}, GPU={using_gpu}).")
    log_message("Nhan q: thoat | s: snapshot | m: tang process_every | n: giam process_every")

    window_name = "Security monitoring - YOLO stranger detection"
    blank_frame_count = 0
    frame_index = 0
    latest_display_frame = None

    # --- FPS tracking ---
    fps_counter = 0
    fps_value = 0.0
    fps_last_time = time.monotonic()

    # --- Adaptive process_every: chỉ điều chỉnh mỗi 2s ---
    last_adapt_time = time.monotonic()

    state = ProcessingState()
    state_lock = threading.Lock()
    worker_condition = threading.Condition()
    stop_worker = False
    pending_frame = None
    pending_source = str(args.source)

    def submit_frame_for_processing(frame_to_process) -> None:
        nonlocal pending_frame, pending_source
        with worker_condition:
            pending_frame = frame_to_process.copy()
            pending_source = str(args.source)
            worker_condition.notify()

    def processing_worker() -> None:
        nonlocal pending_frame, pending_source, stop_worker, state
        while True:
            with worker_condition:
                while pending_frame is None and not stop_worker:
                    worker_condition.wait()
                if stop_worker:
                    return
                frame_to_process = pending_frame
                source = pending_source
                pending_frame = None
            with state_lock:
                state.busy = True
            try:
                next_state = process_security_frame(
                    frame_to_process,
                    source,
                    detector,
                    recognizer,
                    voter,
                    liveness,
                    alerts,
                    args,
                )
            except Exception as exc:
                log_message(f"Loi xu ly frame giam sat: {exc}")
                next_state = ProcessingState(updated_at=time.monotonic())
            with state_lock:
                next_state.busy = False
                state = next_state

    worker = threading.Thread(target=processing_worker, daemon=True)
    worker.start()

    while True:
        ok, frame = camera.read()
        if not ok:
            log_message("Camera khong doc duoc khung hinh moi.")
            break
        frame_index += 1

        # --- FPS ---
        fps_counter += 1
        now = time.monotonic()
        elapsed_fps = now - fps_last_time
        if elapsed_fps >= 1.0:
            fps_value = fps_counter / elapsed_fps
            fps_counter = 0
            fps_last_time = now

        # --- Adaptive process_every (chế độ auto) ---
        if process_mode == "auto" and now - last_adapt_time >= 2.0:
            snap = perf_sampler.snapshot()
            process_every = _adaptive_process_every(process_every, snap, detector.last_inference_ms, using_gpu)
            last_adapt_time = now

        if float(frame.std()) < 1.0:
            blank_frame_count += 1
            if blank_frame_count == 1 or blank_frame_count % 60 == 0:
                log_message(
                    "Camera dang tra ve khung hinh den. Hay mo app webcam tren dien thoai, "
                    "chon camera khac hoac bam Quet lai trong man hinh quan ly."
                )
            cv2.putText(
                frame,
                "Camera dang tra ve khung hinh den",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1 or key == ord("q"):
                break
            continue
        blank_frame_count = 0

        if frame_index % process_every == 0:
            submit_frame_for_processing(frame)

        display_frame = frame.copy()
        with state_lock:
            annotations = list(state.annotations)
            is_processing = state.busy
        draw_annotations(display_frame, annotations)
        if is_processing:
            cv2.putText(
                display_frame,
                "Dang xu ly...",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

        # --- Vẽ HUD hiệu suất ---
        perf_snap = perf_sampler.snapshot()
        draw_perf_hud(
            display_frame,
            perf_snap,
            detector,
            fps_value,
            process_every,
            process_mode,
            model_name,
        )

        latest_display_frame = display_frame
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("q"):
            break
        if key == ord("s"):
            snapshot_frame = latest_display_frame if latest_display_frame is not None else frame
            snapshot = alerts.save_snapshot(snapshot_frame)
            log_message(f"Da luu snapshot: {snapshot}")
        # --- Phím m/n: điều chỉnh process_every thủ công ---
        if key == ord("m"):
            process_every = min(process_every + 1, _PROC_EVERY_MAX)
            process_mode = "manual"
            log_message(f"[Manual] process_every -> {process_every}")
        if key == ord("n"):
            process_every = max(process_every - 1, _PROC_EVERY_MIN)
            process_mode = "manual"
            log_message(f"[Manual] process_every -> {process_every}")

    with worker_condition:
        stop_worker = True
        worker_condition.notify()
    worker.join(timeout=2.0)
    perf_sampler.stop()
    camera.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
