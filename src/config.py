from pathlib import Path
import shutil
import sys


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_path(relative_path: str) -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return app_base_dir() / relative_path


BASE_DIR = app_base_dir()
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
UNKNOWN_FACES_DIR = BASE_DIR / "unknown_faces"
CAPTURES_DIR = BASE_DIR / "captures"
LOGS_DIR = BASE_DIR / "logs"

DEFAULT_YOLO_MODEL = str(resource_path("yolov8n.pt"))
DEFAULT_OPENFACE_MODEL = str(resource_path("models/openface.nn4.small2.v1.t7"))
DEFAULT_CONFIDENCE = 0.45
DEFAULT_ALERT_INTERVAL_SECONDS = 8
# --- Ngưỡng nhận diện (đã được tối ưu hóa cân bằng giữa độ chính xác và độ nhạy) ---
# face_recognition: distance <= threshold mới nhận là người quen (tăng nhẹ từ 0.42 → 0.48)
DEFAULT_FACE_TOLERANCE = 0.48
# OpenCV fallback: score >= threshold mới nhận là người quen (giảm từ 0.72 → 0.65)
DEFAULT_FALLBACK_THRESHOLD = 0.65
# LBPH: score phải >= 0.78 mới nhận là người quen (tăng nhẹ độ nhạy từ 20.0 → 26.0)
DEFAULT_LBPH_THRESHOLD = 26.0
# Đặt bằng THRESHOLD: xóa hoàn toàn vùng borderline, không có candidate
DEFAULT_LBPH_CANDIDATE_THRESHOLD = 28.0
# OpenFace embedding: distance <= threshold (nâng từ 0.18 → 0.23 để nhận diện tốt hơn)
DEFAULT_EMBEDDING_THRESHOLD = 0.23
# Profile được tính tự động, nhưng giới hạn trên là embedding_threshold
DEFAULT_PROFILE_STD_MULTIPLIER = 2.0
DEFAULT_PROFILE_MIN_THRESHOLD = 0.18
DEFAULT_PROFILE_MARGIN = 0.03
# Yêu cầu ít nhất 3 ảnh mẫu khớp trước khi nhận là người quen (giảm từ 4 → 3)
DEFAULT_EMBEDDING_MIN_SUPPORT = 3
DEFAULT_EMBEDDING_TOPK = 3
DEFAULT_MIN_FACE_SIZE = 45
DEFAULT_MIN_FACE_SHARPNESS = 0.0
# Margin tối thiểu giữa người tốt nhất và người thứ 2 (giảm từ 0.08 → 0.05 để tăng độ nhạy)
DEFAULT_UNKNOWN_MARGIN = 0.05
# Bỏ phiếu: cần nhiều frame khớp hơn trước khi xác nhận người quen
DEFAULT_RECOGNITION_VOTE_WINDOW = 9
DEFAULT_RECOGNITION_VOTE_MIN = 5
DEFAULT_RECOGNITION_CANDIDATE_VOTE_MIN = 5


def ensure_project_dirs() -> None:
    for directory in (KNOWN_FACES_DIR, UNKNOWN_FACES_DIR, CAPTURES_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    seed_known_faces()


def seed_known_faces() -> None:
    if not getattr(sys, "frozen", False):
        return
    bundled_known_faces = resource_path("known_faces")
    if not bundled_known_faces.exists() or bundled_known_faces == KNOWN_FACES_DIR:
        return
    if any(KNOWN_FACES_DIR.iterdir()):
        return

    for source_path in bundled_known_faces.rglob("*"):
        relative_path = source_path.relative_to(bundled_known_faces)
        target_path = KNOWN_FACES_DIR / relative_path
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
