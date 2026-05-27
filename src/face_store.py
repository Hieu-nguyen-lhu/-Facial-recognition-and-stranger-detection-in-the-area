from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import cv2

from src.config import KNOWN_FACES_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
POSES = [
    ("front", "Chính diện"),
    ("left", "Xoay trái"),
    ("right", "Xoay phải"),
]


@dataclass(frozen=True)
class KnownPerson:
    name: str
    folder: Path
    images: list[Path]


def safe_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    return normalized.strip("_") or "known_person"


def display_name_from_path(path: Path) -> str:
    if path.is_dir():
        return path.name.replace("_", " ").strip()
    if path.parent != KNOWN_FACES_DIR:
        return path.parent.name.replace("_", " ").strip()
    return path.stem.replace("_", " ").strip()


def person_folder(name: str, known_dir: Path = KNOWN_FACES_DIR) -> Path:
    return known_dir / safe_name(name)


def iter_known_face_images(known_dir: Path = KNOWN_FACES_DIR) -> list[tuple[str, Path]]:
    if not known_dir.exists():
        return []

    images: list[tuple[str, Path]] = []
    for path in sorted(known_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append((path.stem.replace("_", " ").strip(), path))
            continue
        if not path.is_dir():
            continue
        name = path.name.replace("_", " ").strip()
        for image_path in sorted(path.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append((name, image_path))
    return images


def list_known_people(known_dir: Path = KNOWN_FACES_DIR) -> list[KnownPerson]:
    if not known_dir.exists():
        return []

    people: list[KnownPerson] = []
    for path in sorted(known_dir.iterdir()):
        if path.is_dir():
            images = [
                image_path
                for image_path in sorted(path.iterdir())
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            people.append(KnownPerson(display_name_from_path(path), path, images))
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            people.append(KnownPerson(display_name_from_path(path), path, [path]))
    return people


def list_known_faces(known_dir: Path = KNOWN_FACES_DIR) -> list[Path]:
    return [image_path for _name, image_path in iter_known_face_images(known_dir)]


def save_registered_face(
    name: str,
    image_bgr,
    known_dir: Path = KNOWN_FACES_DIR,
    pose: str | None = None,
) -> Path:
    known_dir.mkdir(parents=True, exist_ok=True)
    face = crop_largest_face(image_bgr)
    if face is None:
        face = image_bgr

    folder = person_folder(name, known_dir)
    folder.mkdir(parents=True, exist_ok=True)

    if pose:
        file_path = next_available_pose_path(folder, pose)
    else:
        file_path = folder / "front.jpg"
        counter = 2
        while file_path.exists():
            file_path = folder / f"extra_{counter}.jpg"
            counter += 1

    cv2.imwrite(str(file_path), face)
    return file_path


def save_pose_set(
    name: str,
    pose_images: dict[str, object],
    known_dir: Path = KNOWN_FACES_DIR,
) -> list[Path]:
    saved_paths: list[Path] = []
    for pose, image_or_images in pose_images.items():
        images = image_or_images if isinstance(image_or_images, list) else [image_or_images]
        for image_bgr in images:
            saved_paths.append(save_registered_face(name, image_bgr, known_dir, pose=pose))
    return saved_paths


def next_available_pose_path(folder: Path, pose: str) -> Path:
    base_path = folder / f"{pose}.jpg"
    if not base_path.exists():
        return base_path
    counter = 2
    while True:
        file_path = folder / f"{pose}_{counter}.jpg"
        if not file_path.exists():
            return file_path
        counter += 1


def crop_largest_face(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    boxes = [tuple(map(int, face)) for face in cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)]
    boxes.extend(
        tuple(map(int, face))
        for face in profile_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4)
    )
    flipped = cv2.flip(gray, 1)
    width = gray.shape[1]
    for x, y, w, h in profile_cascade.detectMultiScale(flipped, scaleFactor=1.08, minNeighbors=4):
        boxes.append((int(width - x - w), int(y), int(w), int(h)))

    if not boxes:
        return None
    x, y, w, h = max(boxes, key=lambda item: item[2] * item[3])
    pad = int(max(w, h) * 0.18)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_bgr.shape[1], x + w + pad)
    y2 = min(image_bgr.shape[0], y + h + pad)
    return image_bgr[y1:y2, x1:x2]
