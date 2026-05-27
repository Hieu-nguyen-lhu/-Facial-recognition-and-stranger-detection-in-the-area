from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.config import KNOWN_FACES_DIR, ensure_project_dirs
from src.face_store import save_registered_face


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a known face.")
    parser.add_argument("--name", required=True, help="Person name.")
    parser.add_argument("--image", type=Path, help="Path to an existing face image.")
    parser.add_argument("--camera", type=int, help="Webcam index for live capture.")
    return parser.parse_args()


def register_from_image(name: str, image_path: Path) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Khong doc duoc anh: {image_path}")
        return 1
    saved = save_registered_face(name, image, KNOWN_FACES_DIR)
    print(f"Da dang ky khuon mat: {saved}")
    return 0


def register_from_camera(name: str, camera_index: int) -> int:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Khong mo duoc webcam index {camera_index}")
        return 1

    print("Nhan c de chup dang ky, q de thoat.")
    saved_path: Path | None = None
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        cv2.imshow("Register known face", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            saved_path = save_registered_face(name, frame, KNOWN_FACES_DIR)
            print(f"Da dang ky khuon mat: {saved_path}")
            break

    camera.release()
    cv2.destroyAllWindows()
    return 0 if saved_path else 1


def main() -> int:
    args = parse_args()
    ensure_project_dirs()
    if args.image is None and args.camera is None:
        print("Can truyen --image hoac --camera.")
        return 1
    if args.image is not None:
        return register_from_image(args.name, args.image)
    return register_from_camera(args.name, args.camera)


if __name__ == "__main__":
    raise SystemExit(main())
