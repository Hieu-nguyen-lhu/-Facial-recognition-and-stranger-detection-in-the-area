from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2


AUTO_CAMERA_SOURCE = "auto"
DEFAULT_CAMERA_INDEXES = tuple(range(10))
CAMERA_WARMUP_READS = 8
MIN_FRAME_STDDEV = 1.0


@dataclass(frozen=True)
class CameraOpenResult:
    camera: cv2.VideoCapture | None
    source: int | str | None
    backend_name: str
    error: str | None = None


@dataclass(frozen=True)
class CameraDevice:
    source: int
    label: str
    backend_name: str


def normalize_camera_source(source: str | int) -> int | str:
    if isinstance(source, int):
        return source
    source = str(source).strip()
    if source.lower() == AUTO_CAMERA_SOURCE:
        return AUTO_CAMERA_SOURCE
    return int(source) if source.isdigit() else source


def open_camera_source(source: str | int = AUTO_CAMERA_SOURCE) -> CameraOpenResult:
    normalized = normalize_camera_source(source)
    if normalized == AUTO_CAMERA_SOURCE:
        return open_first_available_camera()
    return _try_open_source(normalized)


def open_first_available_camera(indexes: Iterable[int] = DEFAULT_CAMERA_INDEXES) -> CameraOpenResult:
    failures: list[str] = []
    for index in indexes:
        result = _try_open_source(index)
        if result.camera is not None:
            return result
        if result.error:
            failures.append(result.error)

    checked = ", ".join(str(index) for index in indexes)
    detail = "\n".join(failures[:6])
    if detail:
        detail = "\n\nChi tiet thu nghiem:\n" + detail
    return CameraOpenResult(
        None,
        None,
        "",
        (
            f"Khong mo duoc camera nao trong cac index: {checked}."
            "\nHay kiem tra Camera privacy cua Windows, dong Zoom/Teams/OBS/DroidCam neu dang giu camera,"
            " hoac cam lai webcam roi thu lai."
            f"{detail}"
        ),
    )


def list_available_cameras(indexes: Iterable[int] = DEFAULT_CAMERA_INDEXES) -> list[CameraDevice]:
    devices: list[CameraDevice] = []
    for index in indexes:
        result = _try_open_source(index)
        if result.camera is None:
            continue
        result.camera.release()
        devices.append(
            CameraDevice(
                source=index,
                label=f"Camera {index} ({result.backend_name})",
                backend_name=result.backend_name,
            )
        )
    return devices


def _try_open_source(source: int | str) -> CameraOpenResult:
    backends: list[tuple[int | None, str]]
    if isinstance(source, int):
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (None, "OpenCV default"),
        ]
    else:
        backends = [(None, "OpenCV default")]

    errors: list[str] = []
    for backend, backend_name in backends:
        camera = cv2.VideoCapture(source) if backend is None else cv2.VideoCapture(source, backend)
        if camera.isOpened():
            ok, frame = _read_first_usable_frame(camera)
            if ok:
                return CameraOpenResult(camera, source, backend_name)
            errors.append(f"{source} bang {backend_name}: mo duoc nhung khung hinh den/khong doc duoc")
        else:
            errors.append(f"{source} bang {backend_name}: khong mo duoc")
        camera.release()

    return CameraOpenResult(None, source, "", "; ".join(errors))


def _read_first_usable_frame(camera: cv2.VideoCapture) -> tuple[bool, object | None]:
    last_frame = None
    for _ in range(CAMERA_WARMUP_READS):
        ok, frame = camera.read()
        if not ok or frame is None:
            continue
        last_frame = frame
        if _is_usable_frame(frame):
            return True, frame
    return False, last_frame


def _is_usable_frame(frame) -> bool:
    if frame.size == 0:
        return False
    return float(frame.std()) >= MIN_FRAME_STDDEV
