from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class RecognitionResult:
    name: str | None
    score: float
    backend: str
    face_found: bool
    distance: float | None = None
    reason: str = ""
    candidate_name: str | None = None

    @property
    def is_known(self) -> bool:
        return self.name is not None


# Ngưỡng cứng tuyệt đối — không thể bị override bởi bất kỳ threshold nào
# Khoảng cách > giá trị này sẽ KHÔNG BAO GIỜ được nhận là người quen
EMBEDDING_HARD_CAP_DISTANCE   = 0.25   # score < 0.75 → người lạ (embedding - nâng từ 0.20 lên 0.25)
FACE_RECOG_HARD_CAP_DISTANCE   = 0.48   # score < 0.52 → người lạ (face_recognition - nâng từ 0.22 lên 0.48)
# LBPH: distance > cap → người lạ. score = 1 - distance/120
LBPH_HARD_CAP_DISTANCE         = 32.0   # score < 0.73 → người lạ (LBPH - nâng từ 24.0 lên 32.0)


class FaceRecognizer:
    def __init__(
        self,
        known_dir: Path,
        tolerance: float = 0.42,
        fallback_threshold: float = 0.72,
        lbph_threshold: float = 26.0,
        lbph_candidate_threshold: float = 30.0,
        embedding_model_path: str | Path | None = None,
        embedding_threshold: float = 0.18,
        unknown_dir: Path | None = None,
        profile_std_multiplier: float = 2.0,
        profile_min_threshold: float = 0.08,
        profile_margin: float = 0.03,
        embedding_min_support: int = 4,
        embedding_topk: int = 3,
        min_face_size: int = 45,
        min_sharpness: float = 0.0,
        unknown_margin: float = 0.12,
        crop_padding: float = 0.18,
    ) -> None:
        self.known_dir = Path(known_dir)
        self.unknown_dir = Path(unknown_dir) if unknown_dir else None
        self.tolerance = tolerance
        self.fallback_threshold = fallback_threshold
        self.lbph_threshold = lbph_threshold
        self.lbph_candidate_threshold = lbph_candidate_threshold
        self.embedding_model_path = Path(embedding_model_path) if embedding_model_path else None
        self.embedding_threshold = embedding_threshold
        self.profile_std_multiplier = profile_std_multiplier
        self.profile_min_threshold = profile_min_threshold
        self.profile_margin = profile_margin
        self.embedding_min_support = embedding_min_support
        self.embedding_topk = embedding_topk
        self.min_face_size = min_face_size
        self.min_sharpness = min_sharpness
        self.unknown_margin = unknown_margin
        self.crop_padding = crop_padding
        self.backend = "opencv"
        self._known_encodings: list[np.ndarray] = []
        self._known_names: list[str] = []
        self._known_fallback: list[tuple[str, np.ndarray, np.ndarray | None]] = []
        self._known_embeddings: list[tuple[str, np.ndarray]] = []
        self._unknown_embeddings: list[np.ndarray] = []
        self._embedding_profiles: dict[str, dict[str, float | np.ndarray]] = {}
        self._embedding_net = self._try_load_embedding_model()
        self._lbph_model = None
        self._lbph_labels: dict[int, str] = {}
        self._face_recognition = None if self._embedding_net is not None else self._try_load_face_recognition()
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self._eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self._has_lbph = hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")
        if self._embedding_net is not None:
            self.backend = "openface_embedding"
        elif not self._face_recognition and self._has_lbph:
            self.backend = "opencv_lbph"
        self.reload()

    @property
    def known_count(self) -> int:
        if self._face_recognition:
            return len(self._known_encodings)
        if self._embedding_net is not None:
            return len(self._known_embeddings)
        if self._lbph_model is not None:
            return len(self._lbph_labels)
        return len(self._known_fallback)

    def reload(self) -> None:
        self._known_encodings.clear()
        self._known_names.clear()
        self._known_fallback.clear()
        self._known_embeddings.clear()
        self._unknown_embeddings.clear()
        self._embedding_profiles.clear()
        self._lbph_model = None
        self._lbph_labels.clear()
        lbph_faces: list[np.ndarray] = []
        lbph_label_ids: list[int] = []
        label_by_name: dict[str, int] = {}

        for name, image_path in self._iter_known_face_images():
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue

            if self._embedding_net is not None:
                self._load_embedding_image(image_bgr, name)
            elif self._face_recognition:
                self._load_face_recognition_image(image_bgr, name)
            elif self._has_lbph:
                face = self._prepare_lbph_face(image_bgr)
                if face is None:
                    continue
                if name not in label_by_name:
                    label_id = len(label_by_name)
                    label_by_name[name] = label_id
                    self._lbph_labels[label_id] = name
                for sample in self._augment_lbph_face(face):
                    lbph_faces.append(sample)
                    lbph_label_ids.append(label_by_name[name])
            else:
                self._load_opencv_image(image_bgr, name)

        if self._embedding_net is not None:
            self._build_embedding_profiles()

        if self._has_lbph and lbph_faces:
            self._lbph_model = cv2.face.LBPHFaceRecognizer_create(
                radius=2,
                neighbors=12,
                grid_x=8,
                grid_y=8,
                threshold=max(self.lbph_candidate_threshold, self.lbph_threshold),
            )
            self._lbph_model.train(lbph_faces, np.array(lbph_label_ids, dtype=np.int32))

    def recognize(self, image_bgr: np.ndarray, assume_face: bool = False) -> RecognitionResult:
        if image_bgr.size == 0 or self.known_count == 0:
            return RecognitionResult(None, 0.0, self.backend, False)

        if not self._passes_quality_gate(image_bgr):
            return RecognitionResult(None, 0.0, self.backend, False, reason="low_quality")

        if self._embedding_net is not None:
            return self._recognize_with_embedding(image_bgr, assume_face=assume_face)
        if self._face_recognition:
            return self._recognize_with_face_recognition(image_bgr)
        if self._lbph_model is not None:
            return self._recognize_with_lbph(image_bgr, assume_face=assume_face)
        return self._recognize_with_opencv(image_bgr, assume_face=assume_face)

    def crop_largest_face(self, image_bgr: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face = self._detect_largest_face_box(gray)
        if face is None:
            return None
        return self._crop_box(image_bgr, face)

    def find_largest_face_box(self, image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        if image_bgr.size == 0:
            return None
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return self._detect_largest_face_box(gray)

    def find_face_boxes(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        if image_bgr.size == 0:
            return []
        
        # Tối ưu hóa hiệu năng: Nếu ảnh có độ phân giải lớn, ta scale xuống độ rộng tối đa 640px
        # để Haar Cascade chạy nhanh hơn gấp nhiều lần, sau đó nhân ngược tỷ lệ tọa độ trả về.
        height, width = image_bgr.shape[:2]
        max_scan_width = 640
        if width > max_scan_width:
            scale = max_scan_width / width
            resized = cv2.resize(image_bgr, (max_scan_width, int(height * scale)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            boxes = self._detect_face_boxes(gray)
            restored_boxes = []
            for x, y, w, h in boxes:
                restored_boxes.append((int(x / scale), int(y / scale), int(w / scale), int(h / scale)))
            return restored_boxes
            
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return self._detect_face_boxes(gray)

    def _detect_largest_face_box(self, gray: np.ndarray) -> tuple[int, int, int, int] | None:
        boxes = self._detect_face_boxes(gray)
        if not boxes:
            return None
        return boxes[0]

    def _detect_face_boxes(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        boxes: list[tuple[int, int, int, int]] = []
        prepared = self._prepare_gray(gray)
        frontal = self._cascade.detectMultiScale(prepared, scaleFactor=1.1, minNeighbors=5)
        boxes.extend(tuple(map(int, box)) for box in frontal)

        profiles = self._profile_cascade.detectMultiScale(prepared, scaleFactor=1.08, minNeighbors=4)
        boxes.extend(tuple(map(int, box)) for box in profiles)

        flipped = cv2.flip(prepared, 1)
        flipped_profiles = self._profile_cascade.detectMultiScale(flipped, scaleFactor=1.08, minNeighbors=4)
        width = gray.shape[1]
        for x, y, w, h in flipped_profiles:
            boxes.append((int(width - x - w), int(y), int(w), int(h)))

        boxes = [box for box in boxes if min(box[2], box[3]) >= self.min_face_size]
        if not boxes:
            return []
        return self._dedupe_face_boxes(boxes)

    def _dedupe_face_boxes(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        sorted_boxes = sorted(boxes, key=lambda item: item[2] * item[3], reverse=True)
        kept: list[tuple[int, int, int, int]] = []
        for box in sorted_boxes:
            if all(self._face_box_iou(box, kept_box) < 0.35 for kept_box in kept):
                kept.append(box)
        return kept

    def _face_box_iou(
        self,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (aw * ah) + (bw * bh) - intersection
        return 0.0 if union <= 0 else intersection / union

    def _crop_box(self, image_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box
        pad = int(max(w, h) * self.crop_padding)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image_bgr.shape[1], x + w + pad)
        y2 = min(image_bgr.shape[0], y + h + pad)
        return image_bgr[y1:y2, x1:x2]

    def _prepare_gray(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _enhance_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        image_bgr = self._upscale_small_face(image_bgr)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self._prepare_gray(l_channel)
        enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(enhanced_bgr, (0, 0), 1.0)
        return cv2.addWeighted(enhanced_bgr, 1.25, blurred, -0.25, 0)

    def _upscale_small_face(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        smallest_side = min(width, height)
        if smallest_side >= 120:
            return image_bgr
        scale = 120 / max(1, smallest_side)
        return cv2.resize(
            image_bgr,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

    def _passes_quality_gate(self, image_bgr: np.ndarray) -> bool:
        height, width = image_bgr.shape[:2]
        if min(width, height) < self.min_face_size:
            return False
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if self.min_sharpness <= 0:
            return True
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return sharpness >= self.min_sharpness

    def _try_load_face_recognition(self):
        try:
            import face_recognition
        except Exception:
            self.backend = "opencv"
            return None
        self.backend = "face_recognition"
        return face_recognition

    def _try_load_embedding_model(self):
        if self.embedding_model_path is None or not self.embedding_model_path.exists():
            return None
        try:
            return cv2.dnn.readNetFromTorch(str(self.embedding_model_path))
        except Exception:
            return None

    def _load_embedding_image(self, image_bgr: np.ndarray, name: str) -> None:
        embedding = self._embedding_from_face_or_image(image_bgr)
        if embedding is not None:
            self._known_embeddings.append((name, embedding))

    def _build_embedding_profiles(self) -> None:
        embeddings_by_name: dict[str, list[np.ndarray]] = {}
        for name, embedding in self._known_embeddings:
            embeddings_by_name.setdefault(name, []).append(embedding)

        for name, embeddings in embeddings_by_name.items():
            matrix = np.vstack(embeddings)
            distances: list[float] = []
            if len(embeddings) > 1:
                for index, embedding in enumerate(embeddings):
                    other_embeddings = [other for other_index, other in enumerate(embeddings) if other_index != index]
                    nearest = min(float(1.0 - np.dot(embedding, other)) for other in other_embeddings)
                    distances.append(nearest)
            
            # Cải tiến quan trọng: Nếu số lượng ảnh mẫu dưới 3 ảnh, ta không tính theo công thức thống kê
            # vì nó sẽ bóp ngưỡng nhận diện xuống cực kỳ thấp (0.08 - 0.11) khiến camera không thể nhận dạng.
            # Thay vào đó, sử dụng thẳng ngưỡng mặc định (embedding_threshold).
            if len(embeddings) < 3:
                threshold = self.embedding_threshold
                mean = 0.0
                std = 0.0
            else:
                mean = float(np.mean(distances)) if distances else 0.0
                std = float(np.std(distances)) if len(distances) > 1 else 0.0
                threshold = max(
                    self.profile_min_threshold,
                    mean + (std * self.profile_std_multiplier) + self.profile_margin,
                )
                threshold = min(self.embedding_threshold, threshold)

            self._embedding_profiles[name] = {
                "samples": matrix,
                "threshold": threshold,
                "mean": mean,
                "std": std,
            }

    def _embedding_from_face_or_image(self, image_bgr: np.ndarray) -> np.ndarray | None:
        face = self.crop_largest_face(image_bgr)
        if face is None:
            face = image_bgr
        return self._embedding(face)

    def _embedding(self, image_bgr: np.ndarray) -> np.ndarray | None:
        if image_bgr.size == 0 or self._embedding_net is None:
            return None
        enhanced = self._enhance_bgr(image_bgr)
        blob = cv2.dnn.blobFromImage(
            enhanced,
            scalefactor=1.0 / 255,
            size=(96, 96),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        try:
            self._embedding_net.setInput(blob)
            embedding = self._embedding_net.forward().flatten().astype(np.float32)
        except Exception:
            return None
        norm = float(np.linalg.norm(embedding))
        if norm <= 0:
            return None
        return embedding / norm

    def _recognize_with_embedding(self, image_bgr: np.ndarray, assume_face: bool = False) -> RecognitionResult:
        if assume_face and not self._has_visible_face_features(image_bgr):
            return RecognitionResult(None, 0.0, self.backend, False, reason="no_face_features")

        face = image_bgr if assume_face else self.crop_largest_face(image_bgr)
        if face is None:
            return RecognitionResult(None, 0.0, self.backend, False, reason="no_face")

        query = self._embedding(face)
        if query is None:
            return RecognitionResult(None, 0.0, self.backend, True, reason="no_embedding")

        if not self._embedding_profiles:
            return RecognitionResult(None, 0.0, self.backend, True, reason="no_known_embeddings")

        # Tính distance với mỗi người dùng trung bình top-k ảnh gần nhất
        ranked = []
        for name, profile in self._embedding_profiles.items():
            distances = sorted(float(1.0 - np.dot(query, emb)) for emb in profile["samples"])
            topk = distances[: max(1, min(self.embedding_topk, len(distances)))]
            topk_mean = float(np.mean(topk))
            ranked.append((name, topk_mean))

        ranked.sort(key=lambda item: item[1])
        best_name, best_distance = ranked[0]
        second_distance = ranked[1][1] if len(ranked) > 1 else float("inf")
        score = max(0.0, min(1.0, 1.0 - best_distance))

        # ── Lớp 1: Hard cap tuyệt đối ──────────────────────────────────────
        # Độ trùng khuôn mặt phải đủ cao: distance > cap → người lạ chắc chắn
        # score < 0.80 (distance > 0.20) → KHÔNG BAO GIỜ là người quen
        if best_distance > EMBEDDING_HARD_CAP_DISTANCE:
            return RecognitionResult(None, score, self.backend, True, best_distance, "hard_cap_exceeded", best_name)

        # ── Lớp 2: Margin phân biệt ─────────────────────────────────────────
        # Người tốt nhất phải cách người thứ 2 ít nhất unknown_margin
        # Tránh nhận nhầm khi hai người có distance gần nhau
        margin = second_distance - best_distance
        if margin < self.unknown_margin:
            return RecognitionResult(None, score, self.backend, True, best_distance, "ambiguous_match", best_name)

        # ── Lớp 3: Ngưỡng embedding tuyệt đối ──────────────────────────────
        # Distance còn phải <= embedding_threshold (0.18)
        if best_distance > self.embedding_threshold:
            return RecognitionResult(None, score, self.backend, True, best_distance, "outside_threshold", best_name)

        return RecognitionResult(best_name, score, self.backend, True, best_distance)

    def _load_face_recognition_image(self, image_bgr: np.ndarray, name: str) -> None:
        enhanced = self._enhance_bgr(image_bgr)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        locations = self._face_recognition.face_locations(rgb, model="hog")
        encodings = self._face_recognition.face_encodings(rgb, locations)
        for encoding in encodings:
            self._known_encodings.append(encoding)
            self._known_names.append(name)

    def _recognize_with_face_recognition(self, image_bgr: np.ndarray) -> RecognitionResult:
        enhanced = self._enhance_bgr(image_bgr)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        locations = self._face_recognition.face_locations(rgb, model="hog")
        if not locations:
            return RecognitionResult(None, 0.0, self.backend, False, reason="no_face")

        encodings = self._face_recognition.face_encodings(rgb, locations)
        if not encodings:
            return RecognitionResult(None, 0.0, self.backend, False, reason="no_encoding")

        distances = self._face_recognition.face_distance(self._known_encodings, encodings[0])
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        score = max(0.0, 1.0 - best_distance)
        best_name = self._known_names[best_index]
        margin_ok = self._distance_margin_is_clear(distances, best_name)
        # Hard cap tuyệt đối: distance > cap → người lạ, bất kể tolerance
        if best_distance > FACE_RECOG_HARD_CAP_DISTANCE:
            return RecognitionResult(None, score, self.backend, True, best_distance, "hard_cap_exceeded")
        if best_distance <= self.tolerance and margin_ok:
            return RecognitionResult(best_name, score, self.backend, True, best_distance)
        reason = "close_match" if best_distance <= self.tolerance else "above_tolerance"
        if best_distance <= self.tolerance and not margin_ok:
            reason = "ambiguous_match"
        return RecognitionResult(None, score, self.backend, True, best_distance, reason)

    def _prepare_lbph_face(self, image_bgr: np.ndarray, assume_face: bool = False) -> np.ndarray | None:
        face = None if assume_face else self.crop_largest_face(image_bgr)
        if assume_face or face is None:
            face = image_bgr
        enhanced = self._enhance_bgr(face)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        interpolation = cv2.INTER_AREA if min(gray.shape[:2]) >= 180 else cv2.INTER_CUBIC
        gray = cv2.resize(gray, (180, 180), interpolation=interpolation)
        gray = self._prepare_gray(gray)
        gray = self._focus_face_core(gray)
        return gray

    def _focus_face_core(self, gray: np.ndarray) -> np.ndarray:
        height, width = gray.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, int(height * 0.60))
        axes = (int(width * 0.31), int(height * 0.34))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask[: int(height * 0.25), :] = 0
        mask[:, : int(width * 0.18)] = 0
        mask[:, int(width * 0.82) :] = 0

        neutral = int(np.median(gray[mask > 0])) if np.any(mask > 0) else 128
        focused = np.full_like(gray, neutral)
        focused[mask > 0] = gray[mask > 0]
        return focused

    def _augment_lbph_face(self, gray: np.ndarray) -> list[np.ndarray]:
        samples = [gray, cv2.flip(gray, 1)]
        height, width = gray.shape[:2]
        center = (width // 2, height // 2)

        for angle in (-7, 7):
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            samples.append(cv2.warpAffine(gray, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE))

        for shift_x, shift_y in ((-8, 0), (8, 0), (0, -8), (0, 8)):
            matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            samples.append(cv2.warpAffine(gray, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE))

        for alpha, beta in ((0.85, 8), (1.15, -8)):
            samples.append(cv2.convertScaleAbs(gray, alpha=alpha, beta=beta))

        for scale in (0.92, 1.08):
            resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            samples.append(self._center_crop_or_pad(resized, width, height))

        for kernel_size in (5, 7):
            samples.append(cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0))

        for scale in (0.65,):
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            restored = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
            samples.append(restored)

        darker = cv2.convertScaleAbs(gray, alpha=0.8, beta=-4)
        brighter = cv2.convertScaleAbs(gray, alpha=1.2, beta=4)
        samples.extend([darker, brighter])

        return samples

    def _center_crop_or_pad(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        canvas = np.zeros((height, width), dtype=image.dtype)
        src_h, src_w = image.shape[:2]
        crop_w = min(width, src_w)
        crop_h = min(height, src_h)
        src_x = max(0, (src_w - crop_w) // 2)
        src_y = max(0, (src_h - crop_h) // 2)
        dst_x = max(0, (width - crop_w) // 2)
        dst_y = max(0, (height - crop_h) // 2)
        canvas[dst_y : dst_y + crop_h, dst_x : dst_x + crop_w] = image[
            src_y : src_y + crop_h,
            src_x : src_x + crop_w,
        ]
        return canvas

    def _recognize_with_lbph(self, image_bgr: np.ndarray, assume_face: bool = False) -> RecognitionResult:
        if assume_face and not self._has_visible_face_features(image_bgr):
            return RecognitionResult(None, 0.0, self.backend, False, reason="no_face_features")

        face_found = assume_face or self.crop_largest_face(image_bgr) is not None
        face = self._prepare_lbph_face(image_bgr, assume_face=assume_face)
        if face is None:
            return RecognitionResult(None, 0.0, self.backend, False)

        label_id, distance = self._lbph_model.predict(face)
        distance = float(distance)
        score = max(0.0, min(1.0, 1.0 - (distance / 120.0)))
        name = self._lbph_labels.get(label_id)

        # Hard cap tuyệt đối: distance > cap → người lạ ngay, không bỏ phiếu
        if distance > LBPH_HARD_CAP_DISTANCE:
            return RecognitionResult(None, score, self.backend, face_found, distance, "hard_cap_exceeded")

        if distance <= self.lbph_threshold:
            return RecognitionResult(name, score, self.backend, face_found, distance)

        # Đã xóa vùng borderline: distance > threshold → người lạ, không candidate
        return RecognitionResult(None, score, self.backend, face_found, distance, "above_threshold")

    def _has_visible_face_features(self, image_bgr: np.ndarray) -> bool:
        """
        Kiểm tra khuôn mặt có đủ đặc trưng để nhận diện không.
        Yêu cầu: phải thấy rõ ĐỦ 2 MẮT và vùng giữa mặt không bị che.
        Che miệng = ok | Che 1 mắt = từ chối | Che toàn bộ = từ chối
        """
        if image_bgr.size == 0:
            return False
        enhanced = self._enhance_bgr(image_bgr)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        # ── Bước 1: Phát hiện mắt ───────────────────────────────────────────
        # Chỉ tìm trong nửa trên của mặt (tránh nhầm miệng/mũi thành mắt)
        upper_face = gray[: max(1, int(height * 0.60)), :]
        eyes = self._eye_cascade.detectMultiScale(
            upper_face,
            scaleFactor=1.08,
            minNeighbors=4,          # giảm từ 5→4 để tăng độ nhạy
            minSize=(max(12, width // 10), max(10, height // 16)),
        )

        # ── Bước 2: Hai mắt phải ở hai phía trái/phải của mặt ──────────────
        # Tránh trường hợp cascade phát hiện 2 vùng cùng 1 bên. Với camera
        # giám sát, chấp nhận mặt hơi nghiêng nếu vẫn thấy ít nhất 1 mắt và
        # vùng giữa mặt còn đủ nét.
        eye_centers_x = sorted([int(ex + ew / 2) for ex, ey, ew, eh in eyes])
        has_two_separated_eyes = False
        if len(eye_centers_x) >= 2:
            left_eye_x = eye_centers_x[0]
            right_eye_x = eye_centers_x[-1]
            has_two_separated_eyes = right_eye_x - left_eye_x >= int(width * 0.20)

        # ── Bước 3: Vùng giữa mặt (mũi) không bị che ───────────────────────
        # Nếu mũi bị che bằng tay, vùng này sẽ có texture khác thường
        nose_region = gray[int(height * 0.40): int(height * 0.70),
                           int(width  * 0.25): int(width  * 0.75)]
        nose_sharpness = 0.0
        if nose_region.size > 0:
            # Vùng bị che bởi tay/vật thường có texture đều (không có edge)
            # Ngưỡng sharpness thấp = vùng bị che hoặc quá mờ
            nose_sharpness = float(cv2.Laplacian(nose_region, cv2.CV_64F).var())
            if nose_sharpness < 10.0: # Giảm từ 15.0 xuống 10.0 để tránh loại bỏ khi mờ nhẹ hoặc ánh sáng phẳng
                return False

        if has_two_separated_eyes:
            return True

        return len(eyes) >= 1 and nose_sharpness >= 18.0 # Giảm từ 28.0 xuống 18.0 để nhận diện ổn định hơn

    def _load_opencv_image(self, image_bgr: np.ndarray, name: str) -> None:
        face = self.crop_largest_face(image_bgr)
        if face is None:
            face = image_bgr
        histogram, descriptors = self._fallback_features(face)
        self._known_fallback.append((name, histogram, descriptors))

    def _recognize_with_opencv(self, image_bgr: np.ndarray, assume_face: bool = False) -> RecognitionResult:
        face = None if assume_face else self.crop_largest_face(image_bgr)
        face_found = assume_face or face is not None
        target = image_bgr if assume_face or face is None else face
        histogram, descriptors = self._fallback_features(target)

        scores_by_name: dict[str, float] = {}
        for name, known_histogram, known_descriptors in self._known_fallback:
            histogram_score = cv2.compareHist(histogram, known_histogram, cv2.HISTCMP_CORREL)
            histogram_score = float(max(0.0, min(1.0, histogram_score)))
            orb_score = self._orb_similarity(descriptors, known_descriptors)
            score = (histogram_score * 0.65) + (orb_score * 0.35)
            scores_by_name[name] = max(scores_by_name.get(name, 0.0), score)

        ranked = sorted(scores_by_name.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return RecognitionResult(None, 0.0, self.backend, face_found)
        best_name, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin_ok = best_score - second_score >= self.unknown_margin
        if best_name and best_score >= self.fallback_threshold and margin_ok:
            return RecognitionResult(best_name, best_score, self.backend, face_found)
        reason = "ambiguous_match" if best_score >= self.fallback_threshold else "below_threshold"
        return RecognitionResult(None, best_score, self.backend, face_found, reason=reason)

    def _fallback_features(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        resized = cv2.resize(image_bgr, (160, 160), interpolation=cv2.INTER_AREA)
        enhanced = self._enhance_bgr(resized)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray = self._prepare_gray(gray)
        histogram = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        orb = cv2.ORB_create(nfeatures=250)
        _, descriptors = orb.detectAndCompute(gray, None)
        return histogram, descriptors

    def _orb_similarity(self, descriptors_a, descriptors_b) -> float:
        if descriptors_a is None or descriptors_b is None:
            return 0.0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors_a, descriptors_b)
        if not matches:
            return 0.0
        distances = np.array([match.distance for match in matches], dtype=np.float32)
        good_matches = np.count_nonzero(distances < 64)
        return float(min(1.0, good_matches / max(12, len(matches))))

    def _distance_margin_is_clear(self, distances: np.ndarray, best_name: str) -> bool:
        best_by_name: dict[str, float] = {}
        for name, distance in zip(self._known_names, distances):
            current = best_by_name.get(name)
            value = float(distance)
            if current is None or value < current:
                best_by_name[name] = value
        other_distances = [distance for name, distance in best_by_name.items() if name != best_name]
        if not other_distances:
            return True
        return min(other_distances) - best_by_name[best_name] >= self.unknown_margin

    def _name_from_file(self, image_path: Path) -> str:
        if image_path.parent != self.known_dir:
            return image_path.parent.name.replace("_", " ").strip()
        return image_path.stem.replace("_", " ").strip()

    def _iter_known_face_images(self) -> list[tuple[str, Path]]:
        if not self.known_dir.exists():
            return []

        images: list[tuple[str, Path]] = []
        for path in sorted(self.known_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append((self._name_from_file(path), path))
                continue
            if not path.is_dir():
                continue
            name = path.name.replace("_", " ").strip()
            for image_path in sorted(path.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append((name, image_path))
        return images

    def _iter_unknown_face_images(self) -> list[Path]:
        if self.unknown_dir is None or not self.unknown_dir.exists():
            return []

        images: list[Path] = []
        for path in sorted(self.unknown_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)
        return images
