"""Deep facial embedding extraction and People Album clustering using OpenCV YuNet and SFace."""

from __future__ import annotations

import base64
import io
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from photo_organizer.models import PersonCluster, PhotoInfo
from photo_organizer.vision.detectors import _get_thread_face_cascades

_thread_local = threading.local()

# Model filenames
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"


def _find_face_model_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Locates the YuNet and SFace ONNX models in package or user cache directories."""
    candidates = [
        Path(__file__).resolve().parent.parent / "models_data",
        Path.home() / ".cache" / "photo_organizer" / "models",
    ]
    yunet_path = None
    sface_path = None

    for d in candidates:
        if not yunet_path:
            p = d / YUNET_FILENAME
            if p.exists() and p.stat().st_size > 10000:
                yunet_path = p
        if not sface_path:
            p = d / SFACE_FILENAME
            if p.exists() and p.stat().st_size > 1000000:
                sface_path = p

    return yunet_path, sface_path


def _get_deep_face_models() -> Tuple[Optional["cv2.FaceDetectorYN"], Optional["cv2.FaceRecognizerSF"]]:
    """Returns thread-cached instances of YuNet Face Detector and SFace Face Recognizer."""
    if cv2 is None or not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
        return None, None

    if hasattr(_thread_local, "yunet_detector") and hasattr(_thread_local, "sface_recognizer"):
        return _thread_local.yunet_detector, _thread_local.sface_recognizer

    yunet_path, sface_path = _find_face_model_paths()
    if not yunet_path or not sface_path:
        return None, None

    try:
        detector = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            (320, 320),
            score_threshold=0.65,
            nms_threshold=0.3,
            top_k=5000,
        )
        recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
        _thread_local.yunet_detector = detector
        _thread_local.sface_recognizer = recognizer
        return detector, recognizer
    except Exception:
        return None, None


def extract_face_crops_and_features(
    photo: PhotoInfo, max_faces: int = 5
) -> List[Tuple[Tuple[int, int, int, int], "np.ndarray", str]]:
    """Detects faces in a photo and extracts invariant 128-d deep biometric embeddings (SFace)
    and base64 preview thumbnails.
    Returns a list of ((x, y, w, h), feature_vector_128d, base64_thumbnail).
    """
    if Image is None or cv2 is None or np is None:
        return []

    detector, recognizer = _get_deep_face_models()

    results = []
    try:
        with Image.open(photo.path) as img:
            working = ImageOps.exif_transpose(img).convert("RGB")
            orig_w, orig_h = working.size
            if orig_w == 0 or orig_h == 0:
                return []

            # Scale to standard analysis dimensions (max 640px) for fast and accurate inference
            scale = 640.0 / max(orig_w, orig_h) if max(orig_w, orig_h) > 640 else 1.0
            if scale < 1.0:
                work_dim = (int(orig_w * scale), int(orig_h * scale))
                analysis = working.resize(work_dim, Image.Resampling.BILINEAR)
            else:
                work_dim = (orig_w, orig_h)
                analysis = working

            rgb_arr = np.array(analysis)
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

            if detector is not None and recognizer is not None:
                detector.setInputSize(work_dim)
                _, raw_faces = detector.detect(bgr_arr)

                if raw_faces is not None and len(raw_faces) > 0:
                    for face_row in raw_faces[:max_faces]:
                        score = float(face_row[14]) if len(face_row) > 14 else 1.0
                        if score < 0.65:
                            continue

                        fx, fy, fw, fh = face_row[0:4]
                        x1 = max(0, int(fx / scale))
                        y1 = max(0, int(fy / scale))
                        x2 = min(orig_w, int((fx + fw) / scale))
                        y2 = min(orig_h, int((fy + fh) / scale))
                        if x2 - x1 < 24 or y2 - y1 < 24:
                            continue

                        # Extract 128-dimensional deep metric embedding using aligned face
                        try:
                            aligned = recognizer.alignCrop(bgr_arr, face_row)
                            feat = recognizer.feature(aligned).astype("float32").flatten()
                            norm = np.linalg.norm(feat)
                            if norm > 0:
                                feat /= norm
                            else:
                                continue
                        except Exception:
                            continue

                        # Padded portrait crop for visual avatar preview
                        pad_x = int(0.20 * (x2 - x1))
                        pad_y = int(0.20 * (y2 - y1))
                        px1 = max(0, x1 - pad_x)
                        py1 = max(0, y1 - pad_y)
                        px2 = min(orig_w, x2 + pad_x)
                        py2 = min(orig_h, y2 + pad_y)

                        face_crop = working.crop((px1, py1, px2, py2)).resize((96, 96), Image.Resampling.BILINEAR)
                        buf = io.BytesIO()
                        face_crop.save(buf, format="JPEG", quality=85)
                        b64_data = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

                        results.append(((x1, y1, x2 - x1, y2 - y1), feat, b64_data))

            else:
                # Fallback to Haar Cascade if Deep Learning models are unavailable
                cascades = _get_thread_face_cascades()
                if cascades:
                    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
                    found = []
                    for cascade in cascades:
                        f = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                        if len(f) > 0:
                            found = f
                            break
                    for fx, fy, fw, fh in found[:max_faces]:
                        x1 = max(0, int(fx / scale))
                        y1 = max(0, int(fy / scale))
                        x2 = min(orig_w, int((fx + fw) / scale))
                        y2 = min(orig_h, int((fy + fh) / scale))
                        face_crop = working.crop((x1, y1, x2, y2)).resize((64, 64), Image.Resampling.BILINEAR)
                        face_arr = np.array(face_crop)
                        hsv = cv2.cvtColor(face_arr, cv2.COLOR_RGB2HSV)
                        feat = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).flatten().astype("float32")
                        norm = np.linalg.norm(feat)
                        if norm > 0:
                            feat /= norm
                        buf = io.BytesIO()
                        face_crop.save(buf, format="JPEG", quality=80)
                        b64_data = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
                        results.append(((x1, y1, x2 - x1, y2 - y1), feat, b64_data))

    except Exception:
        return []

    return results


def cluster_people_albums(
    photos: Sequence[PhotoInfo], similarity_threshold: float = 0.48
) -> List[PersonCluster]:
    """Clusters detected faces across library into People Albums using deep biometric cosine metric.
    similarity_threshold: Cosine similarity in [0, 1] required to belong to the same person.
    For SFace embeddings, similarity >= 0.48 indicates the same identity.
    """
    if np is None or cv2 is None:
        return []

    face_records: List[Tuple[PhotoInfo, "np.ndarray", str]] = []

    for photo in photos:
        if photo.is_video:
            continue
        crops = extract_face_crops_and_features(photo)
        for _, feat, b64_thumb in crops:
            face_records.append((photo, feat, b64_thumb))

    if not face_records:
        return []

    # Centroid clustering using Cosine Similarity on normalized 128-d vectors
    clusters: List[Dict] = []

    for photo, feat, b64_thumb in face_records:
        best_cluster = None
        best_similarity = -1.0

        for cluster in clusters:
            # Cosine similarity between unit-normalized vectors is simply the dot product
            sim = float(np.dot(feat, cluster["center"]))
            if sim > best_similarity:
                best_similarity = sim
                best_cluster = cluster

        if best_cluster is not None and best_similarity >= similarity_threshold:
            best_cluster["members"].append((photo, b64_thumb))
            best_cluster["_feats"].append(feat)
            # Recompute centroid as mean vector, then project back to unit sphere
            new_center = np.mean(best_cluster["_feats"], axis=0)
            norm = np.linalg.norm(new_center)
            if norm > 0:
                best_cluster["center"] = new_center / norm
        else:
            clusters.append({
                "center": feat.copy(),
                "members": [(photo, b64_thumb)],
                "_feats": [feat.copy()],
                "rep_thumb": b64_thumb,
            })

    # Sort clusters by member count descending
    clusters.sort(key=lambda c: len(c["members"]), reverse=True)

    people_albums: List[PersonCluster] = []
    for idx, c in enumerate(clusters):
        unique_paths = list(dict.fromkeys(str(m[0].path) for m in c["members"]))
        people_albums.append(
            PersonCluster(
                person_id=f"person_{idx + 1}",
                name=f"Person {idx + 1}",
                face_count=len(c["members"]),
                representative_crop_data=c["rep_thumb"],
                photo_paths=unique_paths,
            )
        )

    return people_albums
