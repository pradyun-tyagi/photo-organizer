"""Face cropping, feature vector extraction, and People Album clustering."""

from __future__ import annotations

import base64
import io
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


def extract_face_crops_and_features(
    photo: PhotoInfo, max_faces: int = 5
) -> List[Tuple[Tuple[int, int, int, int], "np.ndarray", str]]:
    """Detects faces in photo, extracts spatial color/texture feature vector and base64 preview URI.
    Returns list of ((x, y, w, h), feature_vector, base64_thumbnail).
    """
    if Image is None or cv2 is None or np is None:
        return []

    cascades = _get_thread_face_cascades()
    if not cascades:
        return []

    results = []
    try:
        with Image.open(photo.path) as img:
            working = ImageOps.exif_transpose(img).convert("RGB")
            orig_w, orig_h = working.size
            if orig_w == 0 or orig_h == 0:
                return []

            # Scale to standard analysis dimension
            scale = 800.0 / max(orig_w, orig_h)
            if scale < 1.0:
                work_dim = (int(orig_w * scale), int(orig_h * scale))
                analysis = working.resize(work_dim, Image.Resampling.BILINEAR)
            else:
                scale = 1.0
                analysis = working

            rgb_arr = np.array(analysis)
            gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)

            faces = []
            for cascade in cascades:
                found = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                )
                if len(found) > 0:
                    faces = found
                    break

            for idx, (fx, fy, fw, fh) in enumerate(faces[:max_faces]):
                # Map back to original coordinate system
                x1 = max(0, int(fx / scale))
                y1 = max(0, int(fy / scale))
                x2 = min(orig_w, int((fx + fw) / scale))
                y2 = min(orig_h, int((fy + fh) / scale))
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue

                face_crop = working.crop((x1, y1, x2, y2))
                face_crop_small = face_crop.resize((64, 64), Image.Resampling.BILINEAR)

                # Feature vector: 64-dim normalized spatial luminance + color histogram
                face_arr = np.array(face_crop_small)
                hsv = cv2.cvtColor(face_arr, cv2.COLOR_RGB2HSV)
                hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
                hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
                hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
                hist_gray = cv2.calcHist([face_arr], [0], None, [16], [0, 256]).flatten()
                feat = np.concatenate([hist_h, hist_s, hist_v, hist_gray]).astype("float32")
                norm = np.linalg.norm(feat)
                if norm > 0:
                    feat /= norm

                # Create base64 thumbnail for instant UI display
                buf = io.BytesIO()
                face_crop_small.save(buf, format="JPEG", quality=85)
                b64_data = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

                results.append(((x1, y1, x2 - x1, y2 - y1), feat, b64_data))
    except Exception:
        return []

    return results


def cluster_people_albums(
    photos: Sequence[PhotoInfo], distance_threshold: float = 0.55
) -> List[PersonCluster]:
    """Clusters detected faces across library into People Albums."""
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

    # Simple greedy incremental clustering
    clusters: List[Dict] = []  # list of dicts: {"center": feat, "members": [(photo, b64)]}

    for photo, feat, b64_thumb in face_records:
        best_cluster = None
        min_dist = float("inf")

        for cluster in clusters:
            dist = float(np.linalg.norm(feat - cluster["center"]))
            if dist < min_dist:
                min_dist = dist
                best_cluster = cluster

        if best_cluster is not None and min_dist <= distance_threshold:
            best_cluster["members"].append((photo, b64_thumb))
            # Update center vector
            all_feats = [feat] + [m[0] for m in best_cluster.get("_feats", [])]
            best_cluster["_feats"].append((feat, photo))
            new_center = np.mean([f[0] for f in best_cluster["_feats"]], axis=0)
            best_cluster["center"] = new_center / np.linalg.norm(new_center)
        else:
            clusters.append({
                "center": feat,
                "members": [(photo, b64_thumb)],
                "_feats": [(feat, photo)],
                "rep_thumb": b64_thumb,
            })

    # Convert to PersonCluster models, filtering small 1-face anomalies
    people_albums: List[PersonCluster] = []
    # Sort clusters by number of photos descending
    clusters.sort(key=lambda c: len(c["members"]), reverse=True)

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
