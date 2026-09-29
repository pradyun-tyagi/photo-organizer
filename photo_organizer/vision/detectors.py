"""Face and person detectors using OpenCV Haar cascades, HOG, and DNNs."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional, Tuple

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

FACE_CASCADES: Optional[List[object]] = None
HOG_PERSON_DETECTOR: Optional[object] = None
OPENCV_DETECT_LOCK = threading.RLock()
_local_detectors = threading.local()


def load_face_cascades() -> List[object]:
    if cv2 is None:
        return []

    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_profileface.xml",
    ]
    cascades = []
    for name in cascade_names:
        cascade_path = str(Path(cv2.data.haarcascades) / name)
        cascade = cv2.CascadeClassifier(cascade_path)
        if not cascade.empty():
            cascades.append(cascade)
    return cascades


def _get_thread_face_cascades() -> List[object]:
    cascades = getattr(_local_detectors, "cascades", None)
    if cascades is None:
        cascades = load_face_cascades()
        _local_detectors.cascades = cascades
    return cascades


def _get_thread_hog() -> Optional[object]:
    if cv2 is None:
        return None
    hog = getattr(_local_detectors, "hog", None)
    if hog is None:
        try:
            h = cv2.HOGDescriptor()
            h.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            _local_detectors.hog = h
            hog = h
        except Exception:
            return None
    return hog


def detect_face(image: "Image.Image") -> Tuple[bool, bool]:
    global FACE_CASCADES

    import sys
    root_mod = sys.modules.get("photo_organizer")
    active_cv2 = getattr(root_mod, "cv2", cv2)
    active_np = getattr(root_mod, "np", np)

    if active_cv2 is None or active_np is None:
        return False, False

    active_cascades = getattr(root_mod, "FACE_CASCADES", None)
    if active_cascades is None:
        active_cascades = _get_thread_face_cascades()

    if not active_cascades:
        return False, False

    try:
        analysis_fn = getattr(root_mod, "analysis_image", None)
        if analysis_fn is not None:
            working = analysis_fn(image, max_dimension=800)
        elif ImageOps is not None:
            working = ImageOps.exif_transpose(image).convert("RGB")
            working.thumbnail((800, 800), Image.Resampling.BILINEAR)
        else:
            working = image

        gray = active_cv2.cvtColor(active_np.array(working), active_cv2.COLOR_RGB2GRAY)
        flipped = active_cv2.flip(gray, 1)
    except Exception:
        return False, False

    for cascade in active_cascades:
        try:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(28, 28),
            )
            if len(faces) > 0:
                return True, True

            faces = cascade.detectMultiScale(
                flipped,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(28, 28),
            )
            if len(faces) > 0:
                return True, True
        except active_cv2.error:
            return False, False

    return False, True


def detect_person(image: "Image.Image") -> Tuple[bool, bool]:
    global HOG_PERSON_DETECTOR

    if cv2 is None or np is None or Image is None:
        return False, False

    import sys
    root_mod = sys.modules.get("photo_organizer")
    hog = getattr(root_mod, "HOG_PERSON_DETECTOR", None) or _get_thread_hog()
    if hog is None:
        return False, False

    try:
        working = ImageOps.exif_transpose(image).convert("RGB")
        working.thumbnail((500, 500), Image.Resampling.BILINEAR)
        array = cv2.cvtColor(np.array(working), cv2.COLOR_RGB2BGR)
        people, _ = hog.detectMultiScale(
            array,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.10,
        )
        return len(people) > 0, True
    except (cv2.error, Exception):
        return False, False

