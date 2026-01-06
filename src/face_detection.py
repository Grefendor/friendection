from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class FaceBox:
    """Absolute pixel bounding box for a detected face."""
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


class TaskFaceDetector:
    """
    MediaPipe Tasks Face Detector wrapper.
    Works with running_mode=VIDEO by default for camera/stream usage.

    Docs: FaceDetectorOptions + detect_for_video(). :contentReference[oaicite:3]{index=3}
    """

    def __init__(
        self,
        model_path: str,
        min_detection_confidence: float = 0.6,
        min_suppression_threshold: float = 0.3,
        running_mode: str = "VIDEO",  # "IMAGE" or "VIDEO"
    ) -> None:
        self._BaseOptions = mp.tasks.BaseOptions
        self._FaceDetector = mp.tasks.vision.FaceDetector
        self._FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        self._RunningMode = mp.tasks.vision.RunningMode

        if running_mode.upper() not in {"IMAGE", "VIDEO"}:
            raise ValueError("running_mode must be 'IMAGE' or 'VIDEO'")

        self._mode = (
            self._RunningMode.IMAGE
            if running_mode.upper() == "IMAGE"
            else self._RunningMode.VIDEO
        )

        options = self._FaceDetectorOptions(
            base_options=self._BaseOptions(model_asset_path=model_path),
            running_mode=self._mode,
            min_detection_confidence=float(min_detection_confidence),
            min_suppression_threshold=float(min_suppression_threshold),
        )

        # Keep detector open (faster than creating per-frame)
        self._detector = self._FaceDetector.create_from_options(options)

    def close(self) -> None:
        """Release native resources."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> List[FaceBox]:
        """
        Detect faces in a BGR frame.

        - If running_mode=IMAGE: timestamp_ms is ignored and detect() is used.
        - If running_mode=VIDEO: timestamp_ms must be provided and detect_for_video() is used. :contentReference[oaicite:4]{index=4}

        Returns:
            List[FaceBox] with absolute pixel coordinates.
        """
        if self._detector is None:
            raise RuntimeError("Detector is closed. Create a new TaskFaceDetector.")

        if frame_bgr is None or frame_bgr.ndim != 3:
            return []

        h, w = frame_bgr.shape[:2]

        # MediaPipe mp.Image wants SRGB data; convert BGR->RGB.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)  # :contentReference[oaicite:5]{index=5}

        if self._mode == self._RunningMode.IMAGE:
            result = self._detector.detect(mp_image)
        else:
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required in VIDEO mode")
            result = self._detector.detect_for_video(mp_image, int(timestamp_ms))

        detections = getattr(result, "detections", None)
        if not detections:
            return []

        faces: List[FaceBox] = []

        for det in detections:
            # Bounding box is absolute pixels in result output examples. :contentReference[oaicite:6]{index=6}
            bbox = det.bounding_box  # origin_x, origin_y, width, height

            x1 = int(bbox.origin_x)
            y1 = int(bbox.origin_y)
            x2 = int(bbox.origin_x + bbox.width)
            y2 = int(bbox.origin_y + bbox.height)

            # Clamp
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # Score: first category score (single-class face detector)
            score = 0.0
            cats = getattr(det, "categories", None)
            if cats and len(cats) > 0 and hasattr(cats[0], "score"):
                score = float(cats[0].score)

            faces.append(FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score))

        return faces

    def __enter__(self) -> "TaskFaceDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
