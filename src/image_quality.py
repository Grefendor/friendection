import cv2
import numpy as np


def calculate_sharpness(image: np.ndarray) -> float:
    """
    Calculate sharpness score using Laplacian variance.
    Higher value = sharper image.
    """
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """Check if image is blurry based on threshold."""
    return calculate_sharpness(image) < threshold