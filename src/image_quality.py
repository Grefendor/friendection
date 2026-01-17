"""Image quality assessment utilities for face capture optimization."""

import cv2
import numpy as np

# Reusable buffer for grayscale conversion
_sharp_gray: np.ndarray = None

def calculate_sharpness(image: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    
    Args:
        image: Input image (BGR or grayscale).
        
    Returns:
        Sharpness score. Higher values indicate sharper images.
    """
    global _sharp_gray
    if image is None or image.size == 0:
        return 0.0
    
    h, w = image.shape[:2]
    if len(image.shape) == 3:
        if _sharp_gray is None or _sharp_gray.shape != (h, w):
            _sharp_gray = np.empty((h, w), dtype=np.uint8)
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, dst=_sharp_gray)
        gray = _sharp_gray
    else:
        gray = image
    
    laplacian = cv2.Laplacian(gray, cv2.CV_16S)
    mean = laplacian.mean()
    return float(((laplacian - mean) ** 2).mean())


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Check if an image is blurry.
    
    Args:
        image: Input image to evaluate.
        threshold: Sharpness threshold (default: 100.0).
        
    Returns:
        True if sharpness is below threshold.
    """
    return calculate_sharpness(image) < threshold