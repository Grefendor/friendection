import cv2
import numpy as np

# Pre-allocated buffer for sharpness calculation
_sharp_gray: np.ndarray = None

def calculate_sharpness(image: np.ndarray) -> float:
    """
    Calculate sharpness score using Laplacian variance.
    Higher value = sharper image.
    """
    global _sharp_gray
    if image is None or image.size == 0:
        return 0.0
    
    h, w = image.shape[:2]
    if len(image.shape) == 3:
        # Reuse grayscale buffer
        if _sharp_gray is None or _sharp_gray.shape != (h, w):
            _sharp_gray = np.empty((h, w), dtype=np.uint8)
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, dst=_sharp_gray)
        gray = _sharp_gray
    else:
        gray = image
    
    # CV_16S is faster than CV_64F, variance computation handles the rest
    laplacian = cv2.Laplacian(gray, cv2.CV_16S)
    # Compute variance manually to avoid float64 intermediate array
    mean = laplacian.mean()
    return float(((laplacian - mean) ** 2).mean())


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """Check if image is blurry based on threshold."""
    return calculate_sharpness(image) < threshold