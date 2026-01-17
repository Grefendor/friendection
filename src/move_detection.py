"""Motion detection using background subtraction."""

import cv2 as cv
import numpy as np

# Reusable frame buffers
_gray_buffer: np.ndarray = None
_small_frame: np.ndarray = None
_DOWNSAMPLE = 2  # Downscale factor for motion processing

def process_mog2(frame,
                 backSub,
                 prev_brightness,
                 kernel,
                 BRIGHTNESS_RESET_DELTA,
                 ADAPT_LR,
                 MOTION_THRESHOLD,
                 draw_overlay: bool = True):
    """
    Process frame for motion detection using MOG2 background subtraction.
    
    Automatically reinitializes background model on significant brightness changes.
    
    Args:
        frame: Input BGR frame.
        backSub: Background subtractor instance.
        prev_brightness: Previous frame brightness for change detection.
        kernel: Morphological operation kernel.
        BRIGHTNESS_RESET_DELTA: Brightness change threshold for model reset.
        ADAPT_LR: Background model learning rate.
        MOTION_THRESHOLD: Pixel count threshold for motion detection.
        draw_overlay: Whether to draw motion indicator on frame.
        
    Returns:
        Tuple of (backSub, prev_brightness, fgMask, motion_detected).
    """
    global _gray_buffer, _small_frame
    h, w = frame.shape[:2]
    small_h, small_w = h // _DOWNSAMPLE, w // _DOWNSAMPLE
    
    if _small_frame is None or _small_frame.shape[:2] != (small_h, small_w):
        _small_frame = np.empty((small_h, small_w, 3), dtype=np.uint8)
        _gray_buffer = np.empty((small_h, small_w), dtype=np.uint8)
    
    cv.resize(frame, (small_w, small_h), dst=_small_frame, interpolation=cv.INTER_NEAREST)
    cv.cvtColor(_small_frame, cv.COLOR_BGR2GRAY, dst=_gray_buffer)
    brightness = int(_gray_buffer.mean())

    if prev_brightness is None:
        prev_brightness = brightness

    if abs(brightness - prev_brightness) > BRIGHTNESS_RESET_DELTA:
        backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        print("Background model reinitialized due to lighting change")
        prev_brightness = brightness

    fgMask = backSub.apply(_small_frame, learningRate=ADAPT_LR)
    cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel, dst=fgMask)

    motion_pixels = cv.countNonZero(fgMask)
    adjusted_threshold = MOTION_THRESHOLD // (_DOWNSAMPLE * _DOWNSAMPLE)
    motion_detected = motion_pixels > adjusted_threshold

    if motion_detected:
        print(f"Motion detected! Pixels: {motion_pixels}")
        backSub.apply(_small_frame, learningRate=0)
        if draw_overlay:
            cv.putText(frame, "MOTION DETECTED", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return backSub, prev_brightness, fgMask, motion_detected