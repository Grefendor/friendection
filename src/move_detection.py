import cv2 as cv
import numpy as np
def process_mog2(frame,
                 backSub,
                 prev_brightness,
                 kernel,
                 BRIGHTNESS_RESET_DELTA,
                 ADAPT_LR,
                 MOTION_THRESHOLD,
                 draw_overlay: bool = True):
    """Apply MOG2, handle brightness reinit, morphology and motion detection.
    Returns: (backSub, prev_brightness, fgMask, motion_detected)
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    brightness = int(gray.mean())

    if prev_brightness is None:
        prev_brightness = brightness

    if abs(brightness - prev_brightness) > BRIGHTNESS_RESET_DELTA:
        backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        print("Background model reinitialized due to lighting change")
        prev_brightness = brightness

    fgMask = backSub.apply(frame, learningRate=ADAPT_LR)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)

    motion_pixels = cv.countNonZero(fgMask)
    motion_detected = motion_pixels > MOTION_THRESHOLD

    if motion_detected:
        print(f"Motion detected! Pixels: {motion_pixels}")
        backSub.apply(frame, learningRate=0)
        if draw_overlay:
            cv.putText(frame, "MOTION DETECTED", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return backSub, prev_brightness, fgMask, motion_detected