import time
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from pathlib import Path

# statt BlazeFaceDetector:
from src.face_detection import TaskFaceDetector

from src.move_detection import process_mog2
from src.person_detection import check_for_person

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
prev_brightness = None
BRIGHTNESS_RESET_DELTA = 40
ADAPT_LR = 0.01  # langsam anpassen

root = Path(__file__).resolve().parents[1]

# ensure images directory exists
images_dir = Path("images")
images_dir.mkdir(parents=True, exist_ok=True)

# YOLO Modell
model_path = Path("models") / "yolo12n.pt"
model = YOLO(str(model_path))

# Motion detection threshold
MOTION_THRESHOLD = 7000  # number of pixels that need to change
person_counter = 0
frame_index = 0

# Face detector (MediaPipe Tasks)
face_model_path = Path("models") / "blaze_face_short_range.tflite"
detector = TaskFaceDetector(
    model_path=str(face_model_path),
    min_detection_confidence=0.7,
    running_mode="VIDEO",
)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if frame_index == 30:
            frame_index = 0
            backSub, prev_brightness, fgMask, motion = process_mog2(
                frame,
                backSub,
                prev_brightness,
                kernel,
                BRIGHTNESS_RESET_DELTA,
                ADAPT_LR,
                MOTION_THRESHOLD,
            )

            if motion:
                if check_for_person(frame, model):
                    person_counter += 1
                    if person_counter == 4:
                        print("Person detected!")

                        ts_ms = int(time.time() * 1000)
                        faces = detector.detect(frame, timestamp_ms=ts_ms)
                        i = 0
                        for f in faces:
                            cropped_face = frame[f.y1:f.y2, f.x1:f.x2]
                            cv.imwrite(str(images_dir / f"detected_face_{ts_ms}_{i}.png"), cropped_face)
                            i += 1
                        break

            cv.imshow("fgMask", fgMask)

        if cv.waitKey(1) == ord("q"):
            break

        frame_index += 1

finally:
    cap.release()
    detector.close()
    cv.destroyAllWindows()
