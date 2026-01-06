import time
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from pathlib import Path

from src.face_detection import TaskFaceDetector
from src.move_detection import process_mog2
from src.person_detection import check_for_person
from src.image_quality import calculate_sharpness, is_blurry

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
prev_brightness = None
BRIGHTNESS_RESET_DELTA = 40
ADAPT_LR = 0.01

root = Path(__file__).resolve().parents[1]

images_dir = Path("images")
images_dir.mkdir(parents=True, exist_ok=True)

model_path = Path("models") / "yolo12n.pt"
model = YOLO(str(model_path))

MOTION_THRESHOLD = 7000
BLUR_THRESHOLD = 100.0
CAPTURE_ATTEMPTS = 5

person_counter = 0
frame_index = 0

face_model_path = Path("models") / "blaze_face_short_range.tflite"
detector = TaskFaceDetector(
    model_path=str(face_model_path),
    min_detection_confidence=0.7,
    running_mode="VIDEO",
)


def capture_sharpest_faces(cap, detector, num_attempts: int = 5):
    """
    Capture multiple frames and return the sharpest one with detected faces.
    Memory efficient - only keeps the best candidate.
    """
    best_frame = None
    best_faces = None
    best_sharpness = 0.0
    best_attempt = -1

    for attempt in range(num_attempts):
        ret, frame = cap.read()
        if not ret:
            continue

        ts_ms = int(time.time() * 1000)
        faces = detector.detect(frame, timestamp_ms=ts_ms)

        if not faces:
            continue

        # Calculate average sharpness of all detected faces
        total_sharpness = 0.0
        for f in faces:
            crop = frame[f.y1:f.y2, f.x1:f.x2]
            if crop.size > 0:
                total_sharpness += calculate_sharpness(crop)

        avg_sharpness = total_sharpness / len(faces)
        print(f"  Attempt {attempt + 1}: sharpness={avg_sharpness:.2f}, faces={len(faces)}")

        # Keep only if sharper than previous best
        if avg_sharpness > best_sharpness:
            best_sharpness = avg_sharpness
            best_frame = frame
            best_faces = faces
            best_attempt = attempt + 1

    if best_attempt > 0:
        print(f"  Selected attempt {best_attempt} with sharpness {best_sharpness:.2f}")

    return best_frame, best_faces, best_sharpness


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

                        best_frame, best_faces, sharpness = capture_sharpest_faces(
                            cap, detector, CAPTURE_ATTEMPTS
                        )

                        if best_frame is not None and best_faces:
                            ts_ms = int(time.time() * 1000)
                            for i, f in enumerate(best_faces):
                                cropped_face = best_frame[f.y1:f.y2, f.x1:f.x2]
                                if cropped_face.size > 0 and not is_blurry(cropped_face, BLUR_THRESHOLD):
                                    cv.imwrite(str(images_dir / f"detected_face_{ts_ms}_{i}.png"), cropped_face)
                                    print(f"Saved face {i} with sharpness: {sharpness:.2f}")
                                else:
                                    print(f"Skipped blurry face {i}")

                        # Clear references
                        best_frame = None
                        best_faces = None
                        break
            else:
                person_counter = 0

            cv.imshow("fgMask", fgMask)

        if cv.waitKey(1) == ord("q"):
            break

        frame_index += 1

finally:
    cap.release()
    detector.close()
    cv.destroyAllWindows()
