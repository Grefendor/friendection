import time
import shutil
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from pathlib import Path
from typing import Dict

from src.face_detection import TaskFaceDetector
from src.move_detection import process_mog2
from src.person_detection import check_for_person
from src.image_quality import calculate_sharpness, is_blurry
from src.friend_detection import DoorFaceRecognizer
from src.telegram_bot import DoorNotifier

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

to_be_labeled_dir = Path("ToBeLabeled")
to_be_labeled_dir.mkdir(parents=True, exist_ok=True)

model_path = Path("models") / "yolo12n.pt"
model = YOLO(str(model_path))

MOTION_THRESHOLD = 7000
BLUR_THRESHOLD = 100.0
CAPTURE_ATTEMPTS = 5
RECOGNITION_THRESHOLD = 2  # Require this many successful recognitions to confirm
COOLDOWN_SECONDS = 10  # Wait before re-recognizing same person
AUTO_LABEL_THRESHOLD = 0.80  # Auto-add to gallery if confidence >= 80%

person_counter = 0
frame_index = 0

# Track recognition votes per person across detection events
recognition_votes: Dict[str, int] = {}
last_confirmed: Dict[str, float] = {}  # name -> timestamp of last confirmed recognition

face_model_path = Path("models") / "blaze_face_short_range.tflite"
detector = TaskFaceDetector(
    model_path=str(face_model_path),
    min_detection_confidence=0.7,
    running_mode="VIDEO",
)

# Build face gallery once at startup (friends_db/Name/*.jpg)
friends_db = Path("friends_db")
recognizer = DoorFaceRecognizer(providers=["CPUExecutionProvider"], det_size=(640, 640))
if friends_db.exists():
    recognizer.build_gallery(str(friends_db))
    if not recognizer.gallery:
        print(f"No embeddings built from {friends_db.resolve()}")
else:
    print(f"Gallery folder not found: {friends_db.resolve()}")

# Initialize Telegram notifier
notifier = DoorNotifier()
if notifier.is_configured():
    print("Telegram notifications enabled")
else:
    print("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.")

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
                            crops = []
                            for i, f in enumerate(best_faces):
                                cropped_face = best_frame[f.y1:f.y2, f.x1:f.x2]
                                if cropped_face.size > 0 and not is_blurry(cropped_face, BLUR_THRESHOLD):
                                    crops.append(cropped_face)
                                    cv.imwrite(str(images_dir / f"detected_face_{ts_ms}_{i}.png"), cropped_face)
                                    print(f"Saved face {i} with sharpness: {sharpness:.2f}")
                                else:
                                    print(f"Skipped blurry face {i}")

                            # Identify all faces in crops
                            if crops and recognizer.gallery:
                                results = recognizer.identify_all(crops, sim_thresh=0.70)
                                current_time = time.time()
                                
                                for idx, (name, sim) in enumerate(results):
                                    # Get the saved photo path for this crop
                                    photo_path = str(images_dir / f"detected_face_{ts_ms}_{idx}.png")
                                    
                                    if name:
                                        # Check cooldown
                                        if name in last_confirmed:
                                            elapsed = current_time - last_confirmed[name]
                                            if elapsed < COOLDOWN_SECONDS:
                                                print(f"  {name} on cooldown ({COOLDOWN_SECONDS - elapsed:.1f}s remaining)")
                                                continue
                                        
                                        # Add vote
                                        recognition_votes[name] = recognition_votes.get(name, 0) + 1
                                        print(f"  Vote for {name}: {recognition_votes[name]}/{RECOGNITION_THRESHOLD} (sim={sim:.3f})")
                                        
                                        # Check if threshold reached
                                        if recognition_votes[name] >= RECOGNITION_THRESHOLD:
                                            print(f"*** CONFIRMED: {name} at the door! ***")
                                            last_confirmed[name] = current_time
                                            recognition_votes[name] = 0  # Reset votes
                                            
                                            # Send Telegram notification
                                            notifier.notify_person(name, sim, photo_path)
                                            
                                            # Auto-add to gallery if high confidence, else to ToBeLabeled
                                            if sim >= AUTO_LABEL_THRESHOLD:
                                                dest_dir = friends_db / name
                                                dest_dir.mkdir(parents=True, exist_ok=True)
                                                dest_path = dest_dir / Path(photo_path).name
                                                shutil.move(photo_path, dest_path)
                                                print(f"  Auto-added to {dest_dir.name}/ (sim={sim:.0%})")
                                            else:
                                                dest_path = to_be_labeled_dir / f"{name}_{Path(photo_path).name}"
                                                shutil.move(photo_path, dest_path)
                                                print(f"  Moved to ToBeLabeled/ for review (sim={sim:.0%})")
                                    else:
                                        print(f"  Unknown visitor (best_sim={sim:.3f})")
                                        # Move unknown faces to ToBeLabeled
                                        if Path(photo_path).exists():
                                            dest_path = to_be_labeled_dir / f"unknown_{Path(photo_path).name}"
                                            shutil.move(photo_path, dest_path)
                                            print(f"  Moved unknown face to ToBeLabeled/")
                            elif not recognizer.gallery:
                                print("Gallery empty; skip recognition.")

                        # Clear references and reset person counter for next detection
                        best_frame = None
                        best_faces = None
                        person_counter = 0
            else:
                person_counter = 0

        # Small sleep to prevent CPU spinning (no GUI needed)
        time.sleep(0.01)

        frame_index += 1

finally:
    cap.release()
    detector.close()
