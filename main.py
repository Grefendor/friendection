import time
import shutil
import os
import numpy as np
import cv2 as cv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

# Suppress ONNX Runtime info messages (0=verbose, 3=error only)
os.environ["ONNXRUNTIME_LOG_LEVEL"] = "3"

from src.move_detection import process_mog2
from src.image_quality import calculate_sharpness, is_blurry
from src.friend_detection import DoorFaceRecognizer
from src.telegram_bot import DoorNotifier


class CameraBuffer:
    """
    Threaded camera capture to eliminate frame read blocking.
    Continuously reads frames in background, always provides latest frame.
    """
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ret = False
        self.stopped = False
        self._thread = Thread(target=self._reader, daemon=True)
        self._thread.start()
    
    def _reader(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
    
    def read(self):
        return self.ret, self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.stopped = True
        self._thread.join(timeout=1.0)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Optimization: Reduce resolution for faster processing on edge devices
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Threaded camera buffer - eliminates blocking on frame reads
camera = CameraBuffer(cap)

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

MOTION_THRESHOLD = 7000
BLUR_THRESHOLD = 100.0
CAPTURE_ATTEMPTS = 5
RECOGNITION_THRESHOLD = 2  # Require this many successful recognitions to confirm
COOLDOWN_SECONDS = 300  # 5 minutes - wait before re-notifying for same person
AUTO_LABEL_THRESHOLD = 0.80  # Auto-add to gallery if confidence >= 80%

person_counter = 0

# Thread pool for async Telegram notifications (non-blocking)
notification_executor = ThreadPoolExecutor(max_workers=1)

# Time-based processing instead of frame counting (more efficient)
PROCESS_INTERVAL = 0.5  # Process every 0.5 seconds instead of every 30 frames
last_process_time = 0.0

# Track recognition votes per person across detection events
recognition_votes: Dict[str, int] = {}
last_confirmed: Dict[str, float] = {}  # name -> timestamp of last confirmed recognition

# Build face gallery once at startup (friends_db/Name/*.jpg)
# Using InsightFace for BOTH detection and recognition (single model, more efficient)
friends_db = Path("friends_db")
recognizer = DoorFaceRecognizer(
    providers=["CPUExecutionProvider"],
    det_size=(320, 320)  # Smaller detection size for faster processing on edge devices
)
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

def capture_sharpest_faces(cap, recognizer: DoorFaceRecognizer, num_attempts: int = 5) -> Tuple[Optional[np.ndarray], List, float]:
    """
    Capture multiple frames and return the sharpest one with detected faces.
    Uses InsightFace for detection (same model used for recognition).
    Memory efficient - only keeps the best candidate.
    Early exit if sharpness is good enough.
    """
    GOOD_SHARPNESS_THRESHOLD = 200.0  # If sharpness exceeds this, return immediately
    
    best_frame = None
    best_faces = None
    best_sharpness = 0.0
    best_attempt = -1

    for attempt in range(num_attempts):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Use InsightFace for detection (reuses the same model as recognition)
        faces = recognizer.app.get(frame)

        if not faces:
            continue

        # Calculate average sharpness of all detected faces
        total_sharpness = 0.0
        for f in faces:
            bbox = f.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            crop = frame[y1:y2, x1:x2]
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
            
            # Early exit if sharpness is good enough
            if avg_sharpness >= GOOD_SHARPNESS_THRESHOLD:
                print(f"  Early exit: sharpness {avg_sharpness:.2f} exceeds threshold")
                break

    if best_attempt > 0:
        print(f"  Selected attempt {best_attempt} with sharpness {best_sharpness:.2f}")

    return best_frame, best_faces, best_sharpness


try:
    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.01)  # Wait for camera to provide frame
            continue

        current_time = time.time()
        
        # Time-based processing: only process at specified interval
        if current_time - last_process_time >= PROCESS_INTERVAL:
            last_process_time = current_time
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
                # Use InsightFace directly for face detection (no YOLO needed)
                # If we detect a face, a person is definitely there
                quick_faces = recognizer.app.get(frame)
                
                if quick_faces:
                    person_counter += 1
                    if person_counter == 4:
                        print("Face detected - capturing best frame!")

                        best_frame, best_faces, sharpness = capture_sharpest_faces(
                            cap, recognizer, CAPTURE_ATTEMPTS
                        )

                        if best_frame is not None and best_faces:
                            ts_ms = int(time.time() * 1000)
                            crops = []
                            for i, f in enumerate(best_faces):
                                bbox = f.bbox.astype(int)
                                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                                cropped_face = best_frame[y1:y2, x1:x2]
                                if cropped_face.size > 0 and not is_blurry(cropped_face, BLUR_THRESHOLD):
                                    crops.append(cropped_face)
                                    # Use JPEG for faster saves (5-10x faster than PNG)
                                    cv.imwrite(str(images_dir / f"detected_face_{ts_ms}_{i}.jpg"), cropped_face,
                                               [cv.IMWRITE_JPEG_QUALITY, 85])
                                    print(f"Saved face {i} with sharpness: {sharpness:.2f}")
                                else:
                                    print(f"Skipped blurry face {i}")

                            # Identify faces using embeddings already extracted by InsightFace
                            if best_faces and recognizer.gallery:
                                # Use embeddings directly from detection (more efficient)
                                results = recognizer.identify_from_faces(best_faces, sim_thresh=0.70)
                                current_time = time.time()
                                
                                for idx, (name, sim) in enumerate(results):
                                    # Get the saved photo path for this crop
                                    photo_path = str(images_dir / f"detected_face_{ts_ms}_{idx}.jpg")
                                    
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
                                            
                                            # Move file first, then send notification with final path
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
                                            
                                            # Send Telegram notification asynchronously (non-blocking)
                                            # Use dest_path since file was already moved
                                            notification_executor.submit(notifier.notify_person, name, sim, str(dest_path))
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

finally:
    camera.stop()
    cap.release()
    notification_executor.shutdown(wait=False)  # Don't wait for pending notifications
