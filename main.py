import time
import os
import numpy as np
import cv2 as cv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

# ONNX Runtime configuration
os.environ["ONNXRUNTIME_LOG_LEVEL"] = "3"  # Suppress info messages (error only)
os.environ.setdefault("OMP_NUM_THREADS", "4")  # Thread count for inference
os.environ.setdefault("ORT_DISABLE_TENSORRT", "1")  # Disable TensorRT by default

# OpenCV optimization settings
cv.setUseOptimized(True)
cv.setNumThreads(4)

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

# Camera initialization
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

camera = CameraBuffer(cap)

# Background subtraction configuration
backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
prev_brightness = None
BRIGHTNESS_RESET_DELTA = 40  # Threshold for brightness change detection
ADAPT_LR = 0.01  # Background model learning rate

root = Path(__file__).resolve().parents[1]

to_be_labeled_dir = Path("ToBeLabeled")
to_be_labeled_dir.mkdir(parents=True, exist_ok=True)

# Detection and recognition thresholds
MOTION_THRESHOLD = 16000  # Minimum pixels for motion detection
BLUR_THRESHOLD = 100.0  # Laplacian variance threshold for blur detection
CAPTURE_ATTEMPTS = 5  # Number of frames to capture for best quality selection
RECOGNITION_THRESHOLD = 2  # Required consecutive recognitions for confirmation
COOLDOWN_SECONDS = 300  # Notification cooldown per person (seconds)
AUTO_LABEL_THRESHOLD = 0.80  # Minimum confidence for automatic gallery addition

person_counter = 0

# Async executors for non-blocking I/O operations
notification_executor = ThreadPoolExecutor(max_workers=1)
save_executor = ThreadPoolExecutor(max_workers=2)

# Frame processing interval (seconds)
PROCESS_INTERVAL = 0.5
last_process_time = 0.0

# Recognition state tracking
recognition_votes: Dict[str, int] = {}  # Vote count per person
last_confirmed: Dict[str, float] = {}  # Last confirmation timestamp per person

# Face recognition setup
friends_db = Path("friends_db")
recognizer = DoorFaceRecognizer(
    providers=None,  # Auto-detect: TensorRT > CUDA > CPU
    det_size=(160, 160)  # Detection resolution (lower = faster)
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

def capture_sharpest_faces(
    cap,
    recognizer: DoorFaceRecognizer,
    num_attempts: int = 5,
    initial_frame: Optional[np.ndarray] = None,
    initial_faces: Optional[List] = None,
) -> Tuple[Optional[np.ndarray], List, float]:
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

    def _evaluate(frame: np.ndarray, faces: Optional[List], attempt_no: int) -> bool:
        nonlocal best_frame, best_faces, best_sharpness, best_attempt
        if frame is None:
            return False

        # Use InsightFace for detection (reuses the same model as recognition)
        if faces is None:
            faces = recognizer.app.get(frame)

        if not faces:
            return False

        # Calculate average sharpness of all detected faces
        total_sharpness = 0.0
        for f in faces:
            bbox = f.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                total_sharpness += calculate_sharpness(crop)

        avg_sharpness = total_sharpness / len(faces)
        print(f"  Attempt {attempt_no}: sharpness={avg_sharpness:.2f}, faces={len(faces)}")

        if avg_sharpness > best_sharpness:
            best_sharpness = avg_sharpness
            best_frame = frame
            best_faces = faces
            best_attempt = attempt_no

            if avg_sharpness >= GOOD_SHARPNESS_THRESHOLD:
                print(f"  Early exit: sharpness {avg_sharpness:.2f} exceeds threshold")
                return True
        return False

    attempt_no = 1
    if initial_frame is not None:
        if _evaluate(initial_frame, initial_faces, attempt_no):
            return best_frame, best_faces, best_sharpness
        attempt_no += 1

    for _ in range(attempt_no - 1, num_attempts):
        ret, frame = cap.read()
        if not ret or frame is None:
            attempt_no += 1
            continue
        if _evaluate(frame, None, attempt_no):
            break
        attempt_no += 1

    if best_attempt > 0:
        print(f"  Selected attempt {best_attempt} with sharpness {best_sharpness:.2f}")

    return best_frame, best_faces, best_sharpness


app_get = recognizer.app.get
is_blurry_fn = is_blurry
time_time = time.time


def submit_save(dest_path: Path, crop: np.ndarray, notify_args: Optional[Tuple[str, float, str]] = None) -> None:
    crop_copy = crop.copy()
    def _save() -> bool:
        return cv.imwrite(str(dest_path), crop_copy, [cv.IMWRITE_JPEG_QUALITY, 85])

    future = save_executor.submit(_save)

    if notify_args is not None:
        def _after_save(fut):
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if ok:
                name, sim, path_str = notify_args
                notification_executor.submit(notifier.notify_person, name, sim, path_str)
        future.add_done_callback(_after_save)

try:
    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.01)  # Wait for camera to provide frame
            continue

        current_time = time_time()
        
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
                draw_overlay=False,
            )

            if motion:
                quick_faces = app_get(frame)
                
                if quick_faces:
                    person_counter += 1
                    if person_counter == 4:
                        print("Face detected - capturing best frame!")

                        best_frame, best_faces, sharpness = capture_sharpest_faces(
                            cap,
                            recognizer,
                            CAPTURE_ATTEMPTS,
                            initial_frame=frame,
                            initial_faces=quick_faces,
                        )

                        if best_frame is not None and best_faces:
                            ts_ms = int(time.time() * 1000)
                            crops: List[Optional[np.ndarray]] = []
                            for i, f in enumerate(best_faces):
                                bbox = f.bbox.astype(int)
                                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                                cropped_face = best_frame[y1:y2, x1:x2]
                                if cropped_face.size > 0 and not is_blurry_fn(cropped_face, BLUR_THRESHOLD):
                                    crops.append(cropped_face)
                                    print(f"Accepted face {i} with sharpness: {sharpness:.2f}")
                                else:
                                    crops.append(None)
                                    print(f"Skipped blurry face {i}")

                            if best_faces and recognizer.gallery:
                                results = recognizer.identify_from_faces(best_faces, sim_thresh=0.70)
                                current_time = time_time()
                                
                                for idx, (name, sim) in enumerate(results):
                                    crop = crops[idx] if idx < len(crops) else None
                                    if crop is None:
                                        print("  Skipping notification/save: crop not available")
                                        continue
                                    
                                    if name:
                                        if name in last_confirmed:
                                            elapsed = current_time - last_confirmed[name]
                                            if elapsed < COOLDOWN_SECONDS:
                                                print(f"  {name} on cooldown ({COOLDOWN_SECONDS - elapsed:.1f}s remaining)")
                                                continue
                                        
                                        recognition_votes[name] = recognition_votes.get(name, 0) + 1
                                        print(f"  Vote for {name}: {recognition_votes[name]}/{RECOGNITION_THRESHOLD} (sim={sim:.3f})")
                                        
                                        if recognition_votes[name] >= RECOGNITION_THRESHOLD:
                                            print(f"*** CONFIRMED: {name} at the door! ***")
                                            last_confirmed[name] = current_time
                                            recognition_votes[name] = 0
                                            
                                            if sim >= AUTO_LABEL_THRESHOLD:
                                                dest_dir = friends_db / name
                                                dest_dir.mkdir(parents=True, exist_ok=True)
                                                dest_path = dest_dir / f"{ts_ms}_{idx}.jpg"
                                                submit_save(dest_path, crop, (name, sim, str(dest_path)))
                                                print(f"  Auto-added to {dest_dir.name}/ (sim={sim:.0%})")
                                            else:
                                                dest_path = to_be_labeled_dir / f"{name}_{ts_ms}_{idx}.jpg"
                                                submit_save(dest_path, crop, (name, sim, str(dest_path)))
                                                print(f"  Saved to ToBeLabeled/ for review (sim={sim:.0%})")
                                    else:
                                        print(f"  Unknown visitor (best_sim={sim:.3f})")
                                        dest_path = to_be_labeled_dir / f"unknown_{ts_ms}_{idx}.jpg"
                                        submit_save(dest_path, crop)
                                        print(f"  Saved unknown face to ToBeLabeled/")
                            elif not recognizer.gallery:
                                print("Gallery empty; skip recognition.")

                        # Clear references and reset person counter for next detection
                        best_frame = None
                        best_faces = None
                        person_counter = 0
            else:
                person_counter = 0

        time.sleep(0.01)

finally:
    camera.stop()
    cap.release()
    notification_executor.shutdown(wait=False)
    save_executor.shutdown(wait=False)
