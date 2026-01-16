# Real-Time Person And Face Capture

Real-time camera pipeline that detects motion, confirms a person with YOLO, then captures and saves the sharpest non-blurry face crops using MediaPipe Tasks face detection. Images are saved under `images/` for later review.

## Features
- Motion gating via MOG2 with automatic reinitialization on large brightness changes.
- Person confirmation with a YOLOv8 nano model before triggering face capture.
- Face detection with the BlazeFace short-range model (MediaPipe Tasks).
- Sharpness scoring and blur filtering to keep only crisp face crops.
- Lightweight CPU defaults; configurable thresholds in code.

## Project Layout
- [main.py](main.py): Orchestrates camera loop, motion gating, person check, and face capture.
- [src/face_detection.py](src/face_detection.py): MediaPipe Tasks face detector wrapper and `FaceBox` dataclass.
- [src/move_detection.py](src/move_detection.py): Background subtraction, brightness reset, and motion detection.
- [src/person_detection.py](src/person_detection.py): YOLO-based person presence check.
- [src/image_quality.py](src/image_quality.py): Sharpness/blur helpers.
- models/: Holds `yolo12n.pt` and `blaze_face_short_range.tflite`.
- images/: Output face crops are saved here.

## Requirements
- Python 3.10+ recommended.
- Camera accessible at device index 0.
- Python packages: `opencv-python`, `numpy`, `ultralytics`, `mediapipe`.

Install dependencies (preferably in a virtual environment):
```bash
pip install opencv-python numpy ultralytics mediapipe
```

## Models
- YOLO: `models/yolo12n.pt` (lightweight COCO model). Replace with another YOLO checkpoint if desired; keep class names intact for "person" detection.
- Face detector: `models/blaze_face_short_range.tflite` (MediaPipe Tasks). Use the short-range model for near-field webcam use.

## Running
```bash
python main.py
```

- Press `q` to exit.
- Face crops are written to `images/detected_face_<timestamp>_<idx>.png`.

## Processing Flow
1. Read frames from the default camera.
2. Apply MOG2 background subtraction; reset the model if brightness shifts by >40.
3. If motion pixels exceed the threshold (7000), run YOLO person detection on the frame.
4. After four consecutive person-positive checks, attempt up to five captures to pick the sharpest faces.
5. For each detected face, skip if blurry (Laplacian variance below 100); otherwise save the crop.

## Tunable Parameters (in `main.py`)
- `MOTION_THRESHOLD`: Pixels needed to flag motion (default 7000).
- `BRIGHTNESS_RESET_DELTA`: Rebuild background model when lighting changes (default 40).
- `ADAPT_LR`: Learning rate for the background model (default 0.01).
- `CAPTURE_ATTEMPTS`: Number of frames sampled to find the sharpest faces (default 5).
- `BLUR_THRESHOLD`: Blur cutoff for face crops (default 100.0).
- YOLO confidence and MediaPipe detection thresholds are set inside their respective modules.

## Tips and Troubleshooting
- If the camera fails to open, verify device permissions and the index in `cv.VideoCapture(0)`.
- For slower hardware, reduce `CAPTURE_ATTEMPTS` or raise `MOTION_THRESHOLD` to avoid extra work.
- To run on GPU with YOLO, adjust `device` in `src/person_detection.py` to your CUDA device (e.g., `device="0"`).
- Lighting changes can trigger a background reset; adjust `BRIGHTNESS_RESET_DELTA` if it happens too often.

## Next Steps
- Add CLI flags or a config file for thresholds.
- Log events (motion/person/face captures) with timestamps for auditing.
- Optionally store full frames alongside cropped faces.
