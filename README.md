# Project1 – Door Person Recognition 🚪

A lightweight door-camera pipeline that detects motion and persons, captures the sharpest face images, recognizes known people from a local gallery, and optionally sends Telegram alerts.

---

## ✅ Key Capabilities

- Motion detection using OpenCV MOG2 with adaptive background and brightness reinitialization.
- Person detection using Ultralytics YOLO (`yolo12n.pt`).
- Face detection via MediaPipe Tasks (wrapper `TaskFaceDetector`) — works with MediaPipe model assets (e.g. `blaze_face_short_range.tflite`).
- Face recognition using InsightFace (`buffalo_l`) to build per-person prototypes from a `friends_db/` gallery.
- Image-quality filtering (sharpness / blur check) and "capture the sharpest face" across multiple attempts.
- Multi-vote confirmation and cooldown to avoid spurious notifications.
- Auto-labeling: high-confidence recognitions are moved into the gallery; others are moved to `ToBeLabeled/` for review.
- Optional Telegram notifications (text or photo) via `DoorNotifier` (uses environment variables or `.env`).

---

## 🚀 Setup (Linux)

- Python 3.9+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install ultralytics opencv-python insightface onnxruntime scikit-learn numpy python-dotenv python-telegram-bot
```

Note: `python-dotenv` is optional but convenient for loading `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from a `.env` file.

---

## 📁 Models

Place model files in `models/` (the repository keeps the folder but ignores weights):

- `models/yolo12n.pt` — Ultralytics YOLO for person detection
- `models/blaze_face_short_range.tflite` — example MediaPipe face model (used by `TaskFaceDetector`)

InsightFace models are managed by `insightface` / `FaceAnalysis` (the `buffalo_l` model is used by the recognizer by default).

---

## 🧑‍🤝‍🧑 Friend gallery

Add people to `friends_db/` as subfolders:

```
friends_db/
  Alice/
    1.jpg
    2.jpg
  Bob/
    pic1.png
```

- Use 5–10 diverse photos per person (lighting, glasses, slight angles). Full images or face crops are both accepted.
- The gallery is built at startup by `DoorFaceRecognizer.build_gallery()` and stores a single prototype embedding per person.

---

## ⚙️ Running

```bash
source .venv/bin/activate
python main.py
```

- The program reads from the default camera (`cv2.VideoCapture(0)`), continuously looks for motion, verifies a person with YOLO, captures the sharpest face frames, runs recognition, and then acts (notify, auto-label, move files).
- To enable Telegram, set environment variables or create a `.env` in the project root with:

```
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

---

## 🧰 Tuning & Default Parameters

Default values (see `main.py`):

- MOTION_THRESHOLD = 7000 (motion_pixels threshold for detection)
- BLUR_THRESHOLD = 100.0 (Laplacian variance below which an image is considered blurry)
- CAPTURE_ATTEMPTS = 5 (frames to sample and choose the sharpest)
- RECOGNITION_THRESHOLD = 2 (number of votes required to confirm a recognized person)
- COOLDOWN_SECONDS = 10 (seconds to wait before re-confirming the same person)
- AUTO_LABEL_THRESHOLD = 0.80 (>= this similarity will auto-add the photo to the person's gallery)
- Recognition similarity used by default: sim_thresh ≈ 0.70

Tips:
- Increase `det_size` in `DoorFaceRecognizer` (e.g. to `(960, 960)`) if faces are small.
- Adjust `MOTION_THRESHOLD` based on your scene (busy streets vs. quiet porch).
- Increase `BLUR_THRESHOLD` to be stricter about accepting only sharp crops.

---

## 📦 File / Folder Behavior

- Sharp, accepted face crops are saved to `images/` temporarily and then moved:
  - Confirmed/high-confidence matches → appended into `friends_db/<Name>/` (auto-label)
  - Low-confidence or unknown → moved to `ToBeLabeled/` for manual review
- `images/`, `media/`, `models/`, and `friends_db/` contents are ignored by Git; `.gitkeep` keeps directories tracked.

---

## 💡 Notes & Troubleshooting

- If the gallery is empty, recognition is skipped and faces are instead saved for manual labeling.
- `DoorNotifier` prints what it would send when Telegram credentials are missing; set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to enable live notifications.
- The project uses MediaPipe Tasks for face detection, InsightFace for embeddings, and YOLO for person verification.

---

## License

Add your project license here.
