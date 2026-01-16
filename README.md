# Project1 – Door Person Recognition

Detects motion/persons and identifies known friends from a local gallery.

## Setup (Linux)

- Python 3.9+ recommended.
- Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install ultralytics opencv-python insightface onnxruntime scikit-learn numpy
```

## Models

Place model files in `models/`:
- `models/yolo12n.pt` (Ultralytics YOLO model)
- `models/blaze_face_short_range.tflite` (MediaPipe BlazeFace short range)

The repo keeps the folder but ignores contents; add your own weights locally.

## Friend gallery

Add reference images to `friends_db/`:
```
friends_db/
  Alice/
    1.jpg
    2.jpg
  Bob/
    pic1.png
```
- Use 5–10 diverse photos per person (lighting, glasses, slight angles).
- The app builds embeddings at startup and identifies visitors from cropped, non‑blurry face frames.

## Run

```bash
source .venv/bin/activate
python main.py
```

- Press `q` to quit the video window.

## Notes

- Recognition uses InsightFace (ArcFace) embeddings with cosine similarity and voting over multiple crops.
- Tune thresholds in `main.py` / `src/friend_detection.py`:
  - Similarity threshold: ~0.65–0.75
  - Votes needed: 2+
- If faces are small, increase `det_size` in `DoorFaceRecognizer` to `(960, 960)`.

## Git ignore

- `images/`, `media/`, `models/`, and `friends_db/` contents are ignored by Git.
- `.gitkeep` files keep the directories themselves under version control.
