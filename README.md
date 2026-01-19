# Door Person Recognition System

A real-time door camera application that detects motion, captures faces, recognizes known individuals from a local gallery, and sends notifications via Telegram.

---

## ⚠️ Legal Notice & Consent Requirements

**IMPORTANT: This software uses facial recognition technology.**

Before using this software:
- **Get consent** from anyone whose face will be captured and stored
- **Check your local laws** - facial recognition may be restricted in your jurisdiction (GDPR, BIPA, CCPA, etc.)
- **Understand your responsibilities** - you are solely responsible for compliance with applicable laws

This is a personal project created for educational purposes. If you use or modify this software, ensure you comply with all relevant laws and obtain proper consent.

---

## Disclaimer

This README and the comments within the source code were generated with AI assistance.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Parameters](#parameters)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- Motion detection using OpenCV MOG2 with adaptive background modeling and automatic brightness compensation
- Face detection and recognition using InsightFace (buffalo_l model)
- Image quality filtering with sharpness analysis to capture optimal frames
- **Bootstrap mode**: Start with empty gallery - automatically captures faces to `ToBeLabeled/` folder for manual organization into `friends_db/`
- Multi-frame confirmation system to reduce false positives
- Cooldown mechanism to prevent notification spam
- Automatic gallery expansion for high-confidence recognitions (≥80% similarity)
- Manual review queue for uncertain detections in `ToBeLabeled/` folder
- Telegram notifications with photo attachments
- Automatic GPU detection and utilization (CUDA, TensorRT) when available
- Optimized for edge devices (Raspberry Pi, NVIDIA Jetson)

---

## Requirements

- Python 3.9 or higher
- Camera device (USB webcam or CSI camera)
- Supported platforms: Linux (x86_64, ARM64), Windows, macOS

### Hardware Recommendations

| Platform | Notes |
|----------|-------|
| Desktop/Laptop | CPU inference at approximately 10-12 FPS |
| NVIDIA GPU (Compute 6.0+) | Install onnxruntime-gpu for acceleration |
| Raspberry Pi 4/5 | Functional but slower; consider reducing det_size |
| NVIDIA Jetson | Use JetPack's onnxruntime-gpu wheel for TensorRT acceleration |

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Grefendor/friendection.git
cd Project1
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

4. (Optional) For NVIDIA GPU acceleration:

```bash
pip install onnxruntime-gpu
```

---

## Configuration

### Friend Gallery

You can build your friends database in two ways:

#### Option 1: Start from Scratch (Bootstrap Mode)

1. Run the application with an empty or non-existent `friends_db/` folder
2. The system will capture all detected faces to `ToBeLabeled/`
3. Manually organize captured faces by creating person folders:
   ```bash
   mkdir -p friends_db/PersonName
   mv ToBeLabeled/captured_*.jpg friends_db/PersonName/
   ```
4. Restart the application to load the new gallery

#### Option 2: Pre-populate with Photos

Create a directory structure with existing photos:

```
friends_db/
    PersonName1/
        photo1.jpg
        photo2.jpg
    PersonName2/
        photo1.png
        photo2.jpg
```

Guidelines:
- Use 5-10 photos per person with varied lighting and angles
- Both full images and cropped faces are accepted
- Supported formats: JPG, JPEG, PNG, BMP, WEBP
- **IMPORTANT**: Only add photos of individuals who have explicitly consented to facial recognition

### Telegram Notifications

Create a `.env` file in the project root:

```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

To obtain these credentials:
1. Create a bot via BotFather on Telegram
2. Send a message to your bot, then retrieve your chat ID via the Telegram API

---

## Usage

Start the application:

```bash
source .venv/bin/activate
python main.py
```

The application will:
1. Initialize the camera and face recognition models
2. Continuously monitor for motion
3. When motion is detected, attempt face detection
4. Compare detected faces against the gallery (if it exists)
5. Send notifications for confirmed recognitions
6. Automatically manage captured faces:
   - **No gallery exists**: All faces saved to `ToBeLabeled/` as `captured_*.jpg`
   - **Gallery exists, person recognized with ≥80% confidence**: Auto-added to their folder in `friends_db/`
   - **Gallery exists, person recognized with <80% confidence**: Saved to `ToBeLabeled/` with person's name for manual review
   - **Unknown person**: Saved to `ToBeLabeled/` as `unknown_*.jpg`

### Managing Captured Faces

Periodically review the `ToBeLabeled/` folder:

```bash
# Review captured faces
ls -lh ToBeLabeled/

# Add a new person to the gallery (after obtaining consent!)
mkdir -p friends_db/PersonName
mv ToBeLabeled/captured_* friends_db/PersonName/

# Move uncertain matches to confirm identity
mv ToBeLabeled/PersonName_*.jpg friends_db/PersonName/

# Delete unwanted captures
rm ToBeLabeled/unknown_*.jpg
```

After modifying `friends_db/`, restart the application to rebuild the gallery.

To stop the application, press `Ctrl+C`.

---

## Project Structure

```
Project1/
    main.py                 # Main application entry point
    requirements.txt        # Python dependencies
    .env                    # Telegram credentials (create manually)
    friends_db/             # Known person galleries
    ToBeLabeled/            # Unrecognized faces for review
    models/                 # Model files (auto-downloaded)
        buffalo_l/          # InsightFace face recognition model
    src/
        friend_detection.py # Face recognition logic
        move_detection.py   # Motion detection logic
        image_quality.py    # Sharpness calculation
        telegram_bot.py     # Notification system
    scripts/
        benchmark_compare.py # Performance comparison tool
```

---

## Parameters

Key parameters in `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| MOTION_THRESHOLD | 16000 | Pixel count threshold for motion detection |
| BLUR_THRESHOLD | 40.0 | Minimum Laplacian variance (higher = stricter) |
| CAPTURE_ATTEMPTS | 5 | Frames sampled to find sharpest face |
| RECOGNITION_THRESHOLD | 2 | Confirmations required before notification |
| COOLDOWN_SECONDS | 300 | Seconds before re-notifying for same person |
| AUTO_LABEL_THRESHOLD | 0.80 | Similarity threshold for automatic gallery addition |
| det_size | (160, 160) | Face detection resolution (lower = faster) |

### Tuning Recommendations

- Increase `MOTION_THRESHOLD` for high-traffic areas
- Decrease `det_size` to `(128, 128)` on Raspberry Pi for better performance
- Increase `det_size` to `(320, 320)` or higher if faces appear small in frame
- Adjust `BLUR_THRESHOLD` based on camera quality and lighting conditions

---

## Troubleshooting

### No faces detected
- Ensure adequate lighting
- Increase `det_size` if faces are small or distant
- Verify camera is functioning with a simple OpenCV test

### High CPU usage
- Reduce `det_size` to `(128, 128)` or `(160, 160)`
- Increase `PROCESS_INTERVAL` in main.py

### GPU not detected
- Verify CUDA installation with `nvidia-smi`
- Install correct onnxruntime-gpu version for your CUDA version
- For older GPUs (Compute capability below 6.0), CPU fallback is automatic

### Telegram notifications not working
- Verify `.env` file exists and contains valid credentials
- Ensure bot has been started (send /start to your bot)
- Check network connectivity

---

## License

MIT License with Commons Clause

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

**Commons Clause Restriction:**

The Software is provided to you by the Licensor under the License, as defined below, subject to the following condition:

Without limiting other conditions in the License, the grant of rights under the License will not include, and the License does not grant to you, the right to Sell the Software.

For purposes of the foregoing, "Sell" means practicing any or all of the rights granted to you under the License to provide to third parties, for a fee or other consideration (including without limitation fees for hosting or consulting/support services related to the Software), a product or service whose value derives, entirely or substantially, from the functionality of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
