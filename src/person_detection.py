from pathlib import Path
import cv2
from ultralytics import YOLO

def check_for_person(frame, model) -> bool:
    results = model.predict(
        frame,
        device="cpu",
        conf=0.25,
        verbose=False,
        save=False,  # prevent YOLO from writing runs/detect/predict folders
        save_txt=False,
        save_conf=False,
        save_crop=False,
    )

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == "person" and box.conf[0] >= 0.25:
                return True
    return False
