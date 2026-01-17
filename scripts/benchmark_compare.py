#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO


def run_benchmark(repo_path: Path, video_path: Path, frames: int, stride: int) -> dict:
    sys.path.insert(0, str(repo_path))

    import cv2 as cv

    from src.friend_detection import DoorFaceRecognizer
    from src.move_detection import process_mog2

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    prev_brightness = None

    BRIGHTNESS_RESET_DELTA = 40
    ADAPT_LR = 0.01
    MOTION_THRESHOLD = 7000

    friends_db = repo_path / "friends_db"
    recognizer = DoorFaceRecognizer(
        providers=["CPUExecutionProvider"],
        det_size=(320, 320),
    )
    if friends_db.exists():
        recognizer.build_gallery(str(friends_db))

    total_frames = 0
    motion_frames = 0
    face_frames = 0
    recognized_faces = 0

    start = time.perf_counter()

    frame_idx = 0
    while total_frames < frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        total_frames += 1
        frame_idx += 1

        try:
            backSub, prev_brightness, _fgMask, motion = process_mog2(
                frame,
                backSub,
                prev_brightness,
                kernel,
                BRIGHTNESS_RESET_DELTA,
                ADAPT_LR,
                MOTION_THRESHOLD,
                draw_overlay=False,
            )
        except TypeError:
            backSub, prev_brightness, _fgMask, motion = process_mog2(
                frame,
                backSub,
                prev_brightness,
                kernel,
                BRIGHTNESS_RESET_DELTA,
                ADAPT_LR,
                MOTION_THRESHOLD,
            )

        if motion:
            motion_frames += 1
            faces = recognizer.app.get(frame)
            if faces:
                face_frames += 1
                if recognizer.gallery:
                    results = recognizer.identify_from_faces(faces, sim_thresh=0.70)
                    recognized_faces += sum(1 for name, _sim in results if name)

    elapsed = time.perf_counter() - start
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    cap.release()

    return {
        "frames": total_frames,
        "elapsed_sec": elapsed,
        "fps": fps,
        "motion_frames": motion_frames,
        "face_frames": face_frames,
        "recognized_faces": recognized_faces,
    }


def run_self(repo_path: Path, video_path: Path, frames: int, stride: int, label: str) -> dict:
    if os.environ.get("BENCH_QUIET") == "1":
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = run_benchmark(repo_path, video_path, frames, stride)
    else:
        result = run_benchmark(repo_path, video_path, frames, stride)
    result["label"] = label
    return result


def run_subprocess(repo_path: Path, video_path: Path, frames: int, stride: int, label: str) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--repo",
        str(repo_path),
        "--video",
        str(video_path),
        "--frames",
        str(frames),
        "--stride",
        str(stride),
        "--label",
        label,
        "--json",
    ]
    env = os.environ.copy()
    env["BENCH_QUIET"] = "1"
    out = subprocess.check_output(cmd, text=True, env=env)
    return json.loads(out)


def archive_head_to_temp(repo_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="benchmark_head_"))
    archive_data = subprocess.check_output(["git", "-C", str(repo_path), "archive", "HEAD"])
    with tempfile.NamedTemporaryFile(suffix=".tar") as tmp_tar:
        tmp_tar.write(archive_data)
        tmp_tar.flush()
        shutil.unpack_archive(tmp_tar.name, temp_dir)
    return temp_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare benchmark between current working tree and HEAD.")
    parser.add_argument("--repo", type=Path, default=None, help="Run benchmark for this repo path only")
    parser.add_argument("--video", type=Path, default=Path("media/vTest.mkv"))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--label", type=str, default="run")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.repo is not None:
        repo_path = args.repo.resolve()
        video_path = (repo_path / args.video).resolve() if not args.video.is_absolute() else args.video
        result = run_self(repo_path, video_path, args.frames, args.stride, args.label)
        if args.json:
            print(json.dumps(result))
        else:
            print(result)
        return

    repo_path = Path(__file__).resolve().parents[1]
    video_path = (repo_path / args.video).resolve() if not args.video.is_absolute() else args.video

    head_repo = archive_head_to_temp(repo_path)
    head_friends_db = head_repo / "friends_db"
    src_friends_db = repo_path / "friends_db"
    if src_friends_db.exists():
        shutil.copytree(src_friends_db, head_friends_db, dirs_exist_ok=True)

    try:
        head_result = run_subprocess(head_repo, video_path, args.frames, args.stride, "HEAD")
        current_result = run_subprocess(repo_path, video_path, args.frames, args.stride, "WORKTREE")
    finally:
        shutil.rmtree(head_repo, ignore_errors=True)

    print("Benchmark results:")
    print(json.dumps({"HEAD": head_result, "WORKTREE": current_result}, indent=2))

    if head_result["fps"] > 0:
        speedup = current_result["fps"] / head_result["fps"]
        print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
