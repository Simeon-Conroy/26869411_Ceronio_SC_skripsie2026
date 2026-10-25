"""Quick MediaPipe hand landmark display launched from the dashboard."""

from __future__ import annotations

import argparse
import ctypes
import os
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog

import h5py
import numpy as np

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
LIVE_FOLDER = Path(__file__).resolve().parent
DEFAULT_OUTPUT_FOLDER = PROJECT_FOLDER / "recordings" / "mediapipe"
DEFAULT_MODEL = LIVE_FOLDER / "hand_landmarker.task"
LANDMARK_COUNT = 21
COORD_COUNT = 3
MAX_HANDS = 2
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def import_camera_packages():
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("OpenCV is not installed. Install it with: python -m pip install opencv-python") from error
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
    except ModuleNotFoundError as error:
        raise RuntimeError("MediaPipe is not installed. Install it with: python -m pip install mediapipe opencv-python") from error
    return cv2, mp, BaseOptions, vision


def ask_text(title, prompt, initial=""):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    value = simpledialog.askstring(title, prompt, initialvalue=initial, parent=root)
    root.destroy()
    return value.strip() if value and value.strip() else None


def ask_save_path(default_folder: Path):
    default_folder.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filename = filedialog.asksaveasfilename(title="Save MediaPipe recording", initialdir=str(default_folder), initialfile=f"MediaPipe_{datetime.now():%Y%m%d_%H%M%S}.h5", defaultextension=".h5", filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")], parent=root)
    root.destroy()
    return Path(filename) if filename else None


def empty_frame_record(timestamp_s, frame_index, width, height, label=""):
    return {"timestamp_s": float(timestamp_s), "frame_index": int(frame_index), "width": int(width), "height": int(height), "label": label, "hand_count": 0, "handedness": ["", ""], "score": np.zeros((MAX_HANDS,), dtype=np.float32), "landmarks_norm": np.full((MAX_HANDS, LANDMARK_COUNT, COORD_COUNT), np.nan, dtype=np.float32), "landmarks_world": np.full((MAX_HANDS, LANDMARK_COUNT, COORD_COUNT), np.nan, dtype=np.float32)}


def category_label_score(categories):
    if not categories: return "", 0.0
    category = categories[0]
    label = getattr(category, "category_name", "") or getattr(category, "display_name", "") or ""
    score = float(getattr(category, "score", 0.0))
    return label, score


def extract_record(result, timestamp_s, frame_index, width, height, label=""):
    record = empty_frame_record(timestamp_s, frame_index, width, height, label)
    hands = getattr(result, "hand_landmarks", []) or []
    world_hands = getattr(result, "hand_world_landmarks", []) or []
    handedness = getattr(result, "handedness", []) or []
    count = min(MAX_HANDS, len(hands))
    record["hand_count"] = count
    for hand_index in range(count):
        if hand_index < len(handedness):
            label_name, score = category_label_score(handedness[hand_index])
            record["handedness"][hand_index] = label_name
            record["score"][hand_index] = score
        for landmark_index, landmark in enumerate(hands[hand_index][:LANDMARK_COUNT]):
            record["landmarks_norm"][hand_index, landmark_index] = (landmark.x, landmark.y, landmark.z)
        if hand_index < len(world_hands):
            for landmark_index, landmark in enumerate(world_hands[hand_index][:LANDMARK_COUNT]):
                record["landmarks_world"][hand_index, landmark_index] = (landmark.x, landmark.y, landmark.z)
    return record


def draw_landmarks(cv2, image, result):
    height, width = image.shape[:2]
    for hand in getattr(result, "hand_landmarks", []) or []:
        points = []
        for landmark in hand[:LANDMARK_COUNT]:
            x = int(np.clip(landmark.x, 0.0, 1.0) * width)
            y = int(np.clip(landmark.y, 0.0, 1.0) * height)
            points.append((x, y))
        for a, b in HAND_CONNECTIONS:
            if a < len(points) and b < len(points): cv2.line(image, points[a], points[b], (0, 220, 255), 2, cv2.LINE_AA)
        for point in points: cv2.circle(image, point, 4, (0, 255, 90), -1, cv2.LINE_AA)


def save_records(records, output_path: Path, camera_index, model_complexity, max_num_hands, model_path):
    if not records:
        print("Nothing to save.")
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([record["label"] for record in records], dtype=h5py.string_dtype("utf-8"))
    handedness = np.asarray([record["handedness"] for record in records], dtype=h5py.string_dtype("utf-8"))
    with h5py.File(output_path, "w") as file:
        meta = file.create_group("meta")
        meta.attrs["format"] = "Sharpa MediaPipe quick recording"
        meta.attrs["format_version"] = 2
        meta.attrs["created_at"] = datetime.now().astimezone().isoformat()
        meta.attrs["camera_index"] = int(camera_index)
        meta.attrs["model_complexity"] = int(model_complexity)
        meta.attrs["max_num_hands"] = int(max_num_hands)
        meta.attrs["model_path"] = str(model_path)
        meta.attrs["landmark_count"] = LANDMARK_COUNT
        meta.attrs["coordinate_order"] = "x,y,z"
        meta.attrs["landmarks_norm_units"] = "normalized image coordinates"
        meta.attrs["landmarks_world_units"] = "MediaPipe world landmark coordinates"
        group = file.create_group("mediapipe")
        group.create_dataset("timestamp_s", data=np.asarray([record["timestamp_s"] for record in records], dtype=np.float64))
        group.create_dataset("frame_index", data=np.asarray([record["frame_index"] for record in records], dtype=np.int64))
        group.create_dataset("image_size", data=np.asarray([[record["width"], record["height"]] for record in records], dtype=np.int32))
        group.create_dataset("label", data=labels)
        group.create_dataset("hand_count", data=np.asarray([record["hand_count"] for record in records], dtype=np.int8))
        group.create_dataset("handedness", data=handedness)
        group.create_dataset("score", data=np.asarray([record["score"] for record in records], dtype=np.float32), compression="gzip")
        group.create_dataset("landmarks_norm", data=np.asarray([record["landmarks_norm"] for record in records], dtype=np.float32), compression="gzip")
        group.create_dataset("landmarks_world", data=np.asarray([record["landmarks_world"] for record in records], dtype=np.float32), compression="gzip")
    print(f"Saved {len(records)} MediaPipe frames to:\n{output_path}")
    return True



def emit_window_handle(cv2, window_name, dock_slot):
    if os.name != "nt": return
    try:
        cv2.waitKey(1)
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        if hwnd:
            print(f"MEDIAPIPE_HWND {dock_slot} {int(hwnd)}", flush=True)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Quick MediaPipe hand landmark recorder")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--max-hands", type=int, default=1, choices=[1, 2], help="number of hands to track")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1], help="kept for H5 metadata")
    parser.add_argument("--width", type=int, default=960, help="requested webcam width")
    parser.add_argument("--height", type=int, default=540, help="requested webcam height")
    parser.add_argument("--fps", type=float, default=12.0, help="maximum display/tracking FPS")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="MediaPipe hand_landmarker.task path")
    parser.add_argument("--window-title", default="MediaPipe Quick Recorder", help="camera window title")
    parser.add_argument("--dock-slot", type=int, default=0, help="dashboard dock slot")
    args = parser.parse_args()

    try:
        cv2, mp, BaseOptions, vision = import_camera_packages()
    except RuntimeError as error:
        print(error)
        return 1

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"MediaPipe hand model not found:\n{model_path}")
        return 1

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open webcam index {args.camera}.")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    frame_index = 0
    min_frame_period = 1.0 / max(1.0, args.fps)
    next_frame_time = time.monotonic()
    window_name = args.window_title
    handle_emitted = False
    options = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)), running_mode=vision.RunningMode.VIDEO, num_hands=args.max_hands, min_hand_detection_confidence=0.55, min_hand_presence_confidence=0.55, min_tracking_confidence=0.55)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            now = time.monotonic()
            if now < next_frame_time:
                time.sleep(min(0.01, next_frame_time - now))
                continue
            next_frame_time = now + min_frame_period
            ok, frame = cap.read()
            if not ok:
                print("Webcam frame read failed.")
                break
            frame_index += 1
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
            draw_landmarks(cv2, frame, result)
            cv2.imshow(window_name, frame)
            if not handle_emitted:
                emit_window_handle(cv2, window_name, args.dock_slot)
                handle_emitted = True
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27): break
            if key in (ord("r"), ord("R")):
                recording = not recording
                print("Recording started." if recording else f"Recording stopped. Held frames: {len(records)}")
            elif key in (ord("c"), ord("C")):
                label = ask_text("MediaPipe snapshot", "Name this MediaPipe snapshot:", f"pose_{len(records) + 1:03d}")
                if label:
                    records.append(extract_record(result, time.time(), frame_index, width, height, label))
                    print(f"Captured {label} — {len(records)} held frames.")
            elif key in (ord("u"), ord("U")):
                if records:
                    removed = records.pop()
                    print(f"Removed {removed['label'] or 'frame'} — {len(records)} held frames remain.")
                else: print("Nothing to undo.")
            elif key in (ord("s"), ord("S")):
                output_path = ask_save_path(DEFAULT_OUTPUT_FOLDER)
                if output_path: save_records(records, output_path, args.camera, args.model_complexity, args.max_hands, model_path)
                else: print("Save cancelled. Frames are still held in memory.")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
