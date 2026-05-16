#!/usr/bin/env python3
"""
Usage: python tomato.py <image> <part>
Example: python tomato.py photo.jpg nose
"""

import sys
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

TOMATO_GIF = "tomato.gif"

# COCO 17-point pose keypoint indices
KP = {
    "nose":           0,
    "left_eye":       1,
    "right_eye":      2,
    "left_ear":       3,
    "right_ear":      4,
    "left_shoulder":  5,
    "right_shoulder": 6,
    "left_elbow":     7,
    "right_elbow":    8,
    "left_wrist":     9,
    "right_wrist":    10,
    "left_hip":       11,
    "right_hip":      12,
    "left_knee":      13,
    "right_knee":     14,
    "left_ankle":     15,
    "right_ankle":    16,
}

PART_MAP = {
    "nose":           "nose",
    "face":           "nose",
    "eye":            "left_eye",
    "left_eye":       "left_eye",
    "right_eye":      "right_eye",
    "left_ear":       "left_ear",
    "right_ear":      "right_ear",
    "ear":            "left_ear",
    "hand":           "right_wrist",
    "left_hand":      "left_wrist",
    "right_hand":     "right_wrist",
    "shoulder":       "left_shoulder",
    "left_shoulder":  "left_shoulder",
    "right_shoulder": "right_shoulder",
    "chest":          "chest",
    "elbow":          "left_elbow",
    "left_elbow":     "left_elbow",
    "right_elbow":    "right_elbow",
    "hip":            "left_hip",
    "left_hip":       "left_hip",
    "right_hip":      "right_hip",
    "knee":           "left_knee",
    "left_knee":      "left_knee",
    "right_knee":     "right_knee",
    "ankle":          "left_ankle",
    "left_ankle":     "left_ankle",
    "right_ankle":    "right_ankle",
}


def get_point(image_bgr, part_key):
    model = YOLO("yolov8n-pose.pt")
    results = model(image_bgr, verbose=False)
    if not results or results[0].keypoints is None:
        return None
    kps = results[0].keypoints.xy[0].cpu().numpy()
    if part_key == "chest":
        ls, rs = kps[KP["left_shoulder"]], kps[KP["right_shoulder"]]
        x, y = int((ls[0] + rs[0]) / 2), int((ls[1] + rs[1]) / 2)
    else:
        pt = kps[KP[part_key]]
        x, y = int(pt[0]), int(pt[1])
    if x == 0 and y == 0:
        return None
    return x, y


def load_tomato_frames():
    gif = Image.open(TOMATO_GIF)
    duration = gif.info.get("duration", 100)
    frames = []
    try:
        while True:
            frames.append(gif.convert("RGBA"))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames, duration


def build_output_gif(photo_bgr, tomato_frames, point, duration):
    photo_rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(photo_rgb).convert("RGBA")
    h, w = photo_bgr.shape[:2]

    out_frames = []

    for frame in tomato_frames:
        tw, th = int(w * 1.5), int(h * 1.5)
        resized = frame.resize((tw, th), Image.LANCZOS)
        canvas = base.copy()
        x = point[0] - tw // 2
        y = point[1] - th // 2
        canvas.paste(resized, (x, y), resized)
        out_frames.append(canvas.convert("RGB").quantize(256))

    out_path = "output.gif"
    out_frames[0].save(
        out_path,
        save_all=True,
        append_images=out_frames[1:],
        loop=0,
        duration=duration,
        optimize=False,
    )
    return out_path


def main():
    if len(sys.argv) != 2:
        print("Usage: python tomato.py <image>")
        sys.exit(1)

    image_path = sys.argv[1]

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Could not load image: {image_path}")
        sys.exit(1)

    point = get_point(image_bgr, PART_MAP["face"])
    if point is None:
        print("Could not detect a face. Make sure the face is clearly visible.")
        sys.exit(1)

    print(f"Detected face at {point}, compositing...")
    tomato_frames, duration = load_tomato_frames()
    out_path = build_output_gif(image_bgr, tomato_frames, point, duration)
    print(f"Saved → {out_path}  ({len(tomato_frames)} frames)")


if __name__ == "__main__":
    main()
