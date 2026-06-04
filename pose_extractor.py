"""
pose_extractor.py
=================
Extracts 3D pose landmarks from video using MediaPipe Tasks API.
Downloads the pose landmarker model (~6MB) on first run automatically.
"""

import cv2
import numpy as np
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
MODEL_PATH = Path(__file__).parent / "pose_landmarker.task"

LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
    "left_heel": 29,     "right_heel": 30,
    "left_toe": 31,      "right_toe": 32,
}


def _ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        return
    print("  Downloading pose model (~6MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"  Model ready ({MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


def angle_3pt(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def vertical_angle(p1, p2):
    vec = np.array(p2) - np.array(p1)
    vert = np.array([0.0, -1.0, 0.0])
    cos_a = np.dot(vec, vert) / (np.linalg.norm(vec) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(abs(cos_a), -1.0, 1.0))))


def height_diff_percent(p1, p2, frame_h):
    return float(abs(p1[1] - p2[1]) / (frame_h + 1e-8) * 100)


def midpoint(p1, p2):
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2]


def compute_measurements(lm, frame_h):
    m = {}
    hip_width_norm = None

    if all(k in lm for k in ["left_shoulder", "left_elbow", "left_wrist"]):
        m["left_elbow_angle"] = angle_3pt(lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"])
    if all(k in lm for k in ["right_shoulder", "right_elbow", "right_wrist"]):
        m["right_elbow_angle"] = angle_3pt(lm["right_shoulder"], lm["right_elbow"], lm["right_wrist"])

    if all(k in lm for k in ["left_hip", "left_knee", "left_ankle"]):
        m["left_knee_angle"] = angle_3pt(lm["left_hip"], lm["left_knee"], lm["left_ankle"])
    if all(k in lm for k in ["right_hip", "right_knee", "right_ankle"]):
        m["right_knee_angle"] = angle_3pt(lm["right_hip"], lm["right_knee"], lm["right_ankle"])

    if "left_hip" in lm and "right_hip" in lm:
        hip_width_norm = abs(lm["left_hip"][0] - lm["right_hip"][0]) + 1e-6
        m["hip_width_norm"] = float(hip_width_norm)
        m["hip_midpoint"] = midpoint(lm["left_hip"], lm["right_hip"])
        m["hip_height_diff_pct"] = height_diff_percent(lm["left_hip"], lm["right_hip"], frame_h)

    if "left_shoulder" in lm and "right_shoulder" in lm:
        m["shoulder_midpoint"] = midpoint(lm["left_shoulder"], lm["right_shoulder"])
        m["shoulder_height_diff_pct"] = height_diff_percent(lm["left_shoulder"], lm["right_shoulder"], frame_h)

    if "hip_midpoint" in m and "shoulder_midpoint" in m:
        m["trunk_lean_angle"] = vertical_angle(m["hip_midpoint"], m["shoulder_midpoint"])
        # Anterior tilt proxy: horizontal offset between hip and shoulder midpoints
        # normalised by vertical distance so it's scale-invariant
        vert_dist = abs(m["shoulder_midpoint"][1] - m["hip_midpoint"][1]) + 1e-6
        horiz_offset = m["hip_midpoint"][0] - m["shoulder_midpoint"][0]
        import math
        m["anterior_tilt_deg"] = math.degrees(math.atan2(abs(horiz_offset), vert_dist))

    if "nose" in lm and "hip_midpoint" in m:
        m["spine_vertical_angle"] = vertical_angle(lm["nose"], m["hip_midpoint"])

    if "left_heel" in lm and "left_ankle" in lm:
        m["left_heel_rise_pct"] = height_diff_percent(lm["left_heel"], lm["left_ankle"], frame_h)
    if "right_heel" in lm and "right_ankle" in lm:
        m["right_heel_rise_pct"] = height_diff_percent(lm["right_heel"], lm["right_ankle"], frame_h)

    if "left_knee" in lm and "left_toe" in lm:
        m["left_knee_toe_offset"] = abs(lm["left_knee"][0] - lm["left_toe"][0])
        if hip_width_norm:
            m["left_knee_toe_offset_norm"] = m["left_knee_toe_offset"] / hip_width_norm
    if "right_knee" in lm and "right_toe" in lm:
        m["right_knee_toe_offset"] = abs(lm["right_knee"][0] - lm["right_toe"][0])
        if hip_width_norm:
            m["right_knee_toe_offset_norm"] = m["right_knee_toe_offset"] / hip_width_norm

    # Knee height difference (proxy for working-leg height in retiré/développé)
    if "left_knee" in lm and "right_knee" in lm:
        m["knee_height_diff_pct"] = height_diff_percent(lm["left_knee"], lm["right_knee"], frame_h)

    # Foot spread (horizontal distance between toes — proxy for 2nd position opening)
    if "left_toe" in lm and "right_toe" in lm:
        m["foot_spread_norm"] = abs(lm["left_toe"][0] - lm["right_toe"][0])

    if "left_wrist" in lm and "left_elbow" in lm:
        m["left_wrist_drop_pct"] = height_diff_percent(lm["left_wrist"], lm["left_elbow"], frame_h)
    if "right_wrist" in lm and "right_elbow" in lm:
        m["right_wrist_drop_pct"] = height_diff_percent(lm["right_wrist"], lm["right_elbow"], frame_h)

    if all(k in lm for k in ["left_hip", "right_hip", "right_ankle"]):
        m["arabesque_angle"] = angle_3pt(lm["left_hip"], lm["right_hip"], lm["right_ankle"])

    if "left_hip" in lm and "right_hip" in lm:
        hip_vec = np.array(lm["right_hip"]) - np.array(lm["left_hip"])
        m["hip_rotation_deg"] = float(abs(np.degrees(np.arctan2(hip_vec[2], hip_vec[0]))))

    return m


def extract_frames(video_path, sample_every_n=3):
    """
    Extract pose landmarks from a video file.
    Returns dict: { frames, fps, frame_w, frame_h, total_frames, sampled_frames }
    """
    _ensure_model()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frame_idx = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n == 0:
                timestamp_ms = int(frame_idx / fps * 1000)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, timestamp_ms)

                if result.pose_landmarks:
                    raw = result.pose_landmarks[0]
                    named, vis = {}, {}
                    for name, idx in LM.items():
                        if idx < len(raw):
                            p = raw[idx]
                            named[name] = [p.x, p.y, p.z]
                            vis[name] = getattr(p, "visibility", 0.9)

                    frames.append({
                        "frame_idx": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "landmarks": named,
                        "measurements": compute_measurements(named, frame_h),
                        "visibility": vis,
                    })

            frame_idx += 1

    cap.release()
    return {
        "frames": frames,
        "fps": fps,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "total_frames": total_frames,
        "sampled_frames": len(frames),
    }
