"""
exercise_analyser.py
====================
Database-driven analyser for non-plie ballet exercises.
Uses ballet_database.json terms/corrections and frame measurements from pose_extractor.
"""

import json
from collections import deque
from pathlib import Path


DB_PATH = Path(__file__).parent / "ballet_database.json"


def _load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


DB = _load_db()
TERMS = {t.get("id"): t for t in DB.get("terms", [])}


class EMA:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if new_value is None:
            return self.value
        if self.value is None:
            self.value = float(new_value)
        else:
            self.value = self.alpha * float(new_value) + (1.0 - self.alpha) * self.value
        return self.value


class NofMBuffer:
    def __init__(self, n, m):
        self.n = n
        self.history = deque(maxlen=m)

    def update(self, triggered):
        self.history.append(1 if triggered else 0)
        return sum(self.history) >= self.n


class Hysteresis:
    def __init__(self, on_thresh, off_thresh):
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh
        self.active = False

    def update(self, value):
        if value is None:
            self.active = False
            return False
        if not self.active and value >= self.on_thresh:
            self.active = True
        elif self.active and value < self.off_thresh:
            self.active = False
        return self.active


def gate(visibility, *landmark_names, threshold=0.6):
    return all(visibility.get(name, 0.0) >= threshold for name in landmark_names)


def _ms_to_ts(ms):
    s = int(ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def _grade_from_score(score):
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 55:
        return "Needs Work"
    return "Significant Corrections"


def compute_score(corrections, total_frames):
    if total_frames == 0:
        return 100
    penalty = 0.0
    for c in corrections:
        max_per = 18 if c["severity"] == "major" else 10
        # Cap frequency ratio at 0.75 so sustained corrections don't zero the score
        freq_ratio = min(c["frequency"] / max(total_frames, 1), 0.75)
        penalty += max_per * freq_ratio
    return max(0, round(100 - penalty))


MIN_SEGMENT_MS = 600  # ignore corrections shorter than this
MIN_SEGMENT_FRAME_PCT = 0.04  # must appear in ≥4% of frames to count


def build_correction_segments(all_triggered, frames, max_gap_frames=5):
    segments_by_id = {}
    for idx, triggered in enumerate(all_triggered):
        ts = frames[idx]["timestamp_ms"]
        for tid in triggered:
            segments_by_id.setdefault(tid, []).append((idx, ts))

    compact = {}
    for tid, points in segments_by_id.items():
        points.sort(key=lambda x: x[0])
        merged = []
        start_idx, start_ms = points[0]
        end_idx, end_ms = points[0]

        for idx, ts in points[1:]:
            if idx - end_idx <= max_gap_frames + 1:
                end_idx, end_ms = idx, ts
                continue
            merged.append(_make_segment(start_idx, end_idx, start_ms, end_ms))
            start_idx, start_ms = idx, ts
            end_idx, end_ms = idx, ts

        merged.append(_make_segment(start_idx, end_idx, start_ms, end_ms))
        compact[tid] = merged

    return compact


def _make_segment(start_idx, end_idx, start_ms, end_ms):
    return {
        "start_frame": start_idx,
        "end_frame": end_idx,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "start_ts": _ms_to_ts(start_ms),
        "end_ts": _ms_to_ts(end_ms),
        "duration_ms": max(0, end_ms - start_ms),
        "frames": (end_idx - start_idx) + 1,
    }


def _aggregate_corrections(all_triggered, frames, trigger_meta):
    counts = {}
    first_seen = {}
    segments_by_id = build_correction_segments(all_triggered, frames)

    total_frames = max(len(frames), 1)

    # Drop segments that are too short to be meaningful
    for tid in list(segments_by_id.keys()):
        segments_by_id[tid] = [
            s for s in segments_by_id[tid]
            if s["duration_ms"] >= MIN_SEGMENT_MS
        ]
        if not segments_by_id[tid]:
            del segments_by_id[tid]

    for i, triggered in enumerate(all_triggered):
        ts = frames[i]["timestamp_ms"]
        for tid in triggered:
            counts[tid] = counts.get(tid, 0) + 1
            if tid not in first_seen:
                first_seen[tid] = ts

    corrections = []
    for tid, meta in trigger_meta.items():
        if tid not in counts:
            continue
        # Require correction to appear in enough frames to be meaningful
        if counts[tid] / total_frames < MIN_SEGMENT_FRAME_PCT:
            continue
        # Require at least one valid segment after duration filtering
        if tid not in segments_by_id:
            continue
        corrections.append(
            {
                "id": tid,
                "severity": meta.get("severity", "warning"),
                "cue_pt": meta.get("cue_pt", ""),
                "cue_en": meta.get("cue_en", ""),
                "frequency": counts[tid],
                "first_seen_ms": first_seen[tid],
                "first_seen_ts": _ms_to_ts(first_seen[tid]),
                "segments": segments_by_id.get(tid, []),
            }
        )

    corrections.sort(key=lambda c: (0 if c["severity"] == "major" else 1, -c["frequency"]))
    return corrections


def _metric_name_from_trigger(trigger_id, condition):
    if trigger_id.startswith("arab_"):
        if trigger_id == "arab_001":
            return "arabesque_angle"
        if trigger_id == "arab_002":
            return "hip_rotation_deg"
        if trigger_id == "arab_003":
            return "trunk_lean_angle"
        return "shoulder_height_diff_pct"
    if trigger_id.startswith("tendu_"):
        if condition == "supporting_knee_bent":
            return "knee_angle"
        if condition == "hip_hiking":
            return "hip_height_diff_pct"
        if condition == "trunk_instability":
            return "trunk_lean_angle"
        return "foot_articulation_proxy"
    if trigger_id.startswith("bras_"):
        if trigger_id == "bras_001":
            return "shoulder_height_diff_pct"
        if trigger_id == "bras_002":
            return "elbow_angle"
        return "wrist_drop_pct"
    if trigger_id.startswith("gbat_"):
        if trigger_id == "gbat_001":
            return "hip_height_diff_pct"
        if trigger_id == "gbat_002":
            return "knee_angle"
        return "trunk_lean_angle"
    return "unknown_metric"


def _enrich_corrections(corrections, total_frames, trigger_meta, presence_confidence):
    enriched = []
    for c in corrections:
        segs = c.get("segments", [])
        longest_frames = max((s.get("frames", 0) for s in segs), default=0)
        freq_ratio = c.get("frequency", 0) / max(1, total_frames)
        seg_ratio = longest_frames / max(1, total_frames)
        severity_bonus = 0.12 if c.get("severity") == "major" else 0.04
        confidence = min(0.99, max(0.0, 0.4 * freq_ratio + 0.3 * seg_ratio + 0.2 * presence_confidence + 0.1 + severity_bonus))
        level = "high" if confidence >= 0.67 else ("medium" if confidence >= 0.4 else "low")

        meta = trigger_meta.get(c.get("id"), {})
        c2 = dict(c)
        c2["confidence"] = round(confidence, 3)
        c2["confidence_level"] = level
        c2["evidence"] = {
            "metric_name": _metric_name_from_trigger(c.get("id", ""), meta.get("condition", "")),
            "threshold": meta.get("threshold", "db_rule"),
            "frequency_ratio": round(freq_ratio, 3),
            "longest_segment_frames": longest_frames,
            "presence_confidence": round(presence_confidence, 3),
        }
        enriched.append(c2)
    return enriched


def _motion_profile(frames):
    if not frames:
        return {"motion": [], "peak_idx": 0}
    y = [f["measurements"].get("hip_midpoint", [0.5, 0.5, 0])[1] for f in frames]
    motion = [0.0]
    for i in range(1, len(y)):
        motion.append(abs(y[i] - y[i - 1]))
    peak_idx = max(range(len(motion)), key=lambda i: motion[i]) if motion else 0
    return {"motion": motion, "peak_idx": peak_idx}


def smooth_frames(frames, alpha=0.3):
    if not frames:
        return frames
    keys = [
        "left_knee_angle",
        "right_knee_angle",
        "trunk_lean_angle",
        "hip_height_diff_pct",
        "shoulder_height_diff_pct",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_wrist_drop_pct",
        "right_wrist_drop_pct",
        "arabesque_angle",
        "hip_rotation_deg",
        "left_knee_toe_offset_norm",
        "right_knee_toe_offset_norm",
    ]
    ema = {k: EMA(alpha=alpha) for k in keys}
    out = []
    for frame in frames:
        copied = dict(frame)
        m = dict(frame.get("measurements", {}))
        for k in keys:
            if k in m:
                m[k] = ema[k].update(m[k])
        copied["measurements"] = m
        out.append(copied)
    return out


def _signal_phases(signal, n, flat_thresh_pct=0.15, bottom_thresh_pct=0.20, rising=True):
    """
    Generic signal-based phase detector for cyclic movements.
    rising=True  → peaks are the active phase (relevé, tendu, arabesque)
    rising=False → valleys are the active phase (fondu, plié-like)
    Returns list of phase strings per frame.
    """
    if not signal or n == 0:
        return ["preparation"] * n

    sig_min = min(signal)
    sig_max = max(signal)
    sig_range = sig_max - sig_min
    if sig_range < 1e-3:
        return ["preparation"] * n

    # Normalise to 0-1
    norm = [(s - sig_min) / sig_range for s in signal]

    flat_low  = flat_thresh_pct          # below this = flat (preparation/finish)
    flat_high = 1.0 - flat_thresh_pct
    active    = 1.0 - bottom_thresh_pct  # above this = peak (for rising)
    active_lo = bottom_thresh_pct        # below this = valley (for falling)

    # Derivative
    deriv = [0.0] * n
    for i in range(1, n - 1):
        deriv[i] = norm[i + 1] - norm[i - 1]
    if n > 1:
        deriv[0]     = norm[1] - norm[0]
        deriv[n - 1] = norm[n - 1] - norm[n - 2]

    phases = []
    for i, v in enumerate(norm):
        is_start = i < n * 0.12
        is_end   = i > n * 0.88
        if rising:
            if v >= active:
                phase = "peak"
            elif v <= flat_low:
                phase = "preparation" if is_start else "finish"
            elif deriv[i] > 0.02:
                phase = "rising"
            elif deriv[i] < -0.02:
                phase = "descent"
            else:
                phase = phases[-1] if phases else "preparation"
        else:
            if v <= active_lo:
                phase = "peak"
            elif v >= flat_high:
                phase = "preparation" if is_start else "finish"
            elif deriv[i] < -0.02:
                phase = "rising"   # going into the movement
            elif deriv[i] > 0.02:
                phase = "descent"  # returning from the movement
            else:
                phase = phases[-1] if phases else "preparation"
        phases.append(phase)

    # Smooth single-frame noise
    for i in range(1, n - 1):
        if phases[i] != phases[i - 1] and phases[i] != phases[i + 1]:
            phases[i] = phases[i - 1]

    return phases


def _detect_exercise_phases(frames, exercise):
    """Signal-based phase detection per exercise. Returns list of phase strings."""
    n = len(frames)
    if n == 0:
        return []
    if n < 6:
        return ["execution"] * n

    meas = [f.get("measurements", {}) for f in frames]

    if exercise == "releve":
        signal = [max(m.get("left_heel_rise_pct", 0), m.get("right_heel_rise_pct", 0)) for m in meas]
        return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=True)

    if exercise == "fondu":
        # Knee angle drops as dancer bends → valley = active phase
        signal = [min(m.get("left_knee_angle", 180), m.get("right_knee_angle", 180)) for m in meas]
        return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=False)

    if exercise == "battement_tendu":
        signal = [max(m.get("left_knee_toe_offset_norm", 0), m.get("right_knee_toe_offset_norm", 0)) for m in meas]
        return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=True)

    if exercise == "retire_passe":
        # Hip height asymmetry increases as working leg rises
        signal = [m.get("hip_height_diff_pct", 0) for m in meas]
        return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=True)

    if exercise in ("arabesque", "grand_battement"):
        # arabesque_angle decreases as leg rises (angle at hip-hip-ankle)
        signal = [m.get("arabesque_angle", 180) for m in meas]
        return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=False)

    # port_de_bras: arm variation — use elbow angle range as proxy
    if exercise == "port_de_bras":
        signal = [abs(m.get("left_elbow_angle", 150) - 150) + abs(m.get("right_elbow_angle", 150) - 150) for m in meas]
        return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=True)

    return ["execution"] * n


def _check_condition(exercise, trigger_id, condition, m, values, phase="execution"):
    ACTIVE = {"rising", "peak", "descent", "execution"}  # phases where movement is happening

    # releve — only check during active movement, not at rest
    if exercise == "releve":
        if phase not in ACTIVE:
            return False
        if condition == "knees_not_extended":
            # Only meaningful when actually rising/at peak
            return phase in {"rising", "peak"} and (
                m.get("left_knee_angle", 180) < 170 or m.get("right_knee_angle", 180) < 170
            )
        if condition == "insufficient_rise":
            # Only fire at the peak — if max rise is still too low
            return phase == "peak" and max(m.get("left_heel_rise_pct", 0), m.get("right_heel_rise_pct", 0)) < 10
        if condition == "asymmetric_rise":
            return phase in {"rising", "peak"} and (
                abs(m.get("left_heel_rise_pct", 0) - m.get("right_heel_rise_pct", 0)) > 5
            )
        if condition == "hip_shifting":
            return phase in {"rising", "peak"} and m.get("hip_height_diff_pct", 0) > 2.5
        if condition == "trunk_compensating":
            return phase in {"rising", "peak"} and m.get("trunk_lean_angle", 0) > 6

    # fondu — check during the bend and at the lowest point
    if exercise == "fondu":
        if phase not in ACTIVE:
            return False
        if condition == "supporting_knee_collapsing":
            return phase in {"rising", "peak"} and (
                m.get("left_knee_toe_offset_norm", 0) > 0.06
                or m.get("right_knee_toe_offset_norm", 0) > 0.06
            )
        if condition == "hip_not_level":
            return phase in {"rising", "peak"} and m.get("hip_height_diff_pct", 0) > 3
        if condition == "trunk_compensating":
            return phase in {"rising", "peak"} and m.get("trunk_lean_angle", 0) > 8

    # battement_tendu — check during brush out and extension
    if exercise == "battement_tendu":
        if phase not in ACTIVE:
            return False
        max_bend = values.get("supporting_knee", {}).get("max_bend_degrees", 5)
        min_knee = 180 - max_bend
        max_hip_diff = values.get("hip_stability", {}).get("max_height_diff_pct", 2)
        max_lean = values.get("trunk_stability", {}).get("max_lean_degrees", 5)

        if condition == "supporting_knee_bent":
            return phase in {"rising", "peak", "descent"} and (
                m.get("left_knee_angle", 180) < min_knee or m.get("right_knee_angle", 180) < min_knee
            )
        if condition == "hip_hiking":
            return phase in {"rising", "peak"} and m.get("hip_height_diff_pct", 0) > max_hip_diff
        if condition == "incorrect_foot_lead":
            # Only meaningful at peak extension — offset should be clearly asymmetric
            return phase == "peak" and (
                abs(m.get("left_knee_toe_offset_norm", 0) - m.get("right_knee_toe_offset_norm", 0)) < 0.03
            )
        if condition == "trunk_instability":
            return phase in {"rising", "peak"} and m.get("trunk_lean_angle", 0) > max_lean

    # retire_passe — check during draw-up and at peak
    if exercise == "retire_passe":
        if phase not in ACTIVE:
            return False
        if condition == "supporting_knee_bent":
            return phase in {"rising", "peak"} and (
                m.get("left_knee_angle", 180) < 175 or m.get("right_knee_angle", 180) < 175
            )
        if condition == "hip_hiking":
            return phase in {"rising", "peak"} and m.get("hip_height_diff_pct", 0) > 2.5
        if condition == "trunk_instability":
            return phase in {"rising", "peak"} and m.get("trunk_lean_angle", 0) > 5

    # arabesque — only check during the held arabesque phase
    if exercise == "arabesque":
        if phase not in {"peak", "execution"}:
            return False
        if trigger_id == "arab_001":
            min_angle = values.get("working_leg_height", {}).get("min_angle_degrees", 90)
            return m.get("arabesque_angle", 180) < min_angle
        if trigger_id == "arab_002":
            max_rot = values.get("hip_squareness", {}).get("max_rotation_degrees", 5)
            return m.get("hip_rotation_deg", 0) > max_rot
        if trigger_id == "arab_003":
            max_dev = values.get("spine_vertical", {}).get("max_deviation_pct", 8)
            return m.get("trunk_lean_angle", 0) > max_dev
        if trigger_id == "arab_004":
            max_shoulder_diff = values.get("shoulder_level", {}).get("max_height_diff_pct", 4)
            return m.get("shoulder_height_diff_pct", 0) > max_shoulder_diff

    # port_de_bras — check during arm movement
    if exercise == "port_de_bras":
        if phase not in ACTIVE:
            return False
        min_elbow = values.get("elbow_curve", {}).get("min_angle_degrees", 140)
        max_elbow = values.get("elbow_curve", {}).get("max_angle_degrees", 160)
        if trigger_id == "bras_001":
            return m.get("shoulder_height_diff_pct", 0) > values.get("shoulder_tension", {}).get("max_elevation_pct", 2)
        if trigger_id == "bras_002":
            return phase in {"rising", "peak"} and (
                m.get("left_elbow_angle", 150) > max_elbow
                or m.get("right_elbow_angle", 150) > max_elbow
                or m.get("left_elbow_angle", 150) < min_elbow
                or m.get("right_elbow_angle", 150) < min_elbow
            )
        if trigger_id == "bras_003":
            return phase in {"rising", "peak"} and (
                m.get("left_wrist_drop_pct", 0) > 6 or m.get("right_wrist_drop_pct", 0) > 6
            )

    # grand_battement — only during kick and peak
    if exercise == "grand_battement":
        if phase not in ACTIVE:
            return False
        max_hip_diff = values.get("hip_stability", {}).get("max_height_diff_pct", 4)
        min_angle = values.get("working_leg_height", {}).get("min_angle_degrees", 90)
        if trigger_id == "gbat_001":
            return phase in {"rising", "peak"} and m.get("hip_height_diff_pct", 0) > max_hip_diff
        if trigger_id == "gbat_002":
            return phase in {"rising", "peak", "descent"} and (
                m.get("left_knee_angle", 180) < 175 or m.get("right_knee_angle", 180) < 175
            )
        if trigger_id == "gbat_003":
            # Uncontrolled return — trunk compensation on the way down
            return phase == "descent" and (
                m.get("trunk_lean_angle", 0) > 9 or m.get("arabesque_angle", 180) < min_angle * 0.7
            )

    return False


def _visibility_ok(exercise, trigger_id, visibility):
    if exercise == "releve":
        if trigger_id in ("releve_001",):
            return gate(visibility, "left_hip", "left_knee", "left_ankle") or gate(visibility, "right_hip", "right_knee", "right_ankle")
        if trigger_id in ("releve_002",):
            return gate(visibility, "left_hip", "right_hip")
        if trigger_id in ("releve_003", "releve_004"):
            return gate(visibility, "left_heel", "left_ankle") or gate(visibility, "right_heel", "right_ankle")
        if trigger_id in ("releve_005",):
            return gate(visibility, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if exercise == "retire_passe":
        if trigger_id == "retire_001":
            return gate(visibility, "left_hip", "left_knee", "left_ankle") or gate(visibility, "right_hip", "right_knee", "right_ankle")
        if trigger_id == "retire_002":
            return gate(visibility, "left_hip", "right_hip")
        if trigger_id == "retire_003":
            return gate(visibility, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if exercise == "fondu":
        if trigger_id == "fondu_001":
            return gate(visibility, "left_knee", "left_toe") or gate(visibility, "right_knee", "right_toe")
        if trigger_id == "fondu_002":
            return gate(visibility, "left_hip", "right_hip")
        if trigger_id == "fondu_003":
            return gate(visibility, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if exercise == "battement_tendu":
        if trigger_id == "tendu_001":
            return gate(visibility, "left_hip", "left_knee", "left_ankle") or gate(visibility, "right_hip", "right_knee", "right_ankle")
        if trigger_id == "tendu_002":
            return gate(visibility, "left_hip", "right_hip")
        if trigger_id == "tendu_003":
            return gate(visibility, "left_knee", "left_toe") or gate(visibility, "right_knee", "right_toe")
        if trigger_id == "tendu_004":
            return gate(visibility, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if exercise == "arabesque":
        if trigger_id == "arab_001":
            return gate(visibility, "left_hip", "right_hip", "right_ankle")
        if trigger_id == "arab_002":
            return gate(visibility, "left_hip", "right_hip")
        if trigger_id == "arab_003":
            return gate(visibility, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
        if trigger_id == "arab_004":
            return gate(visibility, "left_shoulder", "right_shoulder")
    if exercise == "port_de_bras":
        if trigger_id in ("bras_001", "bras_002"):
            return gate(visibility, "left_shoulder", "right_shoulder", "left_elbow", "right_elbow")
        if trigger_id == "bras_003":
            return gate(visibility, "left_wrist", "right_wrist", "left_elbow", "right_elbow")
    if exercise == "grand_battement":
        if trigger_id == "gbat_001":
            return gate(visibility, "left_hip", "right_hip")
        if trigger_id == "gbat_002":
            return gate(visibility, "left_hip", "left_knee", "left_ankle") or gate(visibility, "right_hip", "right_knee", "right_ankle")
        if trigger_id == "gbat_003":
            return gate(visibility, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    return True


def _signal_value(exercise, trigger_id, m, phase="execution"):
    ACTIVE = {"rising", "peak", "descent", "execution"}
    if phase not in ACTIVE:
        return None  # phase gate — no signal outside active movement

    if exercise == "releve":
        if phase not in {"rising", "peak"}:
            return None
        if trigger_id == "releve_001":
            return max(180 - m.get("left_knee_angle", 180), 180 - m.get("right_knee_angle", 180))
        if trigger_id == "releve_002":
            return m.get("hip_height_diff_pct", 0)
        if trigger_id == "releve_003":
            return max(0, 10 - max(m.get("left_heel_rise_pct", 0), m.get("right_heel_rise_pct", 0))) if phase == "peak" else None
        if trigger_id == "releve_004":
            return abs(m.get("left_heel_rise_pct", 0) - m.get("right_heel_rise_pct", 0))
        if trigger_id == "releve_005":
            return m.get("trunk_lean_angle", 0)
    if exercise == "retire_passe":
        if phase not in {"rising", "peak"}:
            return None
        if trigger_id == "retire_001":
            return max(180 - m.get("left_knee_angle", 180), 180 - m.get("right_knee_angle", 180))
        if trigger_id == "retire_002":
            return m.get("hip_height_diff_pct", 0)
        if trigger_id == "retire_003":
            return m.get("trunk_lean_angle", 0)
    if exercise == "fondu":
        if phase not in {"rising", "peak"}:
            return None
        if trigger_id == "fondu_002":
            return m.get("hip_height_diff_pct", 0)
        if trigger_id == "fondu_003":
            return m.get("trunk_lean_angle", 0)
    if exercise == "battement_tendu":
        if phase not in {"rising", "peak", "descent"}:
            return None
        if trigger_id == "tendu_001":
            return max(180 - m.get("left_knee_angle", 180), 180 - m.get("right_knee_angle", 180))
        if trigger_id == "tendu_002":
            return m.get("hip_height_diff_pct", 0)
        if trigger_id == "tendu_004":
            return m.get("trunk_lean_angle", 0) if phase in {"rising", "peak"} else None
    if exercise == "arabesque":
        if phase not in {"peak", "execution"}:
            return None
        if trigger_id == "arab_001":
            return 180 - m.get("arabesque_angle", 180)
        if trigger_id == "arab_002":
            return m.get("hip_rotation_deg", 0)
        if trigger_id == "arab_003":
            return m.get("trunk_lean_angle", 0)
        if trigger_id == "arab_004":
            return m.get("shoulder_height_diff_pct", 0)
    if exercise == "port_de_bras":
        if trigger_id == "bras_001":
            return m.get("shoulder_height_diff_pct", 0)
        if trigger_id == "bras_003":
            return max(m.get("left_wrist_drop_pct", 0), m.get("right_wrist_drop_pct", 0)) if phase in {"rising", "peak"} else None
    if exercise == "grand_battement":
        if trigger_id == "gbat_001":
            return m.get("hip_height_diff_pct", 0) if phase in {"rising", "peak"} else None
        if trigger_id == "gbat_002":
            return max(180 - m.get("left_knee_angle", 180), 180 - m.get("right_knee_angle", 180)) if phase in {"rising", "peak", "descent"} else None
        if trigger_id == "gbat_003":
            return m.get("trunk_lean_angle", 0) if phase == "descent" else None
    return None


def _default_temporal(trigger_id):
    # Flowing/sustained exercises need more frames to confirm a correction
    if trigger_id.startswith(("bras_", "arab_", "gbat_")):
        return 6, 10
    if trigger_id.startswith(("releve_", "retire_", "fondu_", "tendu_")):
        return 4, 7
    # Plie
    return 3, 6


def _build_temporal_buffers(trigger_meta):
    buffers = {}
    for tid, meta in trigger_meta.items():
        n = int(meta.get("n_of_m_n", _default_temporal(tid)[0]))
        m = int(meta.get("n_of_m_m", _default_temporal(tid)[1]))
        buffers[tid] = NofMBuffer(n=n, m=max(m, n))
    return buffers


def _build_hysteresis(exercise, trigger_meta, values):
    hs = {}
    for tid, meta in trigger_meta.items():
        on_thresh = meta.get("hysteresis_on")
        off_thresh = meta.get("hysteresis_off")

        if on_thresh is None or off_thresh is None:
            if exercise == "battement_tendu" and tid == "tendu_002":
                on_thresh = values.get("hip_stability", {}).get("max_height_diff_pct", 2)
                off_thresh = max(0.5, on_thresh - 1.0)
            elif exercise == "battement_tendu" and tid == "tendu_004":
                on_thresh = values.get("trunk_stability", {}).get("max_lean_degrees", 5)
                off_thresh = max(1.0, on_thresh - 1.5)
            elif exercise == "arabesque" and tid == "arab_002":
                on_thresh = values.get("hip_squareness", {}).get("max_rotation_degrees", 5)
                off_thresh = max(1.0, on_thresh - 1.5)
            elif exercise == "arabesque" and tid == "arab_003":
                on_thresh = values.get("spine_vertical", {}).get("max_deviation_pct", 8)
                off_thresh = max(1.0, on_thresh - 2.0)
            elif exercise == "arabesque" and tid == "arab_004":
                on_thresh = values.get("shoulder_level", {}).get("max_height_diff_pct", 4)
                off_thresh = max(0.8, on_thresh - 1.0)
            elif exercise == "port_de_bras" and tid == "bras_001":
                on_thresh = values.get("shoulder_tension", {}).get("max_elevation_pct", 2)
                off_thresh = max(0.6, on_thresh - 0.8)
            elif exercise == "port_de_bras" and tid == "bras_003":
                on_thresh = 6
                off_thresh = 4.5
            elif exercise == "grand_battement" and tid == "gbat_001":
                on_thresh = values.get("hip_stability", {}).get("max_height_diff_pct", 4)
                off_thresh = max(1.0, on_thresh - 1.2)
            elif exercise == "grand_battement" and tid == "gbat_003":
                on_thresh = 9
                off_thresh = 7
            elif exercise == "releve" and tid == "releve_001":
                on_thresh = 10
                off_thresh = 6
            elif exercise == "releve" and tid == "releve_002":
                on_thresh = 2
                off_thresh = 0.8
            elif exercise == "releve" and tid == "releve_004":
                on_thresh = 5
                off_thresh = 3
            elif exercise == "releve" and tid == "releve_005":
                on_thresh = 5
                off_thresh = 3
            elif exercise == "retire_passe" and tid == "retire_001":
                on_thresh = 5
                off_thresh = 2
            elif exercise == "retire_passe" and tid == "retire_002":
                on_thresh = 2
                off_thresh = 0.8
            elif exercise == "retire_passe" and tid == "retire_003":
                on_thresh = 5
                off_thresh = 3
            elif exercise == "fondu" and tid == "fondu_002":
                on_thresh = 3
                off_thresh = 1.5
            elif exercise == "fondu" and tid == "fondu_003":
                on_thresh = 8
                off_thresh = 6

        if on_thresh is not None and off_thresh is not None:
            hs[tid] = Hysteresis(float(on_thresh), float(off_thresh))
    return hs


def _exercise_presence_confidence(exercise_id, frames, values):
    """
    Rough movement-presence confidence (0..1) used to avoid cross-exercise false positives.
    """
    if not frames:
        return 0.0

    measurements = [f.get("measurements", {}) for f in frames]

    def longest_streak(flags):
        best = cur = 0
        for f in flags:
            if f:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    if exercise_id == "arabesque":
        # Leg must reach clearly above horizontal — angle < 60° (standing baseline is ~90-100°)
        angles = [m.get("arabesque_angle") for m in measurements if m.get("arabesque_angle") is not None]
        if not angles:
            return 0.0
        baseline = sorted(angles)[int(len(angles) * 0.7)]  # 70th percentile = typical standing angle
        peak = min(angles)
        drop = baseline - peak  # how much below standing the leg got
        active = sum(1 for a in angles if a < baseline - drop * 0.5)
        streak = longest_streak([a < baseline - drop * 0.5 for a in angles])
        if drop < 20:  # leg never really lifted
            return 0.0
        c1 = active / max(1, len(angles))
        c2 = streak / max(1, len(angles))
        return min(1.0, 0.6 * c1 + 0.4 * c2)

    if exercise_id == "battement_tendu":
        # Detect actual extension MOVEMENT — offset must peak then return (variation), not just be high
        offs = [
            max(m.get("left_knee_toe_offset_norm", 0), m.get("right_knee_toe_offset_norm", 0))
            for m in measurements
        ]
        if not offs:
            return 0.0
        baseline = sorted(offs)[int(len(offs) * 0.3)]  # 30th percentile = resting offset
        peak = max(offs)
        extension = peak - baseline  # actual extension beyond resting position
        if extension < 0.25:  # turnout alone won't exceed this
            return 0.0
        active = sum(1 for v in offs if v > baseline + extension * 0.5)
        return min(1.0, active / max(1, len(offs)))

    if exercise_id == "grand_battement":
        # Same as arabesque — requires clear leg elevation above standing baseline
        angles = [m.get("arabesque_angle") for m in measurements if m.get("arabesque_angle") is not None]
        if not angles:
            return 0.0
        baseline = sorted(angles)[int(len(angles) * 0.7)]
        peak = min(angles)
        drop = baseline - peak
        if drop < 25:  # leg never reached grand battement height
            return 0.0
        active = sum(1 for a in angles if a < baseline - drop * 0.5)
        return min(1.0, active / max(1, len(angles)))

    if exercise_id == "port_de_bras":
        # Conservative: only mark present if both arms show notable angular variation.
        l = [m.get("left_elbow_angle") for m in measurements if m.get("left_elbow_angle") is not None]
        r = [m.get("right_elbow_angle") for m in measurements if m.get("right_elbow_angle") is not None]
        if not l or not r:
            return 0.0
        l_var = (max(l) - min(l)) / 40.0
        r_var = (max(r) - min(r)) / 40.0
        return max(0.0, min(1.0, (l_var + r_var) / 2.0))

    if exercise_id == "releve":
        l = [m.get("left_heel_rise_pct", 0) for m in measurements]
        r = [m.get("right_heel_rise_pct", 0) for m in measurements]
        if not l:
            return 0.0
        max_rise = max(max(l), max(r))
        active = sum(1 for lv, rv in zip(l, r) if max(lv, rv) > 6)
        return min(1.0, 0.5 * (active / len(l)) + 0.5 * min(1.0, max_rise / 15.0))

    if exercise_id == "retire_passe":
        l = [m.get("left_heel_rise_pct", 0) for m in measurements]
        r = [m.get("right_heel_rise_pct", 0) for m in measurements]
        if not l:
            return 0.0
        asym = [abs(lv - rv) for lv, rv in zip(l, r)]
        active = sum(1 for a in asym if a > 4)
        return min(1.0, active / max(1, len(asym)))

    if exercise_id == "fondu":
        ml = [m.get("left_knee_angle", 180) for m in measurements]
        mr = [m.get("right_knee_angle", 180) for m in measurements]
        if not ml:
            return 0.0
        asym = [abs(lv - rv) for lv, rv in zip(ml, mr)]
        active = sum(1 for a in asym if a > 15)
        return min(1.0, active / max(1, len(asym)))

    return 0.0


def analyse_exercise(pose_data, exercise_id):
    frames = pose_data.get("frames", [])
    fps = pose_data.get("fps", 0)

    if not frames:
        return {"error": "No pose data extracted from video."}

    term = TERMS.get(exercise_id)
    if not term:
        return {"error": f"Unsupported exercise '{exercise_id}'."}

    trigger_meta = {t["id"]: t for t in term.get("correction_triggers", []) if "id" in t}
    values = term.get("correct_values", {})
    frames = smooth_frames(frames, alpha=0.15)
    presence_confidence = _exercise_presence_confidence(exercise_id, frames, values)
    presence_threshold = {
        "arabesque": 0.50,
        "grand_battement": 0.50,
        "battement_tendu": 0.55,
        "port_de_bras": 0.50,
        "releve": 0.45,
        "retire_passe": 0.40,
        "fondu": 0.45,
    }.get(exercise_id, 0.45)
    movement_detected = presence_confidence >= presence_threshold

    if not movement_detected:
        return {
            "exercise": term.get("term_fr") or exercise_id,
            "exercise_id": exercise_id,
            "movement_detected": False,
            "presence_confidence": round(presence_confidence, 3),
            "score": 100,
            "grade": "Not Detected",
            "corrections": [],
            "correction_timeline": [],
            "phases_summary": {},
            "total_frames_analysed": len(frames),
            "duration_seconds": round(frames[-1]["timestamp_ms"] / 1000, 1) if frames else 0,
            "fps": fps,
        }

    phases = _detect_exercise_phases(frames, exercise_id)
    buffers = _build_temporal_buffers(trigger_meta)
    hysteresis_map = _build_hysteresis(exercise_id, trigger_meta, values)

    all_triggered = []
    for frame, phase in zip(frames, phases):
        m = frame.get("measurements", {})
        vis = frame.get("visibility", {})
        frame_hits = []
        for tid, t in trigger_meta.items():
            condition = t.get("condition", "")
            if not _visibility_ok(exercise_id, tid, vis):
                raw = False
            else:
                signal = _signal_value(exercise_id, tid, m, phase)
                if signal is not None and tid in hysteresis_map:
                    raw = hysteresis_map[tid].update(signal)
                else:
                    raw = _check_condition(exercise_id, tid, condition, m, values, phase)

            confirmed = buffers[tid].update(raw)
            if confirmed:
                frame_hits.append(tid)
        all_triggered.append(frame_hits)

    corrections = _aggregate_corrections(all_triggered, frames, trigger_meta)
    corrections = _enrich_corrections(corrections, len(frames), trigger_meta, presence_confidence)
    correction_timeline = [{"id": c["id"], "severity": c["severity"], "segments": c["segments"]} for c in corrections]

    phase_counts = {}
    for p in phases:
        phase_counts[p] = phase_counts.get(p, 0) + 1

    score = compute_score(corrections, len(frames))
    return {
        "exercise": term.get("term_fr") or exercise_id,
        "exercise_id": exercise_id,
        "movement_detected": True,
        "presence_confidence": round(presence_confidence, 3),
        "score": score,
        "grade": _grade_from_score(score),
        "corrections": corrections,
        "correction_timeline": correction_timeline,
        "phases_summary": phase_counts,
        "total_frames_analysed": len(frames),
        "duration_seconds": round(frames[-1]["timestamp_ms"] / 1000, 1) if frames else 0,
        "fps": fps,
    }
