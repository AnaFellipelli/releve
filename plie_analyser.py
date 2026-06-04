"""
plie_analyser.py
================
Phase detection + correction engine for plié analysis.
Reads ballet_database.json for thresholds and bilingual cues.
Produces a structured report with score, corrections, and timestamps.
"""

import json
from collections import deque
from pathlib import Path


DB_PATH = Path(__file__).parent / "ballet_database.json"


def _load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


DB = _load_db()
PLIE_TERM = next((t for t in DB.get("terms", []) if t.get("id") == "plie"), {})
PLIE_CORRECT_VALUES = PLIE_TERM.get("correct_values", {})
PLIE_DB_TRIGGERS = {t["id"]: t for t in PLIE_TERM.get("correction_triggers", []) if "id" in t}


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


def smooth_frames(frames, alpha=0.3):
    """Apply EMA smoothing to noisy scalar measurements frame-by-frame."""
    if not frames:
        return frames
    keys = [
        "left_knee_angle",
        "right_knee_angle",
        "left_heel_rise_pct",
        "right_heel_rise_pct",
        "trunk_lean_angle",
        "hip_height_diff_pct",
        "anterior_tilt_deg",
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


def gate(visibility, *landmark_names, threshold=0.6):
    return all(visibility.get(name, 0.0) >= threshold for name in landmark_names)


def _visibility_ok(frame, trigger_id):
    vis = frame.get("visibility", {})
    if trigger_id == "plie_001":
        left_ok = gate(vis, "left_knee", "left_toe", "left_hip")
        right_ok = gate(vis, "right_knee", "right_toe", "right_hip")
        return left_ok or right_ok
    if trigger_id in ("plie_003", "plie_005"):
        left_ok = gate(vis, "left_heel", "left_ankle")
        right_ok = gate(vis, "right_heel", "right_ankle")
        return left_ok or right_ok
    if trigger_id in ("plie_002", "plie_008"):
        return gate(vis, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if trigger_id == "plie_009":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id == "plie_011":
        return gate(vis, "left_knee", "right_knee", "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    return True


# ── PHASE DETECTION ───────────────────────────────────────────────────

def detect_phases(frames):
    """
    Classify each frame into plié phase. Handles multiple repetitions by
    detecting descent/ascent direction from the knee-angle derivative.
    Phases: preparation | descent | bottom | ascent | finish
    """
    if not frames:
        return []

    def avg_knee(f):
        m = f["measurements"]
        angles = [m.get("left_knee_angle"), m.get("right_knee_angle")]
        valid = [a for a in angles if a is not None]
        return sum(valid) / len(valid) if valid else 180.0

    knee_angles = [avg_knee(f) for f in frames]
    n = len(knee_angles)
    global_min = min(knee_angles)
    global_max = max(knee_angles)
    range_deg = global_max - global_min

    # If range is too small the person barely moved — treat as preparation
    if range_deg < 10:
        return ["preparation"] * n

    # Bottom = within 20% of the deepest point; top = within 10% of standing
    bottom_thresh = global_min + range_deg * 0.20
    top_thresh    = global_max - range_deg * 0.10

    # Compute smoothed derivative to determine direction
    deriv = [0.0] * n
    for i in range(1, n - 1):
        deriv[i] = knee_angles[i + 1] - knee_angles[i - 1]
    deriv[0]     = knee_angles[1] - knee_angles[0] if n > 1 else 0
    deriv[n - 1] = knee_angles[n - 1] - knee_angles[n - 2] if n > 1 else 0

    phases = []
    for i, angle in enumerate(knee_angles):
        if angle >= top_thresh:
            phase = "preparation" if i < n * 0.15 else "finish"
        elif angle <= bottom_thresh:
            phase = "bottom"
        elif deriv[i] > 0.3:
            phase = "ascent"
        elif deriv[i] < -0.3:
            phase = "descent"
        else:
            # Flat section not at top or bottom — inherit from neighbour
            phase = phases[-1] if phases else "preparation"
        phases.append(phase)

    # Smooth single-frame noise
    for i in range(1, n - 1):
        if phases[i] != phases[i - 1] and phases[i] != phases[i + 1]:
            phases[i] = phases[i - 1]

    return phases


# ── CORRECTION TRIGGERS ───────────────────────────────────────────────

TRIGGERS = [
    {
        "id": "plie_001",
        "severity": PLIE_DB_TRIGGERS.get("plie_001", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("descent", "bottom", "ascent") and
            (
                m.get("left_knee_toe_offset_norm", 0) >
                PLIE_CORRECT_VALUES.get("knee_over_toe_alignment", {}).get("max_offset_normalized", 0.25)
                or
                m.get("right_knee_toe_offset_norm", 0) >
                PLIE_CORRECT_VALUES.get("knee_over_toe_alignment", {}).get("max_offset_normalized", 0.25)
            )
        ),
        "cue_pt": PLIE_DB_TRIGGERS.get("plie_001", {}).get("cue_pt", "Joelhos caindo para dentro."),
        "cue_en": PLIE_DB_TRIGGERS.get("plie_001", {}).get("cue_en", "Knees falling inward."),
    },
    {
        "id": "plie_002",
        "severity": PLIE_DB_TRIGGERS.get("plie_002", {}).get("severity", "major"),
        "check": lambda m, phase: (
            m.get("anterior_tilt_deg", 0)
            > PLIE_CORRECT_VALUES.get("pelvic_tilt", {}).get("max_anterior_tilt_degrees", 5)
        ),
        "cue_pt": PLIE_DB_TRIGGERS.get("plie_002", {}).get("cue_pt", "Mantenha a pelve neutra."),
        "cue_en": PLIE_DB_TRIGGERS.get("plie_002", {}).get("cue_en", "Keep the pelvis neutral."),
    },
    {
        "id": "plie_003",
        "severity": PLIE_DB_TRIGGERS.get("plie_003", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase == "descent" and
            (
                m.get("left_heel_rise_pct", 0)
                > PLIE_CORRECT_VALUES.get("heel_contact_demi", {}).get("max_rise_pct", 3)
                or
                m.get("right_heel_rise_pct", 0)
                > PLIE_CORRECT_VALUES.get("heel_contact_demi", {}).get("max_rise_pct", 3)
            )
        ),
        "cue_pt": PLIE_DB_TRIGGERS.get("plie_003", {}).get("cue_pt", "Calcanhares subindo no demi-plié."),
        "cue_en": PLIE_DB_TRIGGERS.get("plie_003", {}).get("cue_en", "Heels rising in demi-plié."),
    },
    {
        "id": "plie_005",
        "severity": PLIE_DB_TRIGGERS.get("plie_005", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase == "bottom" and
            (
                m.get("left_heel_rise_pct", 0)
                > PLIE_CORRECT_VALUES.get("heel_contact_demi", {}).get("max_rise_pct", 3)
                or
                m.get("right_heel_rise_pct", 0)
                > PLIE_CORRECT_VALUES.get("heel_contact_demi", {}).get("max_rise_pct", 3)
            )
        ),
        "cue_pt": PLIE_DB_TRIGGERS.get("plie_005", {}).get("cue_pt", "Calcanhares subindo cedo no grand plié."),
        "cue_en": PLIE_DB_TRIGGERS.get("plie_005", {}).get("cue_en", "Heels rising too early in grand plié."),
    },
    {
        "id": "plie_006",
        "severity": PLIE_DB_TRIGGERS.get("plie_006", {}).get("severity", "major"),
        "check": lambda m, phase: False,  # aggregate check handled separately
        "cue_pt": PLIE_DB_TRIGGERS.get("plie_006", {}).get("cue_pt", "Pausa no fundo do plié."),
        "cue_en": PLIE_DB_TRIGGERS.get("plie_006", {}).get("cue_en", "Pause at the bottom of the plié."),
    },
    {
        "id": "plie_007",
        "severity": PLIE_DB_TRIGGERS.get("plie_007", {}).get("severity", "warning"),
        "check": lambda m, phase: False,  # velocity check handled in aggregate
        "cue_pt": PLIE_DB_TRIGGERS.get("plie_007", {}).get("cue_pt", "Plié não fluido."),
        "cue_en": PLIE_DB_TRIGGERS.get("plie_007", {}).get("cue_en", "Non-fluid plié."),
    },
    {
        "id": "plie_008",
        "severity": PLIE_DB_TRIGGERS.get("plie_008", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("descent", "bottom")
            and m.get("trunk_lean_angle", 0)
            > PLIE_CORRECT_VALUES.get("trunk_lean", {}).get("max_forward_lean_degrees", 10)
        ),
        "cue_pt": PLIE_DB_TRIGGERS.get("plie_008", {}).get("cue_pt", "Tronco muito inclinado para frente."),
        "cue_en": PLIE_DB_TRIGGERS.get("plie_008", {}).get("cue_en", "Trunk leaning too far forward."),
    },
    {
        "id": "plie_009",
        "severity": PLIE_DB_TRIGGERS.get("plie_009", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("descent", "bottom", "ascent") and
            m.get("hip_height_diff_pct", 0)
            > PLIE_CORRECT_VALUES.get("hip_level", {}).get("max_height_diff_pct", 3)
        ),
        "cue_pt": PLIE_DB_TRIGGERS.get("plie_009", {}).get("cue_pt", "Quadril deslocando lateralmente."),
        "cue_en": PLIE_DB_TRIGGERS.get("plie_009", {}).get("cue_en", "Hip shifting laterally."),
    },
    {
        "id": "plie_011",
        "severity": PLIE_DB_TRIGGERS.get("plie_011", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase == "bottom" and
            (m.get("left_knee_angle", 180) < 60 or m.get("right_knee_angle", 180) < 60) and
            m.get("trunk_lean_angle", 0) > 8
        ),
        "cue_pt": PLIE_DB_TRIGGERS.get("plie_011", {}).get("cue_pt", "Sentando na articulação no fundo do plié."),
        "cue_en": PLIE_DB_TRIGGERS.get("plie_011", {}).get("cue_en", "Sitting into the joint at the bottom."),
    },
]


N_OF_M_CONFIG = {
    "plie_001": (5, 8),
    "plie_002": (5, 8),
    "plie_003": (4, 7),
    "plie_005": (4, 7),
    "plie_006": (1, 2),
    "plie_007": (3, 6),
    "plie_008": (5, 8),
    "plie_009": (5, 8),
    "plie_011": (4, 6),
}

HYSTERESIS_CONFIG = {
    "plie_001": (
        (_on := PLIE_CORRECT_VALUES.get("knee_over_toe_alignment", {}).get("max_offset_normalized", 0.25)),
        max(0.0, _on * 0.72),  # off_thresh scales with on_thresh to avoid inverted band
    ),
    "plie_002": (PLIE_CORRECT_VALUES.get("pelvic_tilt", {}).get("max_anterior_tilt_degrees", 5), 4),
    "plie_003": (PLIE_CORRECT_VALUES.get("heel_contact_demi", {}).get("max_rise_pct", 3), 2),
    "plie_005": (PLIE_CORRECT_VALUES.get("heel_contact_demi", {}).get("max_rise_pct", 3), 2),
    "plie_008": (PLIE_CORRECT_VALUES.get("trunk_lean", {}).get("max_forward_lean_degrees", 10), 8),
    "plie_009": (PLIE_CORRECT_VALUES.get("hip_level", {}).get("max_height_diff_pct", 3), 2),
}

TRIGGER_EVIDENCE = {
    "plie_001": ("knee_toe_offset_norm", PLIE_CORRECT_VALUES.get("knee_over_toe_alignment", {}).get("max_offset_normalized", 0.25)),
    "plie_002": ("anterior_tilt_deg", PLIE_CORRECT_VALUES.get("pelvic_tilt", {}).get("max_anterior_tilt_degrees", 5)),
    "plie_003": ("heel_rise_pct_descent", PLIE_CORRECT_VALUES.get("heel_contact_demi", {}).get("max_rise_pct", 3)),
    "plie_005": ("heel_rise_pct_bottom", PLIE_CORRECT_VALUES.get("heel_contact_demi", {}).get("max_rise_pct", 3)),
    "plie_006": ("bottom_pause_frames", PLIE_CORRECT_VALUES.get("bottom_pause", {}).get("max_stationary_frames", 2)),
    "plie_007": ("descent_velocity_cv", PLIE_CORRECT_VALUES.get("velocity_consistency", {}).get("max_cv", 0.08)),
    "plie_008": ("trunk_lean_angle", PLIE_CORRECT_VALUES.get("trunk_lean", {}).get("max_forward_lean_degrees", 10)),
    "plie_009": ("hip_height_diff_pct", PLIE_CORRECT_VALUES.get("hip_level", {}).get("max_height_diff_pct", 3)),
    "plie_011": ("knee_depth_with_lean", 1),
}


def _trigger_signal(trigger_id, m, phase):
    if trigger_id == "plie_001" and phase in ("descent", "bottom", "ascent"):
        return max(m.get("left_knee_toe_offset_norm", 0), m.get("right_knee_toe_offset_norm", 0))
    if trigger_id == "plie_002":
        return m.get("anterior_tilt_deg", 0)
    if trigger_id == "plie_003" and phase == "descent":
        return max(m.get("left_heel_rise_pct", 0), m.get("right_heel_rise_pct", 0))
    if trigger_id == "plie_005" and phase == "bottom":
        return max(m.get("left_heel_rise_pct", 0), m.get("right_heel_rise_pct", 0))
    if trigger_id == "plie_008" and phase in ("descent", "bottom"):
        return m.get("trunk_lean_angle", 0)
    if trigger_id == "plie_009" and phase in ("descent", "bottom", "ascent"):
        return m.get("hip_height_diff_pct", 0)
    return None


def evaluate_frame(frame, phase):
    """Run all trigger checks against a single frame. Returns list of triggered IDs."""
    m = frame["measurements"]
    triggered = []
    for t in TRIGGERS:
        if t["id"] == "plie_007":
            continue  # handled separately
        if t["check"](m, phase):
            triggered.append(t["id"])
    return triggered


def evaluate_frame_with_temporal_filters(frame, phase, buffers, hysteresis_map):
    """Evaluate frame with visibility gating, hysteresis, and N-of-M confirmation."""
    m = frame["measurements"]
    triggered = []
    for t in TRIGGERS:
        tid = t["id"]
        if tid == "plie_007":
            continue  # handled separately

        if not _visibility_ok(frame, tid):
            raw = False
        else:
            signal = _trigger_signal(tid, m, phase)
            if signal is not None and tid in hysteresis_map:
                raw = hysteresis_map[tid].update(signal)
            else:
                raw = t["check"](m, phase)

        confirmed = buffers[tid].update(raw)
        if confirmed:
            triggered.append(tid)
    return triggered


def check_velocity_uniformity(frames, phases):
    """
    Detect mechanical, resistance-free plié via hip midpoint velocity variance.
    Returns True if velocity is suspiciously uniform (robotic).
    """
    descent_frames = [
        f for f, p in zip(frames, phases) if p == "descent" and "hip_midpoint" in f["measurements"]
    ]
    if len(descent_frames) < 4:
        return False

    positions = [f["measurements"]["hip_midpoint"][1] for f in descent_frames]
    velocities = [abs(positions[i+1] - positions[i]) for i in range(len(positions)-1)]
    if not velocities or max(velocities) < 1e-5:
        return False

    mean_v = sum(velocities) / len(velocities)
    variance = sum((v - mean_v)**2 for v in velocities) / len(velocities)
    cv = (variance**0.5) / (mean_v + 1e-8)
    max_cv = PLIE_CORRECT_VALUES.get("velocity_consistency", {}).get("max_cv", 0.08)
    return cv < max_cv


def check_pause_at_bottom(frames, phases):
    """Detect pause at the bottom of plié (> 2 consecutive frames barely moving)."""
    bottom_frames = [
        f for f, p in zip(frames, phases)
        if p == "bottom" and "hip_midpoint" in f["measurements"]
    ]
    if len(bottom_frames) < 3:
        return False

    positions = [f["measurements"]["hip_midpoint"][1] for f in bottom_frames]
    paused_count = sum(1 for i in range(len(positions)-1) if abs(positions[i+1]-positions[i]) < 0.002)
    required = PLIE_CORRECT_VALUES.get("bottom_pause", {}).get("max_stationary_frames", 2)
    return paused_count >= required


# ── REPORT GENERATION ─────────────────────────────────────────────────

MIN_SEGMENT_MS = 600
MIN_SEGMENT_FRAME_PCT = 0.03


def build_correction_segments(all_triggered, frames, max_gap_frames=5):
    """
    Build timestamp segments per correction ID from frame-level trigger hits.
    Small gaps (<= max_gap_frames) are merged into one continuous segment.
    """
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


def aggregate_corrections(all_triggered, frames):
    """
    Group trigger hits across all frames.
    Returns list of correction dicts sorted by severity + frequency.
    """
    counts = {}
    first_seen = {}
    segments_by_id = build_correction_segments(all_triggered, frames)
    for i, triggered in enumerate(all_triggered):
        ts = frames[i]["timestamp_ms"]
        for tid in triggered:
            counts[tid] = counts.get(tid, 0) + 1
            if tid not in first_seen:
                first_seen[tid] = ts

    corrections = []
    for t in TRIGGERS:
        tid = t["id"]
        if tid in counts:
            corrections.append({
                "id": tid,
                "severity": t["severity"],
                "cue_pt": t["cue_pt"],
                "cue_en": t["cue_en"],
                "frequency": counts[tid],
                "first_seen_ms": first_seen.get(tid, 0),
                "first_seen_ts": _ms_to_ts(first_seen.get(tid, 0)),
                "segments": segments_by_id.get(tid, []),
            })

    # Sort: major first, then by frequency
    corrections.sort(key=lambda c: (0 if c["severity"] == "major" else 1, -c["frequency"]))
    return corrections


def filter_corrections_by_persistence(corrections, total_frames):
    if total_frames <= 0:
        return corrections

    filtered = []
    for c in corrections:
        segs = c.get("segments", [])
        # Drop all sub-threshold segments first
        valid_segs = [s for s in segs if s.get("duration_ms", 0) >= MIN_SEGMENT_MS]
        if not valid_segs:
            continue
        freq = c.get("frequency", 0)
        if freq / max(total_frames, 1) < MIN_SEGMENT_FRAME_PCT:
            continue
        c2 = dict(c)
        c2["segments"] = valid_segs
        filtered.append(c2)
    return filtered


def enrich_corrections_with_confidence(corrections, total_frames):
    enriched = []
    for c in corrections:
        segs = c.get("segments", [])
        longest_frames = max((s.get("frames", 0) for s in segs), default=0)
        freq_ratio = c.get("frequency", 0) / max(1, total_frames)
        seg_ratio = longest_frames / max(1, total_frames)
        severity_bonus = 0.15 if c.get("severity") == "major" else 0.05
        confidence = min(0.99, max(0.0, 0.45 * freq_ratio + 0.35 * seg_ratio + 0.2 + severity_bonus))
        level = "high" if confidence >= 0.67 else ("medium" if confidence >= 0.4 else "low")

        metric_name, threshold = TRIGGER_EVIDENCE.get(c.get("id"), ("unknown_metric", 0))
        c2 = dict(c)
        c2["confidence"] = round(confidence, 3)
        c2["confidence_level"] = level
        c2["evidence"] = {
            "metric_name": metric_name,
            "threshold": threshold,
            "frequency_ratio": round(freq_ratio, 3),
            "longest_segment_frames": longest_frames,
        }
        enriched.append(c2)
    return enriched


def _ms_to_ts(ms):
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def compute_score(corrections, total_frames):
    if total_frames == 0:
        return 100
    penalty = 0.0
    for c in corrections:
        max_per = 18 if c["severity"] == "major" else 10
        freq_ratio = min(c["frequency"] / max(total_frames, 1), 0.75)
        penalty += max_per * freq_ratio
    return max(0, round(100 - penalty))


def analyse_plie(pose_data):
    """
    Full plié analysis pipeline.

    Args:
        pose_data: output from pose_extractor.extract_frames()

    Returns:
        dict with: score, grade, corrections, phases_summary, metadata
    """
    frames = pose_data["frames"]
    fps = pose_data["fps"]

    if not frames:
        return {"error": "No pose data extracted from video."}

    # 1) Smooth noisy trajectories before phase/trigger evaluation.
    frames = smooth_frames(frames, alpha=0.15)
    phases = detect_phases(frames)

    # 2) Temporal filters per trigger: N-of-M + hysteresis.
    buffers = {
        tid: NofMBuffer(n=N_OF_M_CONFIG.get(tid, (2, 3))[0], m=N_OF_M_CONFIG.get(tid, (2, 3))[1])
        for tid in [t["id"] for t in TRIGGERS]
    }
    hysteresis_map = {
        tid: Hysteresis(on_thresh=cfg[0], off_thresh=cfg[1])
        for tid, cfg in HYSTERESIS_CONFIG.items()
    }

    all_triggered = []
    for frame, phase in zip(frames, phases):
        triggered = evaluate_frame_with_temporal_filters(frame, phase, buffers, hysteresis_map)
        all_triggered.append(triggered)

    # Velocity and pause checks
    if check_velocity_uniformity(frames, phases):
        for triggered in all_triggered:
            if "plie_007" not in triggered:
                triggered.append("plie_007")

    if check_pause_at_bottom(frames, phases):
        for idx, phase in enumerate(phases):
            if phase == "bottom":
                all_triggered[idx].append("plie_006")
                break

    corrections = aggregate_corrections(all_triggered, frames)
    corrections = filter_corrections_by_persistence(corrections, len(frames))
    corrections = enrich_corrections_with_confidence(corrections, len(frames))
    correction_timeline = [
        {
            "id": c["id"],
            "severity": c["severity"],
            "segments": c["segments"],
        }
        for c in corrections
    ]

    # Phase summary
    phase_counts = {}
    for p in phases:
        phase_counts[p] = phase_counts.get(p, 0) + 1

    score = compute_score(corrections, len(frames))
    grade = (
        "Excellent" if score >= 90 else
        "Good" if score >= 75 else
        "Needs Work" if score >= 55 else
        "Significant Corrections"
    )

    return {
        "exercise": "Plié",
        "score": score,
        "grade": grade,
        "corrections": corrections,
        "correction_timeline": correction_timeline,
        "phases_summary": phase_counts,
        "total_frames_analysed": len(frames),
        "duration_seconds": round(frames[-1]["timestamp_ms"] / 1000, 1) if frames else 0,
        "fps": fps,
    }
