"""
arabesque_analyser.py
=====================
Dedicated phase-detection + correction engine for Arabesque.
Reads ballet_database.json for thresholds and bilingual cues.
Produces a structured report identical in shape to analyse_plie().

Entry point: analyse_arabesque(pose_data)
"""

import json
from pathlib import Path

from analyser_utils import (
    EMA, NofMBuffer, Hysteresis,
    gate, _ms_to_ts,
    smooth_frames, _signal_phases,
    compute_score, _grade_from_score,
    aggregate_corrections, enrich_corrections_with_confidence,
)


# ── DATABASE ───────────────────────────────────────────────────────────

_DB_PATH = Path(__file__).parent / "ballet_database.json"

with open(_DB_PATH, "r", encoding="utf-8") as _f:
    _DB = json.load(_f)

_TERM = next((t for t in _DB.get("terms", []) if t.get("id") == "arabesque"), {})
_TERM_FR = _TERM.get("term_fr", "Arabesque")
_CV = _TERM.get("correct_values", {})
_DB_TRIGGERS = {t["id"]: t for t in _TERM.get("correction_triggers", []) if "id" in t}

# Thresholds from DB correct_values
_MIN_ARAB_ANGLE  = _CV.get("working_leg_height", {}).get("min_angle_degrees", 90)
_MAX_HIP_ROT     = _CV.get("hip_squareness", {}).get("max_rotation_degrees", 5)
_MAX_TRUNK_DEV   = _CV.get("spine_vertical", {}).get("max_deviation_pct", 8)
_MAX_SHOULDER    = _CV.get("shoulder_level", {}).get("max_height_diff_pct", 4)


# ── PHASE DETECTION ────────────────────────────────────────────────────

def _detect_phases(frames):
    """
    Phase signal: arabesque_angle — DECREASES as leg rises (angle at hip vertex).
    Falling signal — valley corresponds to leg at peak height.
    Phases: preparation | rising | peak | descent | finish
    """
    n = len(frames)
    if n == 0:
        return []
    if n < 6:
        return ["execution"] * n

    meas = [f.get("measurements", {}) for f in frames]
    signal = [m.get("arabesque_angle", 180) for m in meas]
    return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=False)


# ── PRESENCE CONFIDENCE ────────────────────────────────────────────────

def _presence_confidence(frames):
    """
    Leg must reach clearly above horizontal — arabesque_angle drops well
    below the standing baseline.
    """
    if not frames:
        return 0.0
    meas = [f.get("measurements", {}) for f in frames]
    angles = [m.get("arabesque_angle") for m in meas if m.get("arabesque_angle") is not None]
    if not angles:
        return 0.0
    baseline = sorted(angles)[int(len(angles) * 0.7)]
    peak = min(angles)
    drop = baseline - peak
    if drop < 20:
        return 0.0
    active = sum(1 for a in angles if a < baseline - drop * 0.5)
    streak = 0
    best_streak = 0
    for a in angles:
        if a < baseline - drop * 0.5:
            streak += 1
            best_streak = max(best_streak, streak)
        else:
            streak = 0
    c1 = active / max(1, len(angles))
    c2 = best_streak / max(1, len(angles))
    return min(1.0, 0.6 * c1 + 0.4 * c2)


# ── VISIBILITY GATING ──────────────────────────────────────────────────

def _visibility_ok(frame, trigger_id):
    vis = frame.get("visibility", {})
    if trigger_id == "arab_001":
        return gate(vis, "left_hip", "right_hip", "right_ankle")
    if trigger_id == "arab_002":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id == "arab_003":
        return gate(vis, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if trigger_id == "arab_004":
        return gate(vis, "left_shoulder", "right_shoulder")
    return True


# ── CORRECTION TRIGGERS ────────────────────────────────────────────────

TRIGGERS = [
    {
        "id": "arab_001",
        "severity": _DB_TRIGGERS.get("arab_001", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("peak", "execution") and
            m.get("arabesque_angle", 180) > _MIN_ARAB_ANGLE
        ),
        "cue_pt": _DB_TRIGGERS.get("arab_001", {}).get("cue_pt",
            "Arabesque abaixo de 90°. Eleve a perna — inicie o movimento do quadril, não do joelho."),
        "cue_en": _DB_TRIGGERS.get("arab_001", {}).get("cue_en",
            "Arabesque below 90°. Lift the leg — initiate from the hip, not the knee."),
    },
    {
        "id": "arab_002",
        "severity": _DB_TRIGGERS.get("arab_002", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("peak", "execution") and
            m.get("hip_rotation_deg", 0) > _MAX_HIP_ROT
        ),
        "cue_pt": _DB_TRIGGERS.get("arab_002", {}).get("cue_pt",
            "Quadril abrindo. Mantenha ambos os quadris de frente."),
        "cue_en": _DB_TRIGGERS.get("arab_002", {}).get("cue_en",
            "Hip opening. Keep both hips square to the front."),
    },
    {
        "id": "arab_003",
        "severity": _DB_TRIGGERS.get("arab_003", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("peak", "execution") and
            m.get("trunk_lean_angle", 0) > _MAX_TRUNK_DEV
        ),
        "cue_pt": _DB_TRIGGERS.get("arab_003", {}).get("cue_pt",
            "Tronco inclinando muito para frente. Mantenha o eixo ereto."),
        "cue_en": _DB_TRIGGERS.get("arab_003", {}).get("cue_en",
            "Trunk leaning too far forward. Maintain upright axis."),
    },
    {
        "id": "arab_004",
        "severity": _DB_TRIGGERS.get("arab_004", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("peak", "execution") and
            m.get("shoulder_height_diff_pct", 0) > _MAX_SHOULDER
        ),
        "cue_pt": _DB_TRIGGERS.get("arab_004", {}).get("cue_pt",
            "Ombros desiguais. Épaulement — ombros quadrados."),
        "cue_en": _DB_TRIGGERS.get("arab_004", {}).get("cue_en",
            "Shoulders uneven. Épaulement — shoulders square."),
    },
]

N_OF_M_CONFIG = {
    "arab_001": (6, 10),
    "arab_002": (6, 10),
    "arab_003": (6, 10),
    "arab_004": (6, 10),
}

HYSTERESIS_CONFIG = {
    "arab_001": (_MIN_ARAB_ANGLE + 5, _MIN_ARAB_ANGLE - 5),  # fires when angle > 95° (leg too low)
    "arab_002": (_MAX_HIP_ROT, max(1.0, _MAX_HIP_ROT - 1.5)),
    "arab_003": (_MAX_TRUNK_DEV, max(1.0, _MAX_TRUNK_DEV - 2.0)),
    "arab_004": (_MAX_SHOULDER, max(0.8, _MAX_SHOULDER - 1.0)),
}

TRIGGER_EVIDENCE = {
    "arab_001": ("arabesque_angle", _MIN_ARAB_ANGLE),
    "arab_002": ("hip_rotation_deg", _MAX_HIP_ROT),
    "arab_003": ("trunk_lean_angle", _MAX_TRUNK_DEV),
    "arab_004": ("shoulder_height_diff_pct", _MAX_SHOULDER),
}


# ── SIGNAL FOR HYSTERESIS ──────────────────────────────────────────────

def _trigger_signal(trigger_id, m, phase):
    if phase not in ("peak", "execution"):
        return None
    if trigger_id == "arab_001":
        # Raw angle: higher = leg lower. Hysteresis (95, 85): fires when angle > 95°.
        return m.get("arabesque_angle", 180)
    if trigger_id == "arab_002":
        return m.get("hip_rotation_deg", 0)
    if trigger_id == "arab_003":
        return m.get("trunk_lean_angle", 0)
    if trigger_id == "arab_004":
        return m.get("shoulder_height_diff_pct", 0)
    return None


# ── FRAME EVALUATION ───────────────────────────────────────────────────

def _evaluate_frame(frame, phase, buffers, hysteresis_map):
    m = frame["measurements"]
    triggered = []
    for t in TRIGGERS:
        tid = t["id"]
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


# ── MAIN ENTRY POINT ───────────────────────────────────────────────────

def analyse_arabesque(pose_data):
    """
    Full arabesque analysis pipeline.

    Args:
        pose_data: output from pose_extractor.extract_frames()

    Returns:
        dict with: exercise, exercise_id, movement_detected, presence_confidence,
                   score, grade, corrections, correction_timeline,
                   phases_summary, total_frames_analysed, duration_seconds, fps
    """
    frames = pose_data.get("frames", [])
    fps = pose_data.get("fps", 0)

    if not frames:
        return {"error": "No pose data extracted from video."}

    frames = smooth_frames(frames, alpha=0.15)
    presence_conf = _presence_confidence(frames)
    movement_detected = presence_conf >= 0.50

    if not movement_detected:
        return {
            "exercise": _TERM_FR,
            "exercise_id": "arabesque",
            "movement_detected": False,
            "presence_confidence": round(presence_conf, 3),
            "score": 100,
            "grade": "Not Detected",
            "corrections": [],
            "correction_timeline": [],
            "phases_summary": {},
            "total_frames_analysed": len(frames),
            "duration_seconds": round(frames[-1]["timestamp_ms"] / 1000, 1) if frames else 0,
            "fps": fps,
        }

    phases = _detect_phases(frames)

    buffers = {
        tid: NofMBuffer(n=N_OF_M_CONFIG[tid][0], m=N_OF_M_CONFIG[tid][1])
        for tid in N_OF_M_CONFIG
    }
    hysteresis_map = {
        tid: Hysteresis(cfg[0], cfg[1])
        for tid, cfg in HYSTERESIS_CONFIG.items()
    }

    all_triggered = []
    for frame, phase in zip(frames, phases):
        all_triggered.append(_evaluate_frame(frame, phase, buffers, hysteresis_map))

    corrections = aggregate_corrections(all_triggered, frames, TRIGGERS, TRIGGER_EVIDENCE)
    corrections = enrich_corrections_with_confidence(
        corrections, len(frames), TRIGGER_EVIDENCE, presence_conf
    )
    correction_timeline = [
        {"id": c["id"], "severity": c["severity"], "segments": c["segments"]}
        for c in corrections
    ]

    phase_counts = {}
    for p in phases:
        phase_counts[p] = phase_counts.get(p, 0) + 1

    score = compute_score(corrections, len(frames))
    return {
        "exercise": _TERM_FR,
        "exercise_id": "arabesque",
        "movement_detected": True,
        "presence_confidence": round(presence_conf, 3),
        "score": score,
        "grade": _grade_from_score(score),
        "corrections": corrections,
        "correction_timeline": correction_timeline,
        "phases_summary": phase_counts,
        "total_frames_analysed": len(frames),
        "duration_seconds": round(frames[-1]["timestamp_ms"] / 1000, 1) if frames else 0,
        "fps": fps,
    }
