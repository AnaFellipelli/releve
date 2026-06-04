"""
releve_analyser.py
==================
Dedicated phase-detection + correction engine for Relevé.
Reads ballet_database.json for thresholds and bilingual cues.
Produces a structured report identical in shape to analyse_plie().

Entry point: analyse_releve(pose_data)
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

_TERM = next((t for t in _DB.get("terms", []) if t.get("id") == "releve"), {})
_TERM_FR = _TERM.get("term_fr", "Relevé")
_CV = _TERM.get("correct_values", {})
_DB_TRIGGERS = {t["id"]: t for t in _TERM.get("correction_triggers", []) if "id" in t}

# Thresholds from DB correct_values
_MIN_KNEE_ANGLE  = _CV.get("knees", {}).get("min_angle_degrees", 170)
_MIN_RISE_PCT    = _CV.get("heel_rise", {}).get("min_rise_pct", 10)
_MAX_ASYM_PCT    = _CV.get("heel_symmetry", {}).get("max_asymmetry_pct", 5)
_MAX_HIP_DIFF    = _CV.get("hip_stability", {}).get("max_height_diff_pct", 2)
_MAX_LEAN        = _CV.get("trunk_stability", {}).get("max_lean_degrees", 5)


# ── PHASE DETECTION ────────────────────────────────────────────────────

def _detect_phases(frames):
    """
    Phase signal: max(left_heel_rise_pct, right_heel_rise_pct)
    Rising signal — peaks at full relevé.
    Phases: preparation | rising | peak | descent | finish
    """
    n = len(frames)
    if n == 0:
        return []
    if n < 6:
        return ["execution"] * n

    meas = [f.get("measurements", {}) for f in frames]
    signal = [
        max(m.get("left_heel_rise_pct", 0), m.get("right_heel_rise_pct", 0))
        for m in meas
    ]
    return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=True)


# ── PRESENCE CONFIDENCE ────────────────────────────────────────────────

def _presence_confidence(frames):
    if not frames:
        return 0.0
    meas = [f.get("measurements", {}) for f in frames]
    l_vals = [m.get("left_heel_rise_pct", 0) for m in meas]
    r_vals = [m.get("right_heel_rise_pct", 0) for m in meas]
    if not l_vals:
        return 0.0
    max_rise = max(max(l_vals), max(r_vals))
    active = sum(1 for lv, rv in zip(l_vals, r_vals) if max(lv, rv) > 6)
    return min(1.0, 0.5 * (active / max(1, len(l_vals))) + 0.5 * min(1.0, max_rise / 15.0))


# ── VISIBILITY GATING ──────────────────────────────────────────────────

def _visibility_ok(frame, trigger_id):
    vis = frame.get("visibility", {})
    if trigger_id == "releve_001":
        return (gate(vis, "left_hip", "left_knee", "left_ankle") or
                gate(vis, "right_hip", "right_knee", "right_ankle"))
    if trigger_id == "releve_002":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id in ("releve_003", "releve_004"):
        return (gate(vis, "left_heel", "left_ankle") or
                gate(vis, "right_heel", "right_ankle"))
    if trigger_id == "releve_005":
        return gate(vis, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    return True


# ── CORRECTION TRIGGERS ────────────────────────────────────────────────

TRIGGERS = [
    {
        "id": "releve_001",
        "severity": _DB_TRIGGERS.get("releve_001", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and (
                m.get("left_knee_angle", 180) < _MIN_KNEE_ANGLE or
                m.get("right_knee_angle", 180) < _MIN_KNEE_ANGLE
            )
        ),
        "cue_pt": _DB_TRIGGERS.get("releve_001", {}).get("cue_pt",
            "Joelhos não estendidos no relevé. Estique completamente ambos os joelhos."),
        "cue_en": _DB_TRIGGERS.get("releve_001", {}).get("cue_en",
            "Knees not fully extended in relevé. Fully straighten both knees."),
    },
    {
        "id": "releve_002",
        "severity": _DB_TRIGGERS.get("releve_002", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("hip_height_diff_pct", 0) > _MAX_HIP_DIFF
        ),
        "cue_pt": _DB_TRIGGERS.get("releve_002", {}).get("cue_pt",
            "Quadril deslocando lateralmente no relevé. Suba de forma simétrica."),
        "cue_en": _DB_TRIGGERS.get("releve_002", {}).get("cue_en",
            "Hip shifting laterally in relevé. Rise symmetrically."),
    },
    {
        "id": "releve_003",
        "severity": _DB_TRIGGERS.get("releve_003", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase == "peak" and
            max(m.get("left_heel_rise_pct", 0), m.get("right_heel_rise_pct", 0)) < _MIN_RISE_PCT
        ),
        "cue_pt": _DB_TRIGGERS.get("releve_003", {}).get("cue_pt",
            "Subida insuficiente no relevé. Eleve os calcanhares ao máximo."),
        "cue_en": _DB_TRIGGERS.get("releve_003", {}).get("cue_en",
            "Insufficient rise in relevé. Lift the heels to maximum."),
    },
    {
        "id": "releve_004",
        "severity": _DB_TRIGGERS.get("releve_004", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            abs(m.get("left_heel_rise_pct", 0) - m.get("right_heel_rise_pct", 0)) > _MAX_ASYM_PCT
        ),
        "cue_pt": _DB_TRIGGERS.get("releve_004", {}).get("cue_pt",
            "Subida assimétrica — um calcanhar sobe mais que o outro."),
        "cue_en": _DB_TRIGGERS.get("releve_004", {}).get("cue_en",
            "Asymmetric rise — one heel higher than the other."),
    },
    {
        "id": "releve_005",
        "severity": _DB_TRIGGERS.get("releve_005", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("trunk_lean_angle", 0) > _MAX_LEAN
        ),
        "cue_pt": _DB_TRIGGERS.get("releve_005", {}).get("cue_pt",
            "Tronco compensando no relevé. Mantenha o eixo vertical."),
        "cue_en": _DB_TRIGGERS.get("releve_005", {}).get("cue_en",
            "Trunk compensating in relevé. Keep the axis vertical."),
    },
]

N_OF_M_CONFIG = {
    "releve_001": (4, 7),
    "releve_002": (4, 7),
    "releve_003": (4, 7),
    "releve_004": (4, 7),
    "releve_005": (4, 7),
}

HYSTERESIS_CONFIG = {
    "releve_001": (10.0, 6.0),   # bend signal: 180 - knee_angle
    "releve_002": (2.0, 0.8),
    "releve_003": (3.0, 1.0),    # shortfall signal: _MIN_RISE_PCT - heel_rise; fires when rise < 7%
    "releve_004": (5.0, 3.0),
    "releve_005": (5.0, 3.0),
}

TRIGGER_EVIDENCE = {
    "releve_001": ("knee_angle_deficit", _MIN_KNEE_ANGLE),
    "releve_002": ("hip_height_diff_pct", _MAX_HIP_DIFF),
    "releve_003": ("heel_rise_pct", _MIN_RISE_PCT),
    "releve_004": ("heel_rise_asymmetry_pct", _MAX_ASYM_PCT),
    "releve_005": ("trunk_lean_angle", _MAX_LEAN),
}


# ── SIGNAL FOR HYSTERESIS ──────────────────────────────────────────────

def _trigger_signal(trigger_id, m, phase):
    if trigger_id == "releve_001" and phase in ("rising", "peak"):
        return max(
            180 - m.get("left_knee_angle", 180),
            180 - m.get("right_knee_angle", 180),
        )
    if trigger_id == "releve_002" and phase in ("rising", "peak"):
        return m.get("hip_height_diff_pct", 0)
    if trigger_id == "releve_003" and phase == "peak":
        return max(0, _MIN_RISE_PCT - max(
            m.get("left_heel_rise_pct", 0), m.get("right_heel_rise_pct", 0)
        ))
    if trigger_id == "releve_004" and phase in ("rising", "peak"):
        return abs(m.get("left_heel_rise_pct", 0) - m.get("right_heel_rise_pct", 0))
    if trigger_id == "releve_005" and phase in ("rising", "peak"):
        return m.get("trunk_lean_angle", 0)
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

def analyse_releve(pose_data):
    """
    Full relevé analysis pipeline.

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
    movement_detected = presence_conf >= 0.45

    if not movement_detected:
        return {
            "exercise": _TERM_FR,
            "exercise_id": "releve",
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
        "exercise_id": "releve",
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
