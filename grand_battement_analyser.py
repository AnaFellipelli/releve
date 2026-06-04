"""
grand_battement_analyser.py
===========================
Dedicated phase-detection + correction engine for Grand Battement.
Reads ballet_database.json for thresholds and bilingual cues.
Produces a structured report identical in shape to analyse_plie().

Entry point: analyse_grand_battement(pose_data)
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

_TERM = next((t for t in _DB.get("terms", []) if t.get("id") == "grand_battement"), {})
_TERM_FR = _TERM.get("term_fr", "Grand Battement")
_CV = _TERM.get("correct_values", {})
_DB_TRIGGERS = {t["id"]: t for t in _TERM.get("correction_triggers", []) if "id" in t}

# Thresholds from DB correct_values
_MAX_HIP_DIFF   = _CV.get("hip_stability", {}).get("max_height_diff_pct", 4)
_MIN_ARAB_ANGLE = _CV.get("working_leg_height", {}).get("min_angle_degrees", 90)
_MAX_TRUNK_LEAN = 9   # degrees on descent — from exercise_analyser.py gbat_003 constant


# ── PHASE DETECTION ────────────────────────────────────────────────────

def _detect_phases(frames):
    """
    Phase signal: arabesque_angle — DECREASES as leg rises (same as arabesque).
    Falling signal — valley = peak height.
    Phases: preparation | rising (throw) | peak | descent (controlled return) | finish
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
    Grand battement requires a notably larger leg elevation than arabesque
    — arabesque_angle drops at least 25° below standing baseline.
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
    if drop < 25:
        return 0.0
    active = sum(1 for a in angles if a < baseline - drop * 0.5)
    return min(1.0, active / max(1, len(angles)))


# ── VISIBILITY GATING ──────────────────────────────────────────────────

def _visibility_ok(frame, trigger_id):
    vis = frame.get("visibility", {})
    if trigger_id == "gbat_001":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id == "gbat_002":
        return (gate(vis, "left_hip", "left_knee", "left_ankle") or
                gate(vis, "right_hip", "right_knee", "right_ankle"))
    if trigger_id == "gbat_003":
        return gate(vis, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if trigger_id == "gbat_004":
        return gate(vis, "left_hip", "right_hip", "right_ankle")
    if trigger_id == "gbat_005":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id == "gbat_006":
        return gate(vis, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    return True


# ── CORRECTION TRIGGERS ────────────────────────────────────────────────

TRIGGERS = [
    {
        "id": "gbat_001",
        "severity": _DB_TRIGGERS.get("gbat_001", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("hip_height_diff_pct", 0) > _MAX_HIP_DIFF
        ),
        "cue_pt": _DB_TRIGGERS.get("gbat_001", {}).get("cue_pt",
            "Quadril subindo com a perna. Estabilize o quadril."),
        "cue_en": _DB_TRIGGERS.get("gbat_001", {}).get("cue_en",
            "Hip riding up with the leg. Stabilize the hip."),
    },
    {
        "id": "gbat_002",
        "severity": _DB_TRIGGERS.get("gbat_002", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak", "descent") and (
                m.get("left_knee_angle", 180) < 175 or
                m.get("right_knee_angle", 180) < 175
            )
        ),
        "cue_pt": _DB_TRIGGERS.get("gbat_002", {}).get("cue_pt",
            "Joelho de apoio dobrado. Mantenha o joelho completamente esticado."),
        "cue_en": _DB_TRIGGERS.get("gbat_002", {}).get("cue_en",
            "Supporting knee bending. Keep the knee fully extended throughout."),
    },
    {
        "id": "gbat_003",
        "severity": _DB_TRIGGERS.get("gbat_003", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase == "descent" and
            m.get("trunk_lean_angle", 0) > _MAX_TRUNK_LEAN
        ),
        "cue_pt": _DB_TRIGGERS.get("gbat_003", {}).get("cue_pt",
            "Retorno descontrolado. Controle a descida com a mesma energia da subida."),
        "cue_en": _DB_TRIGGERS.get("gbat_003", {}).get("cue_en",
            "Uncontrolled return. Control the descent with the same energy as the lift."),
    },
    {
        "id": "gbat_004",
        "severity": _DB_TRIGGERS.get("gbat_004", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase == "peak" and
            m.get("arabesque_angle", 180) > 135
        ),
        "cue_pt": _DB_TRIGGERS.get("gbat_004", {}).get("cue_pt",
            "Batimento insuficiente. Eleva a perna pelo menos a 45°."),
        "cue_en": _DB_TRIGGERS.get("gbat_004", {}).get("cue_en",
            "Insufficient battement height. Lift the leg at least 45°."),
    },
    {
        "id": "gbat_005",
        "severity": _DB_TRIGGERS.get("gbat_005", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("hip_rotation_deg", 0) > 15
        ),
        "cue_pt": _DB_TRIGGERS.get("gbat_005", {}).get("cue_pt",
            "Quadril a rodar no batimento. Mantém o quadril estabilizado."),
        "cue_en": _DB_TRIGGERS.get("gbat_005", {}).get("cue_en",
            "Hip twisting on battement. Keep the hip stabilized."),
    },
    {
        "id": "gbat_006",
        "severity": _DB_TRIGGERS.get("gbat_006", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase == "rising" and
            m.get("anterior_tilt_deg", 0) > 10
        ),
        "cue_pt": _DB_TRIGGERS.get("gbat_006", {}).get("cue_pt",
            "Tronco a compensar na subida. Controla o eixo durante o batimento."),
        "cue_en": _DB_TRIGGERS.get("gbat_006", {}).get("cue_en",
            "Trunk compensating on the kick. Control the axis during the battement."),
    },
]

N_OF_M_CONFIG = {
    "gbat_001": (6, 10),
    "gbat_002": (6, 10),
    "gbat_003": (6, 10),
    "gbat_004": (5,9), "gbat_005": (5,9), "gbat_006": (5,9),
}

HYSTERESIS_CONFIG = {
    "gbat_001": (_MAX_HIP_DIFF, max(1.0, _MAX_HIP_DIFF - 1.2)),
    "gbat_002": (5.0, 3.0),    # signal: 180 - knee_angle
    "gbat_003": (_MAX_TRUNK_LEAN, 7.0),
    "gbat_005": (15.0, 10.0), "gbat_006": (10.0, 6.0),
}

TRIGGER_EVIDENCE = {
    "gbat_001": ("hip_height_diff_pct", _MAX_HIP_DIFF),
    "gbat_002": ("knee_angle", 175),
    "gbat_003": ("trunk_lean_angle", _MAX_TRUNK_LEAN),
    "gbat_004": ("arabesque_angle", 135), "gbat_005": ("hip_rotation_deg", 15), "gbat_006": ("anterior_tilt_deg", 10),
}


# ── SIGNAL FOR HYSTERESIS ──────────────────────────────────────────────

def _trigger_signal(trigger_id, m, phase):
    if trigger_id == "gbat_001" and phase in ("rising", "peak"):
        return m.get("hip_height_diff_pct", 0)
    if trigger_id == "gbat_002" and phase in ("rising", "peak", "descent"):
        return max(
            180 - m.get("left_knee_angle", 180),
            180 - m.get("right_knee_angle", 180),
        )
    if trigger_id == "gbat_003" and phase == "descent":
        return m.get("trunk_lean_angle", 0)
    if trigger_id == "gbat_005" and phase in ("rising", "peak"):
        return m.get("hip_rotation_deg", 0)
    if trigger_id == "gbat_006" and phase == "rising":
        return m.get("anterior_tilt_deg", 0)
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

def analyse_grand_battement(pose_data):
    """
    Full grand battement analysis pipeline.

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
            "exercise_id": "grand_battement",
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
        "exercise_id": "grand_battement",
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
