"""
fondu_analyser.py
=================
Dedicated phase-detection + correction engine for Fondu.
Reads ballet_database.json for thresholds and bilingual cues.
Produces a structured report identical in shape to analyse_plie().

Entry point: analyse_fondu(pose_data)
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

_TERM = next((t for t in _DB.get("terms", []) if t.get("id") == "fondu"), {})
_TERM_FR = _TERM.get("term_fr", "Fondu")
_CV = _TERM.get("correct_values", {})
_DB_TRIGGERS = {t["id"]: t for t in _TERM.get("correction_triggers", []) if "id" in t}

# Thresholds from DB correct_values
_MAX_KNEE_OFFSET = _CV.get("supporting_knee", {}).get("max_offset_normalized", 0.06)
_MAX_HIP_DIFF    = _CV.get("hip_stability", {}).get("max_height_diff_pct", 3)
_MAX_LEAN        = _CV.get("trunk_stability", {}).get("max_lean_degrees", 8)


# ── PHASE DETECTION ────────────────────────────────────────────────────

def _detect_phases(frames):
    """
    Phase signal: min(left_knee_angle, right_knee_angle)
    Falling signal — knee angle DROPS as dancer bends (valley = peak phase).
    Phases: preparation | rising (into bend) | peak (deepest) | descent (return) | finish
    """
    n = len(frames)
    if n == 0:
        return []
    if n < 6:
        return ["execution"] * n

    meas = [f.get("measurements", {}) for f in frames]
    signal = [
        min(m.get("left_knee_angle", 180), m.get("right_knee_angle", 180))
        for m in meas
    ]
    return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=False)


# ── PRESENCE CONFIDENCE ────────────────────────────────────────────────

def _presence_confidence(frames):
    """
    Detect fondu via asymmetric knee bend: one knee bends while the other
    stays extended (single-leg fondu).
    """
    if not frames:
        return 0.0
    meas = [f.get("measurements", {}) for f in frames]
    ml = [m.get("left_knee_angle", 180) for m in meas]
    mr = [m.get("right_knee_angle", 180) for m in meas]
    if not ml:
        return 0.0
    asym = [abs(lv - rv) for lv, rv in zip(ml, mr)]
    active = sum(1 for a in asym if a > 15)
    return min(1.0, active / max(1, len(asym)))


# ── VISIBILITY GATING ──────────────────────────────────────────────────

def _visibility_ok(frame, trigger_id):
    vis = frame.get("visibility", {})
    if trigger_id == "fondu_001":
        return (gate(vis, "left_knee", "left_toe") or
                gate(vis, "right_knee", "right_toe"))
    if trigger_id == "fondu_002":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id == "fondu_003":
        return gate(vis, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if trigger_id == "fondu_004":
        return gate(vis, "left_hip", "right_hip", "right_ankle")
    if trigger_id == "fondu_005":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id == "fondu_006":
        return gate(vis, "nose") or gate(vis, "left_hip", "right_hip")
    return True


# ── CORRECTION TRIGGERS ────────────────────────────────────────────────

TRIGGERS = [
    {
        "id": "fondu_001",
        "severity": _DB_TRIGGERS.get("fondu_001", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and (
                m.get("left_knee_toe_offset_norm", 0) > _MAX_KNEE_OFFSET or
                m.get("right_knee_toe_offset_norm", 0) > _MAX_KNEE_OFFSET
            )
        ),
        "cue_pt": _DB_TRIGGERS.get("fondu_001", {}).get("cue_pt",
            "Joelho de apoio caindo para dentro no fondu. Mantenha o joelho sobre o 2º e 3º dedos."),
        "cue_en": _DB_TRIGGERS.get("fondu_001", {}).get("cue_en",
            "Supporting knee collapsing inward in fondu. Keep the knee over the 2nd and 3rd toes."),
    },
    {
        "id": "fondu_002",
        "severity": _DB_TRIGGERS.get("fondu_002", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("hip_height_diff_pct", 0) > _MAX_HIP_DIFF
        ),
        "cue_pt": _DB_TRIGGERS.get("fondu_002", {}).get("cue_pt",
            "Quadril desnivelado no fondu. Mantenha os quadris nivelados."),
        "cue_en": _DB_TRIGGERS.get("fondu_002", {}).get("cue_en",
            "Hip not level in fondu. Keep hips level."),
    },
    {
        "id": "fondu_003",
        "severity": _DB_TRIGGERS.get("fondu_003", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("trunk_lean_angle", 0) > _MAX_LEAN
        ),
        "cue_pt": _DB_TRIGGERS.get("fondu_003", {}).get("cue_pt",
            "Tronco compensando no fondu. Mantenha o eixo longo."),
        "cue_en": _DB_TRIGGERS.get("fondu_003", {}).get("cue_en",
            "Trunk compensating in fondu. Keep the axis long."),
    },
    {
        "id": "fondu_004",
        "severity": _DB_TRIGGERS.get("fondu_004", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("descent", "finish") and
            m.get("arabesque_angle", 180) > 155
        ),
        "cue_pt": _DB_TRIGGERS.get("fondu_004", {}).get("cue_pt",
            "Perna de trabalho não se estende no fondu. Estende completamente no final do movimento."),
        "cue_en": _DB_TRIGGERS.get("fondu_004", {}).get("cue_en",
            "Working leg not extending in fondu. Fully extend at the end of the movement."),
    },
    {
        "id": "fondu_005",
        "severity": _DB_TRIGGERS.get("fondu_005", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("hip_rotation_deg", 0) > 12
        ),
        "cue_pt": _DB_TRIGGERS.get("fondu_005", {}).get("cue_pt",
            "Quadril a rodar no fondu. Mantém o en dehors sem rodar o quadril."),
        "cue_en": _DB_TRIGGERS.get("fondu_005", {}).get("cue_en",
            "Hip rotating in fondu. Maintain en dehors without twisting the hip."),
    },
    {
        "id": "fondu_006",
        "severity": _DB_TRIGGERS.get("fondu_006", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("spine_vertical_angle", 0) > 12
        ),
        "cue_pt": _DB_TRIGGERS.get("fondu_006", {}).get("cue_pt",
            "Coluna a ceder no fondu. Alonga a coluna — cresce para cima."),
        "cue_en": _DB_TRIGGERS.get("fondu_006", {}).get("cue_en",
            "Spine collapsing in fondu. Lengthen the spine — grow upward."),
    },
]

N_OF_M_CONFIG = {
    "fondu_001": (4, 7),
    "fondu_002": (4, 7),
    "fondu_003": (4, 7),
    "fondu_004": (4,7), "fondu_005": (4,7), "fondu_006": (4,7),
}

HYSTERESIS_CONFIG = {
    "fondu_001": (_MAX_KNEE_OFFSET, max(0.03, _MAX_KNEE_OFFSET - 0.02)),
    "fondu_002": (_MAX_HIP_DIFF, max(1.0, _MAX_HIP_DIFF - 1.5)),
    "fondu_003": (_MAX_LEAN, max(4.0, _MAX_LEAN - 2.0)),
    "fondu_005": (12.0, 8.0), "fondu_006": (12.0, 8.0),
}

TRIGGER_EVIDENCE = {
    "fondu_001": ("knee_toe_offset_norm", _MAX_KNEE_OFFSET),
    "fondu_002": ("hip_height_diff_pct", _MAX_HIP_DIFF),
    "fondu_003": ("trunk_lean_angle", _MAX_LEAN),
    "fondu_004": ("arabesque_angle", 155), "fondu_005": ("hip_rotation_deg", 12), "fondu_006": ("spine_vertical_angle", 12),
}


# ── SIGNAL FOR HYSTERESIS ──────────────────────────────────────────────

def _trigger_signal(trigger_id, m, phase):
    if trigger_id == "fondu_001" and phase in ("rising", "peak"):
        return max(
            m.get("left_knee_toe_offset_norm", 0),
            m.get("right_knee_toe_offset_norm", 0),
        )
    if trigger_id == "fondu_002" and phase in ("rising", "peak"):
        return m.get("hip_height_diff_pct", 0)
    if trigger_id == "fondu_003" and phase in ("rising", "peak"):
        return m.get("trunk_lean_angle", 0)
    if trigger_id == "fondu_005" and phase in ("rising", "peak"):
        return m.get("hip_rotation_deg", 0)
    if trigger_id == "fondu_006" and phase in ("rising", "peak"):
        return m.get("spine_vertical_angle", 0)
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

def analyse_fondu(pose_data):
    """
    Full fondu analysis pipeline.

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
            "exercise_id": "fondu",
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
        "exercise_id": "fondu",
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
