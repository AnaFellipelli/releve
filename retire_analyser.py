"""
retire_analyser.py
==================
Dedicated phase-detection + correction engine for Retiré / Passé.
Reads ballet_database.json for thresholds and bilingual cues.
Produces a structured report identical in shape to analyse_plie().

Entry point: analyse_retire(pose_data)
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

_TERM = next((t for t in _DB.get("terms", []) if t.get("id") == "retire_passe"), {})
_TERM_FR = _TERM.get("term_fr", "Retiré / Passé")
_CV = _TERM.get("correct_values", {})
_DB_TRIGGERS = {t["id"]: t for t in _TERM.get("correction_triggers", []) if "id" in t}

# Thresholds from DB correct_values
_MAX_KNEE_BEND  = _CV.get("supporting_knee", {}).get("max_bend_degrees", 5)
_MIN_KNEE       = 180 - _MAX_KNEE_BEND                       # 175°
_MAX_HIP_DIFF   = _CV.get("hip_stability", {}).get("max_height_diff_pct", 2)
_MAX_LEAN       = _CV.get("trunk_stability", {}).get("max_lean_degrees", 5)


# ── PHASE DETECTION ────────────────────────────────────────────────────

def _detect_phases(frames):
    """
    Phase signal: knee_height_diff_pct — increases as working knee rises.
    Rising signal — peaks at full retiré (working knee at hip height).
    Note: hip_height_diff_pct is a correction metric (trigger retire_002),
    not a movement signal — it should stay level throughout.
    Phases: preparation | rising | peak | descent | finish
    """
    n = len(frames)
    if n == 0:
        return []
    if n < 6:
        return ["execution"] * n

    meas = [f.get("measurements", {}) for f in frames]
    signal = [m.get("knee_height_diff_pct", 0) for m in meas]
    return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=True)


# ── PRESENCE CONFIDENCE ────────────────────────────────────────────────

def _presence_confidence(frames):
    """
    Detect retiré via knee height asymmetry: the working knee rises while
    the supporting knee stays at hip level, creating a knee_height_diff_pct spike.
    heel_rise_pct was incorrect — in retiré the whole working leg rises,
    keeping heel-to-ankle distance constant.
    """
    if not frames:
        return 0.0
    meas = [f.get("measurements", {}) for f in frames]
    vals = [m.get("knee_height_diff_pct", 0) for m in meas]
    if not vals:
        return 0.0
    peak = max(vals)
    if peak < 6:
        return 0.0
    active = sum(1 for v in vals if v > 4)
    return min(1.0, active / max(1, len(vals)))


# ── VISIBILITY GATING ──────────────────────────────────────────────────

def _visibility_ok(frame, trigger_id):
    vis = frame.get("visibility", {})
    if trigger_id == "retire_001":
        return (gate(vis, "left_hip", "left_knee", "left_ankle") or
                gate(vis, "right_hip", "right_knee", "right_ankle"))
    if trigger_id == "retire_002":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id == "retire_003":
        return gate(vis, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if trigger_id == "retire_004":
        return gate(vis, "left_knee", "right_knee", "left_hip", "right_hip")
    if trigger_id == "retire_005":
        return gate(vis, "left_shoulder", "right_shoulder")
    if trigger_id == "retire_006":
        return gate(vis, "left_hip", "right_hip")
    return True


# ── CORRECTION TRIGGERS ────────────────────────────────────────────────

TRIGGERS = [
    {
        "id": "retire_001",
        "severity": _DB_TRIGGERS.get("retire_001", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and (
                m.get("left_knee_angle", 180) < _MIN_KNEE or
                m.get("right_knee_angle", 180) < _MIN_KNEE
            )
        ),
        "cue_pt": _DB_TRIGGERS.get("retire_001", {}).get("cue_pt",
            "Joelho de apoio dobrado no retiré. Estique completamente o joelho de suporte."),
        "cue_en": _DB_TRIGGERS.get("retire_001", {}).get("cue_en",
            "Supporting knee bending in retiré. Fully extend the supporting knee."),
    },
    {
        "id": "retire_002",
        "severity": _DB_TRIGGERS.get("retire_002", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("hip_height_diff_pct", 0) > _MAX_HIP_DIFF
        ),
        "cue_pt": _DB_TRIGGERS.get("retire_002", {}).get("cue_pt",
            "Quadril subindo no retiré. Não eleve o quadril da perna de trabalho."),
        "cue_en": _DB_TRIGGERS.get("retire_002", {}).get("cue_en",
            "Hip hiking in retiré. Do not lift the working-leg hip."),
    },
    {
        "id": "retire_003",
        "severity": _DB_TRIGGERS.get("retire_003", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("trunk_lean_angle", 0) > _MAX_LEAN
        ),
        "cue_pt": _DB_TRIGGERS.get("retire_003", {}).get("cue_pt",
            "Tronco instável no retiré. Mantenha o eixo ereto."),
        "cue_en": _DB_TRIGGERS.get("retire_003", {}).get("cue_en",
            "Trunk instability in retiré. Keep the axis upright."),
    },
    {
        "id": "retire_004",
        "severity": _DB_TRIGGERS.get("retire_004", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase == "peak" and
            m.get("knee_height_diff_pct", 100) < 8
        ),
        "cue_pt": _DB_TRIGGERS.get("retire_004", {}).get("cue_pt",
            "Joelho de trabalho baixo demais no retiré. Eleva o joelho até à altura do quadril."),
        "cue_en": _DB_TRIGGERS.get("retire_004", {}).get("cue_en",
            "Working knee too low in retiré. Lift the knee to hip height."),
    },
    {
        "id": "retire_005",
        "severity": _DB_TRIGGERS.get("retire_005", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("shoulder_height_diff_pct", 0) > 3
        ),
        "cue_pt": _DB_TRIGGERS.get("retire_005", {}).get("cue_pt",
            "Ombro a subir no retiré. Mantém os ombros nivelados e relaxados."),
        "cue_en": _DB_TRIGGERS.get("retire_005", {}).get("cue_en",
            "Shoulder lifting in retiré. Keep shoulders level and relaxed."),
    },
    {
        "id": "retire_006",
        "severity": _DB_TRIGGERS.get("retire_006", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("hip_rotation_deg", 0) > 10
        ),
        "cue_pt": _DB_TRIGGERS.get("retire_006", {}).get("cue_pt",
            "Quadril a abrir no retiré. Mantém os quadris quadrados."),
        "cue_en": _DB_TRIGGERS.get("retire_006", {}).get("cue_en",
            "Hip opening in retiré. Keep hips square."),
    },
]

N_OF_M_CONFIG = {
    "retire_001": (4, 7),
    "retire_002": (4, 7),
    "retire_003": (4, 7),
    "retire_004": (4,7), "retire_005": (4,7), "retire_006": (4,7),
}

HYSTERESIS_CONFIG = {
    "retire_001": (5.0, 2.0),   # signal: 180 - knee_angle
    "retire_002": (2.0, 0.8),
    "retire_003": (5.0, 3.0),
    "retire_005": (3.0, 1.5), "retire_006": (10.0, 6.0),
}

TRIGGER_EVIDENCE = {
    "retire_001": ("knee_angle_deficit", _MIN_KNEE),
    "retire_002": ("hip_height_diff_pct", _MAX_HIP_DIFF),
    "retire_003": ("trunk_lean_angle", _MAX_LEAN),
    "retire_004": ("knee_height_diff_pct", 8), "retire_005": ("shoulder_height_diff_pct", 3), "retire_006": ("hip_rotation_deg", 10),
}


# ── SIGNAL FOR HYSTERESIS ──────────────────────────────────────────────

def _trigger_signal(trigger_id, m, phase):
    if trigger_id == "retire_001" and phase in ("rising", "peak"):
        return max(
            180 - m.get("left_knee_angle", 180),
            180 - m.get("right_knee_angle", 180),
        )
    if trigger_id == "retire_002" and phase in ("rising", "peak"):
        return m.get("hip_height_diff_pct", 0)
    if trigger_id == "retire_003" and phase in ("rising", "peak"):
        return m.get("trunk_lean_angle", 0)
    if trigger_id == "retire_005" and phase in ("rising", "peak"):
        return m.get("shoulder_height_diff_pct", 0)
    if trigger_id == "retire_006" and phase in ("rising", "peak"):
        return m.get("hip_rotation_deg", 0)
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

def analyse_retire(pose_data):
    """
    Full retiré / passé analysis pipeline.

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
    movement_detected = presence_conf >= 0.40

    if not movement_detected:
        return {
            "exercise": _TERM_FR,
            "exercise_id": "retire_passe",
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
        "exercise_id": "retire_passe",
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
