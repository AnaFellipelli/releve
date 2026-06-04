"""
port_de_bras_analyser.py
========================
Dedicated phase-detection + correction engine for Port de Bras.
Reads ballet_database.json for thresholds and bilingual cues.
Produces a structured report identical in shape to analyse_plie().

Entry point: analyse_port_de_bras(pose_data)
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

_TERM = next((t for t in _DB.get("terms", []) if t.get("id") == "port_de_bras"), {})
_TERM_FR = _TERM.get("term_fr", "Port de Bras")
_CV = _TERM.get("correct_values", {})
_DB_TRIGGERS = {t["id"]: t for t in _TERM.get("correction_triggers", []) if "id" in t}

# Thresholds from DB correct_values
_MIN_ELBOW      = _CV.get("elbow_curve", {}).get("min_angle_degrees", 140)
_MAX_ELBOW      = _CV.get("elbow_curve", {}).get("max_angle_degrees", 160)
_MAX_SHOULDER   = _CV.get("shoulder_tension", {}).get("max_elevation_pct", 2)
_MAX_WRIST_DROP = 6   # degrees — from exercise_analyser.py bras_003 constant


# ── PHASE DETECTION ────────────────────────────────────────────────────

def _detect_phases(frames):
    """
    Phase signal: total elbow deviation from target 150°.
    abs(left_elbow_angle - 150) + abs(right_elbow_angle - 150)
    Rising signal — peaks when arms are furthest from neutral.
    Phases: preparation | rising | peak | descent | finish
    """
    n = len(frames)
    if n == 0:
        return []
    if n < 6:
        return ["execution"] * n

    meas = [f.get("measurements", {}) for f in frames]
    signal = [
        abs(m.get("left_elbow_angle", 150) - 150) + abs(m.get("right_elbow_angle", 150) - 150)
        for m in meas
    ]
    return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=True)


# ── PRESENCE CONFIDENCE ────────────────────────────────────────────────

def _presence_confidence(frames):
    """
    Detect port de bras via notable angular variation in both elbows.
    Conservative: requires both arms to show variation.
    """
    if not frames:
        return 0.0
    meas = [f.get("measurements", {}) for f in frames]
    l_vals = [m.get("left_elbow_angle") for m in meas if m.get("left_elbow_angle") is not None]
    r_vals = [m.get("right_elbow_angle") for m in meas if m.get("right_elbow_angle") is not None]
    if not l_vals or not r_vals:
        return 0.0
    l_var = (max(l_vals) - min(l_vals)) / 40.0
    r_var = (max(r_vals) - min(r_vals)) / 40.0
    return max(0.0, min(1.0, (l_var + r_var) / 2.0))


# ── VISIBILITY GATING ──────────────────────────────────────────────────

def _visibility_ok(frame, trigger_id):
    vis = frame.get("visibility", {})
    if trigger_id in ("bras_001", "bras_002"):
        return gate(vis, "left_shoulder", "right_shoulder", "left_elbow", "right_elbow")
    if trigger_id == "bras_003":
        return gate(vis, "left_wrist", "right_wrist", "left_elbow", "right_elbow")
    return True


# ── CORRECTION TRIGGERS ────────────────────────────────────────────────

TRIGGERS = [
    {
        "id": "bras_001",
        "severity": _DB_TRIGGERS.get("bras_001", {}).get("severity", "major"),
        "check": lambda m, phase: (
            m.get("shoulder_height_diff_pct", 0) > _MAX_SHOULDER
        ),
        "cue_pt": _DB_TRIGGERS.get("bras_001", {}).get("cue_pt",
            "Ombros subindo durante o port de bras. Relaxe os ombros para baixo."),
        "cue_en": _DB_TRIGGERS.get("bras_001", {}).get("cue_en",
            "Shoulders rising during port de bras. Release shoulders down."),
    },
    {
        "id": "bras_002",
        "severity": _DB_TRIGGERS.get("bras_002", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and (
                m.get("left_elbow_angle", 150) > _MAX_ELBOW or
                m.get("right_elbow_angle", 150) > _MAX_ELBOW or
                m.get("left_elbow_angle", 150) < _MIN_ELBOW or
                m.get("right_elbow_angle", 150) < _MIN_ELBOW
            )
        ),
        "cue_pt": _DB_TRIGGERS.get("bras_002", {}).get("cue_pt",
            "Cotovelos muito retos. Mantenha a curva suave do braço."),
        "cue_en": _DB_TRIGGERS.get("bras_002", {}).get("cue_en",
            "Elbows too straight. Maintain the soft arm curve."),
    },
    {
        "id": "bras_003",
        "severity": _DB_TRIGGERS.get("bras_003", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and (
                m.get("left_wrist_drop_pct", 0) > _MAX_WRIST_DROP or
                m.get("right_wrist_drop_pct", 0) > _MAX_WRIST_DROP
            )
        ),
        "cue_pt": _DB_TRIGGERS.get("bras_003", {}).get("cue_pt",
            "Pulso quebrado. Alinhe o pulso com o antebraço."),
        "cue_en": _DB_TRIGGERS.get("bras_003", {}).get("cue_en",
            "Broken wrist. Align the wrist with the forearm."),
    },
]

N_OF_M_CONFIG = {
    "bras_001": (6, 10),
    "bras_002": (6, 10),
    "bras_003": (6, 10),
}

HYSTERESIS_CONFIG = {
    "bras_001": (_MAX_SHOULDER, max(0.6, _MAX_SHOULDER - 0.8)),
    "bras_003": (_MAX_WRIST_DROP, 4.5),
}

TRIGGER_EVIDENCE = {
    "bras_001": ("shoulder_height_diff_pct", _MAX_SHOULDER),
    "bras_002": ("elbow_angle", f"[{_MIN_ELBOW}, {_MAX_ELBOW}]"),
    "bras_003": ("wrist_drop_pct", _MAX_WRIST_DROP),
}


# ── SIGNAL FOR HYSTERESIS ──────────────────────────────────────────────

def _trigger_signal(trigger_id, m, phase):
    if trigger_id == "bras_001":
        return m.get("shoulder_height_diff_pct", 0)
    if trigger_id == "bras_003" and phase in ("rising", "peak"):
        return max(m.get("left_wrist_drop_pct", 0), m.get("right_wrist_drop_pct", 0))
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

def analyse_port_de_bras(pose_data):
    """
    Full port de bras analysis pipeline.

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
            "exercise_id": "port_de_bras",
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
        "exercise_id": "port_de_bras",
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
