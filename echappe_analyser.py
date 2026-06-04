"""
echappe_analyser.py
===================
Dedicated phase-detection + correction engine for Échappé Relevé.
Entry point: analyse_echappe(pose_data)
"""

import json
from pathlib import Path

from analyser_utils import (
    EMA, NofMBuffer, Hysteresis,
    gate, _ms_to_ts,
    smooth_frames, _signal_phases,
    compute_score, _grade_from_score,
    build_correction_segments, aggregate_corrections,
    filter_corrections_by_persistence, enrich_corrections_with_confidence,
)

_DB_PATH = Path(__file__).parent / "ballet_database.json"
with open(_DB_PATH, "r", encoding="utf-8") as _f:
    _DB = json.load(_f)

_TERM = next((t for t in _DB.get("terms", []) if t.get("id") == "echappe"), {})
_TERM_FR = _TERM.get("term_fr", "Échappé Relevé")
_CV = _TERM.get("correct_values", {})
_DB_TRIGGERS = {t["id"]: t for t in _TERM.get("correction_triggers", []) if "id" in t}

_MIN_KNEE_ANGLE   = _CV.get("knee_extension", {}).get("min_angle_degrees", 168)
_MIN_SPREAD       = _CV.get("foot_opening", {}).get("min_spread_norm", 0.25)
_MAX_CLOSE_KNEE   = _CV.get("landing_plie", {}).get("max_knee_angle_at_close", 165)
_MAX_HIP_DIFF     = _CV.get("hip_stability", {}).get("max_height_diff_pct", 2.5)
_MAX_LEAN         = _CV.get("trunk_stability", {}).get("max_lean_degrees", 6)


def _detect_phases(frames):
    n = len(frames)
    if n == 0: return []
    if n < 6: return ["execution"] * n
    meas = [f.get("measurements", {}) for f in frames]
    signal = [
        max(m.get("left_heel_rise_pct", 0), m.get("right_heel_rise_pct", 0))
        for m in meas
    ]
    return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=True)


def _presence_confidence(frames):
    if not frames: return 0.0
    meas = [f.get("measurements", {}) for f in frames]
    bilateral = [
        min(m.get("left_heel_rise_pct", 0), m.get("right_heel_rise_pct", 0))
        for m in meas
    ]
    active = sum(1 for v in bilateral if v > 4)
    return min(1.0, active / max(1, len(bilateral)))


def _visibility_ok(frame, trigger_id):
    vis = frame.get("visibility", {})
    if trigger_id == "echappe_001":
        return (gate(vis, "left_hip", "left_knee", "left_ankle") or
                gate(vis, "right_hip", "right_knee", "right_ankle"))
    if trigger_id == "echappe_002":
        return (gate(vis, "left_toe") and gate(vis, "right_toe"))
    if trigger_id == "echappe_003":
        return (gate(vis, "left_hip", "left_knee", "left_ankle") or
                gate(vis, "right_hip", "right_knee", "right_ankle"))
    if trigger_id == "echappe_004":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id == "echappe_005":
        return gate(vis, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    return True


TRIGGERS = [
    {
        "id": "echappe_001",
        "severity": _DB_TRIGGERS.get("echappe_001", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase == "peak" and (
                m.get("left_knee_angle", 180) < _MIN_KNEE_ANGLE or
                m.get("right_knee_angle", 180) < _MIN_KNEE_ANGLE
            )
        ),
        "cue_pt": _DB_TRIGGERS.get("echappe_001", {}).get("cue_pt",
            "Joelhos não estendidos no échappé. Estica completamente os dois joelhos na subida."),
        "cue_en": _DB_TRIGGERS.get("echappe_001", {}).get("cue_en",
            "Knees not extended in échappé. Fully extend both knees at the top."),
    },
    {
        "id": "echappe_002",
        "severity": _DB_TRIGGERS.get("echappe_002", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase == "peak" and
            m.get("foot_spread_norm", 1.0) < _MIN_SPREAD
        ),
        "cue_pt": _DB_TRIGGERS.get("echappe_002", {}).get("cue_pt",
            "Pés não abrem suficientemente para a 2ª posição. Abre completamente no échappé."),
        "cue_en": _DB_TRIGGERS.get("echappe_002", {}).get("cue_en",
            "Feet not opening fully to 2nd position. Open fully in the échappé."),
    },
    {
        "id": "echappe_003",
        "severity": _DB_TRIGGERS.get("echappe_003", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("descent", "finish") and
            min(m.get("left_knee_angle", 180), m.get("right_knee_angle", 180)) > _MAX_CLOSE_KNEE
        ),
        "cue_pt": _DB_TRIGGERS.get("echappe_003", {}).get("cue_pt",
            "Fecho sem demi-plié. Amortece o fechamento com os joelhos."),
        "cue_en": _DB_TRIGGERS.get("echappe_003", {}).get("cue_en",
            "Closing without demi-plié. Absorb the closing with the knees."),
    },
    {
        "id": "echappe_004",
        "severity": _DB_TRIGGERS.get("echappe_004", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            m.get("hip_height_diff_pct", 0) > _MAX_HIP_DIFF
        ),
        "cue_pt": _DB_TRIGGERS.get("echappe_004", {}).get("cue_pt",
            "Quadril desnivelado no échappé. Mantém os quadris simétricos."),
        "cue_en": _DB_TRIGGERS.get("echappe_004", {}).get("cue_en",
            "Hip not level in échappé. Keep hips symmetrical."),
    },
    {
        "id": "echappe_005",
        "severity": _DB_TRIGGERS.get("echappe_005", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            m.get("trunk_lean_angle", 0) > _MAX_LEAN
        ),
        "cue_pt": _DB_TRIGGERS.get("echappe_005", {}).get("cue_pt",
            "Tronco a compensar no échappé. Mantém o eixo vertical."),
        "cue_en": _DB_TRIGGERS.get("echappe_005", {}).get("cue_en",
            "Trunk compensating in échappé. Keep the axis vertical."),
    },
]

N_OF_M_CONFIG = {t["id"]: (4, 7) for t in TRIGGERS}

HYSTERESIS_CONFIG = {
    "echappe_004": (_MAX_HIP_DIFF, max(1.0, _MAX_HIP_DIFF - 1.0)),
    "echappe_005": (_MAX_LEAN, max(3.0, _MAX_LEAN - 2.0)),
}

TRIGGER_EVIDENCE = {
    "echappe_001": ("knee_angle", _MIN_KNEE_ANGLE),
    "echappe_002": ("foot_spread_norm", _MIN_SPREAD),
    "echappe_003": ("knee_angle", _MAX_CLOSE_KNEE),
    "echappe_004": ("hip_height_diff_pct", _MAX_HIP_DIFF),
    "echappe_005": ("trunk_lean_angle", _MAX_LEAN),
}


def _trigger_signal(trigger_id, m, phase):
    if trigger_id == "echappe_004":
        return m.get("hip_height_diff_pct", 0)
    if trigger_id == "echappe_005":
        return m.get("trunk_lean_angle", 0)
    return None


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


def analyse_echappe(pose_data):
    frames = pose_data.get("frames", [])
    fps = pose_data.get("fps", 0)
    if not frames:
        return {"error": "No pose data extracted from video."}

    frames = smooth_frames(frames, alpha=0.15)
    presence_conf = _presence_confidence(frames)
    movement_detected = presence_conf >= 0.30

    if not movement_detected:
        return {
            "exercise": _TERM_FR, "exercise_id": "echappe",
            "movement_detected": False, "presence_confidence": round(presence_conf, 3),
            "score": 100, "grade": "Not Detected",
            "corrections": [], "correction_timeline": [], "phases_summary": {},
            "total_frames_analysed": len(frames),
            "duration_seconds": round(frames[-1]["timestamp_ms"] / 1000, 1) if frames else 0,
            "fps": fps,
        }

    phases = _detect_phases(frames)
    buffers = {tid: NofMBuffer(n=N_OF_M_CONFIG[tid][0], m=N_OF_M_CONFIG[tid][1]) for tid in N_OF_M_CONFIG}
    hysteresis_map = {tid: Hysteresis(cfg[0], cfg[1]) for tid, cfg in HYSTERESIS_CONFIG.items()}

    all_triggered = []
    for frame, phase in zip(frames, phases):
        all_triggered.append(_evaluate_frame(frame, phase, buffers, hysteresis_map))

    corrections = aggregate_corrections(all_triggered, frames, TRIGGERS, TRIGGER_EVIDENCE)
    corrections = enrich_corrections_with_confidence(corrections, len(frames), TRIGGER_EVIDENCE, presence_conf)
    correction_timeline = [{"id": c["id"], "severity": c["severity"], "segments": c["segments"]} for c in corrections]

    phase_counts = {}
    for p in phases: phase_counts[p] = phase_counts.get(p, 0) + 1

    score = compute_score(corrections, len(frames))
    return {
        "exercise": _TERM_FR, "exercise_id": "echappe",
        "movement_detected": True, "presence_confidence": round(presence_conf, 3),
        "score": score, "grade": _grade_from_score(score),
        "corrections": corrections, "correction_timeline": correction_timeline,
        "phases_summary": phase_counts, "total_frames_analysed": len(frames),
        "duration_seconds": round(frames[-1]["timestamp_ms"] / 1000, 1) if frames else 0,
        "fps": fps,
    }
