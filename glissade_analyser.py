"""
glissade_analyser.py
====================
Dedicated phase-detection + correction engine for Glissade.
Reads ballet_database.json for thresholds and bilingual cues.
Produces a structured report identical in shape to analyse_plie().

Entry point: analyse_glissade(pose_data)
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

_TERM = next((t for t in _DB.get("terms", []) if t.get("id") == "glissade"), {})
_TERM_FR = _TERM.get("term_fr", "Glissade")
_CV = _TERM.get("correct_values", {})
_DB_TRIGGERS = {t["id"]: t for t in _TERM.get("correction_triggers", []) if "id" in t}

_MAX_LANDING_KNEE = _CV.get("landing_plie", {}).get("max_knee_angle_at_landing", 165)
_MIN_EXT_NORM     = _CV.get("leg_extension", {}).get("min_offset_norm", 0.20)
_MAX_HIP_DIFF     = _CV.get("hip_level", {}).get("max_height_diff_pct", 4)
_MAX_LEAN         = _CV.get("trunk_stability", {}).get("max_lean_degrees", 8)


def _detect_phases(frames):
    """
    Phase signal: min(knee_angle) — falls as dancer dips into demi-plié.
    rising=False: valley = peak phase (deepest demi-plié).
    hip_height_diff_pct was incorrect — it is the glissade_003 correction metric,
    not a movement signal.
    """
    n = len(frames)
    if n == 0: return []
    if n < 6: return ["execution"] * n
    meas = [f.get("measurements", {}) for f in frames]
    signal = [min(m.get("left_knee_angle", 180), m.get("right_knee_angle", 180)) for m in meas]
    return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=False)


def _presence_confidence(frames):
    if not frames: return 0.0
    meas = [f.get("measurements", {}) for f in frames]
    ml = [m.get("left_knee_angle", 180) for m in meas]
    mr = [m.get("right_knee_angle", 180) for m in meas]
    min_knee = [min(lv, rv) for lv, rv in zip(ml, mr)]
    active = sum(1 for v in min_knee if v < 165)
    hip_vals = [m.get("hip_height_diff_pct", 0) for m in meas]
    has_transfer = max(hip_vals) > 2.0 if hip_vals else False
    if not has_transfer: return 0.0
    return min(1.0, active / max(1, len(min_knee)))


def _visibility_ok(frame, trigger_id):
    vis = frame.get("visibility", {})
    if trigger_id == "glissade_001":
        return (gate(vis, "left_hip", "left_knee", "left_ankle") or
                gate(vis, "right_hip", "right_knee", "right_ankle"))
    if trigger_id == "glissade_002":
        return (gate(vis, "left_knee", "left_toe") or gate(vis, "right_knee", "right_toe"))
    if trigger_id == "glissade_003":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id == "glissade_004":
        return gate(vis, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    return True


TRIGGERS = [
    {
        "id": "glissade_001",
        "severity": _DB_TRIGGERS.get("glissade_001", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("descent", "finish") and
            min(m.get("left_knee_angle", 180), m.get("right_knee_angle", 180)) > _MAX_LANDING_KNEE
        ),
        "cue_pt": _DB_TRIGGERS.get("glissade_001", {}).get("cue_pt",
            "Aterragem sem demi-plié. Amortece a chegada com o joelho."),
        "cue_en": _DB_TRIGGERS.get("glissade_001", {}).get("cue_en",
            "Landing without demi-plié. Absorb the landing with the knee."),
    },
    {
        "id": "glissade_002",
        "severity": _DB_TRIGGERS.get("glissade_002", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase == "peak" and
            max(m.get("left_knee_toe_offset_norm", 0), m.get("right_knee_toe_offset_norm", 0)) < _MIN_EXT_NORM
        ),
        "cue_pt": _DB_TRIGGERS.get("glissade_002", {}).get("cue_pt",
            "Perna de trabalho não se estende durante a glissade. Desliza com a perna completamente estendida."),
        "cue_en": _DB_TRIGGERS.get("glissade_002", {}).get("cue_en",
            "Working leg not extending during glissade. Slide with the leg fully extended."),
    },
    {
        "id": "glissade_003",
        "severity": _DB_TRIGGERS.get("glissade_003", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("hip_height_diff_pct", 0) > _MAX_HIP_DIFF
        ),
        "cue_pt": _DB_TRIGGERS.get("glissade_003", {}).get("cue_pt",
            "Quadril desnivelado na glissade. Mantém os quadris a nível."),
        "cue_en": _DB_TRIGGERS.get("glissade_003", {}).get("cue_en",
            "Hip not level in glissade. Keep hips level."),
    },
    {
        "id": "glissade_004",
        "severity": _DB_TRIGGERS.get("glissade_004", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            m.get("trunk_lean_angle", 0) > _MAX_LEAN
        ),
        "cue_pt": _DB_TRIGGERS.get("glissade_004", {}).get("cue_pt",
            "Tronco a inclinar na glissade. Mantém o eixo central durante o deslize."),
        "cue_en": _DB_TRIGGERS.get("glissade_004", {}).get("cue_en",
            "Trunk leaning in glissade. Maintain the central axis during the slide."),
    },
]

N_OF_M_CONFIG = {t["id"]: (4, 7) for t in TRIGGERS}

HYSTERESIS_CONFIG = {
    "glissade_003": (_MAX_HIP_DIFF, max(2.0, _MAX_HIP_DIFF - 1.5)),
    "glissade_004": (_MAX_LEAN, max(5.0, _MAX_LEAN - 3.0)),
}

TRIGGER_EVIDENCE = {
    "glissade_001": ("knee_angle", _MAX_LANDING_KNEE),
    "glissade_002": ("knee_toe_offset_norm", _MIN_EXT_NORM),
    "glissade_003": ("hip_height_diff_pct", _MAX_HIP_DIFF),
    "glissade_004": ("trunk_lean_angle", _MAX_LEAN),
}


def _trigger_signal(trigger_id, m, phase):
    if trigger_id == "glissade_003" and phase in ("rising", "peak"):
        return m.get("hip_height_diff_pct", 0)
    if trigger_id == "glissade_004":
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


def analyse_glissade(pose_data):
    frames = pose_data.get("frames", [])
    fps = pose_data.get("fps", 0)

    if not frames:
        return {"error": "No pose data extracted from video."}

    frames = smooth_frames(frames, alpha=0.15)
    presence_conf = _presence_confidence(frames)
    movement_detected = presence_conf >= 0.35

    if not movement_detected:
        return {
            "exercise": _TERM_FR, "exercise_id": "glissade",
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
    for p in phases:
        phase_counts[p] = phase_counts.get(p, 0) + 1

    score = compute_score(corrections, len(frames))
    return {
        "exercise": _TERM_FR, "exercise_id": "glissade",
        "movement_detected": True, "presence_confidence": round(presence_conf, 3),
        "score": score, "grade": _grade_from_score(score),
        "corrections": corrections, "correction_timeline": correction_timeline,
        "phases_summary": phase_counts, "total_frames_analysed": len(frames),
        "duration_seconds": round(frames[-1]["timestamp_ms"] / 1000, 1) if frames else 0,
        "fps": fps,
    }
