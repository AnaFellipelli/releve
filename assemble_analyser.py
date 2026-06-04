"""
assemble_analyser.py
====================
Dedicated phase-detection + correction engine for Assemblé.
Entry point: analyse_assemble(pose_data)
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

_TERM = next((t for t in _DB.get("terms", []) if t.get("id") == "assemble"), {})
_TERM_FR = _TERM.get("term_fr", "Assemblé")
_CV = _TERM.get("correct_values", {})
_DB_TRIGGERS = {t["id"]: t for t in _TERM.get("correction_triggers", []) if "id" in t}

_MAX_LAND_KNEE    = _CV.get("landing_plie", {}).get("max_knee_angle_at_landing", 162)
_MAX_SPREAD_LAND  = _CV.get("feet_assembly", {}).get("max_spread_norm_at_landing", 0.12)
_MAX_HIP_DIFF     = _CV.get("hip_stability", {}).get("max_height_diff_pct", 3)
_MAX_LEAN         = _CV.get("trunk_stability", {}).get("max_lean_degrees", 8)


def _detect_phases(frames):
    n = len(frames)
    if n == 0: return []
    if n < 6: return ["execution"] * n
    meas = [f.get("measurements", {}) for f in frames]
    # Phase signal: min knee angle (falling — deepest point is the landing plié)
    signal = [
        min(m.get("left_knee_angle", 180), m.get("right_knee_angle", 180))
        for m in meas
    ]
    return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=False)


def _presence_confidence(frames):
    if not frames: return 0.0
    meas = [f.get("measurements", {}) for f in frames]
    min_knee = [min(m.get("left_knee_angle", 180), m.get("right_knee_angle", 180)) for m in meas]
    spread = [m.get("foot_spread_norm", 0) for m in meas]
    has_bend = sum(1 for v in min_knee if v < 165) >= 2
    has_spread_change = (max(spread) - min(spread)) > 0.05 if spread else False
    if not has_bend: return 0.0
    active = sum(1 for v in min_knee if v < 165)
    score = active / max(1, len(min_knee))
    if has_spread_change:
        score = min(1.0, score * 1.5)
    return min(1.0, score)


def _visibility_ok(frame, trigger_id):
    vis = frame.get("visibility", {})
    if trigger_id == "assemble_001":
        return (gate(vis, "left_hip", "left_knee", "left_ankle") or
                gate(vis, "right_hip", "right_knee", "right_ankle"))
    if trigger_id == "assemble_002":
        return (gate(vis, "left_toe") and gate(vis, "right_toe"))
    if trigger_id == "assemble_003":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id == "assemble_004":
        return gate(vis, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    return True


TRIGGERS = [
    {
        "id": "assemble_001",
        "severity": _DB_TRIGGERS.get("assemble_001", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("peak", "descent") and
            min(m.get("left_knee_angle", 180), m.get("right_knee_angle", 180)) > _MAX_LAND_KNEE
        ),
        "cue_pt": _DB_TRIGGERS.get("assemble_001", {}).get("cue_pt",
            "Aterragem sem demi-plié. Amortece a chegada com os joelhos."),
        "cue_en": _DB_TRIGGERS.get("assemble_001", {}).get("cue_en",
            "Landing without demi-plié. Absorb the landing with the knees."),
    },
    {
        "id": "assemble_002",
        "severity": _DB_TRIGGERS.get("assemble_002", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("peak", "descent") and
            m.get("foot_spread_norm", 0) > _MAX_SPREAD_LAND
        ),
        "cue_pt": _DB_TRIGGERS.get("assemble_002", {}).get("cue_pt",
            "Pés não se juntam corretamente. Fecha os pés na 5ª posição no ar."),
        "cue_en": _DB_TRIGGERS.get("assemble_002", {}).get("cue_en",
            "Feet not assembling tightly. Close feet to 5th position in the air."),
    },
    {
        "id": "assemble_003",
        "severity": _DB_TRIGGERS.get("assemble_003", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            m.get("hip_height_diff_pct", 0) > _MAX_HIP_DIFF
        ),
        "cue_pt": _DB_TRIGGERS.get("assemble_003", {}).get("cue_pt",
            "Quadril desnivelado no assemblé. Mantém os quadris estabilizados."),
        "cue_en": _DB_TRIGGERS.get("assemble_003", {}).get("cue_en",
            "Hip not level in assemblé. Keep hips stabilized."),
    },
    {
        "id": "assemble_004",
        "severity": _DB_TRIGGERS.get("assemble_004", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            m.get("trunk_lean_angle", 0) > _MAX_LEAN
        ),
        "cue_pt": _DB_TRIGGERS.get("assemble_004", {}).get("cue_pt",
            "Tronco a inclinar no assemblé. Mantém o eixo central."),
        "cue_en": _DB_TRIGGERS.get("assemble_004", {}).get("cue_en",
            "Trunk leaning in assemblé. Maintain the central axis."),
    },
]

N_OF_M_CONFIG = {t["id"]: (4, 7) for t in TRIGGERS}

HYSTERESIS_CONFIG = {
    "assemble_002": (_MAX_SPREAD_LAND, max(0.06, _MAX_SPREAD_LAND - 0.04)),
    "assemble_003": (_MAX_HIP_DIFF, max(1.5, _MAX_HIP_DIFF - 1.0)),
    "assemble_004": (_MAX_LEAN, max(4.0, _MAX_LEAN - 3.0)),
}

TRIGGER_EVIDENCE = {
    "assemble_001": ("knee_angle", _MAX_LAND_KNEE),
    "assemble_002": ("foot_spread_norm", _MAX_SPREAD_LAND),
    "assemble_003": ("hip_height_diff_pct", _MAX_HIP_DIFF),
    "assemble_004": ("trunk_lean_angle", _MAX_LEAN),
}


def _trigger_signal(trigger_id, m, phase):
    if trigger_id == "assemble_002" and phase in ("peak", "descent"):
        return m.get("foot_spread_norm", 0)
    if trigger_id == "assemble_003":
        return m.get("hip_height_diff_pct", 0)
    if trigger_id == "assemble_004":
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


def analyse_assemble(pose_data):
    frames = pose_data.get("frames", [])
    fps = pose_data.get("fps", 0)
    if not frames:
        return {"error": "No pose data extracted from video."}

    frames = smooth_frames(frames, alpha=0.15)
    presence_conf = _presence_confidence(frames)
    movement_detected = presence_conf >= 0.35

    if not movement_detected:
        return {
            "exercise": _TERM_FR, "exercise_id": "assemble",
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
        "exercise": _TERM_FR, "exercise_id": "assemble",
        "movement_detected": True, "presence_confidence": round(presence_conf, 3),
        "score": score, "grade": _grade_from_score(score),
        "corrections": corrections, "correction_timeline": correction_timeline,
        "phases_summary": phase_counts, "total_frames_analysed": len(frames),
        "duration_seconds": round(frames[-1]["timestamp_ms"] / 1000, 1) if frames else 0,
        "fps": fps,
    }
