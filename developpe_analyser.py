"""
developpe_analyser.py
=====================
Dedicated phase-detection + correction engine for Développé.
Reads ballet_database.json for thresholds and bilingual cues.

Entry point: analyse_developpe(pose_data)
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

_TERM = next((t for t in _DB.get("terms", []) if t.get("id") == "developpe"), {})
_TERM_FR = _TERM.get("term_fr", "Développé")
_CV = _TERM.get("correct_values", {})
_DB_TRIGGERS = {t["id"]: t for t in _TERM.get("correction_triggers", []) if "id" in t}

_MAX_KNEE_BEND  = _CV.get("supporting_knee", {}).get("max_bend_degrees", 7)
_MIN_KNEE       = 180 - _MAX_KNEE_BEND
_MAX_HIP_DIFF   = _CV.get("hip_stability", {}).get("max_height_diff_pct", 3)
_MAX_ARAB_ANGLE = _CV.get("extension_height", {}).get("max_arabesque_angle_at_peak", 140)
_MAX_LEAN       = _CV.get("trunk_stability", {}).get("max_lean_degrees", 7)
_MAX_SHOULDER   = _CV.get("shoulder_level", {}).get("max_height_diff_pct", 3)


def _detect_phases(frames):
    n = len(frames)
    if n == 0: return []
    if n < 6: return ["execution"] * n
    meas = [f.get("measurements", {}) for f in frames]
    signal = [m.get("arabesque_angle", 180) for m in meas]
    return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=False)


def _presence_confidence(frames):
    if not frames: return 0.0
    meas = [f.get("measurements", {}) for f in frames]
    angles = [m.get("arabesque_angle") for m in meas if m.get("arabesque_angle") is not None]
    if not angles: return 0.0
    baseline = sorted(angles)[int(len(angles) * 0.7)]
    peak = min(angles)
    drop = baseline - peak
    if drop < 15: return 0.0
    active = sum(1 for a in angles if a < baseline - drop * 0.4)
    return min(1.0, active / max(1, len(angles)))


def _visibility_ok(frame, trigger_id):
    vis = frame.get("visibility", {})
    if trigger_id == "dev_001":
        return (gate(vis, "left_hip", "left_knee", "left_ankle") or
                gate(vis, "right_hip", "right_knee", "right_ankle"))
    if trigger_id == "dev_002":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id == "dev_003":
        return gate(vis, "left_hip", "right_hip", "right_ankle")
    if trigger_id == "dev_004":
        return gate(vis, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if trigger_id == "dev_005":
        return gate(vis, "left_shoulder", "right_shoulder")
    return True


TRIGGERS = [
    {
        "id": "dev_001",
        "severity": _DB_TRIGGERS.get("dev_001", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and (
                m.get("left_knee_angle", 180) < _MIN_KNEE or
                m.get("right_knee_angle", 180) < _MIN_KNEE
            )
        ),
        "cue_pt": _DB_TRIGGERS.get("dev_001", {}).get("cue_pt",
            "Joelho de apoio dobrado no développé. Estica completamente."),
        "cue_en": _DB_TRIGGERS.get("dev_001", {}).get("cue_en",
            "Supporting knee bending in développé. Fully extend."),
    },
    {
        "id": "dev_002",
        "severity": _DB_TRIGGERS.get("dev_002", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("hip_height_diff_pct", 0) > _MAX_HIP_DIFF
        ),
        "cue_pt": _DB_TRIGGERS.get("dev_002", {}).get("cue_pt",
            "Quadril a subir durante o desenvolvimento. Estabiliza o quadril."),
        "cue_en": _DB_TRIGGERS.get("dev_002", {}).get("cue_en",
            "Hip hiking during development. Stabilize the hip."),
    },
    {
        "id": "dev_003",
        "severity": _DB_TRIGGERS.get("dev_003", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase == "peak" and
            m.get("arabesque_angle", 180) > _MAX_ARAB_ANGLE
        ),
        "cue_pt": _DB_TRIGGERS.get("dev_003", {}).get("cue_pt",
            "Extensão insuficiente no développé. Desenvolve a perna até à altura máxima."),
        "cue_en": _DB_TRIGGERS.get("dev_003", {}).get("cue_en",
            "Insufficient extension in développé. Develop the leg to maximum height."),
    },
    {
        "id": "dev_004",
        "severity": _DB_TRIGGERS.get("dev_004", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("trunk_lean_angle", 0) > _MAX_LEAN
        ),
        "cue_pt": _DB_TRIGGERS.get("dev_004", {}).get("cue_pt",
            "Tronco a compensar no développé. Mantém o eixo ereto."),
        "cue_en": _DB_TRIGGERS.get("dev_004", {}).get("cue_en",
            "Trunk compensating in développé. Keep the axis upright."),
    },
    {
        "id": "dev_005",
        "severity": _DB_TRIGGERS.get("dev_005", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("shoulder_height_diff_pct", 0) > _MAX_SHOULDER
        ),
        "cue_pt": _DB_TRIGGERS.get("dev_005", {}).get("cue_pt",
            "Ombro a subir no développé. Relaxa os ombros para baixo."),
        "cue_en": _DB_TRIGGERS.get("dev_005", {}).get("cue_en",
            "Shoulder lifting in développé. Relax shoulders down."),
    },
]

N_OF_M_CONFIG = {t["id"]: (4, 7) for t in TRIGGERS}

HYSTERESIS_CONFIG = {
    "dev_001": (7.0, 4.0),
    "dev_002": (_MAX_HIP_DIFF, max(1.0, _MAX_HIP_DIFF - 1.0)),
    "dev_004": (_MAX_LEAN, max(3.0, _MAX_LEAN - 2.0)),
    "dev_005": (_MAX_SHOULDER, max(1.5, _MAX_SHOULDER - 1.0)),
}

TRIGGER_EVIDENCE = {
    "dev_001": ("knee_angle", _MIN_KNEE),
    "dev_002": ("hip_height_diff_pct", _MAX_HIP_DIFF),
    "dev_003": ("arabesque_angle", _MAX_ARAB_ANGLE),
    "dev_004": ("trunk_lean_angle", _MAX_LEAN),
    "dev_005": ("shoulder_height_diff_pct", _MAX_SHOULDER),
}


def _trigger_signal(trigger_id, m, phase):
    if trigger_id == "dev_001" and phase in ("rising", "peak"):
        return max(180 - m.get("left_knee_angle", 180), 180 - m.get("right_knee_angle", 180))
    if trigger_id == "dev_002" and phase in ("rising", "peak"):
        return m.get("hip_height_diff_pct", 0)
    if trigger_id == "dev_004" and phase in ("rising", "peak"):
        return m.get("trunk_lean_angle", 0)
    if trigger_id == "dev_005" and phase in ("rising", "peak"):
        return m.get("shoulder_height_diff_pct", 0)
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


def analyse_developpe(pose_data):
    frames = pose_data.get("frames", [])
    fps = pose_data.get("fps", 0)

    if not frames:
        return {"error": "No pose data extracted from video."}

    frames = smooth_frames(frames, alpha=0.15)
    presence_conf = _presence_confidence(frames)
    movement_detected = presence_conf >= 0.40

    if not movement_detected:
        return {
            "exercise": _TERM_FR, "exercise_id": "developpe",
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
        "exercise": _TERM_FR, "exercise_id": "developpe",
        "movement_detected": True, "presence_confidence": round(presence_conf, 3),
        "score": score, "grade": _grade_from_score(score),
        "corrections": corrections, "correction_timeline": correction_timeline,
        "phases_summary": phase_counts, "total_frames_analysed": len(frames),
        "duration_seconds": round(frames[-1]["timestamp_ms"] / 1000, 1) if frames else 0,
        "fps": fps,
    }
