"""
tendu_analyser.py
=================
Dedicated phase-detection + correction engine for Battement Tendu.
Reads ballet_database.json for thresholds and bilingual cues.
Produces a structured report identical in shape to analyse_plie().

Entry point: analyse_tendu(pose_data)
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


# ── DATABASE ───────────────────────────────────────────────────────────

_DB_PATH = Path(__file__).parent / "ballet_database.json"

with open(_DB_PATH, "r", encoding="utf-8") as _f:
    _DB = json.load(_f)

_TERM = next((t for t in _DB.get("terms", []) if t.get("id") == "battement_tendu"), {})
_TERM_FR = _TERM.get("term_fr", "Battement Tendu")
_CV = _TERM.get("correct_values", {})
_DB_TRIGGERS = {t["id"]: t for t in _TERM.get("correction_triggers", []) if "id" in t}

# Pull thresholds from DB correct_values (with sensible fallbacks)
_MAX_KNEE_BEND   = _CV.get("supporting_knee", {}).get("max_bend_degrees", 5)
_MIN_KNEE        = 180 - _MAX_KNEE_BEND                            # 175°
_MAX_HIP_DIFF    = _CV.get("hip_stability", {}).get("max_height_diff_pct", 2)
_MAX_LEAN        = _CV.get("trunk_stability", {}).get("max_lean_degrees", 5)


# ── PHASE DETECTION ────────────────────────────────────────────────────

def _detect_phases(frames):
    """
    Phase signal: max(left_knee_toe_offset_norm, right_knee_toe_offset_norm)
    Rising signal — peaks at full extension.
    Phases: preparation | rising | peak | descent | finish
    """
    n = len(frames)
    if n == 0:
        return []
    if n < 6:
        return ["execution"] * n

    meas = [f.get("measurements", {}) for f in frames]
    signal = [
        max(m.get("left_knee_toe_offset_norm", 0), m.get("right_knee_toe_offset_norm", 0))
        for m in meas
    ]
    return _signal_phases(signal, n, flat_thresh_pct=0.12, bottom_thresh_pct=0.22, rising=True)


# ── PRESENCE CONFIDENCE ────────────────────────────────────────────────

def _presence_confidence(frames):
    """
    Detect actual extension movement: offset must peak then return
    (variation), not just be statically high.
    """
    if not frames:
        return 0.0
    meas = [f.get("measurements", {}) for f in frames]
    offs = [
        max(m.get("left_knee_toe_offset_norm", 0), m.get("right_knee_toe_offset_norm", 0))
        for m in meas
    ]
    if not offs:
        return 0.0
    baseline = sorted(offs)[int(len(offs) * 0.3)]
    peak = max(offs)
    extension = peak - baseline
    if extension < 0.10:
        return 0.0
    active = sum(1 for v in offs if v > baseline + extension * 0.35)
    return min(1.0, 4.0 * active / max(1, len(offs)))


# ── VISIBILITY GATING ──────────────────────────────────────────────────

def _visibility_ok(frame, trigger_id):
    vis = frame.get("visibility", {})
    if trigger_id == "tendu_001":
        return (gate(vis, "left_hip", "left_knee", "left_ankle") or
                gate(vis, "right_hip", "right_knee", "right_ankle"))
    if trigger_id == "tendu_002":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id == "tendu_003":
        return (gate(vis, "left_knee", "left_toe") or
                gate(vis, "right_knee", "right_toe"))
    if trigger_id == "tendu_004":
        return gate(vis, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if trigger_id == "tendu_005":
        return gate(vis, "left_hip", "right_hip")
    if trigger_id == "tendu_006":
        return gate(vis, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if trigger_id == "tendu_007":
        return (gate(vis, "left_elbow", "left_wrist") or gate(vis, "right_elbow", "right_wrist"))
    return True


# ── CORRECTION TRIGGERS ────────────────────────────────────────────────

TRIGGERS = [
    {
        "id": "tendu_001",
        "severity": _DB_TRIGGERS.get("tendu_001", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak", "descent") and (
                m.get("left_knee_angle", 180) < _MIN_KNEE or
                m.get("right_knee_angle", 180) < _MIN_KNEE
            )
        ),
        "cue_pt": _DB_TRIGGERS.get("tendu_001", {}).get("cue_pt",
            "Joelho de apoio dobrado. Estique completamente o joelho de suporte."),
        "cue_en": _DB_TRIGGERS.get("tendu_001", {}).get("cue_en",
            "Supporting knee bending. Fully straighten the supporting knee."),
    },
    {
        "id": "tendu_002",
        "severity": _DB_TRIGGERS.get("tendu_002", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("hip_height_diff_pct", 0) > _MAX_HIP_DIFF
        ),
        "cue_pt": _DB_TRIGGERS.get("tendu_002", {}).get("cue_pt",
            "Quadril subindo na direção da perna de trabalho. Mantenha ambos os quadris nivelados."),
        "cue_en": _DB_TRIGGERS.get("tendu_002", {}).get("cue_en",
            "Hip hiking toward working leg. Keep both hips level."),
    },
    {
        "id": "tendu_003",
        "severity": _DB_TRIGGERS.get("tendu_003", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase == "peak" and
            abs(m.get("left_knee_toe_offset_norm", 0) - m.get("right_knee_toe_offset_norm", 0)) < 0.03
        ),
        "cue_pt": _DB_TRIGGERS.get("tendu_003", {}).get("cue_pt",
            "Articulação do pé incorreta. Para fora: calcanhar lidera. Para dentro: dedos lideram."),
        "cue_en": _DB_TRIGGERS.get("tendu_003", {}).get("cue_en",
            "Incorrect foot articulation. Going out: heel leads. Coming in: toes lead."),
    },
    {
        "id": "tendu_004",
        "severity": _DB_TRIGGERS.get("tendu_004", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("trunk_lean_angle", 0) > _MAX_LEAN
        ),
        "cue_pt": _DB_TRIGGERS.get("tendu_004", {}).get("cue_pt",
            "Tronco instável durante o tendu. Mantenha o eixo central fixo."),
        "cue_en": _DB_TRIGGERS.get("tendu_004", {}).get("cue_en",
            "Trunk instability during tendu. Keep the central axis fixed."),
    },
    {
        "id": "tendu_005",
        "severity": _DB_TRIGGERS.get("tendu_005", {}).get("severity", "major"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("hip_rotation_deg", 0) > 12
        ),
        "cue_pt": _DB_TRIGGERS.get("tendu_005", {}).get("cue_pt",
            "Quadril a rodar no tendu. Mantém os quadris quadrados à frente."),
        "cue_en": _DB_TRIGGERS.get("tendu_005", {}).get("cue_en",
            "Hip rotating during tendu. Keep hips square to the front."),
    },
    {
        "id": "tendu_006",
        "severity": _DB_TRIGGERS.get("tendu_006", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            phase in ("rising", "peak") and
            m.get("anterior_tilt_deg", 0) > 8
        ),
        "cue_pt": _DB_TRIGGERS.get("tendu_006", {}).get("cue_pt",
            "Peso a transferir para a frente. Mantém o centro sobre a perna de apoio."),
        "cue_en": _DB_TRIGGERS.get("tendu_006", {}).get("cue_en",
            "Weight shifting forward. Keep centre over the supporting leg."),
    },
    {
        "id": "tendu_007",
        "severity": _DB_TRIGGERS.get("tendu_007", {}).get("severity", "warning"),
        "check": lambda m, phase: (
            max(m.get("left_wrist_drop_pct", 0), m.get("right_wrist_drop_pct", 0)) > 10
        ),
        "cue_pt": _DB_TRIGGERS.get("tendu_007", {}).get("cue_pt",
            "Pulso a cair no tendu. Mantém os braços na posição."),
        "cue_en": _DB_TRIGGERS.get("tendu_007", {}).get("cue_en",
            "Wrist dropping in tendu. Maintain arm position."),
    },
]

N_OF_M_CONFIG = {
    "tendu_001": (4, 7),
    "tendu_002": (4, 7),
    "tendu_003": (4, 7),
    "tendu_004": (4, 7),
    "tendu_005": (4, 7), "tendu_006": (4, 7), "tendu_007": (4, 7),
}

HYSTERESIS_CONFIG = {
    "tendu_001": (5.0, 2.0),  # signal: 180 - knee_angle; matches _MAX_KNEE_BEND = 5
    "tendu_002": (_MAX_HIP_DIFF, max(0.5, _MAX_HIP_DIFF - 1.0)),
    # tendu_003: inverted signal — fires when offset diff LOW (poor articulation)
    # signal = 0.06 - abs(diff); on=0.03 (diff<0.03), off=0.01 (diff>0.05)
    "tendu_003": (0.03, 0.01),
    "tendu_004": (_MAX_LEAN, max(1.0, _MAX_LEAN - 1.5)),
    "tendu_005": (12.0, 7.0), "tendu_006": (8.0, 5.0), "tendu_007": (10.0, 6.0),
}

TRIGGER_EVIDENCE = {
    "tendu_001": ("knee_angle", _MIN_KNEE),
    "tendu_002": ("hip_height_diff_pct", _MAX_HIP_DIFF),
    "tendu_003": ("foot_articulation_proxy", 0.03),
    "tendu_004": ("trunk_lean_angle", _MAX_LEAN),
    "tendu_005": ("hip_rotation_deg", 12), "tendu_006": ("anterior_tilt_deg", 8), "tendu_007": ("wrist_drop_pct", 10),
}


# ── SIGNAL FOR HYSTERESIS ──────────────────────────────────────────────

def _trigger_signal(trigger_id, m, phase):
    if trigger_id == "tendu_001" and phase in ("rising", "peak", "descent"):
        return max(
            180 - m.get("left_knee_angle", 180),
            180 - m.get("right_knee_angle", 180),
        )
    if trigger_id == "tendu_002" and phase in ("rising", "peak"):
        return m.get("hip_height_diff_pct", 0)
    if trigger_id == "tendu_003" and phase == "peak":
        # Invert: high signal = poor articulation (diff near zero)
        diff = abs(m.get("left_knee_toe_offset_norm", 0) - m.get("right_knee_toe_offset_norm", 0))
        return max(0.0, 0.06 - diff)
    if trigger_id == "tendu_004" and phase in ("rising", "peak"):
        return m.get("trunk_lean_angle", 0)
    if trigger_id == "tendu_005" and phase in ("rising", "peak"):
        return m.get("hip_rotation_deg", 0)
    if trigger_id == "tendu_006" and phase in ("rising", "peak"):
        return m.get("anterior_tilt_deg", 0)
    if trigger_id == "tendu_007":
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

def analyse_tendu(pose_data):
    """
    Full battement tendu analysis pipeline.

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
    movement_detected = presence_conf >= 0.20

    if not movement_detected:
        return {
            "exercise": _TERM_FR,
            "exercise_id": "battement_tendu",
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
        "exercise_id": "battement_tendu",
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
