"""
analyser_utils.py
=================
Shared utilities imported by all individual ballet exercise analysers.
Provides core data structures (EMA, NofMBuffer, Hysteresis), helper
functions (gate, _ms_to_ts), scoring, segment building, and the generic
phase-detection engine used by every exercise analyser.

Do NOT import exercise-specific logic here — this module must remain
exercise-agnostic.
"""

from collections import deque


# ── SMOOTHING ──────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average filter for noisy scalar time-series."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if new_value is None:
            return self.value
        if self.value is None:
            self.value = float(new_value)
        else:
            self.value = self.alpha * float(new_value) + (1.0 - self.alpha) * self.value
        return self.value


# ── TEMPORAL FILTERS ───────────────────────────────────────────────────

class NofMBuffer:
    """
    N-of-M confirmation buffer.
    Returns True only when at least N of the last M frames were flagged.
    """

    def __init__(self, n, m):
        self.n = n
        self.history = deque(maxlen=m)

    def update(self, triggered):
        self.history.append(1 if triggered else 0)
        return sum(self.history) >= self.n


class Hysteresis:
    """
    Two-threshold hysteresis to avoid rapid on/off toggling.
    Activates when value >= on_thresh; deactivates when value < off_thresh.
    """

    def __init__(self, on_thresh, off_thresh):
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh
        self.active = False

    def update(self, value):
        if value is None:
            self.active = False
            return False
        if not self.active and value >= self.on_thresh:
            self.active = True
        elif self.active and value < self.off_thresh:
            self.active = False
        return self.active


# ── VISIBILITY GATING ──────────────────────────────────────────────────

def gate(visibility, *landmark_names, threshold=0.6):
    """Return True iff every named landmark has confidence >= threshold."""
    return all(visibility.get(name, 0.0) >= threshold for name in landmark_names)


# ── TIMESTAMP HELPERS ──────────────────────────────────────────────────

def _ms_to_ts(ms):
    """Convert milliseconds to MM:SS string."""
    s = int(ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


# ── FRAME SMOOTHING ────────────────────────────────────────────────────

def smooth_frames(frames, alpha=0.3):
    """
    Apply per-key EMA smoothing to noisy scalar measurements.
    Covers all measurement keys used across all exercises.
    """
    if not frames:
        return frames
    keys = [
        "left_knee_angle",
        "right_knee_angle",
        "trunk_lean_angle",
        "hip_height_diff_pct",
        "shoulder_height_diff_pct",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_wrist_drop_pct",
        "right_wrist_drop_pct",
        "arabesque_angle",
        "hip_rotation_deg",
        "left_knee_toe_offset_norm",
        "right_knee_toe_offset_norm",
        "left_heel_rise_pct",
        "right_heel_rise_pct",
        "anterior_tilt_deg",
        "spine_vertical_angle",
        "foot_spread_norm",
        "knee_height_diff_pct",
    ]
    ema = {k: EMA(alpha=alpha) for k in keys}
    out = []
    for frame in frames:
        copied = dict(frame)
        m = dict(frame.get("measurements", {}))
        for k in keys:
            if k in m:
                m[k] = ema[k].update(m[k])
        copied["measurements"] = m
        out.append(copied)
    return out


# ── GENERIC PHASE DETECTION ────────────────────────────────────────────

def _signal_phases(signal, n, flat_thresh_pct=0.15, bottom_thresh_pct=0.20, rising=True):
    """
    Generic signal-based phase detector for cyclic / held movements.

    rising=True  → peaks are the active phase (relevé, tendu, arabesque)
    rising=False → valleys are the active phase (fondu, plié-like)

    Returns a list of phase strings (length n):
      preparation | rising | peak | descent | finish
    """
    if not signal or n == 0:
        return ["preparation"] * n

    sig_min = min(signal)
    sig_max = max(signal)
    sig_range = sig_max - sig_min
    if sig_range < 1e-3:
        return ["preparation"] * n

    norm = [(s - sig_min) / sig_range for s in signal]

    flat_low  = flat_thresh_pct
    flat_high = 1.0 - flat_thresh_pct
    active    = 1.0 - bottom_thresh_pct
    active_lo = bottom_thresh_pct

    deriv = [0.0] * n
    for i in range(1, n - 1):
        deriv[i] = norm[i + 1] - norm[i - 1]
    if n > 1:
        deriv[0]     = norm[1] - norm[0]
        deriv[n - 1] = norm[n - 1] - norm[n - 2]

    phases = []
    for i, v in enumerate(norm):
        is_start = i < n * 0.12
        is_end   = i > n * 0.88
        if rising:
            if v >= active:
                phase = "peak"
            elif v <= flat_low:
                phase = "preparation" if is_start else "finish"
            elif deriv[i] > 0.02:
                phase = "rising"
            elif deriv[i] < -0.02:
                phase = "descent"
            else:
                phase = phases[-1] if phases else "preparation"
        else:
            if v <= active_lo:
                phase = "peak"
            elif v >= flat_high:
                phase = "preparation" if is_start else "finish"
            elif deriv[i] < -0.02:
                phase = "rising"
            elif deriv[i] > 0.02:
                phase = "descent"
            else:
                phase = phases[-1] if phases else "preparation"
        phases.append(phase)

    # Smooth single-frame noise
    for i in range(1, n - 1):
        if phases[i] != phases[i - 1] and phases[i] != phases[i + 1]:
            phases[i] = phases[i - 1]

    return phases


# ── SCORING ────────────────────────────────────────────────────────────

def compute_score(corrections, total_frames):
    """
    Compute 0–100 score. Each correction reduces score proportional to
    its frequency; major corrections carry a higher ceiling penalty.
    """
    if total_frames == 0:
        return 100
    penalty = 0.0
    for c in corrections:
        max_per = 18 if c["severity"] == "major" else 10
        freq_ratio = min(c["frequency"] / max(total_frames, 1), 0.75)
        penalty += max_per * freq_ratio
    return max(0, round(100 - penalty))


def _grade_from_score(score):
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 55:
        return "Needs Work"
    return "Significant Corrections"


# ── CORRECTION SEGMENT BUILDING ────────────────────────────────────────

MIN_SEGMENT_MS = 600        # ignore segments shorter than this
MIN_SEGMENT_FRAME_PCT = 0.04  # correction must appear in ≥4% of frames


def _make_segment(start_idx, end_idx, start_ms, end_ms):
    return {
        "start_frame": start_idx,
        "end_frame": end_idx,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "start_ts": _ms_to_ts(start_ms),
        "end_ts": _ms_to_ts(end_ms),
        "duration_ms": max(0, end_ms - start_ms),
        "frames": (end_idx - start_idx) + 1,
    }


def build_correction_segments(all_triggered, frames, max_gap_frames=5):
    """
    Build timestamp segments per correction ID from frame-level trigger hits.
    Small gaps (<= max_gap_frames) are merged into one continuous segment.
    """
    segments_by_id = {}
    for idx, triggered in enumerate(all_triggered):
        ts = frames[idx]["timestamp_ms"]
        for tid in triggered:
            segments_by_id.setdefault(tid, []).append((idx, ts))

    compact = {}
    for tid, points in segments_by_id.items():
        points.sort(key=lambda x: x[0])
        merged = []
        start_idx, start_ms = points[0]
        end_idx, end_ms = points[0]

        for idx, ts in points[1:]:
            if idx - end_idx <= max_gap_frames + 1:
                end_idx, end_ms = idx, ts
                continue
            merged.append(_make_segment(start_idx, end_idx, start_ms, end_ms))
            start_idx, start_ms = idx, ts
            end_idx, end_ms = idx, ts

        merged.append(_make_segment(start_idx, end_idx, start_ms, end_ms))
        compact[tid] = merged

    return compact


def filter_corrections_by_persistence(corrections, total_frames):
    """Drop corrections that are too brief or too rare to be reliable."""
    if total_frames <= 0:
        return corrections

    filtered = []
    for c in corrections:
        segs = c.get("segments", [])
        valid_segs = [s for s in segs if s.get("duration_ms", 0) >= MIN_SEGMENT_MS]
        if not valid_segs:
            continue
        freq = c.get("frequency", 0)
        if freq / max(total_frames, 1) < MIN_SEGMENT_FRAME_PCT:
            continue
        c2 = dict(c)
        c2["segments"] = valid_segs
        filtered.append(c2)
    return filtered


def aggregate_corrections(all_triggered, frames, triggers, trigger_evidence=None):
    """
    Group trigger hits across all frames.

    Args:
        all_triggered: list of lists of trigger IDs (one list per frame)
        frames:        raw frame list (for timestamps)
        triggers:      list of trigger dicts with id/severity/cue_pt/cue_en keys
        trigger_evidence: optional dict {tid: (metric_name, threshold)}

    Returns:
        list of correction dicts sorted by severity + frequency.
    """
    counts = {}
    first_seen = {}
    segments_by_id = build_correction_segments(all_triggered, frames)

    # Drop segments too short to count
    for tid in list(segments_by_id.keys()):
        segments_by_id[tid] = [
            s for s in segments_by_id[tid]
            if s["duration_ms"] >= MIN_SEGMENT_MS
        ]
        if not segments_by_id[tid]:
            del segments_by_id[tid]

    for i, triggered in enumerate(all_triggered):
        ts = frames[i]["timestamp_ms"]
        for tid in triggered:
            counts[tid] = counts.get(tid, 0) + 1
            if tid not in first_seen:
                first_seen[tid] = ts

    total_frames = max(len(frames), 1)
    corrections = []
    for t in triggers:
        tid = t["id"]
        if tid not in counts:
            continue
        if counts[tid] / total_frames < MIN_SEGMENT_FRAME_PCT:
            continue
        if tid not in segments_by_id:
            continue
        corrections.append({
            "id": tid,
            "severity": t["severity"],
            "cue_pt": t["cue_pt"],
            "cue_en": t["cue_en"],
            "frequency": counts[tid],
            "first_seen_ms": first_seen.get(tid, 0),
            "first_seen_ts": _ms_to_ts(first_seen.get(tid, 0)),
            "segments": segments_by_id.get(tid, []),
        })

    corrections.sort(key=lambda c: (0 if c["severity"] == "major" else 1, -c["frequency"]))
    return corrections


def enrich_corrections_with_confidence(corrections, total_frames, trigger_evidence=None,
                                       presence_confidence=1.0):
    """
    Attach confidence score and evidence metadata to each correction.

    trigger_evidence: dict {tid: (metric_name, threshold)} — optional.
    presence_confidence: float 0..1 from exercise presence detection.
    """
    enriched = []
    for c in corrections:
        segs = c.get("segments", [])
        longest_frames = max((s.get("frames", 0) for s in segs), default=0)
        freq_ratio = c.get("frequency", 0) / max(1, total_frames)
        seg_ratio = longest_frames / max(1, total_frames)
        severity_bonus = 0.12 if c.get("severity") == "major" else 0.04
        confidence = min(
            0.99,
            max(0.0, 0.4 * freq_ratio + 0.3 * seg_ratio + 0.2 * presence_confidence + 0.1 + severity_bonus)
        )
        level = "high" if confidence >= 0.67 else ("medium" if confidence >= 0.4 else "low")

        tid = c.get("id", "")
        if trigger_evidence and tid in trigger_evidence:
            metric_name, threshold = trigger_evidence[tid]
        else:
            metric_name, threshold = "unknown_metric", "db_rule"

        c2 = dict(c)
        c2["confidence"] = round(confidence, 3)
        c2["confidence_level"] = level
        c2["evidence"] = {
            "metric_name": metric_name,
            "threshold": threshold,
            "frequency_ratio": round(freq_ratio, 3),
            "longest_segment_frames": longest_frames,
            "presence_confidence": round(presence_confidence, 3),
        }
        enriched.append(c2)
    return enriched
