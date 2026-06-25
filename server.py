"""
server.py
=========
Flask server for (r)elevē — ballet correction analysis.

Routes:
  GET  /          → Main app UI
  POST /analyse   → Video upload → pose extraction → analysis → JSON report
  GET  /api/sessions → List previous sessions
  GET  /health    → Health check

Run:
  pip install flask mediapipe opencv-python numpy boto3 python-dotenv supabase
  python server.py
  open http://localhost:5000

Storage: Cloudflare R2 (S3-compatible) via boto3
Database: Supabase (PostgreSQL)
"""

import os
import json
import traceback
import boto3
from supabase import create_client, Client
from botocore.config import Config
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

from pose_extractor import extract_frames
from plie_analyser import analyse_plie
from tendu_analyser import analyse_tendu
from releve_analyser import analyse_releve
from retire_analyser import analyse_retire
from fondu_analyser import analyse_fondu
from arabesque_analyser import analyse_arabesque
from port_de_bras_analyser import analyse_port_de_bras
from grand_battement_analyser import analyse_grand_battement
from glissade_analyser import analyse_glissade
from degage_analyser import analyse_degage
from rond_de_jambe_analyser import analyse_rond_de_jambe
from developpe_analyser import analyse_developpe
from echappe_analyser import analyse_echappe
from assemble_analyser import analyse_assemble
from frappe_analyser import analyse_frappe
from attitude_analyser import analyse_attitude

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

ALLOWED   = {"mp4", "mov", "avi", "webm", "mkv"}
EXERCISES = {"plie", "battement_tendu", "arabesque", "port_de_bras",
             "grand_battement", "releve", "retire_passe", "fondu",
             "glissade", "battement_degage", "rond_de_jambe", "developpe",
             "echappe", "assemble", "frappe", "attitude"}
BASE_DIR   = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploaded_videos"
UPLOADS_DIR.mkdir(exist_ok=True)
LOCAL_SESSIONS_FILE = BASE_DIR / "analysis_sessions.json"

# ── Supabase ──────────────────────────────────────────────────────────
_supabase_url = os.environ.get("SUPABASE_URL", "")
_supabase_key = os.environ.get("SUPABASE_SERVICE_KEY", "")
_supabase = (
    create_client(_supabase_url, _supabase_key)
    if _supabase_url and _supabase_key
    else None
)

# ── Cloudflare R2 ─────────────────────────────────────────────────────
_r2_account_id = os.environ.get("R2_ACCOUNT_ID", "")
_r2 = (
    boto3.client(
        "s3",
        endpoint_url=f'https://{_r2_account_id}.r2.cloudflarestorage.com',
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    if _r2_account_id
    else None
)
R2_BUCKET     = os.environ.get("R2_BUCKET_NAME", "")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")


# ── Helpers ───────────────────────────────────────────────────────────

def _grade_from_score(score):
    if score >= 90: return "Excellent"
    if score >= 75: return "Good"
    if score >= 55: return "Needs Work"
    return "Significant Corrections"


def _local_video_url(stored_filename: str) -> str:
    return f"http://localhost:5000/videos/{stored_filename}"


def _upload_video(local_path: Path, stored_filename: str) -> str:
    if not _r2:
        return _local_video_url(stored_filename)
    try:
        ext = stored_filename.rsplit(".", 1)[-1].lower()
        content_type = "video/x-matroska" if ext == "mkv" else f"video/{ext}"
        with open(local_path, "rb") as f:
            _r2.upload_fileobj(
                f, R2_BUCKET, stored_filename,
                ExtraArgs={"ContentType": content_type},
            )
        return f"{R2_PUBLIC_URL}/{stored_filename}"
    except Exception as e:
        print(f"[server] R2 upload failed ({e}), serving video locally")
        return _local_video_url(stored_filename)


def _save_session_local(session: dict):
    try:
        data = json.loads(LOCAL_SESSIONS_FILE.read_text()) if LOCAL_SESSIONS_FILE.exists() else []
    except Exception:
        data = []
    data = [s for s in data if s.get("id") != session["id"]]
    data.insert(0, session)
    LOCAL_SESSIONS_FILE.write_text(json.dumps(data, indent=2, default=str))


def _save_session(session: dict):
    if not _supabase:
        _save_session_local(session)
        return
    try:
        _supabase.table("sessions").upsert({
            "id":                session["id"],
            "created_at":        session["created_at"],
            "video_filename":    session["video_filename"],
            "video_url":         session["video_url"],
            "exercise_id":       session["exercise_id"],
            "exercise":          session["exercise"],
            "score":             session["score"],
            "grade":             session["grade"],
            "corrections_count": session["corrections_count"],
            "duration_seconds":  session["duration_seconds"],
            "report":            session["report"],
        }).execute()
    except Exception as e:
        print(f"[server] Supabase save failed ({e}), saving locally")
        _save_session_local(session)


def _load_sessions():
    if not _supabase:
        try:
            return json.loads(LOCAL_SESSIONS_FILE.read_text()) if LOCAL_SESSIONS_FILE.exists() else []
        except Exception:
            return []
    try:
        q = _supabase.table("sessions").select("*").order("created_at", desc=True)
        return q.execute().data or []
    except Exception as e:
        print(f"[server] Supabase load failed ({e}), reading local sessions")
        try:
            return json.loads(LOCAL_SESSIONS_FILE.read_text()) if LOCAL_SESSIONS_FILE.exists() else []
        except Exception:
            return []


def _merge_reports(reports):
    merged_corrections, merged_timeline = [], []
    phases_summary, scores, durations, frames_counts = {}, [], [], []

    for exercise_id, report in reports.items():
        if report.get("error"):
            continue
        if report.get("movement_detected") is False:
            continue
        scores.append(report.get("score", 0))
        durations.append(report.get("duration_seconds", 0))
        frames_counts.append(report.get("total_frames_analysed", 0))
        phases_summary[exercise_id] = report.get("phases_summary", {})
        for c in report.get("corrections", []):
            item = dict(c)
            item["exercise_id"] = exercise_id
            item["exercise"] = report.get("exercise", exercise_id)
            merged_corrections.append(item)
        for t in report.get("correction_timeline", []):
            item = dict(t)
            item["exercise_id"] = exercise_id
            item["exercise"] = report.get("exercise", exercise_id)
            merged_timeline.append(item)

    def _sort_key(c):
        if c.get("segments"):
            return (c["segments"][0].get("start_ms", c.get("first_seen_ms", 0)), c.get("id", ""))
        return (c.get("first_seen_ms", 0), c.get("id", ""))

    merged_corrections.sort(key=_sort_key)
    merged_timeline.sort(key=lambda t: (
        t.get("segments", [{}])[0].get("start_ms", 0) if t.get("segments") else 0,
        t.get("id", "")
    ))

    score = round(sum(scores) / len(scores)) if scores else 0
    return {
        "exercise_id": "all",
        "exercise": "All Steps",
        "score": score,
        "grade": _grade_from_score(score),
        "corrections": merged_corrections,
        "correction_timeline": merged_timeline,
        "phases_summary": phases_summary,
        "total_frames_analysed": max(frames_counts) if frames_counts else 0,
        "duration_seconds": max(durations) if durations else 0,
    }


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


# ── ROUTES ────────────────────────────────────────────────────────────

EXERCISE_ANALYSERS = {
    "plie":            lambda pd: {**analyse_plie(pd), "exercise_id": "plie"},
    "battement_tendu": lambda pd: analyse_tendu(pd),
    "releve":          lambda pd: analyse_releve(pd),
    "retire_passe":    lambda pd: analyse_retire(pd),
    "fondu":           lambda pd: analyse_fondu(pd),
    "arabesque":       lambda pd: analyse_arabesque(pd),
    "port_de_bras":    lambda pd: analyse_port_de_bras(pd),
    "grand_battement": lambda pd: analyse_grand_battement(pd),
    "glissade":         lambda pd: analyse_glissade(pd),
    "battement_degage": lambda pd: analyse_degage(pd),
    "rond_de_jambe":    lambda pd: analyse_rond_de_jambe(pd),
    "developpe":        lambda pd: analyse_developpe(pd),
    "echappe":  lambda pd: analyse_echappe(pd),
    "assemble": lambda pd: analyse_assemble(pd),
    "frappe":   lambda pd: analyse_frappe(pd),
    "attitude": lambda pd: analyse_attitude(pd),
}


@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "2.0.0"})


@app.route("/analyse", methods=["POST"])
def analyse():
    if "video" not in request.files:
        return jsonify({"error": "No video file. Use field name 'video'."}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Use: {', '.join(ALLOWED)}"}), 400

    suffix = "." + file.filename.rsplit(".", 1)[1].lower()
    session_id = uuid4().hex[:12]
    stored_filename = f"{session_id}{suffix}"
    stored_path = UPLOADS_DIR / stored_filename
    file.save(str(stored_path))

    exercise_id = request.form.get("exercise", "all").strip().lower()
    if exercise_id != "all" and exercise_id not in EXERCISES:
        return jsonify({"error": f"Unsupported exercise '{exercise_id}'."}), 400

    try:
        print(f"\n[analyse] {file.filename} ({os.path.getsize(stored_path)/1024/1024:.1f} MB)")
        print(f"[analyse] Exercise: {exercise_id}")

        print("[analyse] Extracting pose landmarks...")
        pose_data = extract_frames(str(stored_path), sample_every_n=3)
        print(f"[analyse] {pose_data['sampled_frames']} frames @ {pose_data['fps']:.1f} fps")

        if pose_data["sampled_frames"] == 0:
            return jsonify({
                "error": "No pose detected. Ensure full body is visible with good lighting."
            }), 422

        if exercise_id == "all":
            print("[analyse] Running analysis for all exercises...")
            reports = {ex: EXERCISE_ANALYSERS[ex](pose_data) for ex in sorted(EXERCISES)}
            report = _merge_reports(reports)
        else:
            print(f"[analyse] Running {exercise_id}...")
            report = EXERCISE_ANALYSERS[exercise_id](pose_data)

        print(f"[analyse] Score: {report.get('score')} | {len(report.get('corrections', []))} corrections")

        print("[analyse] Uploading to Cloudflare R2...")
        video_url = _upload_video(stored_path, stored_filename)
        if not _r2 or video_url.startswith("http://localhost"):
            print("[analyse] Keeping video locally for playback")
        else:
            stored_path.unlink(missing_ok=True)

        report.update({
            "video_filename":   file.filename,
            "session_id":       session_id,
            "video_url":        video_url,
            "frames_processed": pose_data["sampled_frames"],
            "video_duration_s": round(pose_data["total_frames"] / pose_data["fps"], 1),
        })

        _save_session({
            "id":                session_id,
            "created_at":        datetime.now(timezone.utc).isoformat(),
            "video_filename":    file.filename,
            "video_url":         video_url,
            "exercise_id":       report.get("exercise_id", exercise_id),
            "exercise":          report.get("exercise", ""),
            "score":             report.get("score", 0),
            "grade":             report.get("grade", ""),
            "corrections_count": len(report.get("corrections", [])),
            "duration_seconds":  report.get("duration_seconds", 0),
            "report":            report,
        })

        return jsonify(report)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Analysis failed.", "detail": str(e)}), 500


@app.route("/api/sessions")
def sessions_list():
    return jsonify(_load_sessions())


@app.route("/videos/<path:filename>")
def serve_video(filename):
    return send_from_directory(UPLOADS_DIR, filename)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


if __name__ == "__main__":
    print("\n" + "─"*48)
    print("  (r)elevē — Ballet Analysis Server")
    print("─"*48)
    print("  http://localhost:5000")
    print("─"*48 + "\n")
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug, host="0.0.0.0", port=5000)
