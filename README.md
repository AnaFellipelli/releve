# (r)elevē — Ballet Analysis

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python server.py

# 3. Open in browser
# http://localhost:5000
```

### One-command launch

```bash
./run.sh
```

This installs requirements (if needed), starts the server, and opens the browser automatically.
If you want to disable auto-open:

```bash
AUTO_OPEN_BROWSER=0 ./run.sh
```

On first run, the pose model (~6MB) downloads automatically from Google.

## How it works

1. Home dashboard shows previous analyses and overall progress
2. Upload a ballet video (MP4, MOV, AVI, WEBM, MKV)
3. MediaPipe extracts 33 body landmarks per frame
4. Analyser runs all configured ballet steps
5. Report generated with score, corrections in EN + PT, and timestamp segments
6. Click any previous upload to reopen the two-panel analysis view (video + corrections)

## Files

| File | Purpose |
|------|---------|
| `server.py` | Flask server + frontend UI |
| `pose_extractor.py` | MediaPipe video processing |
| `plie_analyser.py` | Plié correction engine |
| `ballet_database.json` | Correction database (plié, tendu, arabesque…) |
| `requirements.txt` | Python dependencies |

## API

**POST /analyse**
- Body: `multipart/form-data` with field `video`
- Optional field: `exercise` (`all`, `plie`, `battement_tendu`, `arabesque`, `port_de_bras`, `grand_battement`)
- Returns: JSON report

If `exercise` is omitted, backend defaults to `all` and returns combined corrections
in chronological order, with exercise tags per correction.

```json
{
  "exercise_id": "battement_tendu",
  "exercise": "Battement Tendu",
  "score": 74,
  "grade": "Good",
  "corrections": [
    {
      "id": "plie_001",
      "severity": "major",
      "cue_en": "Knees falling inward...",
      "cue_pt": "Joelhos caindo para dentro...",
      "first_seen_ts": "00:03",
      "frequency": 12,
      "segments": [
        {
          "start_ms": 3200,
          "end_ms": 4100,
          "start_ts": "00:03",
          "end_ts": "00:04",
          "frames": 10
        }
      ]
    }
  ],
  "correction_timeline": [
    {
      "id": "plie_001",
      "severity": "major",
      "segments": [
        { "start_ts": "00:03", "end_ts": "00:04" }
      ]
    }
  ],
  "phases_summary": { "descent": 8, "bottom": 3, "ascent": 7 },
  "duration_seconds": 4.2,
  "frames_processed": 48
}
```
