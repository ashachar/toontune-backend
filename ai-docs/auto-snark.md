# Auto-Snark Narrator for Video (Python)
## All-in-One, Overloaded, Production-Minded Guide

Everything you need is inside this single Markdown file. You'll find: environment checks, install steps, design rationale, full Python source code (one file), requirements (inline), test plan, logging/metrics, scalability notes, cost math, and an extensibility roadmap.

**Assumption honored:** you already have a transcript pipeline that outputs a JSON of `{segments:[{start,end,text}]}`.

**Bonus:** works with ElevenLabs TTS (primary) and pyttsx3 fallback (no external keys required).

---

## 🔎 Verify Your Environment (Pre-flight)

### Python
- **Version:** 3.10+ recommended

### FFmpeg
Required by MoviePy/PyDub for audio I/O.

- **macOS:** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install -y ffmpeg`
- **Windows (winget):** `winget install Gyan.FFmpeg`

### Pip Packages
Listed below; install via commands in Install section.

### ElevenLabs (Optional but Recommended)
Set environment variable:
- **macOS/Linux:** `export ELEVENLABS_API_KEY="..."`
- **Windows (PowerShell):** `$env:ELEVENLABS_API_KEY="..."`

*(If unset, code falls back to local pyttsx3 TTS.)*

---

## 🚀 Install (Copy-Paste)

```bash
# Create and activate a clean venv (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install required Python packages
pip install moviepy==1.0.3 pydub==0.25.1 opencv-python==4.9.0.80 requests==2.32.3 pyttsx3==2.90
```

### Requirements File Content
If you prefer a "requirements file", here's the exact content (you can paste into `requirements.txt`):

```
moviepy==1.0.3
pydub==0.25.1
opencv-python==4.9.0.80
requests==2.32.3
pyttsx3==2.90
```

---

## 🧩 What You'll Ship (TL;DR)

A CLI that:
1. **Reads** a video and its timestamped transcript (`segments[{start,end,text}]`)
2. **Finds** comedic beats (silences/gaps, discourse markers like but/actually/anyway, optional shot changes via OpenCV)
3. **Generates** concise snark lines (5–12 words) from template banks (wry/gentle/spicy) with light context glue
4. **Synthesizes** narrator audio with ElevenLabs (multilingual, configurable voice), or local pyttsx3
5. **Auto-mixes** into the video audio with ducking (not hard mute), then exports MP4
6. **Emits** JSON report with insert timestamps, reasons, durations, and cost estimates (chars)

---

## 🧠 Design Choices (Why These Work)

- **Hybrid beat detection** (text + audio timing + optional vision) → robust across content styles
- **Short quips** keep pacing snappy, limit ducking time, and improve comedic timing
- **Ducking** preserves authenticity (no jarring mutes)
- **Safety filters + style control** (wry/gentle/spicy) reduce brand risk
- **Determinism via seed** for reproducible template choices
- **Report JSON** → measurable, A/B-able, cost-aware pipeline

---

## 🏃 Quick Start (One-Liners)

```bash
# (1) With ElevenLabs (recommended: set API key first)
export ELEVENLABS_API_KEY="YOUR_KEY_HERE"   # Windows PowerShell: setx ELEVENLABS_API_KEY "YOUR_KEY_HERE"

# (2) Save the Python code below to snark_narrator.py (exactly as provided)

# (3) Run on your own files
python snark_narrator.py \
  --video input.mp4 \
  --transcript your_transcript.json \
  --out output_snark.mp4 \
  --style wry \
  --max-snarks 10 \
  --min-gap 12 \
  --duck-db -12 \
  --use-vision 1 \
  --log-level INFO
```

**No ElevenLabs key?** Add `--no-elevenlabs` to use local TTS fallback.

---

## 🗂️ Transcript Format (What the Script Expects)

```json
{
  "segments": [
    {"start": 0.10, "end": 2.60, "text": "Welcome back everyone."},
    {"start": 2.80, "end": 7.90, "text": "Today we are testing a bold new idea. Actually, it's a bit experimental."},
    {"start": 9.20, "end": 13.10, "text": "But if it works, it could save hours of editing every week."},
    {"start": 15.50, "end": 19.00, "text": "Anyway, let's jump into the setup and look at the key steps."},
    {"start": 21.20, "end": 25.20, "text": "Okay, first we'll import the assets and configure the project."}
  ]
}
```

---

## 🧱 FULL SOURCE CODE (Single File) — `snark_narrator.py`

✅ Complete, drop-in, no TODOs  
✅ Works online (ElevenLabs) or fully offline (pyttsx3)  
✅ Includes environment checks, hybrid beat detection, mixing, export, and reporting

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snark_narrator.py
A complete CLI to inject witty/sarcastic TTS commentary into videos at detected comedic beats.

Features:
- Hybrid beat detection (pauses, discourse markers, optional shot changes via OpenCV).
- Template-based snark generation (styles: wry, gentle, spicy) with light context glue.
- ElevenLabs TTS synthesis (fallback to local pyttsx3 if API key is absent or --no-elevenlabs).
- Smart ducking mix and final MP4 export.
- JSON report with per-insert details + summary (helpful for metrics & A/B tests).

Usage:
  python snark_narrator.py --video input.mp4 --transcript transcript.json --out output_snark.mp4
"""

import os, sys, json, re, math, time, random, logging, argparse, tempfile, hashlib, shutil, subprocess
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

# Third-party deps
import requests
from pydub import AudioSegment, effects
from moviepy.editor import VideoFileClip, AudioFileClip

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# --------------------------- Defaults & Constants ---------------------------
DEFAULT_STYLE = "wry"               # wry | gentle | spicy
DEFAULT_MAX_SNARKS = 10
DEFAULT_MIN_GAP_S = 12.0            # min seconds between inserts
DEFAULT_PAUSE_THRESHOLD_S = 1.0     # candidate if gap >= this
DEFAULT_DUCK_DB = -12               # attenuation during overlay
DEFAULT_USE_VISION = True           # shot-change detection via OpenCV
DEFAULT_SEED = 1337

ELEVEN_MODEL_ID = "eleven_multilingual_v2"
ELEVEN_VOICE_ID_DEFAULT = "21m00Tcm4TlvDq8ikWAM"  # public example ("Rachel")

# Beat scoring weights
W_PAUSE = 1.0
W_MARKER = 1.2
W_SHOT = 1.0

# Marker lexicon (expand as needed)
MARKERS = [
    "but", "actually", "wait", "anyway", "ok", "okay", "look", "seriously",
    "honestly", "so", "listen", "right", "moving on"
]

# Simple profanity replace (extend in production)
BLOCKLIST = set(["damn", "hell", "idiot", "stupid", "hate", "dumb"])

# Template banks
TEMPLATES = {
    "wry": [
        "Bold choice. Not judging. Okay, maybe a little.",
        "Plot twist no one asked for.",
        "Ah yes, the professional approach.",
        "Not suspicious at all. Totally fine.",
        "Narrator: that did not go as planned.",
        "Confidence level: unverified.",
        "We're calling this… creative efficiency.",
        "Somewhere, a tripod just sighed.",
        "If you blinked, you missed the logic.",
        "Ten out of ten for commitment. Evidence pending."
    ],
    "gentle": [
        "Tiny detour. We'll allow it.",
        "Love the enthusiasm. Math? Later.",
        "That's one way to do it—cute.",
        "We support the vibe. Evidence optional.",
        "Wholesome chaos detected."
    ],
    "spicy": [
        "Choices were made. Regrets loading.",
        "Certified side-quest energy.",
        "This tutorial brought to you by chaos.",
        "Confidence: maximum. Accuracy: we'll see.",
        "Speedrun to confusion—any percent."
    ]
}

STOPWORDS = set("""a an the and or but so to of for with on in at from by is are be as this that it if then than very more most less least just really actually seriously okay ok right look listen anyway""".split())

# ------------------------------ Data Models ------------------------------
@dataclass
class Segment:
    start: float
    end: float
    text: str

@dataclass
class Beat:
    time_s: float
    score: float
    reasons: List[str]

@dataclass
class SnarkInsert:
    time_s: float
    text: str
    style: str
    tts_path: str
    duration_ms: int
    reasons: List[str]

# ------------------------------ Utilities ------------------------------
def check_ffmpeg_present() -> None:
    """Ensure ffmpeg is available to moviepy/pydub."""
    # Try PATH first
    if shutil.which("ffmpeg"):
        return
    # Try calling and catching
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return
    except Exception:
        pass
    raise RuntimeError(
        "FFmpeg not found. Install it and ensure it's on PATH.\n"
        "macOS: brew install ffmpeg\nUbuntu/Debian: sudo apt-get install -y ffmpeg\nWindows: winget install Gyan.FFmpeg"
    )

def load_transcript(path: str) -> List[Segment]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs = []
    for s in data.get("segments", []):
        segs.append(Segment(float(s["start"]), float(s["end"]), str(s["text"])))
    if not segs:
        raise ValueError("Transcript JSON has no segments.")
    return segs

def secs_to_ms(s: float) -> int:
    return int(round(s * 1000.0))

def safe_text(t: str) -> str:
    """Replace basic profanity tokens with *beep* (extend in production)."""
    words = t.split()
    cleaned = []
    for w in words:
        lw = w.lower().strip(",.!?;:")
        cleaned.append("*beep*" if lw in BLOCKLIST else w)
    return " ".join(cleaned)

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

# ------------------------------ Beat Detection ------------------------------
def beats_from_transcript_gaps(segments: List[Segment], pause_threshold: float) -> List[Beat]:
    beats: List[Beat] = []
    for i in range(1, len(segments)):
        prev, cur = segments[i-1], segments[i]
        gap = max(0.0, cur.start - prev.end)
        if gap >= pause_threshold:
            t = prev.end + min(0.2, gap / 2.0)   # slightly after previous line
            score = W_PAUSE * min(gap, 3.0)
            beats.append(Beat(time_s=t, score=score, reasons=[f"pause:{gap:.2f}s"]))
    return beats

def beats_from_markers(segments: List[Segment]) -> List[Beat]:
    beats: List[Beat] = []
    for seg in segments:
        lower = f" {seg.text.lower()} "
        if any(f" {m} " in lower for m in MARKERS):
            t = (seg.start + seg.end) / 2.0
            beats.append(Beat(time_s=t, score=W_MARKER * 1.0, reasons=["marker"]))
    return beats

def detect_shot_changes_cv(video_path: str, sample_fps: int = 5) -> List[float]:
    times: List[float] = []
    if not _HAS_CV2:
        return times
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return times
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    prev_hist = None
    diffs = []
    frame_idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % step == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0,256])
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                diffs.append(diff)
                # adaptive threshold from the last 50 samples
                if len(diffs) > 50:
                    recent = diffs[-50:]
                    mu = sum(recent)/len(recent)
                    sigma = (sum((x - mu)**2 for x in recent)/len(recent))**0.5
                    thr = float(mu + 2.0*sigma)
                else:
                    thr = 0.4
                if diff > thr:
                    t = frame_idx / fps
                    times.append(t)
            prev_hist = hist
        frame_idx += 1
    cap.release()
    return times

def beats_from_shots(video_path: str) -> List[Beat]:
    beats: List[Beat] = []
    for t in detect_shot_changes_cv(video_path):
        beats.append(Beat(time_s=t, score=W_SHOT * 1.0, reasons=["shot_change"]))
    return beats

def merge_and_prune_beats(beats: List[Beat], min_gap_s: float, max_snarks: int) -> List[Beat]:
    if not beats:
        return []
    beats = sorted(beats, key=lambda b: b.time_s)
    merged: List[Beat] = [beats[0]]
    for b in beats[1:]:
        if b.time_s - merged[-1].time_s < min_gap_s:
            if b.score > merged[-1].score:
                merged[-1] = b
        else:
            merged.append(b)
    # keep top-N by score but return in chronological order
    topN = sorted(merged, key=lambda b: b.score, reverse=True)[:max_snarks]
    return sorted(topN, key=lambda b: b.time_s)

# ------------------------------ Snark Generation ------------------------------
def extract_keywords_near(segments: List[Segment], t: float, k: int = 2) -> List[str]:
    # choose segment whose center is closest to t
    best = min(segments, key=lambda s: abs(((s.start + s.end)/2.0) - t))
    words = re.findall(r"[A-Za-z]{4,}", best.text)
    kws = []
    for w in words:
        lw = w.lower()
        if lw not in STOPWORDS and len(kws) < k:
            kws.append(w)
    return kws

def choose_template(style: str) -> str:
    bank = TEMPLATES.get(style, TEMPLATES["wry"]).copy()
    random.shuffle(bank)
    return bank[0]

def generate_snark_text(style: str, segments: List[Segment], t: float) -> str:
    base = choose_template(style)
    kws = extract_keywords_near(segments, t, k=2)
    if kws:
        base = f"{base} ({', '.join(kws).lower()})"
    return safe_text(base)

# ------------------------------ TTS Synthesis ------------------------------
def tts_elevenlabs(text: str, out_path: str, api_key: str,
                   voice_id: str = ELEVEN_VOICE_ID_DEFAULT,
                   model_id: str = ELEVEN_MODEL_ID,
                   retries: int = 3, timeout: int = 45) -> bool:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "accept": "audio/mpeg",
        "content-type": "application/json",
        "xi-api-key": api_key
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.85,
            "style": 0.25,
            "use_speaker_boost": True
        }
    }
    for attempt in range(1, retries+1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200 and resp.content:
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                return True
            logging.warning("ElevenLabs TTS non-200: %s %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logging.warning("ElevenLabs TTS error (attempt %d/%d): %s", attempt, retries, e)
        time.sleep(min(2**attempt, 8))
    return False

def tts_local_pyttsx3(text: str, out_path: str, rate: int = 185) -> bool:
    try:
        import pyttsx3
    except Exception as e:
        logging.error("pyttsx3 not installed, cannot local TTS: %s", e)
        return False
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return True
    except Exception as e:
        logging.error("pyttsx3 TTS failed: %s", e)
        return False

# ------------------------------ Audio Mixing ------------------------------
def overlay_with_ducking(base: AudioSegment, overlay: AudioSegment, start_ms: int, duck_db: int) -> AudioSegment:
    end_ms = start_ms + len(overlay)
    if end_ms > len(base):
        base = base + AudioSegment.silent(duration=end_ms - len(base))
    pre = base[:start_ms]
    mid = base[start_ms:end_ms] + duck_db
    post = base[end_ms:]
    mixed_mid = mid.overlay(overlay)
    return pre + mixed_mid + post

# ------------------------------ Core Pipeline ------------------------------
@dataclass
class RunConfig:
    video_path: str
    transcript_path: str
    out_path: str
    style: str = DEFAULT_STYLE
    max_snarks: int = DEFAULT_MAX_SNARKS
    min_gap_s: float = DEFAULT_MIN_GAP_S
    pause_threshold_s: float = DEFAULT_PAUSE_THRESHOLD_S
    duck_db: int = DEFAULT_DUCK_DB
    use_vision: bool = DEFAULT_USE_VISION
    seed: int = DEFAULT_SEED
    force_local_tts: bool = False

def extract_audio_to_wav(video_path: str, out_wav_path: str) -> None:
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(out_wav_path, verbose=False, logger=None)
    clip.close()

def run_pipeline(cfg: RunConfig) -> Dict:
    check_ffmpeg_present()
    random.seed(cfg.seed)

    segments = load_transcript(cfg.transcript_path)
    with tempfile.TemporaryDirectory() as tmp:
        # Extract base audio
        wav_path = os.path.join(tmp, "audio.wav")
        extract_audio_to_wav(cfg.video_path, wav_path)
        base_audio = AudioSegment.from_file(wav_path)

        # Candidate beats
        candidates: List[Beat] = []
        candidates += beats_from_transcript_gaps(segments, cfg.pause_threshold_s)
        candidates += beats_from_markers(segments)
        if cfg.use_vision:
            candidates += beats_from_shots(cfg.video_path)

        selected_beats = merge_and_prune_beats(candidates, cfg.min_gap_s, cfg.max_snarks)

        # Synthesize snarks
        inserts: List[SnarkInsert] = []
        tts_chars_total = 0
        for b in selected_beats:
            text = generate_snark_text(cfg.style, segments, b.time_s)
            tts_chars_total += len(text)

            # Choose output filename (cache-friendly)
            fname_base = f"snark_{secs_to_ms(b.time_s)}_{hash_text(text)}"
            tts_path_mp3 = os.path.join(tmp, f"{fname_base}.mp3")
            tts_path_wav = os.path.join(tmp, f"{fname_base}.wav")

            ok = False
            if not cfg.force_local_tts:
                api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
                voice_id = os.environ.get("ELEVENLABS_VOICE_ID", ELEVEN_VOICE_ID_DEFAULT).strip()
                if api_key:
                    ok = tts_elevenlabs(text, tts_path_mp3, api_key=api_key, voice_id=voice_id)
            if not ok:
                ok = tts_local_pyttsx3(text, tts_path_wav)
                tts_path = tts_path_wav if ok else ""
            else:
                tts_path = tts_path_mp3

            if not ok or not tts_path:
                logging.error("TTS synthesis failed; skipping beat at %.2fs", b.time_s)
                continue

            ov = AudioSegment.from_file(tts_path)
            inserts.append(SnarkInsert(
                time_s=b.time_s, text=text, style=cfg.style,
                tts_path=tts_path, duration_ms=len(ov), reasons=b.reasons
            ))

        # Mix overlays with ducking
        mixed = base_audio
        for ins in inserts:
            ov = AudioSegment.from_file(ins.tts_path)
            start_ms = max(0, secs_to_ms(ins.time_s) - 120)  # start a tad early
            mixed = overlay_with_ducking(mixed, ov, start_ms, cfg.duck_db)

        # Normalize to avoid clipping
        mixed = effects.normalize(mixed)

        # Export WAV
        final_wav = os.path.join(tmp, "final_audio.wav")
        mixed.export(final_wav, format="wav")

        # Attach back to video
        clip = VideoFileClip(cfg.video_path)
        final_audio_clip = AudioFileClip(final_wav)
        out_clip = clip.set_audio(final_audio_clip)
        out_clip.write_videofile(cfg.out_path, codec="libx264", audio_codec="aac", fps=clip.fps, logger=None)
        out_clip.close(); clip.close(); final_audio_clip.close()

        # Build report
        report = {
            "video": os.path.abspath(cfg.video_path),
            "out": os.path.abspath(cfg.out_path),
            "style": cfg.style,
            "max_snarks": cfg.max_snarks,
            "min_gap_s": cfg.min_gap_s,
            "pause_threshold_s": cfg.pause_threshold_s,
            "duck_db": cfg.duck_db,
            "use_vision": cfg.use_vision,
            "seed": cfg.seed,
            "inserts": [asdict(i) for i in inserts],
            "counts": {
                "candidates": len(candidates),
                "selected": len(inserts),
            },
            "estimates": {
                # if ElevenLabs cost is ~ $30 per 1M chars; ~70 chars/snark
                "tts_total_chars": tts_chars_total,
                "approx_cost_usd": round(30.0 * (tts_chars_total / 1_000_000.0), 5)
            }
        }
        rep_path = cfg.out_path + ".report.json"
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report

# ------------------------------ CLI ------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto‑Snark Narrator for Videos (Python)")
    p.add_argument("--video", required=True, help="Input video path (mp4/mov).")
    p.add_argument("--transcript", required=True, help="Transcript JSON path (segments: [{start,end,text}]).")
    p.add_argument("--out", required=True, help="Output video path (mp4).")
    p.add_argument("--style", default=DEFAULT_STYLE, choices=list(TEMPLATES.keys()), help="Snark style/template bank.")
    p.add_argument("--max-snarks", type=int, default=DEFAULT_MAX_SNARKS, help="Maximum number of inserts.")
    p.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP_S, help="Minimum gap (s) between inserts.")
    p.add_argument("--pause-threshold", type=float, default=DEFAULT_PAUSE_THRESHOLD_S, help="Minimum pause (s) to consider a beat.")
    p.add_argument("--duck-db", type=int, default=DEFAULT_DUCK_DB, help="Duck amount (dB) during TTS.")
    p.add_argument("--use-vision", type=int, default=1 if DEFAULT_USE_VISION else 0, help="Enable (1) or disable (0) shot-change detector.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for template selection.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Logging verbosity.")
    p.add_argument("--no-elevenlabs", action="store_true", help="Force local TTS (pyttsx3).")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    cfg = RunConfig(
        video_path=args.video,
        transcript_path=args.transcript,
        out_path=args.out,
        style=args.style,
        max_snarks=args.max_snarks,
        min_gap_s=args.min_gap,
        pause_threshold_s=args.pause_threshold,
        duck_db=args.duck_db,
        use_vision=bool(args.use_vision),
        seed=args.seed,
        force_local_tts=bool(args.no_elevenlabs)
    )
    report = run_pipeline(cfg)
    logging.info("Done. Report: %s", args.out + ".report.json")

if __name__ == "__main__":
    main()
```

---

## 🧪 Test Plan (Sanity → Robustness)

### Sanity Check (10 min)
Run with `--no-elevenlabs` (offline TTS) on a short clip (20–30 s) + the sample transcript structure above → verify:
- Output MP4 renders, audio present
- Snarks land roughly on gaps/markers
- Ducking is audible but not jarring

### With ElevenLabs
- Set `ELEVENLABS_API_KEY` and (optionally) `ELEVENLABS_VOICE_ID`
- Re-run the same command without `--no-elevenlabs`; confirm higher-quality voice and duration in report

### Edge Cases
- **Dense speech, few gaps:** ensure markers still produce beats; `--max-snarks` caps inserts; `--min-gap` respected
- **No markers:** gaps and shot-changes still create candidates
- **Long pauses:** see that pause score doesn't dominate infinitely (capped via `min(gap, 3.0)` in code)
- **Determinism:** same `--seed` → same template choices

---

## 📈 Logging & Metrics (What to Watch)

### Log Levels
- **INFO:** for pipeline milestones
- **WARNING/ERROR:** for TTS retries/failures

### Report JSON Fields
- `counts.candidates` / `selected`
- Each insert: `time_s`, `text`, `duration_ms`, `reasons` ("pause:1.34s", "marker", "shot_change")
- `estimates.tts_total_chars`, `approx_cost_usd`

### KPIs Over Runs
- Avg snarks/min, chars/snark, TTS latency/snark, % time ducked
- Error rates (TTS failures), clipping avoidance (normalize applied)

---

## ⚙️ Tuning Cheatsheet

- **Make it chattier:** increase `--max-snarks`, lower `--min-gap` to ~8–10
- **More precise timing:** raise `--pause-threshold` to favor real silences
- **Tone down intrusion:** use `--duck-db -8` (less attenuation) and pick `--style gentle`
- **Faster pipeline:** set `--use-vision 0` to skip shot detection

---

## 🛡️ Safety / Brand Controls

- **Profanity replacement** is on by default (simple blocklist). Extend if needed
- **Style control** maps to banks (wry/gentle/spicy). Default is wry (balanced)
- **Legal/voices:** use ElevenLabs stock voices or your licensed custom voices. Do not mimic unlicensed celebrities

---

## 💸 Cost Math (Rule of Thumb)

- Assume ~70 chars/snark × 10 snarks/video → ~700 chars total
- If your ElevenLabs plan is $30 / 1M chars, cost/video ≈ $0.021
- The code logs `approx_cost_usd` in the report for transparency

---

## 🧭 Roadmap Ideas (Next 1–3 Sprints)

1. **Dual-host banter:** two voices alternating quips
2. **Prosody-aware timing:** refine insert point using RMS/zero-crossing/sample-level silence detection
3. **LLM persona learning:** few-shot the creator's humor; store "voice of snark" as JSON persona
4. **Multilingual output:** detect transcript language → auto-select ElevenLabs voice/lang
5. **Review mode:** optional interactive approval (CLI prompt or simple HTML with accept/replace flow)

---

## 🧑‍🔧 Troubleshooting (Targeted)

- **Export fails with MoviePy:** confirm FFmpeg is installed and on PATH; try re-encoding input: `ffmpeg -i input.mp4 -c:v libx264 -c:a aac reinput.mp4`
- **Audio out of sync:** avoid variable-frame-rate inputs; re-mux with ffmpeg; keep source and target fps aligned (script copies source fps)
- **TTS too quiet/loud:** tweak `--duck-db` or normalize after overlay (already done, but you can adjust)
- **OpenCV errors / no opencv:** run with `--use-vision 0` (everything else still works)

---

## ✅ Final Checklist

- [ ] Python 3.10+, FFmpeg installed
- [ ] Pip deps installed
- [ ] Transcript JSON in the expected shape
- [ ] (Optional) `ELEVENLABS_API_KEY` set
- [ ] Run the CLI (see Quick Start), inspect `*.report.json`, iterate on flags