#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snark_narrator_silence_aware.py
Enhanced version with silence-aware snark generation.

Key improvements:
- Analyzes silence FIRST before generating snarks
- Creates snarks that fit within available silence gaps
- No overlap with dialogue
- Automatic duration adjustment

Usage:
  python snark_narrator_silence_aware.py --video input.mp4 --transcript transcript.json --out output.mp4
"""

import os, sys, json, re, math, time, random, logging, argparse, tempfile, hashlib, shutil, subprocess
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Third-party deps
import requests
from pydub import AudioSegment, effects
from pydub.silence import detect_silence
from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# --------------------------- Defaults & Constants ---------------------------
DEFAULT_STYLE = "wry"               
DEFAULT_MAX_SNARKS = 8
DEFAULT_MIN_SILENCE_DURATION = 1.5  # Minimum silence gap for snark (seconds)
DEFAULT_SILENCE_THRESH = -40        # dB threshold for silence detection
DEFAULT_DUCK_DB = -12               
DEFAULT_SEED = 1337

ELEVEN_MODEL_ID = "eleven_turbo_v2_5"  
ELEVEN_VOICE_ID_DEFAULT = "21m00Tcm4TlvDq8ikWAM"  # Rachel

# Adaptive snark templates by duration
DURATION_TEMPLATES = {
    "ultra_short": {  # 1.0-1.5 seconds
        "texts": ["Wow.", "Sure.", "Right.", "Oh boy.", "Noted.", "Mmhmm.", "Cool."],
        "wpm": 180
    },
    "very_short": {  # 1.5-2.0 seconds
        "texts": ["How original.", "Fascinating.", "Thrilling stuff.", "Mind-blowing.", "Genius move."],
        "wpm": 170
    },
    "short": {  # 2.0-3.0 seconds
        "texts": [
            "Never seen that before.",
            "This changes everything.",
            "Pure artistic vision.",
            "Oscar-worthy performance.",
            "Revolutionary concept here."
        ],
        "wpm": 160
    },
    "medium": {  # 3.0-4.0 seconds
        "texts": [
            "And the crowd goes mild.",
            "Alert the media immediately.",
            "Someone call the Nobel committee.",
            "This is peak entertainment value."
        ],
        "wpm": 150
    },
    "long": {  # 4.0+ seconds
        "texts": [
            "Ah yes, exactly what we needed right now.",
            "I'm sure this will age like fine wine.",
            "This is definitely the highlight of my existence.",
            "Truly the pinnacle of human achievement we're witnessing."
        ],
        "wpm": 140
    }
}

STOPWORDS = set("""a an the and or but so to of for with on in at from by is are be as this that it if then than very more most less least just really actually seriously okay ok right look listen anyway""".split())

# ------------------------------ Data Models ------------------------------
@dataclass
class Segment:
    start: float
    end: float
    text: str

@dataclass
class SilencePeriod:
    start_ms: int
    end_ms: int
    duration_ms: int
    start_s: float
    end_s: float
    duration_s: float
    quality: str  # "silent" or "quiet"
    
    @property
    def is_usable(self) -> bool:
        return self.duration_s >= DEFAULT_MIN_SILENCE_DURATION

@dataclass
class AdaptiveSnark:
    period: SilencePeriod
    text: str
    style: str
    duration_estimate: float
    wpm: int
    context: str

@dataclass
class SnarkInsert:
    time_s: float
    text: str
    audio_path: str
    duration_s: float
    style: str

# ------------------------------ Utilities ------------------------------
def check_ffmpeg_present() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return
    except Exception:
        pass
    raise RuntimeError("FFmpeg not found. Install it and ensure it's on PATH.")

def load_transcript(path: str) -> List[Segment]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs = []
    for s in data.get("segments", []):
        segs.append(Segment(float(s["start"]), float(s["end"]), str(s["text"])))
    if not segs:
        raise ValueError("Transcript JSON has no segments.")
    return segs

def safe_text(t: str) -> str:
    """Clean text for TTS"""
    # Remove any existing emotion tags
    t = re.sub(r'<emotion="[^"]+">|</emotion>', '', t)
    return t.strip()

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

# ------------------------------ Silence Analysis ------------------------------
class SilenceAnalyzer:
    """Analyzes video for periods of silence suitable for snark insertion"""
    
    def __init__(self, video_path: str, transcript: List[Segment] = None):
        self.video_path = video_path
        self.transcript = transcript or []
        self.silence_periods = []
        
    def extract_and_analyze_audio(self, 
                                  silence_thresh: int = DEFAULT_SILENCE_THRESH,
                                  min_silence_len: int = 1500) -> List[SilencePeriod]:
        """Extract audio and find all silence periods"""
        
        logging.info("Analyzing audio for silence periods...")
        
        # Extract audio to temporary WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            # Extract audio using moviepy
            clip = VideoFileClip(self.video_path)
            clip.audio.write_audiofile(tmp_path, verbose=False, logger=None)
            clip.close()
            
            # Load with pydub
            audio = AudioSegment.from_file(tmp_path)
            
            # Detect truly silent periods
            silent_ranges = detect_silence(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                seek_step=100
            )
            
            # Detect quieter periods (background music only)
            quiet_ranges = detect_silence(
                audio,
                min_silence_len=int(min_silence_len * 0.8),  # Slightly shorter
                silence_thresh=silence_thresh + 5,  # Less strict threshold
                seek_step=100
            )
            
            # Convert to SilencePeriod objects
            periods = []
            
            # Add truly silent periods
            for start_ms, end_ms in silent_ranges:
                duration_ms = end_ms - start_ms
                periods.append(SilencePeriod(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=duration_ms,
                    start_s=start_ms / 1000,
                    end_s=end_ms / 1000,
                    duration_s=duration_ms / 1000,
                    quality="silent"
                ))
            
            # Add quiet periods that don't overlap
            for start_ms, end_ms in quiet_ranges:
                overlaps = False
                for p in periods:
                    if not (end_ms < p.start_ms or start_ms > p.end_ms):
                        overlaps = True
                        break
                        
                if not overlaps:
                    duration_ms = end_ms - start_ms
                    periods.append(SilencePeriod(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        duration_ms=duration_ms,
                        start_s=start_ms / 1000,
                        end_s=end_ms / 1000,
                        duration_s=duration_ms / 1000,
                        quality="quiet"
                    ))
            
            # Sort by start time
            periods.sort(key=lambda x: x.start_ms)
            
            # Filter to usable periods only
            usable = [p for p in periods if p.is_usable]
            
            logging.info(f"Found {len(periods)} total silence periods, {len(usable)} are usable")
            
            self.silence_periods = usable
            return usable
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def get_context_at_time(self, time_s: float) -> str:
        """Get transcript context at a specific time"""
        if not self.transcript:
            return "scene"
            
        for seg in self.transcript:
            if seg.start <= time_s <= seg.end:
                return seg.text[:50]
                
        # Check what was said just before
        for seg in self.transcript:
            if seg.end <= time_s <= seg.end + 2:
                return f"after: {seg.text[:40]}"
                
        return "transition"
    
    def score_comedic_potential(self, period: SilencePeriod) -> float:
        """Score how good this silence period is for comedy"""
        score = 0.0
        
        # Longer silences are better
        if period.duration_s > 3:
            score += 0.4
        elif period.duration_s > 2:
            score += 0.2
        
        # True silence is better than just quiet
        if period.quality == "silent":
            score += 0.3
        
        # Not too early (let video establish itself)
        if period.start_s > 3:
            score += 0.2
        
        # Check context for good snark opportunities
        context = self.get_context_at_time(period.start_s - 0.5)
        
        # Musical moments are great for snarks
        if any(word in context.lower() for word in ["sing", "song", "music", "do", "re", "mi"]):
            score += 0.4
            
        # After dramatic statements
        if any(word in context.lower() for word in ["very", "always", "never", "must", "will"]):
            score += 0.3
            
        return score

# ------------------------------ Adaptive Snark Generation ------------------------------
class AdaptiveSnarkGenerator:
    """Generates snarks that fit within available silence periods"""
    
    def __init__(self, style: str = DEFAULT_STYLE):
        self.style = style
        self.used_texts = []
        
    def select_template_for_duration(self, duration_s: float) -> Tuple[str, int]:
        """Select appropriate template based on available duration"""
        
        # Account for TTS overhead and safety margin
        safe_duration = duration_s - 0.4
        
        if safe_duration < 1.5:
            category = "ultra_short"
        elif safe_duration < 2.0:
            category = "very_short"
        elif safe_duration < 3.0:
            category = "short"
        elif safe_duration < 4.0:
            category = "medium"
        else:
            category = "long"
            
        templates = DURATION_TEMPLATES[category]
        
        # Avoid repetition
        available = [t for t in templates["texts"] if t not in self.used_texts]
        if not available:
            available = templates["texts"]
            
        text = random.choice(available)
        self.used_texts.append(text)
        
        return text, templates["wpm"]
    
    def generate_for_period(self, period: SilencePeriod, context: str) -> AdaptiveSnark:
        """Generate a snark specifically for this silence period"""
        
        text, wpm = self.select_template_for_duration(period.duration_s)
        
        # Add style-appropriate emotion tags for ElevenLabs v3
        if self.style == "spicy":
            text = f'<emotion="sarcastic">{text}</emotion>'
        elif self.style == "gentle":
            text = f'<emotion="friendly">{text}</emotion>'
        else:  # wry
            text = f'<emotion="deadpan">{text}</emotion>'
        
        # Estimate actual duration
        clean_text = safe_text(text)
        word_count = len(clean_text.split())
        duration_estimate = (word_count / (wpm / 60)) + 0.2  # Add small pause
        
        return AdaptiveSnark(
            period=period,
            text=text,
            style=self.style,
            duration_estimate=min(duration_estimate, period.duration_s - 0.3),
            wpm=wpm,
            context=context[:30]
        )

# ------------------------------ TTS Synthesis ------------------------------
def tts_elevenlabs_v3(text: str, out_path: str, api_key: str, wpm: int = 150,
                      voice_id: str = ELEVEN_VOICE_ID_DEFAULT,
                      model_id: str = ELEVEN_MODEL_ID) -> bool:
    """Enhanced ElevenLabs v3 TTS with speed control"""
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "accept": "audio/mpeg",
        "content-type": "application/json",
        "xi-api-key": api_key
    }
    
    # Adjust voice settings based on emotion and speed
    voice_settings = {
        "stability": 0.45,
        "similarity_boost": 0.85,
        "style": 0.25,
        "use_speaker_boost": True
    }
    
    # Adjust for emotion tags
    if "<emotion=" in text:
        if "sarcastic" in text:
            voice_settings["stability"] = 0.3
            voice_settings["style"] = 0.8
        elif "deadpan" in text:
            voice_settings["stability"] = 0.8
            voice_settings["style"] = 0.1
        elif "friendly" in text:
            voice_settings["stability"] = 0.5
            voice_settings["style"] = 0.6
    
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": voice_settings
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200 and resp.content:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return True
        logging.warning(f"ElevenLabs TTS failed: {resp.status_code}")
    except Exception as e:
        logging.warning(f"ElevenLabs TTS error: {e}")
    
    return False

def tts_system_fallback(text: str, out_path: str, wpm: int = 150) -> bool:
    """System TTS fallback with speed control"""
    
    clean_text = safe_text(text)
    
    # Try macOS 'say' command
    if sys.platform == "darwin":
        try:
            aiff_path = out_path.replace(".mp3", ".aiff")
            subprocess.run([
                "say", "-v", "Samantha",
                "-r", str(wpm),
                "-o", aiff_path,
                clean_text
            ], check=True, capture_output=True)
            
            # Convert to MP3
            subprocess.run([
                "ffmpeg", "-y", "-i", aiff_path,
                "-acodec", "mp3", "-ab", "192k",
                out_path
            ], check=True, capture_output=True)
            
            if os.path.exists(aiff_path):
                os.remove(aiff_path)
            
            return True
        except Exception as e:
            logging.warning(f"System TTS failed: {e}")
    
    # Create silent placeholder as last resort
    try:
        duration_ms = int((len(clean_text.split()) / (wpm / 60)) * 1000)
        silent = AudioSegment.silent(duration=duration_ms)
        silent.export(out_path, format="mp3")
        return True
    except Exception as e:
        logging.error(f"Failed to create placeholder: {e}")
    
    return False

# ------------------------------ Audio Mixing ------------------------------
def create_mixed_audio_smooth(base_audio: AudioSegment, 
                              inserts: List[SnarkInsert],
                              duck_db: int = DEFAULT_DUCK_DB,
                              fade_duration_ms: int = 400) -> AudioSegment:
    """Mix snarks with smooth ducking transitions"""
    
    mixed = base_audio
    
    for insert in inserts:
        if not os.path.exists(insert.audio_path):
            continue
            
        snark_audio = AudioSegment.from_file(insert.audio_path)
        position_ms = int(insert.time_s * 1000)
        duration_ms = len(snark_audio)
        
        # Apply smooth ducking
        fade_in_start = max(0, position_ms - fade_duration_ms)
        fade_out_end = min(len(mixed), position_ms + duration_ms + fade_duration_ms)
        
        # Split audio
        before_fade = mixed[:fade_in_start]
        fade_in = mixed[fade_in_start:position_ms]
        during = mixed[position_ms:position_ms + duration_ms]
        fade_out = mixed[position_ms + duration_ms:fade_out_end]
        after_fade = mixed[fade_out_end:]
        
        # Apply gradual ducking
        if len(fade_in) > 0:
            # Fade down to duck level
            for i in range(len(fade_in)):
                progress = i / len(fade_in)
                attenuation = int(duck_db * progress)
                fade_in = fade_in[:i] + (fade_in[i:i+1] + attenuation) + fade_in[i+1:]
        
        # Duck main section
        during = during + duck_db
        
        # Fade back up
        if len(fade_out) > 0:
            for i in range(len(fade_out)):
                progress = 1 - (i / len(fade_out))
                attenuation = int(duck_db * progress)
                fade_out = fade_out[:i] + (fade_out[i:i+1] + attenuation) + fade_out[i+1:]
        
        # Reconstruct
        mixed = before_fade + fade_in + during + fade_out + after_fade
        
        # Overlay snark
        mixed = mixed.overlay(snark_audio, position=position_ms)
    
    return mixed

# ------------------------------ Main Pipeline ------------------------------
@dataclass
class SilenceAwareConfig:
    video_path: str
    transcript_path: str
    out_path: str
    style: str = DEFAULT_STYLE
    max_snarks: int = DEFAULT_MAX_SNARKS
    min_silence_duration: float = DEFAULT_MIN_SILENCE_DURATION
    silence_thresh: int = DEFAULT_SILENCE_THRESH
    duck_db: int = DEFAULT_DUCK_DB
    seed: int = DEFAULT_SEED
    use_elevenlabs: bool = True

def run_silence_aware_pipeline(cfg: SilenceAwareConfig) -> Dict:
    """Main pipeline for silence-aware snark generation"""
    
    check_ffmpeg_present()
    random.seed(cfg.seed)
    
    # Load transcript
    segments = []
    if cfg.transcript_path and os.path.exists(cfg.transcript_path):
        segments = load_transcript(cfg.transcript_path)
    
    # Analyze silence
    analyzer = SilenceAnalyzer(cfg.video_path, segments)
    silence_periods = analyzer.extract_and_analyze_audio(
        silence_thresh=cfg.silence_thresh,
        min_silence_len=int(cfg.min_silence_duration * 1000)
    )
    
    if not silence_periods:
        logging.warning("No suitable silence periods found!")
        return {"error": "No silence periods found"}
    
    # Score and select best periods for comedy
    scored_periods = []
    for period in silence_periods:
        score = analyzer.score_comedic_potential(period)
        if score > 0.3:  # Minimum threshold
            scored_periods.append((period, score))
    
    # Sort by score and take top N
    scored_periods.sort(key=lambda x: x[1], reverse=True)
    selected_periods = [p for p, _ in scored_periods[:cfg.max_snarks]]
    
    # Generate adaptive snarks
    generator = AdaptiveSnarkGenerator(cfg.style)
    planned_snarks = []
    
    for period in selected_periods:
        context = analyzer.get_context_at_time(period.start_s)
        snark = generator.generate_for_period(period, context)
        planned_snarks.append(snark)
    
    # Sort by time
    planned_snarks.sort(key=lambda x: x.period.start_s)
    
    # Generate TTS audio
    inserts = []
    api_key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
    
    with tempfile.TemporaryDirectory() as tmp:
        for i, snark in enumerate(planned_snarks):
            fname = f"snark_{i+1}_{snark.style}.mp3"
            audio_path = os.path.join(tmp, fname)
            
            # Try ElevenLabs first
            success = False
            if cfg.use_elevenlabs and api_key:
                success = tts_elevenlabs_v3(
                    snark.text, audio_path, api_key, wpm=snark.wpm
                )
            
            # Fallback to system TTS
            if not success:
                success = tts_system_fallback(
                    snark.text, audio_path, wpm=snark.wpm
                )
            
            if success and os.path.exists(audio_path):
                # Get actual duration
                audio = AudioSegment.from_file(audio_path)
                actual_duration = len(audio) / 1000
                
                inserts.append(SnarkInsert(
                    time_s=snark.period.start_s + 0.2,  # Small offset into silence
                    text=safe_text(snark.text),
                    audio_path=audio_path,
                    duration_s=actual_duration,
                    style=snark.style
                ))
        
        # Extract base audio
        clip = VideoFileClip(cfg.video_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            audio_path = tmp_audio.name
            clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        clip.close()
        
        base_audio = AudioSegment.from_file(audio_path)
        
        # Mix with smooth ducking
        mixed_audio = create_mixed_audio_smooth(base_audio, inserts, cfg.duck_db)
        
        # Export mixed audio
        mixed_path = os.path.join(tmp, "mixed_audio.wav")
        mixed_audio.export(mixed_path, format="wav")
        
        # Combine with video
        subprocess.run([
            "ffmpeg", "-y",
            "-i", cfg.video_path,
            "-i", mixed_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            cfg.out_path
        ], check=True, capture_output=True)
        
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
    # Generate report
    report = {
        "video": cfg.video_path,
        "style": cfg.style,
        "silence_analysis": {
            "total_periods": len(silence_periods),
            "usable_periods": len([p for p in silence_periods if p.is_usable]),
            "total_silence_time": sum(p.duration_s for p in silence_periods),
            "selected_periods": len(selected_periods)
        },
        "snarks": [
            {
                "time": insert.time_s,
                "text": insert.text,
                "duration": insert.duration_s,
                "style": insert.style,
                "silence_window": next(
                    p.duration_s for p in selected_periods 
                    if abs(p.start_s - (insert.time_s - 0.2)) < 0.5
                )
            }
            for insert in inserts
        ],
        "settings": {
            "min_silence_duration": cfg.min_silence_duration,
            "silence_threshold": cfg.silence_thresh,
            "duck_level": cfg.duck_db,
            "fade_duration": "400ms"
        }
    }
    
    return report

# ------------------------------ CLI Interface ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Silence-Aware Snark Narrator")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--transcript", help="Transcript JSON path")
    parser.add_argument("--out", required=True, help="Output video path")
    parser.add_argument("--style", default=DEFAULT_STYLE, choices=["wry", "gentle", "spicy"])
    parser.add_argument("--max-snarks", type=int, default=DEFAULT_MAX_SNARKS)
    parser.add_argument("--min-silence", type=float, default=DEFAULT_MIN_SILENCE_DURATION,
                       help="Minimum silence duration in seconds")
    parser.add_argument("--silence-thresh", type=int, default=DEFAULT_SILENCE_THRESH,
                       help="Silence threshold in dB")
    parser.add_argument("--duck-db", type=int, default=DEFAULT_DUCK_DB)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-elevenlabs", action="store_true", help="Use system TTS only")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Run pipeline
    config = SilenceAwareConfig(
        video_path=args.video,
        transcript_path=args.transcript,
        out_path=args.out,
        style=args.style,
        max_snarks=args.max_snarks,
        min_silence_duration=args.min_silence,
        silence_thresh=args.silence_thresh,
        duck_db=args.duck_db,
        seed=args.seed,
        use_elevenlabs=not args.no_elevenlabs
    )
    
    print("=" * 70)
    print("🤫 SILENCE-AWARE SNARK NARRATOR")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Style: {args.style}")
    print(f"Max snarks: {args.max_snarks}")
    print(f"Min silence: {args.min_silence}s")
    print("=" * 70)
    
    try:
        report = run_silence_aware_pipeline(config)
        
        # Save report
        report_path = args.out.replace(".mp4", "_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print("\n✅ SUCCESS!")
        print(f"Output: {args.out}")
        print(f"Report: {report_path}")
        
        if "snarks" in report:
            print(f"\n📊 Generated {len(report['snarks'])} snarks:")
            for s in report['snarks']:
                print(f"  • {s['time']:.1f}s: \"{s['text'][:30]}...\" ({s['duration']:.1f}s)")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()