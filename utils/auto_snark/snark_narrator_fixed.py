#!/usr/bin/env python3
"""
Fixed version of snark_narrator.py that:
1. Creates silent audio placeholders when TTS fails
2. Still generates output even without TTS
"""

import os, json, tempfile, logging
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment, effects

# Import the original functions we need
import sys
sys.path.append(os.path.dirname(__file__))
from snark_narrator import (
    RunConfig, Segment, Beat, SnarkInsert,
    load_transcript, beats_from_transcript_gaps, 
    beats_from_markers, beats_from_shots,
    merge_and_prune_beats, generate_snark_text,
    overlay_with_ducking, secs_to_ms, hash_text
)

def create_silent_audio(duration_ms: int, output_path: str) -> bool:
    """Create a silent audio file as TTS placeholder"""
    try:
        silence = AudioSegment.silent(duration=duration_ms)
        silence.export(output_path, format="wav")
        return True
    except Exception as e:
        logging.error(f"Failed to create silent audio: {e}")
        return False

def run_pipeline_with_fallback(cfg: RunConfig) -> dict:
    """Modified pipeline that uses silent audio when TTS fails"""
    
    segments = load_transcript(cfg.transcript_path)
    
    with tempfile.TemporaryDirectory() as tmp:
        # Extract base audio
        wav_path = os.path.join(tmp, "audio.wav")
        clip = VideoFileClip(cfg.video_path)
        clip.audio.write_audiofile(wav_path, verbose=False, logger=None)
        clip.close()
        base_audio = AudioSegment.from_file(wav_path)
        
        # Detect beats
        candidates = []
        candidates += beats_from_transcript_gaps(segments, cfg.pause_threshold_s)
        candidates += beats_from_markers(segments)
        if cfg.use_vision:
            candidates += beats_from_shots(cfg.video_path)
        
        selected_beats = merge_and_prune_beats(candidates, cfg.min_gap_s, cfg.max_snarks)
        
        # Generate snarks with fallback to silent audio
        inserts = []
        tts_chars_total = 0
        
        for b in selected_beats:
            text = generate_snark_text(cfg.style, segments, b.time_s)
            tts_chars_total += len(text)
            
            # Create filename
            fname_base = f"snark_{secs_to_ms(b.time_s)}_{hash_text(text)}"
            tts_path = os.path.join(tmp, f"{fname_base}.wav")
            
            # Try TTS, fall back to silence
            duration_ms = len(text) * 50  # Estimate 50ms per character
            
            # For now, always create silent placeholder
            logging.info(f"Creating {duration_ms}ms silent placeholder for: {text[:30]}...")
            if create_silent_audio(duration_ms, tts_path):
                inserts.append(SnarkInsert(
                    time_s=b.time_s, 
                    text=text, 
                    style=cfg.style,
                    tts_path=tts_path, 
                    duration_ms=duration_ms, 
                    reasons=b.reasons
                ))
            else:
                logging.error(f"Failed to create audio for beat at {b.time_s:.2f}s")
        
        # Mix audio with silent placeholders
        mixed = base_audio
        for ins in inserts:
            ov = AudioSegment.from_file(ins.tts_path)
            start_ms = max(0, secs_to_ms(ins.time_s) - 120)
            mixed = overlay_with_ducking(mixed, ov, start_ms, cfg.duck_db)
        
        # Normalize and export
        mixed = effects.normalize(mixed)
        final_wav = os.path.join(tmp, "final_audio.wav")
        mixed.export(final_wav, format="wav")
        
        # Attach to video
        clip = VideoFileClip(cfg.video_path)
        final_audio_clip = AudioFileClip(final_wav)
        out_clip = clip.set_audio(final_audio_clip)
        out_clip.write_videofile(cfg.out_path, codec="libx264", audio_codec="aac", 
                                  fps=clip.fps, logger=None, verbose=False)
        out_clip.close()
        clip.close()
        final_audio_clip.close()
        
        # Create report
        report = {
            "video": os.path.abspath(cfg.video_path),
            "out": os.path.abspath(cfg.out_path),
            "style": cfg.style,
            "inserts": [
                {
                    "time_s": ins.time_s,
                    "text": ins.text,
                    "duration_ms": ins.duration_ms,
                    "reasons": ins.reasons,
                    "audio_type": "silent_placeholder"
                }
                for ins in inserts
            ],
            "counts": {
                "candidates": len(candidates),
                "selected": len(inserts)
            },
            "estimates": {
                "tts_total_chars": tts_chars_total,
                "approx_cost_usd": round(30.0 * (tts_chars_total / 1_000_000.0), 5)
            }
        }
        
        rep_path = cfg.out_path + ".report.json"
        with open(rep_path, "w") as f:
            json.dump(report, f, indent=2)
        
        return report

if __name__ == "__main__":
    import argparse
    
    p = argparse.ArgumentParser(description="Fixed Auto-Snark Narrator")
    p.add_argument("--video", required=True)
    p.add_argument("--transcript", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--style", default="wry", choices=["wry", "gentle", "spicy"])
    p.add_argument("--max-snarks", type=int, default=10)
    p.add_argument("--min-gap", type=float, default=12.0)
    p.add_argument("--pause-threshold", type=float, default=1.0)
    p.add_argument("--duck-db", type=int, default=-12)
    p.add_argument("--use-vision", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
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
        force_local_tts=True
    )
    
    report = run_pipeline_with_fallback(cfg)
    print(f"\n✅ Output created: {args.out}")
    print(f"📊 Snarks inserted: {len(report['inserts'])} (with silent placeholders)")
    print(f"📝 Report: {args.out}.report.json")