#!/usr/bin/env python3
"""
Check what's actually at specific timestamps in the video.
"""

import subprocess
import whisper
from pathlib import Path

def extract_and_transcribe(video_path, start_time, duration=2.0):
    """Extract audio at timestamp and transcribe."""
    # Extract audio segment
    audio_file = f"/tmp/check_audio_{start_time}.wav"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_time),
        "-i", str(video_path),
        "-t", str(duration),
        "-vn", "-acodec", "pcm_s16le",
        audio_file
    ]
    subprocess.run(cmd, check=True)
    
    # Transcribe
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    
    return result["text"]

def main():
    video_path = Path("uploads/assets/videos/ai_math1.mp4")
    
    # Check key timestamps where gaps are supposed to be
    timestamps = [
        (24.66, "Gap 2: supposed to be after 'no longer a question of'"),
        (26.08, "Gap 3: supposed to be after 'of if AI will lead'"),
        (38.74, "Gap 5: supposed to be after 'create a new calculus tool'"),
        (42.04, "Gap 7: supposed to be after 'in turn improves AI engines'"),
        (48.98, "Gap 8: supposed to be after 'mental model or a schema'"),
    ]
    
    print("="*70)
    print("TIMESTAMP VERIFICATION")
    print("="*70)
    
    for time, description in timestamps:
        print(f"\n📍 At {time:.2f}s ({description}):")
        transcript = extract_and_transcribe(video_path, time)
        print(f"   Actual content: {transcript[:80]}...")

if __name__ == "__main__":
    main()