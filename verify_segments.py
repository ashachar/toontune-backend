#!/usr/bin/env python3
"""
Quick script to verify individual segments for content duplication.
"""

import subprocess
import json
from pathlib import Path

def transcribe_segment(segment_path):
    """Transcribe a single segment using Whisper."""
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(str(segment_path))
    return result["text"]

def main():
    # Find the most recent work directory
    work_dirs = list(Path("/tmp").glob("precise_gap_pipeline_ai_math1_*"))
    if not work_dirs:
        print("No work directory found")
        return
    
    work_dir = sorted(work_dirs)[-1]
    print(f"Checking segments in: {work_dir}")
    
    segments_dir = work_dir / "segments"
    
    # Check segments around problem areas
    problem_segments = [
        "segment_014_normal.mp4",  # 43.04s - 48.98s
        "segment_015_gap.mp4",      # 48.98s - 49.48s  
        "segment_016_normal.mp4",   # 49.48s - 56.22s
    ]
    
    print("\n" + "="*70)
    print("SEGMENT CONTENT VERIFICATION")
    print("="*70)
    
    for seg_name in problem_segments:
        seg_path = segments_dir / seg_name
        if seg_path.exists():
            print(f"\n📹 {seg_name}:")
            
            # Get duration
            cmd = ["ffprobe", "-v", "error", "-show_entries", 
                   "format=duration", "-of", "json", str(seg_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            duration = float(data["format"]["duration"])
            print(f"   Duration: {duration:.3f}s")
            
            # Transcribe
            transcript = transcribe_segment(seg_path)
            print(f"   Content: {transcript[:100]}...")
            
            # Check for "create a new calculus tool" phrase
            if "create a new calculus tool" in transcript.lower():
                print("   ⚠️ CONTAINS: 'create a new calculus tool'")

if __name__ == "__main__":
    main()