#!/usr/bin/env python3
"""
Trace the ACTUAL issue - follow the exact pipeline execution
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

def trace_pipeline():
    print("="*70)
    print("TRACING ACTUAL PIPELINE EXECUTION")
    print("="*70)
    
    base_dir = Path("uploads/assets/videos/do_re_mi")
    
    # Check what karaoke is doing
    print("\n1. KARAOKE STEP:")
    print("-"*40)
    
    # The karaoke generates to the SAME output path
    # Check if it's failing silently
    karaoke_ass = base_dir / "scenes/edited/karaoke_precise.ass"
    if karaoke_ass.exists():
        print(f"  ✓ Karaoke subtitle file exists: {karaoke_ass.name}")
        print(f"    Size: {karaoke_ass.stat().st_size} bytes")
        
        # Check if FFmpeg command would work
        video_in = base_dir / "scenes/edited/scene_001.mp4"
        print(f"\n  Karaoke command would be:")
        print(f"    ffmpeg -i {video_in.name} -vf ass={karaoke_ass.name} -codec:a copy output.mp4")
        print(f"\n  ⚠️ ISSUE: No video codec specified! Using FFmpeg defaults.")
    else:
        print("  ❌ No karaoke subtitle file found")
    
    # The real issue: Check the karaoke error more carefully
    print("\n2. CHECKING KARAOKE ERROR:")
    print("-"*40)
    
    # Try to run karaoke manually to see the actual error
    from utils.captions.karaoke_precise import PreciseKaraoke
    
    # Load transcript
    words_file = base_dir / "transcripts/transcript_words.json"
    with open(words_file) as f:
        words = json.load(f)['words'][:10]  # Just first 10 words
    
    generator = PreciseKaraoke()
    
    # Create test subtitle file
    test_ass = base_dir / "scenes/edited/test_karaoke.ass"
    
    # Try to generate
    print("  Testing karaoke generation...")
    input_vid = base_dir / "scenes/edited/scene_001.mp4"
    output_vid = base_dir / "scenes/edited/test_karaoke_output.mp4"
    
    import subprocess
    
    # First create the ASS file
    ass_content = generator._create_ass_content(
        frames=[{
            'sentences': [' '.join(w['word'] for w in words)],
            'start': 0,
            'end': 5,
            'color': 'yellow',
            'is_multi_line': False
        }]
    )
    
    with open(test_ass, 'w') as f:
        f.write(ass_content)
    
    # Try the FFmpeg command
    cmd = [
        "ffmpeg",
        "-i", str(input_vid),
        "-vf", f"ass={test_ass}",
        "-codec:a", "copy",
        "-codec:v", "libx264",  # ADD VIDEO CODEC
        "-crf", "18",           # ADD QUALITY
        "-y",
        str(output_vid)
    ]
    
    print(f"  Running: {' '.join(cmd[:4])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("  ✅ Karaoke works with proper video codec!")
        output_vid.unlink()  # Clean up
    else:
        print(f"  ❌ Karaoke failed: {result.stderr[:200]}")
    
    test_ass.unlink(missing_ok=True)

if __name__ == "__main__":
    trace_pipeline()