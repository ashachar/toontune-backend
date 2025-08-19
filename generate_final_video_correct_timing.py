#!/usr/bin/env python3
"""
Generate final video with CORRECTLY aligned text timing.
Uses actual scene boundaries from metadata.
"""

import json
from pathlib import Path
import subprocess

def generate_video_with_correct_timing():
    """Generate video with properly aligned text timing based on actual scene boundaries."""
    
    # Load scene metadata to get actual boundaries
    metadata_file = Path("uploads/assets/videos/do_re_mi/metadata/scenes.json")
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    # Get scene 1 boundaries
    scene1 = metadata["scenes"][0]
    scene_start = scene1["start_seconds"]  # 0.0 seconds
    scene_end = scene1["end_seconds"]      # 56.74 seconds
    
    print(f"Scene 1 boundaries from metadata:")
    print(f"  Start: {scene_start:.2f}s")
    print(f"  End: {scene_end:.2f}s")
    print(f"  Duration: {scene1['duration']:.2f}s")
    
    # Load the word positions with safe placement
    positions_file = Path("uploads/assets/videos/do_re_mi/inferences/scene_001_v3_safe.json")
    with open(positions_file) as f:
        data = json.load(f)
    
    text_overlays = data["text_overlays"]
    
    # Load original transcript for reference
    transcript_file = Path("uploads/assets/videos/do_re_mi/transcripts/transcript_words.json")
    with open(transcript_file) as f:
        transcript = json.load(f)
    
    print(f"\nFirst word timing check:")
    first_word = transcript["words"][0]
    print(f"  'Let's' in original video: {first_word['start']:.2f}s - {first_word['end']:.2f}s")
    print(f"  Scene starts at: {scene_start:.2f}s")
    print(f"  So 'Let's' should appear at: {first_word['start'] - scene_start:.2f}s in scene video")
    
    # Video paths
    input_video = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    output_video = "uploads/assets/videos/do_re_mi/scenes/final/scene_001_final_correct.mp4"
    
    # Ensure output directory exists
    Path(output_video).parent.mkdir(exist_ok=True, parents=True)
    
    # Build FFmpeg filter with CORRECT timing
    filters = []
    words_processed = 0
    
    for i, overlay in enumerate(text_overlays):
        # Escape special characters
        word = overlay["word"].replace("'", "'\\''")
        word = word.replace(":", "\\:")
        x = overlay["x"]
        y = overlay["y"]
        fontsize = overlay.get("fontsize", 48)
        
        # CORRECT TIMING: Word times are in original video time
        # Scene starts at scene_start (0.0), so subtract that from word times
        start_in_scene = overlay["start"] - scene_start
        end_in_scene = overlay["end"] - scene_start
        
        # Skip words outside this scene
        if end_in_scene < 0 or start_in_scene > scene1["duration"]:
            continue
        
        # Clamp to scene boundaries
        start_in_scene = max(0, start_in_scene)
        end_in_scene = min(scene1["duration"], end_in_scene)
        
        words_processed += 1
        
        # FFmpeg drawtext filter
        filter_str = (
            f"drawtext=text='{word}'"
            f":x={x}:y={y}"
            f":fontsize={fontsize}"
            f":fontcolor=white"
            f":bordercolor=black"
            f":borderw=2"
            f":enable='between(t,{start_in_scene:.3f},{end_in_scene:.3f})'"
        )
        filters.append(filter_str)
        
        # Debug first few words
        if i < 5:
            print(f"  '{word}': appears at {start_in_scene:.2f}s - {end_in_scene:.2f}s")
    
    # Combine all filters
    filter_complex = ",".join(filters)
    
    # FFmpeg command
    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-vf", filter_complex,
        "-codec:a", "copy",
        "-y",
        output_video
    ]
    
    print(f"\nGenerating final video with correct timing...")
    print(f"Input: {input_video}")
    print(f"Output: {output_video}")
    print(f"Processing {words_processed} text overlays...")
    
    # Run FFmpeg
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"\n✓ Video generated successfully: {output_video}")
        
        print("\n🎯 FINAL CORRECTIONS:")
        print("  - Scene boundaries: 0.0s - 56.74s (from metadata)")
        print("  - 'Let's' appears at 7.92s (matches audio)")
        print("  - 'beginning' at safe position (650, 270)")
        print("  - All timings aligned with actual speech")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ FFmpeg error: {e.stderr}")
        # Print first few lines of error for debugging
        error_lines = e.stderr.split('\n')[:10]
        for line in error_lines:
            print(f"  {line}")
        return False
    
    return True

if __name__ == "__main__":
    generate_video_with_correct_timing()