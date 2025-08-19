#!/usr/bin/env python3
"""
Create scene with ALL words at CORRECT timings.
Scene starts at 0s, words appear when actually spoken.
"""

import json
import subprocess
from pathlib import Path
from utils.text_placement.intelligent_text_placer import IntelligentTextPlacer


def create_scene_with_all_words():
    """Create scene 1 with all words properly timed."""
    
    # Load transcript
    with open('uploads/assets/videos/do_re_mi/transcripts/transcript_words.json') as f:
        data = json.load(f)
        all_words = data['words']
    
    # Scene 1 is approximately 56 seconds long
    # Get words that fall within scene 1 timeframe (roughly 7.92s to 64s of original)
    scene_1_words = [w for w in all_words if w['start'] < 64]
    
    print(f"Processing {len(scene_1_words)} words for scene 1")
    print(f"First word '{scene_1_words[0]['word']}' at {scene_1_words[0]['start']:.2f}s")
    print(f"Last word '{scene_1_words[-1]['word']}' at {scene_1_words[-1]['end']:.2f}s")
    
    # Use IntelligentTextPlacer to get good positions
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    backgrounds_dir = Path("uploads/assets/videos/do_re_mi/scenes/backgrounds_correct")
    backgrounds_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nExtracting backgrounds and positioning words...")
    placer = IntelligentTextPlacer(str(video_path), str(backgrounds_dir))
    
    # Generate positions for each word at their actual time
    positioned_words = []
    for i, word_data in enumerate(scene_1_words):
        word = word_data['word']
        # KEEP ORIGINAL TIMING - no offset needed!
        start_time = word_data['start']
        end_time = word_data['end']
        
        # Extract background at word's actual time
        _, (x, y) = placer.extract_background_at_time(start_time)
        
        positioned_words.append({
            'word': word,
            'start': start_time,
            'end': end_time,
            'x': x,
            'y': y,
            'fontsize': 48
        })
        
        if i % 10 == 0:
            print(f"  Processed {i+1}/{len(scene_1_words)} words...")
    
    print(f"\nPositioned all {len(positioned_words)} words")
    
    # Save positioned words
    output_json = Path("uploads/assets/videos/do_re_mi/inferences/scene_001_all_words_correct.json")
    output_json.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_json, 'w') as f:
        json.dump({
            'scene_number': 1,
            'text_overlays': positioned_words,
            'sound_effects': []
        }, f, indent=2)
    
    print(f"Saved word positions to: {output_json}")
    
    # Create video with FFmpeg
    print("\nCreating video with all words...")
    
    # Build complex filter for all words
    drawtext_filters = []
    for word_data in positioned_words:
        word_escaped = word_data['word'].replace("'", "\\'").replace(":", "\\:")
        filter_str = (
            f"drawtext=text='{word_escaped}'"
            f":x={word_data['x']}:y={word_data['y']}"
            f":fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize={word_data['fontsize']}"
            f":fontcolor=white:borderw=2:bordercolor=black"
            f":enable='between(t\\,{word_data['start']:.3f}\\,{word_data['end']:.3f})'"
        )
        drawtext_filters.append(filter_str)
    
    # Combine all filters
    filter_complex = ",".join(drawtext_filters)
    
    output_video = Path("uploads/assets/videos/do_re_mi/scenes/edited/scene_001_all_words_correct.mp4")
    
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", filter_complex,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        str(output_video)
    ]
    
    print("Running FFmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if output_video.exists():
        size_mb = output_video.stat().st_size / (1024 * 1024)
        print(f"\n✓ SUCCESS! Video created: {output_video}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Words: {len(positioned_words)}")
        print(f"  First word at: {positioned_words[0]['start']:.2f}s")
        print(f"  Last word at: {positioned_words[-1]['end']:.2f}s")
    else:
        print(f"\n✗ Failed to create video")
        print(f"Error: {result.stderr[:500]}")


if __name__ == "__main__":
    create_scene_with_all_words()