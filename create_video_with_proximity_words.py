#!/usr/bin/env python3
"""
Create video with proximity-based word placement.
Words stay close together to minimize eye movement.
"""

import json
import subprocess
from pathlib import Path
from utils.text_placement.proximity_text_placer import ProximityTextPlacer


def create_video_with_proximity_words():
    """Create scene 1 with proximity-aware word placement."""
    
    # Load transcript
    with open('uploads/assets/videos/do_re_mi/transcripts/transcript_words.json') as f:
        data = json.load(f)
        all_words = data['words']
    
    # Get words for scene 1 (up to ~56 seconds of scene duration)
    # Scene starts at 0s, words should appear at their actual times
    scene_duration = 56.75  # Known duration of scene_001.mp4
    scene_1_words = [w for w in all_words if w['start'] < scene_duration]
    
    print("="*70)
    print("PROXIMITY-BASED WORD PLACEMENT")
    print("="*70)
    print(f"Processing {len(scene_1_words)} words for scene 1")
    print(f"First word '{scene_1_words[0]['word']}' at {scene_1_words[0]['start']:.2f}s")
    print(f"Last word '{scene_1_words[-1]['word']}' at {scene_1_words[-1]['end']:.2f}s")
    
    # Use ProximityTextPlacer
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    backgrounds_dir = Path("uploads/assets/videos/do_re_mi/scenes/proximity_backgrounds")
    
    print("\n" + "="*70)
    print("CALCULATING POSITIONS WITH PROXIMITY AWARENESS")
    print("="*70)
    
    placer = ProximityTextPlacer(str(video_path), str(backgrounds_dir))
    positioned_words = placer.generate_word_positions_with_proximity(scene_1_words)
    
    # Create debug visualization
    placer.create_debug_visualization(
        positioned_words, 
        str(backgrounds_dir / "word_movement_path.png")
    )
    
    # Save positioned words
    output_json = Path("uploads/assets/videos/do_re_mi/inferences/scene_001_proximity.json")
    output_json.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_json, 'w') as f:
        json.dump({
            'scene_number': 1,
            'text_overlays': positioned_words,
            'sound_effects': []
        }, f, indent=2)
    
    print(f"\n✓ Saved proximity-based positions to: {output_json}")
    
    # Create video with FFmpeg
    print("\n" + "="*70)
    print("CREATING VIDEO")
    print("="*70)
    
    # Build filter complex - use filter file approach for many words
    filter_lines = []
    for i, word_data in enumerate(positioned_words):
        # Escape special characters
        word_escaped = (word_data['word']
                       .replace("'", "'\\\\\\\\\\\\\\\\''")
                       .replace('"', '\\\\\\\\"')
                       .replace(':', '\\\\\\\\:')
                       .replace(',', '\\\\\\\\,')
                       .replace('[', '\\\\\\\\[')
                       .replace(']', '\\\\\\\\]'))
        
        if i == 0:
            input_tag = '[0:v]'
        else:
            input_tag = f'[txt{i-1}]'
        
        output_tag = f'[txt{i}]' if i < len(positioned_words) - 1 else ''
        
        filter_str = (
            f"{input_tag}drawtext=text='{word_escaped}'"
            f":x={word_data['x']}:y={word_data['y']}"
            f":fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize={word_data['fontsize']}"
            f":fontcolor=white:borderw=2:bordercolor=black"
            f":enable='between(t,{word_data['start']:.3f},{word_data['end']:.3f})'{output_tag}"
        )
        filter_lines.append(filter_str)
    
    # Write filter to file
    filter_file = Path('temp_proximity_filter.txt')
    with open(filter_file, 'w') as f:
        f.write(';'.join(filter_lines))
    
    # Output video path
    output_video = Path("uploads/assets/videos/do_re_mi/scenes/edited/scene_001_proximity.mp4")
    
    # Run FFmpeg with filter file
    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-filter_complex_script', str(filter_file),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'copy',
        str(output_video)
    ]
    
    print("Running FFmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up filter file
    filter_file.unlink()
    
    if output_video.exists():
        size_mb = output_video.stat().st_size / (1024 * 1024)
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"✓ Video created: {output_video}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Words: {len(positioned_words)}")
        print(f"  First word at: {positioned_words[0]['start']:.2f}s")
        print(f"  Last word at: {positioned_words[-1]['end']:.2f}s")
        print(f"\n✓ Word movement visualization: {backgrounds_dir}/word_movement_path.png")
        print("\nThis video minimizes eye movement by keeping consecutive words close together!")
    else:
        print(f"\n✗ Failed to create video")
        if result.stderr:
            print(f"Error: {result.stderr[:500]}")


if __name__ == "__main__":
    create_video_with_proximity_words()