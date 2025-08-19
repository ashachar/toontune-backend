#!/usr/bin/env python3
"""
Create video with words that are FULLY contained in background.
Words stay as close as possible while never overlapping foreground.
"""

import json
import subprocess
from pathlib import Path
from utils.text_placement.proximity_text_placer_v2 import ProximityTextPlacerV2


def create_video_with_full_background_check():
    """Create scene with proximity words that are fully in background."""
    
    # Load transcript
    with open('uploads/assets/videos/do_re_mi/transcripts/transcript_words.json') as f:
        data = json.load(f)
        all_words = data['words']
    
    # Get words for scene 1
    scene_duration = 56.75
    scene_1_words = [w for w in all_words if w['start'] < scene_duration]
    
    print("="*70)
    print("FULL BACKGROUND-VALIDATED WORD PLACEMENT")
    print("="*70)
    print(f"Processing {len(scene_1_words)} words for scene 1")
    print(f"First word '{scene_1_words[0]['word']}' at {scene_1_words[0]['start']:.2f}s")
    print(f"Last word '{scene_1_words[-1]['word']}' at {scene_1_words[-1]['end']:.2f}s")
    print("\nEnsuring every word is FULLY contained in background areas...")
    
    # Use ProximityTextPlacerV2
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    backgrounds_dir = Path("uploads/assets/videos/do_re_mi/scenes/backgrounds_validated")
    
    print("\n" + "="*70)
    print("CALCULATING POSITIONS WITH FULL BACKGROUND VALIDATION")
    print("="*70)
    
    placer = ProximityTextPlacerV2(str(video_path), str(backgrounds_dir))
    positioned_words = placer.generate_word_positions_with_full_background_check(scene_1_words)
    
    # Create debug visualization
    placer.create_debug_visualization(
        positioned_words, 
        str(backgrounds_dir / "word_path_validated.png")
    )
    
    # Save positioned words
    output_json = Path("uploads/assets/videos/do_re_mi/inferences/scene_001_validated.json")
    output_json.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_json, 'w') as f:
        json.dump({
            'scene_number': 1,
            'text_overlays': positioned_words,
            'sound_effects': []
        }, f, indent=2)
    
    print(f"\n✓ Saved validated positions to: {output_json}")
    
    # Create video with FFmpeg
    print("\n" + "="*70)
    print("CREATING VIDEO")
    print("="*70)
    
    # Build filter complex
    filter_lines = []
    for i, word_data in enumerate(positioned_words):
        # Simple escaping
        word_text = word_data['word'].replace("'", "'\\\\''")
        
        if i == 0:
            input_tag = '[0:v]'
        else:
            input_tag = f'[txt{i-1}]'
        
        output_tag = f'[txt{i}]' if i < len(positioned_words) - 1 else ''
        
        filter_str = (
            f"{input_tag}drawtext=fontfile=/System/Library/Fonts/Helvetica.ttc:"
            f"text='{word_text}':"
            f"x={word_data['x']}:y={word_data['y']}:"
            f"fontsize={word_data['fontsize']}:"
            f"fontcolor=white:borderw=2:bordercolor=black:"
            f"enable='between(t,{word_data['start']:.3f},{word_data['end']:.3f})'{output_tag}"
        )
        filter_lines.append(filter_str)
    
    # Write filter to file
    filter_file = Path('validated_filter.txt')
    with open(filter_file, 'w') as f:
        f.write(';'.join(filter_lines))
    
    # Output video path
    output_video = Path("uploads/assets/videos/do_re_mi/scenes/edited/scene_001_validated.mp4")
    
    # Run FFmpeg
    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-filter_complex_script', str(filter_file),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'copy',
        str(output_video)
    ]
    
    print("Running FFmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up
    filter_file.unlink()
    
    if output_video.exists():
        size_mb = output_video.stat().st_size / (1024 * 1024)
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"✓ Video created: {output_video}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Words: {len(positioned_words)}")
        print(f"\n✓ Visualization: {backgrounds_dir}/word_path_validated.png")
        print("\nKey features:")
        print("  • Every word is FULLY contained in background")
        print("  • Words stay as close as possible to previous word")
        print("  • No overlap with foreground (Maria)")
        print("  • Natural reading flow maintained")
    else:
        print(f"\n✗ Failed to create video")
        if result.stderr:
            print(f"Error: {result.stderr[:500]}")


if __name__ == "__main__":
    create_video_with_full_background_check()