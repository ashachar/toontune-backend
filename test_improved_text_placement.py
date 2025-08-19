#!/usr/bin/env python3
"""
Test improved text placement with rembg background extraction.
Places ALL words from the transcript using intelligent background detection.
"""

import json
from pathlib import Path
from utils.text_placement.intelligent_text_placer import IntelligentTextPlacer
from apply_effects_to_original_scene import OriginalSceneProcessor
import tempfile
import subprocess


def test_scene_with_all_words():
    """Test scene 1 with ALL words from transcript using improved placement."""
    
    # Load transcript with ALL words
    words_file = Path("uploads/assets/videos/do_re_mi/transcripts/transcript_words.json")
    
    if not words_file.exists():
        print("Error: transcript_words.json not found")
        return
    
    # Load words 
    with open(words_file) as f:
        words_data = json.load(f)
        all_words = words_data.get('words', [])
    
    # Scene 1 is approximately first 60 seconds (based on our scene splitting)
    # Get actual duration from video
    import subprocess
    video_path = Path("uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4")
    if video_path.exists():
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True
        )
        scene_duration = float(result.stdout.strip())
    else:
        scene_duration = 60.0  # Default
    
    # Scene 1 starts at beginning of original video
    # Based on the transcript, scene 1 starts around 7.92s (first word "Let's")
    scene_1_start = 7.92  # First word in transcript
    scene_1_end = scene_1_start + scene_duration
    
    print(f"Scene 1: {scene_1_start:.2f}s - {scene_1_end:.2f}s (duration: {scene_duration:.2f}s)")
    
    # Filter words for scene 1 - take first ~80 words
    # (since scene 1 is about 56 seconds and has roughly 80-90 words)
    scene_1_words = all_words[:84]  # First 84 words for scene 1
    
    print(f"Total words in scene 1: {len(scene_1_words)}")
    
    # Adjust timestamps relative to scene start (scene 1 starts at 0 in the video file)
    # The transcript timestamps are from the original full video
    # Scene 1 video starts at what was 7.92s in the original
    for word in scene_1_words:
        word['start'] = max(0, word['start'] - scene_1_start)
        word['end'] = max(0.1, word['end'] - scene_1_start)
    
    # Use IntelligentTextPlacer to get positions
    video_path = Path("uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4")
    backgrounds_dir = Path("uploads/assets/videos/do_re_mi/scenes/backgrounds")
    
    if not video_path.exists():
        print(f"Error: Scene video not found at {video_path}")
        return
    
    print("\n" + "="*70)
    print("EXTRACTING BACKGROUNDS AND POSITIONING WORDS")
    print("="*70)
    
    placer = IntelligentTextPlacer(str(video_path), str(backgrounds_dir))
    positioned_words = placer.generate_word_positions(scene_1_words)
    
    # Convert to text overlays format
    text_overlays = []
    for word_data in positioned_words:
        overlay = {
            "text": word_data['word'],
            "x": word_data['x'],
            "y": word_data['y'],
            "start": word_data['start'],
            "end": word_data['end'],
            "fontsize": 48  # Larger font for visibility
        }
        text_overlays.append(overlay)
    
    # Save the improved inference
    improved_inference = {
        "scene_number": 1,
        "text_overlays": text_overlays,
        "sound_effects": [
            {"sound": "ding", "timestamp": 3.579, "volume": 0.5}
        ]
    }
    
    output_file = Path("uploads/assets/videos/do_re_mi/inferences/scene_001_improved.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w') as f:
        json.dump(improved_inference, f, indent=2)
    
    print(f"\n✓ Saved improved inference to: {output_file}")
    print(f"  Total text overlays: {len(text_overlays)}")
    
    # Show sample of word positions
    print("\nSample word positions (first 10):")
    for i, overlay in enumerate(text_overlays[:10]):
        print(f"  {i+1}. \"{overlay['text']}\" at ({overlay['x']}, {overlay['y']}) from {overlay['start']:.2f}s to {overlay['end']:.2f}s")
    
    # Now process the scene with improved positioning
    print("\n" + "="*70)
    print("APPLYING EFFECTS WITH IMPROVED TEXT PLACEMENT")
    print("="*70)
    
    processor = OriginalSceneProcessor()
    processor.process_scene(
        video_name="do_re_mi",
        scene_number=1,
        text_overlays=text_overlays,
        sound_effects=improved_inference['sound_effects']
    )
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"✓ Improved scene saved to: uploads/assets/videos/do_re_mi/scenes/edited/scene_001.mp4")
    print(f"✓ Background masks saved to: {backgrounds_dir}")
    print(f"✓ Word positions saved to: {backgrounds_dir}/word_positions.json")


if __name__ == "__main__":
    test_scene_with_all_words()