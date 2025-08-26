#!/usr/bin/env python3
"""
Process all words for scene 1 with the new multi-frame algorithm.
"""

import json
from pathlib import Path
from utils.text_placement.proximity_text_placer_v3 import ProximityTextPlacerV3

def process_scene_1():
    """Process all words for scene 1 with multi-frame checking."""
    
    # Load existing words with timing
    transcript_file = Path("uploads/assets/videos/do_re_mi/transcripts/transcript_words.json")
    with open(transcript_file) as f:
        transcript = json.load(f)
    
    # Extract words for scene 1 (7.92s to 56.74s based on previous data)
    scene_1_words = []
    for word_data in transcript["words"]:
        if 7.92 <= word_data["start"] <= 56.74:
            scene_1_words.append(word_data)
    
    print(f"Processing {len(scene_1_words)} words for scene 1")
    
    # Process with new algorithm
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    placer = ProximityTextPlacerV3(video_path, "output/scene_1_v3")
    
    results = placer.process_words(scene_1_words)
    
    # Save as inference file format
    inference_data = {
        "scene_number": 1,
        "text_overlays": results,
        "sound_effects": []
    }
    
    output_file = Path("uploads/assets/videos/do_re_mi/inferences/scene_001_v3_safe.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w') as f:
        json.dump(inference_data, f, indent=2)
    
    print(f"\nSaved {len(results)} word positions to {output_file}")
    
    # Show comparison for key problematic words
    print("\nKey word position changes:")
    for result in results:
        if result["word"] in ["beginning", "myself", "female", "deer"]:
            print(f"  '{result['word']}': ({result['x']}, {result['y']})")

if __name__ == "__main__":
    process_scene_1()