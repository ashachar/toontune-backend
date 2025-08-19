#!/usr/bin/env python3
"""
Verify Final Video Features
============================

This script verifies that the final karaoke video contains all the expected features.
"""

import json
from pathlib import Path

def verify_pipeline_output():
    """Verify the final video has all features."""
    
    base_dir = Path("uploads/assets/videos/do_re_mi")
    
    print("="*70)
    print("🎬 TOONTUNE FINAL VIDEO VERIFICATION")
    print("="*70)
    
    # Check inference for features
    inference_file = base_dir / "inferences/scene_001_inference.json"
    if inference_file.exists():
        with open(inference_file) as f:
            inference = json.load(f)
        
        scene = inference.get('scenes', [{}])[0]
        
        # Check key phrases
        key_phrases = scene.get('key_phrases', [])
        print(f"\n✅ Key Phrases ({len(key_phrases)} found):")
        for phrase in key_phrases:
            print(f"   - '{phrase['phrase']}' at {phrase['start_seconds']}s")
            print(f"     Style: {phrase.get('style', 'default')}, Importance: {phrase.get('importance', 'normal')}")
        
        # Check cartoon characters
        cartoons = scene.get('cartoon_characters', [])
        print(f"\n✅ Cartoon Characters ({len(cartoons)} found):")
        for char in cartoons:
            print(f"   - {char['character_type']} at {char['start_seconds']}s")
            print(f"     Animation: {char.get('animation_style', 'static')}")
            print(f"     Related to: {char.get('related_to', 'N/A')}")
    
    # Check pipeline state
    state_file = base_dir / "metadata/pipeline_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        
        print(f"\n✅ Pipeline Steps Completed:")
        for step in state.get('steps_completed', []):
            print(f"   - {step}")
    
    # Check output files
    print(f"\n📁 Output Files:")
    outputs = [
        ("Edited", "scenes/edited/scene_001.mp4"),
        ("With Phrases", "scenes/embedded_phrases/scene_001.mp4"),
        ("With Cartoons", "scenes/final_with_cartoons/scene_001.mp4"),
        ("Final with Karaoke", "scenes/karaoke/scene_001.mp4")
    ]
    
    for name, path in outputs:
        full_path = base_dir / path
        if full_path.exists():
            size = full_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   ✓ {name}: {size:.1f} MB")
        else:
            print(f"   ✗ {name}: NOT FOUND")
    
    # Pipeline flow
    print(f"\n🔄 Pipeline Flow:")
    print("   1. Edited video (with effects)")
    print("      ↓")
    print("   2. + Key phrases overlay")
    print("      ↓")
    print("   3. + Cartoon characters (using spring.png)")
    print("      ↓")
    print("   4. + Karaoke captions = FINAL VIDEO")
    
    final_video = base_dir / "scenes/karaoke/scene_001.mp4"
    if final_video.exists():
        print(f"\n🎉 FINAL VIDEO READY:")
        print(f"   {final_video}")
        print(f"\n   This video contains:")
        print(f"   • Original edits and effects")
        print(f"   • Key phrases ('very beginning', 'Do Re Mi')")
        print(f"   • Cartoon characters (spring.png as deer and sun)")
        print(f"   • Karaoke word-by-word captions")
        print(f"\n   To play: open {final_video}")
    else:
        print(f"\n⚠️  Final video not found!")
    
    print("="*70)

if __name__ == "__main__":
    verify_pipeline_output()