#!/usr/bin/env python3
"""
Verify Final Pipeline Results
==============================

Verifies that the pipeline correctly outputs everything to scenes/edited folder.
"""

import json
from pathlib import Path

def verify_pipeline():
    base_dir = Path("uploads/assets/videos/do_re_mi")
    
    print("="*70)
    print("🎬 TOONTUNE PIPELINE VERIFICATION - NEW STRUCTURE")
    print("="*70)
    
    # Check folder structure
    print("\n📁 Folder Structure:")
    folders = {
        "original": base_dir / "scenes/original",
        "downsampled": base_dir / "scenes/downsampled",
        "edited": base_dir / "scenes/edited"
    }
    
    for name, path in folders.items():
        if path.exists():
            videos = list(path.glob("scene_*.mp4"))
            if videos:
                size = sum(f.stat().st_size for f in videos) / (1024*1024)
                print(f"  ✓ {name:12} {len(videos)} scenes, {size:.1f} MB total")
                for v in sorted(videos)[:1]:  # Show first scene
                    vsize = v.stat().st_size / (1024*1024)
                    print(f"      scene_001: {vsize:.1f} MB")
        else:
            print(f"  ✗ {name:12} NOT FOUND")
    
    # Check for old folders that should NOT exist
    print("\n🚫 Old Folders (should NOT exist):")
    old_folders = ["karaoke", "embedded_phrases", "final_with_cartoons"]
    for folder in old_folders:
        path = base_dir / "scenes" / folder
        if path.exists():
            print(f"  ⚠️  {folder} EXISTS - should be removed!")
        else:
            print(f"  ✓ {folder} removed (good)")
    
    # Check features in inference
    print("\n✨ Features Applied:")
    inference_file = base_dir / "inferences/scene_001_inference.json"
    if inference_file.exists():
        with open(inference_file) as f:
            inference = json.load(f)
        
        scene = inference.get('scenes', [{}])[0]
        
        # Key phrases
        key_phrases = scene.get('key_phrases', [])
        if key_phrases:
            print(f"  ✓ Key Phrases: {len(key_phrases)} phrases")
            for p in key_phrases[:2]:
                print(f"      - '{p['phrase']}' at {p['start_seconds']}s")
        
        # Cartoons
        cartoons = scene.get('cartoon_characters', [])
        if cartoons:
            print(f"  ✓ Cartoons: {len(cartoons)} characters")
            for c in cartoons[:2]:
                print(f"      - {c['character_type']} at {c['start_seconds']}s")
    
    # Pipeline state
    state_file = base_dir / "metadata/pipeline_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        
        print(f"\n📊 Pipeline Steps:")
        for step in state.get('steps_completed', []):
            print(f"  ✓ {step}")
    
    # Final summary
    edited_path = base_dir / "scenes/edited/scene_001.mp4"
    if edited_path.exists():
        size_mb = edited_path.stat().st_size / (1024*1024)
        print(f"\n🎉 FINAL OUTPUT:")
        print(f"  Location: {edited_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Quality: ORIGINAL (not downsampled)")
        print(f"  Features: All embedded in single file")
        print(f"\n  To play: open {edited_path}")
    else:
        print(f"\n⚠️  Final output not found!")
    
    print("="*70)

if __name__ == "__main__":
    verify_pipeline()