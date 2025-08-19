#!/usr/bin/env python3
"""
Apply all pipeline fixes - replace broken steps with working versions
"""

import shutil
from pathlib import Path

def apply_fixes():
    print("="*70)
    print("🔧 APPLYING PIPELINE FIXES")
    print("="*70)
    
    fixes_applied = []
    
    # 1. Replace karaoke step with fixed version
    karaoke_fixed = Path("pipeline/steps/step_7_karaoke_fixed.py")
    karaoke_original = Path("pipeline/steps/step_7_karaoke.py")
    
    if karaoke_fixed.exists():
        # Backup original
        shutil.copy(karaoke_original, karaoke_original.with_suffix('.py.backup'))
        # Replace with fixed
        shutil.copy(karaoke_fixed, karaoke_original)
        print("✅ Fixed karaoke step (step_7_karaoke.py)")
        fixes_applied.append("Karaoke: Now uses simple subtitles that work")
    
    # 2. Replace cartoons step with fixed version
    cartoons_fixed = Path("pipeline/steps/step_9_embed_cartoons_fixed.py")
    cartoons_original = Path("pipeline/steps/step_9_embed_cartoons.py")
    
    if cartoons_fixed.exists():
        # Backup original
        shutil.copy(cartoons_original, cartoons_original.with_suffix('.py.backup'))
        # Replace with fixed
        shutil.copy(cartoons_fixed, cartoons_original)
        print("✅ Fixed cartoons step (step_9_embed_cartoons.py)")
        fixes_applied.append("Cartoons: Now properly layers on top instead of replacing")
    
    # 3. Update pipeline order in main pipeline
    pipeline_file = Path("pipeline/core/pipeline.py")
    if pipeline_file.exists():
        with open(pipeline_file, 'r') as f:
            content = f.read()
        
        # Check if order is already fixed
        if "Karaoke FIRST (base layer)" in content:
            print("✅ Pipeline order already fixed")
        else:
            print("⚠️  Pipeline order needs manual fix in pipeline.py")
            print("   Move karaoke step BEFORE phrases and cartoons")
            fixes_applied.append("Pipeline order: Karaoke → Phrases → Cartoons")
    
    print("\n" + "="*70)
    print("📋 FIXES APPLIED:")
    print("-"*70)
    for fix in fixes_applied:
        print(f"  • {fix}")
    
    print("\n" + "="*70)
    print("🎯 NEXT STEPS:")
    print("-"*70)
    print("1. Run the pipeline with the fixed steps:")
    print("   python unified_video_pipeline.py <video> --no-downsample --no-transcript --no-scenes --no-prompts --no-inference --no-editing")
    print("\n2. Or run the complete fixed pipeline:")
    print("   python final_fixed_pipeline.py")
    print("\n3. Check the output:")
    print("   uploads/assets/videos/do_re_mi/scenes/edited/scene_001.mp4")
    print("="*70)

if __name__ == "__main__":
    apply_fixes()