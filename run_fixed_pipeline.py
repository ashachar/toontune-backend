#!/usr/bin/env python3
"""
Run the fixed pipeline with all features properly layered
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

# Add to path for imports
sys.path.append(str(Path(__file__).parent))

# Import the FIXED versions
from pipeline.steps.step_7_karaoke_fixed import KaraokeStep
from pipeline.steps.step_8_embed_phrases import EmbedPhrasesStep
from pipeline.steps.step_9_embed_cartoons_fixed import EmbedCartoonsStep

def extract_verification_frames(video_path, output_dir):
    """Extract frames for verification."""
    import subprocess
    
    test_times = [
        (11.5, "Phrase 1: 'very beginning'"),
        (23.0, "Phrase 2: 'Do Re Mi'"),
        (25.0, "Karaoke text"),
        (47.5, "Cartoon 1"),
        (51.5, "Cartoon 2")
    ]
    
    for time, desc in test_times:
        frame_path = output_dir / f"final_{time:.1f}s.png"
        cmd = ['ffmpeg', '-ss', str(time), '-i', str(video_path),
               '-frames:v', '1', '-y', str(frame_path)]
        subprocess.run(cmd, capture_output=True)
        print(f"    {time:.1f}s: {desc}")

def main():
    print("="*80)
    print("🚀 RUNNING FIXED PIPELINE WITH PROPER LAYERING")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")
    
    # Set up paths
    video_dir = Path("uploads/assets/videos/do_re_mi")
    edited_dir = video_dir / "scenes/edited"
    verification_dir = video_dir / "fixed_pipeline_verification"
    verification_dir.mkdir(exist_ok=True)
    
    # Start with clean video
    original = video_dir / "scenes/original/scene_001.mp4"
    edited = edited_dir / "scene_001.mp4"
    
    if original.exists():
        shutil.copy(original, edited)
        print("✓ Reset to clean original video")
    else:
        print("⚠ Using existing edited video")
    
    # Create config object
    class Config:
        def __init__(self):
            self.video_dir = video_dir
            self.video_name = "do_re_mi"
            self.karaoke_style = "default"
    
    config = Config()
    
    # Create pipeline state and dirs for karaoke
    pipeline_state = {'steps_completed': []}
    dirs = {
        'transcripts': video_dir / 'transcripts',
        'scenes': video_dir / 'scenes',
        'scenes_edited': edited_dir,
        'metadata': video_dir / 'metadata'
    }
    
    print("\n📋 PIPELINE ORDER:")
    print("  1. Karaoke (base layer)")
    print("  2. Key phrases (overlay)")
    print("  3. Cartoon characters (top layer)")
    print()
    
    # Step 1: Karaoke FIRST (base layer)
    print("\n[STEP 1] Generating Karaoke Captions")
    print("-"*40)
    try:
        karaoke_step = KaraokeStep(pipeline_state, dirs, config)
        karaoke_step.run()
        print("✅ Karaoke step complete")
    except Exception as e:
        print(f"❌ Karaoke failed: {e}")
    
    # Step 2: Embed phrases on top of karaoke
    print("\n[STEP 2] Embedding Key Phrases")
    print("-"*40)
    try:
        phrases_step = EmbedPhrasesStep(config)
        phrases_step.run()
        print("✅ Phrases step complete")
    except Exception as e:
        print(f"❌ Phrases failed: {e}")
    
    # Step 3: Embed cartoons on top of everything
    print("\n[STEP 3] Embedding Cartoon Characters")
    print("-"*40)
    try:
        cartoons_step = EmbedCartoonsStep(config)
        cartoons_step.run()
        print("✅ Cartoons step complete")
    except Exception as e:
        print(f"❌ Cartoons failed: {e}")
    
    # Verify the result
    print("\n" + "="*80)
    print("📊 VERIFICATION")
    print("="*80)
    
    final_video = edited_dir / "scene_001.mp4"
    if final_video.exists():
        size_mb = final_video.stat().st_size / (1024*1024)
        print(f"\n✅ FINAL VIDEO: {final_video}")
        print(f"   Size: {size_mb:.1f} MB")
        
        print("\n📸 Extracting verification frames:")
        extract_verification_frames(final_video, verification_dir)
        
        # Analyze the frames
        print("\n🔍 Analyzing final frames:")
        from analyze_debug_frames import analyze_frame
        
        for time in [11.5, 23.0, 25.0, 47.5, 51.5]:
            frame_path = verification_dir / f"final_{time:.1f}s.png"
            if frame_path.exists():
                result = analyze_frame(frame_path, f"Final at {time:.1f}s")
                print(result)
    else:
        print("❌ No final video found!")
    
    print("\n" + "="*80)
    print("✨ EXPECTED RESULTS:")
    print("-"*40)
    print("At 11.5s: KARAOKE + PHRASE1 (white)")
    print("At 23.0s: KARAOKE + PHRASE2 (yellow)")  
    print("At 25.0s: KARAOKE")
    print("At 47.5s: CARTOON")
    print("At 51.5s: CARTOON")
    print("="*80)
    
    print(f"\n🎬 TO VERIFY: open {final_video}")
    print(f"📁 CHECK FRAMES: {verification_dir}")

if __name__ == "__main__":
    main()