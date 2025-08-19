#!/usr/bin/env python3
"""
ULTRATHINK DEBUG - Find and fix all pipeline issues
"""

import json
import subprocess
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Add to path
sys.path.append(str(Path(__file__).parent))

def run_command(cmd, description=""):
    """Run command and return success/output."""
    print(f"  Running: {description}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def extract_frame(video_path, time, output_path):
    """Extract a single frame."""
    cmd = ['ffmpeg', '-ss', str(time), '-i', str(video_path),
           '-frames:v', '1', '-y', str(output_path)]
    success, _, _ = run_command(cmd, f"Extract frame at {time}s")
    return success

def get_video_info(video_path):
    """Get video file size and codec info."""
    if not video_path.exists():
        return "NOT FOUND"
    size_mb = video_path.stat().st_size / (1024*1024)
    return f"{size_mb:.1f}MB"

def test_step_individually(step_name, test_func):
    """Test a single step and extract frames."""
    print(f"\n{'='*60}")
    print(f"TESTING: {step_name}")
    print('='*60)
    
    # Run the step
    success = test_func()
    
    if success:
        print(f"✅ {step_name} completed")
    else:
        print(f"❌ {step_name} FAILED")
    
    return success

def main():
    print("="*80)
    print("🔬 ULTRATHINK DEEP DEBUGGING - FINDING ALL ISSUES")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")
    
    base_dir = Path("uploads/assets/videos/do_re_mi")
    edited_dir = base_dir / "scenes/edited"
    debug_dir = base_dir / "ultrathink_debug"
    debug_dir.mkdir(exist_ok=True)
    
    # Keep track of issues found
    issues = []
    
    # 1. CHECK INITIAL STATE
    print("\n📋 INITIAL STATE CHECK:")
    print("-"*60)
    
    original = edited_dir / "scene_001.mp4"
    if not original.exists():
        print("❌ No edited video found to start with!")
        issues.append("No base edited video")
        return
    
    print(f"✓ Base video exists: {get_video_info(original)}")
    
    # Make a backup
    backup = debug_dir / "scene_001_backup.mp4"
    shutil.copy(original, backup)
    print(f"✓ Backup created: {backup.name}")
    
    # 2. TEST KARAOKE GENERATION
    def test_karaoke():
        from pipeline.steps.step_7_karaoke import KaraokeStep
        
        # Prepare for karaoke
        pipeline_state = {'steps_completed': []}
        dirs = {
            'transcripts': base_dir / 'transcripts',
            'scenes': base_dir / 'scenes',
            'scenes_edited': edited_dir,
            'metadata': base_dir / 'metadata'
        }
        
        class Config:
            karaoke_style = "default"
        
        try:
            # First restore backup
            shutil.copy(backup, original)
            
            step = KaraokeStep(pipeline_state, dirs, Config())
            step.run(use_final_with_cartoons=False)
            
            # Check if file was modified
            if original.stat().st_mtime > backup.stat().st_mtime:
                print("  ✓ Video was modified by karaoke")
                return True
            else:
                print("  ❌ Video was NOT modified - karaoke failed")
                issues.append("Karaoke not modifying video")
                return False
        except Exception as e:
            print(f"  ❌ Karaoke exception: {e}")
            issues.append(f"Karaoke exception: {str(e)[:50]}")
            return False
    
    karaoke_ok = test_step_individually("KARAOKE", test_karaoke)
    
    # Extract frame to check karaoke
    if karaoke_ok:
        extract_frame(original, 25.0, debug_dir / "after_karaoke.png")
        print(f"  Frame extracted: after_karaoke.png (should show karaoke at bottom)")
    
    # 3. TEST PHRASE EMBEDDING
    def test_phrases():
        from pipeline.steps.step_8_embed_phrases import EmbedPhrasesStep
        
        class Config:
            video_dir = base_dir
            video_name = "do_re_mi"
        
        try:
            step = EmbedPhrasesStep(Config())
            step.run()
            return True
        except Exception as e:
            print(f"  ❌ Phrases exception: {e}")
            issues.append(f"Phrases exception: {str(e)[:50]}")
            return False
    
    phrases_ok = test_step_individually("PHRASES", test_phrases)
    
    # Extract frames at both phrase times
    if phrases_ok:
        extract_frame(original, 11.5, debug_dir / "after_phrases_1.png")
        extract_frame(original, 23.0, debug_dir / "after_phrases_2.png") 
        print(f"  Frames extracted: Check for 'very beginning' at 11.5s and 'Do Re Mi' at 23s")
    
    # 4. TEST CARTOON EMBEDDING
    def test_cartoons():
        from pipeline.steps.step_9_embed_cartoons import EmbedCartoonsStep
        
        class Config:
            video_dir = base_dir
            video_name = "do_re_mi"
        
        try:
            step = EmbedCartoonsStep(Config())
            step.run()
            return True
        except Exception as e:
            print(f"  ❌ Cartoons exception: {e}")
            issues.append(f"Cartoons exception: {str(e)[:50]}")
            return False
    
    cartoons_ok = test_step_individually("CARTOONS", test_cartoons)
    
    # Extract frames at cartoon times
    if cartoons_ok:
        extract_frame(original, 47.5, debug_dir / "after_cartoons_1.png")
        extract_frame(original, 51.5, debug_dir / "after_cartoons_2.png")
        print(f"  Frames extracted: Check for cartoons at 47.5s and 51.5s")
    
    # 5. DEEP ANALYSIS OF FFMPEG COMMANDS
    print("\n🔍 ANALYZING FFMPEG COMMANDS:")
    print("-"*60)
    
    # Test if multiple drawtext filters work
    print("\nTesting multiple drawtext filters:")
    test_vid = debug_dir / "test_multi_drawtext.mp4"
    cmd = [
        'ffmpeg', '-i', str(backup), '-t', '30',
        '-vf', (
            "drawtext=text='PHRASE 1':fontsize=30:fontcolor=white:x=100:y=100:"
            "enable='between(t,10,14)',"
            "drawtext=text='PHRASE 2':fontsize=30:fontcolor=yellow:x=500:y=300:"
            "enable='between(t,22,26)'"
        ),
        '-c:v', 'libx264', '-crf', '18',
        '-y', str(test_vid)
    ]
    success, _, err = run_command(cmd, "Multiple drawtext test")
    if success:
        print("  ✅ Multiple drawtext filters work")
        extract_frame(test_vid, 11.5, debug_dir / "test_multi_p1.png")
        extract_frame(test_vid, 23.0, debug_dir / "test_multi_p2.png")
    else:
        print(f"  ❌ Multiple drawtext failed: {err[:100]}")
        issues.append("Multiple drawtext filters failing")
    
    # 6. CHECK ACTUAL EMBEDDING COMMANDS
    print("\n📝 CHECKING ACTUAL COMMANDS USED:")
    print("-"*60)
    
    # Load inference to see what should be embedded
    inference_file = base_dir / "inferences/scene_001_inference.json"
    with open(inference_file) as f:
        inference = json.load(f)
    
    scene = inference['scenes'][0]
    phrases = scene.get('key_phrases', [])
    cartoons = scene.get('cartoon_characters', [])
    
    print(f"Expected features:")
    print(f"  Phrases: {len(phrases)}")
    for p in phrases:
        print(f"    - '{p['phrase']}' at {p['start_seconds']}s, pos=({p.get('top_left_pixels',{}).get('x')},{p.get('top_left_pixels',{}).get('y')})")
    
    print(f"  Cartoons: {len(cartoons)}")
    for c in cartoons:
        print(f"    - {c['character_type']} at {c['start_seconds']}s")
    
    # 7. FINAL DIAGNOSIS
    print("\n" + "="*80)
    print("🔬 DIAGNOSIS COMPLETE")
    print("="*80)
    
    if issues:
        print("\n❌ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✅ All steps completed without errors")
    
    print("\n📁 CHECK FRAMES IN:", debug_dir)
    print("\nFrames to examine:")
    frames_to_check = [
        "after_karaoke.png - Should show karaoke captions at bottom",
        "after_phrases_1.png - Should show 'very beginning' at top-right", 
        "after_phrases_2.png - Should show 'Do Re Mi' at top-left",
        "after_cartoons_1.png - Should show cartoon at 47.5s",
        "after_cartoons_2.png - Should show cartoon at 51.5s",
        "test_multi_p1.png - Test of phrase 1",
        "test_multi_p2.png - Test of phrase 2"
    ]
    for frame in frames_to_check:
        print(f"  • {frame}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()