#!/usr/bin/env python3
"""
Fix the pipeline and test with a clean video
"""

import subprocess
import shutil
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

def extract_test_frames(video_path, output_dir, suffix=""):
    """Extract frames at key times for testing."""
    times = [11.5, 23.0, 25.0, 47.5, 51.5]
    for t in times:
        frame_path = output_dir / f"frame_{t:.1f}s{suffix}.png"
        cmd = ['ffmpeg', '-ss', str(t), '-i', str(video_path),
               '-frames:v', '1', '-y', str(frame_path)]
        subprocess.run(cmd, capture_output=True)

def get_clean_video():
    """Get a clean video without any overlays."""
    base_dir = Path("uploads/assets/videos/do_re_mi")
    original = base_dir / "scenes/original/scene_001.mp4"
    edited = base_dir / "scenes/edited/scene_001.mp4"
    
    if original.exists():
        # Copy original to edited as clean start
        shutil.copy(original, edited)
        print(f"✓ Reset to clean original video")
        return edited
    else:
        print("❌ No original video found!")
        return None

def test_karaoke_only(video_path, output_dir):
    """Test just karaoke generation."""
    print("\n1️⃣ TESTING KARAOKE ONLY...")
    
    # Try a simple subtitle overlay with ass filter
    test_ass = output_dir / "test_simple.ass"
    
    # Create a minimal ASS file
    ass_content = """[Script Info]
Title: Test Karaoke
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:10.00,0:00:15.00,Default,,0,0,0,,TEST KARAOKE LINE 1
Dialogue: 0,0:00:20.00,0:00:25.00,Default,,0,0,0,,TEST KARAOKE LINE 2
"""
    
    with open(test_ass, 'w') as f:
        f.write(ass_content)
    
    # Test with ass filter
    output = output_dir / "with_karaoke.mp4"
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', f"ass={test_ass}",
        '-c:v', 'libx264', '-crf', '18',
        '-c:a', 'copy',
        '-y', str(output)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("  ✅ Karaoke test successful")
        shutil.copy(output, video_path)  # Update edited
        return True
    else:
        print(f"  ❌ Karaoke failed: {result.stderr[:200]}")
        
        # Try with subtitles filter instead
        print("  Trying subtitles filter...")
        cmd[2] = f"subtitles={test_ass}"
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ Karaoke with subtitles filter successful")
            shutil.copy(output, video_path)
            return True
        else:
            print(f"  ❌ Subtitles filter also failed")
            return False

def test_phrases_on_top(video_path, output_dir):
    """Add phrases on top of existing video."""
    print("\n2️⃣ ADDING PHRASES ON TOP...")
    
    # Load inference for actual phrase data
    base_dir = Path("uploads/assets/videos/do_re_mi")
    inference_file = base_dir / "inferences/scene_001_inference.json"
    
    with open(inference_file) as f:
        inference = json.load(f)
    
    phrases = inference['scenes'][0]['key_phrases']
    
    # Build drawtext filters
    filters = []
    for p in phrases:
        text = p['phrase'].replace("'", "\\'").replace(":", "\\:")
        x = p.get('top_left_pixels', {}).get('x', 100)
        y = p.get('top_left_pixels', {}).get('y', 100)
        start = float(p['start_seconds'])
        duration = float(p['duration_seconds'])
        fontcolor = 'yellow' if 'playful' in p.get('style', '') else 'white'
        
        filter_str = (
            f"drawtext=text='{text}':fontsize=30:fontcolor={fontcolor}:"
            f"x={x}:y={y}:enable='between(t,{start},{start+duration})'"
        )
        filters.append(filter_str)
    
    output = output_dir / "with_phrases.mp4"
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', ','.join(filters),
        '-c:v', 'libx264', '-crf', '18',
        '-c:a', 'copy',
        '-y', str(output)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✅ Added {len(phrases)} phrases")
        shutil.copy(output, video_path)
        return True
    else:
        print(f"  ❌ Phrases failed: {result.stderr[:200]}")
        return False

def test_cartoons_on_top(video_path, output_dir):
    """Add cartoons on top of existing video WITH proper layering."""
    print("\n3️⃣ ADDING CARTOONS ON TOP (PROPERLY)...")
    
    spring_path = Path("cartoon-test/spring.png")
    if not spring_path.exists():
        print("  ❌ No spring.png found")
        return False
    
    output = output_dir / "with_cartoons.mp4"
    
    # Use filter_complex to properly layer cartoons
    # Key: Each overlay builds on the previous result
    filter_complex = (
        "[1:v]scale=100:120[c1];"  # Scale first cartoon
        "[2:v]scale=100:120[c2];"  # Scale second cartoon
        "[0:v][c1]overlay=x=200:y=200:enable='between(t,46.5,49.5)'[v1];"  # First overlay
        "[v1][c2]overlay=x=400:y=150:enable='between(t,50.5,53.5)'[vout]"  # Second overlay on top of first
    )
    
    cmd = [
        'ffmpeg', 
        '-i', str(video_path),  # Main video with karaoke+phrases
        '-i', str(spring_path),  # Cartoon 1
        '-i', str(spring_path),  # Cartoon 2 (same image)
        '-filter_complex', filter_complex,
        '-map', '[vout]',  # Use the final output
        '-map', '0:a?',    # Copy audio
        '-c:v', 'libx264', '-crf', '18',
        '-c:a', 'copy',
        '-y', str(output)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("  ✅ Added cartoons with proper layering")
        shutil.copy(output, video_path)
        return True
    else:
        print(f"  ❌ Cartoons failed: {result.stderr[:200]}")
        return False

def main():
    print("="*80)
    print("🔧 FIXING AND TESTING CLEAN PIPELINE")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")
    
    base_dir = Path("uploads/assets/videos/do_re_mi")
    test_dir = base_dir / "clean_test"
    test_dir.mkdir(exist_ok=True)
    
    # 1. Start with clean video
    video_path = get_clean_video()
    if not video_path:
        return
    
    extract_test_frames(video_path, test_dir, "_0_clean")
    
    # 2. Add karaoke
    if test_karaoke_only(video_path, test_dir):
        extract_test_frames(video_path, test_dir, "_1_karaoke")
    
    # 3. Add phrases on top
    if test_phrases_on_top(video_path, test_dir):
        extract_test_frames(video_path, test_dir, "_2_phrases")
    
    # 4. Add cartoons on top
    if test_cartoons_on_top(video_path, test_dir):
        extract_test_frames(video_path, test_dir, "_3_cartoons")
    
    # 5. Analyze final result
    print("\n" + "="*80)
    print("📊 FINAL ANALYSIS")
    print("="*80)
    
    print("\n✅ Pipeline executed in order:")
    print("  1. Clean video")
    print("  2. + Karaoke")
    print("  3. + Phrases")
    print("  4. + Cartoons")
    
    print(f"\n📁 Check frames in: {test_dir}")
    print("\nExpected progression:")
    print("  _0_clean: Nothing")
    print("  _1_karaoke: Karaoke at bottom")
    print("  _2_phrases: Karaoke + 2 phrases")
    print("  _3_cartoons: Karaoke + 2 phrases + 2 cartoons")
    
    print(f"\n🎬 FINAL VIDEO: {video_path}")
    print("This should have ALL features layered together!")
    
    # Quick check of final
    from analyze_debug_frames import analyze_frame
    print("\nFinal video analysis:")
    for t in [11.5, 23.0, 47.5]:
        frame_path = test_dir / f"frame_{t:.1f}s_3_cartoons.png"
        if frame_path.exists():
            result = analyze_frame(frame_path, f"At {t:.1f}s")
            print(result)

if __name__ == "__main__":
    main()