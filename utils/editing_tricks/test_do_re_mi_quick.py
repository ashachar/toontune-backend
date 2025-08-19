#!/usr/bin/env python3
"""
Quick test of video effects on do_re_mi.mov - processes shorter segments for speed.
Creates a master showcase with title cards.
"""

import sys
import cv2
import numpy as np
import subprocess
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from color_effects import apply_color_splash
from text_effects import apply_motion_tracking_text
from motion_effects import apply_floating_effect, apply_smooth_zoom
from layout_effects import add_progress_bar


def extract_short_clip(input_video: Path, output_path: Path, start: float = 0, duration: float = 2.0):
    """Extract a short clip from video."""
    cmd = [
        'ffmpeg',
        '-ss', str(start),
        '-i', str(input_video),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-c:a', 'aac',
        '-y',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def add_text_overlay(input_path: Path, text: str, output_path: Path):
    """Add text overlay to video using ffmpeg."""
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-vf', f"drawtext=text='{text}':fontsize=40:fontcolor=white:borderw=2:bordercolor=black:x=(w-text_w)/2:y=50",
        '-c:a', 'copy',
        '-y',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def main():
    """Quick test with selected effects."""
    
    input_video = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi.mov")
    output_dir = Path("do_re_mi_quick_test")
    output_dir.mkdir(exist_ok=True)
    
    if not input_video.exists():
        print(f"Error: Video not found: {input_video}")
        sys.exit(1)
    
    print("="*60)
    print("QUICK VIDEO EFFECTS TEST - DO RE MI")
    print("="*60)
    
    # Extract short clip for faster processing
    print("\n1. Extracting 3-second clip...")
    short_clip = output_dir / "short_clip.mp4"
    extract_short_clip(input_video, short_clip, start=2, duration=3)
    print(f"   ✓ Created {short_clip.name}")
    
    segments = []
    
    # Original with title
    print("\n2. Creating original with title...")
    original_titled = output_dir / "00_original.mp4"
    add_text_overlay(short_clip, "ORIGINAL", original_titled)
    segments.append(str(original_titled))
    print(f"   ✓ Created {original_titled.name}")
    
    # Apply selected effects
    print("\n3. Applying effects...")
    
    # Color Splash
    try:
        print("   - Color Splash...")
        effect_path = output_dir / "01_color_splash_raw.mp4"
        apply_color_splash(short_clip, target_color=(255, 100, 0), tolerance=40, output_path=effect_path)
        
        titled_path = output_dir / "01_color_splash.mp4"
        add_text_overlay(effect_path, "COLOR SPLASH", titled_path)
        segments.append(str(titled_path))
        print(f"     ✓ Done")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    # Floating Effect
    try:
        print("   - Floating Effect...")
        effect_path = output_dir / "02_floating_raw.mp4"
        apply_floating_effect(short_clip, amplitude=20, frequency=0.5, direction="vertical", output_path=effect_path)
        
        titled_path = output_dir / "02_floating.mp4"
        add_text_overlay(effect_path, "FLOATING EFFECT", titled_path)
        segments.append(str(titled_path))
        print(f"     ✓ Done")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    # Smooth Zoom
    try:
        print("   - Smooth Zoom...")
        effect_path = output_dir / "03_zoom_raw.mp4"
        apply_smooth_zoom(short_clip, zoom_factor=1.5, zoom_type="in_out", output_path=effect_path)
        
        titled_path = output_dir / "03_zoom.mp4"
        add_text_overlay(effect_path, "SMOOTH ZOOM", titled_path)
        segments.append(str(titled_path))
        print(f"     ✓ Done")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    # Progress Bar
    try:
        print("   - Progress Bar...")
        effect_path = output_dir / "04_progress_raw.mp4"
        add_progress_bar(short_clip, bar_height=8, bar_color=(0, 255, 0), style="glow", output_path=effect_path)
        
        titled_path = output_dir / "04_progress.mp4"
        add_text_overlay(effect_path, "PROGRESS BAR", titled_path)
        segments.append(str(titled_path))
        print(f"     ✓ Done")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    # Motion Tracking Text
    try:
        print("   - Motion Tracking Text...")
        effect_path = output_dir / "05_motion_text_raw.mp4"
        apply_motion_tracking_text(short_clip, text="TRACKING", font_scale=1.2, output_path=effect_path)
        
        titled_path = output_dir / "05_motion_text.mp4"
        add_text_overlay(effect_path, "MOTION TRACKING", titled_path)
        segments.append(str(titled_path))
        print(f"     ✓ Done")
    except Exception as e:
        print(f"     ✗ Failed: {e}")
    
    # Create final showcase
    print("\n4. Creating final showcase...")
    list_file = output_dir / "segments.txt"
    with open(list_file, 'w') as f:
        for segment in segments:
            f.write(f"file '{segment}'\n")
    
    showcase_path = output_dir / "do_re_mi_showcase.mp4"
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(list_file),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-y',
        str(showcase_path)
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    print(f"   ✓ Created {showcase_path.name}")
    
    # Get duration
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(showcase_path)
    ]
    duration = subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip()
    
    print("\n" + "="*60)
    print("SHOWCASE COMPLETE")
    print("="*60)
    print(f"\n📺 Showcase video: {showcase_path}")
    print(f"⏱  Duration: {float(duration):.1f} seconds")
    print(f"📊 Effects demonstrated: {len(segments)}")
    print("\nTo view:")
    print(f"  open {showcase_path}")


if __name__ == "__main__":
    main()