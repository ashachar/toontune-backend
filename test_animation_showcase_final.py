#!/usr/bin/env python3
"""
Final Animation Showcase - Using 3-second test element
Shows 10 different animations in sequence
"""

import os
import sys
import subprocess
from datetime import datetime

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import animation classes
from utils.animations.fade_in import FadeIn
from utils.animations.zoom_in import ZoomIn
from utils.animations.bounce import Bounce
from utils.animations.wave import Wave
from utils.animations.glitch import Glitch
from utils.animations.spin import Spin
from utils.animations.typewriter import Typewriter
from utils.animations.neon_glow import NeonGlow
from utils.animations.emergence_from_static_point import EmergenceFromStaticPoint
from utils.animations.carousel import Carousel

def create_title_card(text, duration, output_path):
    """Create a title card"""
    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'color=c=black:s=1920x1080:d={duration}:r=30',
        '-vf', f"drawtext=text='{text}':fontcolor=white:fontsize=60:x=(w-text_w)/2:y=(h-text_h)/2:font=Arial",
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

def create_background(duration, output_path):
    """Create background video"""
    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'color=c=0x1a1a2e:s=1920x1080:d={duration}:r=30',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

def main():
    # Use the short test element
    element_path = "test_element_3sec.mp4"
    
    if not os.path.exists(element_path):
        print(f"Creating test element from do_re_mi.mov...")
        cmd = [
            'ffmpeg',
            '-i', 'uploads/assets/videos/do_re_mi.mov',
            '-t', '3',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',
            element_path
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/animation_showcase_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temp directory
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    print("\n🎬 ANIMATION SHOWCASE")
    print("=" * 60)
    print("Demonstrating 10 different animation types")
    print("")
    
    # Center position
    center_x, center_y = 960, 540
    
    # Animation segments
    segments = []
    
    # Opening card
    opening = os.path.join(temp_dir, "opening.mp4")
    create_title_card("Animation Showcase", 2, opening)
    segments.append(opening)
    
    # List of animations to showcase
    animations = [
        ("Fade In", FadeIn, {"center_point": (center_x, center_y), "fade_speed": 0.3}, 2),
        ("Zoom In", ZoomIn, {"start_scale": 0.1, "end_scale": 1.0, "zoom_duration": 40}, 2),
        ("Bounce", Bounce, {"bounce_height": 200, "num_bounces": 2, "bounce_duration": 50}, 2),
        ("Wave", Wave, {"wave_type": "horizontal", "amplitude": 25, "frequency": 3}, 2),
        ("Emergence", EmergenceFromStaticPoint, {"direction": 0, "emergence_speed": 5.0}, 2),
        ("Spin", Spin, {"spin_speed": 12, "spin_axis": "z"}, 2),
        ("Glitch", Glitch, {"glitch_intensity": 0.7, "glitch_frequency": 0.4}, 2),
        ("Neon Glow", NeonGlow, {"glow_color": "#00FFFF", "glow_intensity": 1.5}, 2),
        ("Typewriter", Typewriter, {"text": "ANIMATED!", "chars_per_second": 8, "font_size": 80}, 2),
        ("Carousel", Carousel, {"num_items": 3, "radius": 120, "rotation_speed": 4}, 2),
    ]
    
    # Process each animation
    for i, (name, anim_class, params, duration) in enumerate(animations):
        print(f"\n📍 {i+1}/10: {name}")
        
        # Title card for this animation
        title = os.path.join(temp_dir, f"title_{i}.mp4")
        create_title_card(name, 1, title)
        segments.append(title)
        
        # Background for animation
        bg = os.path.join(temp_dir, f"bg_{i}.mp4")
        create_background(duration, bg)
        
        # Create animation
        anim_output = os.path.join(temp_dir, f"anim_{i}.mp4")
        
        try:
            animation = anim_class(
                element_path=element_path,
                background_path=bg,
                position=(center_x, center_y),
                fps=30,
                duration=duration,
                temp_dir=os.path.join(temp_dir, f"work_{i}"),
                **params
            )
            
            if animation.render(anim_output):
                segments.append(anim_output)
                print(f"   ✓ Completed")
            else:
                print(f"   ✗ Failed")
        except Exception as e:
            print(f"   ✗ Error: {str(e)[:80]}")
    
    # Closing card
    closing = os.path.join(temp_dir, "closing.mp4")
    create_title_card("The End", 2, closing)
    segments.append(closing)
    
    # Combine all segments
    print("\n" + "=" * 60)
    print("🎞️  Combining segments...")
    
    # Create concat list
    concat_file = os.path.join(temp_dir, "concat.txt")
    with open(concat_file, 'w') as f:
        for seg in segments:
            if os.path.exists(seg):
                f.write(f"file '{seg}'\n")
    
    # Final output
    final = os.path.join(output_dir, "showcase.mp4")
    
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',
        final
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"\n✅ SHOWCASE COMPLETE!")
        print(f"📁 Saved to: {final}")
        
        # Get duration
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', final],
            capture_output=True, text=True
        )
        duration = float(probe.stdout.strip()) if probe.stdout else 0
        print(f"⏱️  Duration: {duration:.1f} seconds")
        
        # Open video
        print("\n🎥 Opening video...")
        subprocess.run(['open', final])
    else:
        print(f"\n❌ Failed to create showcase")
        print(result.stderr[:200])

if __name__ == "__main__":
    main()