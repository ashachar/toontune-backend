#!/usr/bin/env python3
"""
Quick Animation Showcase Test
Shows a selection of animations to demonstrate variety
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import selected animation classes
from utils.animations.fade_in import FadeIn
from utils.animations.slide_in import SlideIn
from utils.animations.zoom_in import ZoomIn
from utils.animations.bounce import Bounce
from utils.animations.emergence_from_static_point import EmergenceFromStaticPoint
from utils.animations.warp import Warp
from utils.animations.wave import Wave
from utils.animations.typewriter import Typewriter
from utils.animations.glitch import Glitch
from utils.animations.neon_glow import NeonGlow
from utils.animations.spin import Spin
from utils.animations.carousel import Carousel

def create_title_card(text, output_path, duration=1):
    """Create a title card with text"""
    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'color=c=black:s=1920x1080:d={duration}',
        '-vf', f"drawtext=text='{text}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2",
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return output_path

def create_solid_background(color, duration, output_path):
    """Create a solid color background video"""
    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'color=c={color}:s=1920x1080:d={duration}:r=30',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return output_path

def main():
    # Setup paths
    element_path = "uploads/assets/videos/do_re_mi.mov"
    
    # Check if element exists
    if not os.path.exists(element_path):
        print(f"Error: {element_path} not found!")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/animation_showcase_quick_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory for segments
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Center position for all animations
    center_x, center_y = 960, 540
    
    # List to store all video segments
    segments = []
    segment_list_file = os.path.join(temp_dir, "segments.txt")
    
    print("\n🎬 QUICK ANIMATION SHOWCASE - Selected Animations")
    print("=" * 60)
    
    # Selected animations to showcase variety (name, class, params, duration)
    animations = [
        # Entry Animation
        ("Fade In", FadeIn, {
            "center_point": (center_x, center_y),
            "fade_speed": 0.15
        }, 2),
        
        ("Slide In", SlideIn, {
            "slide_direction": "left",
            "slide_duration": 25,
            "easing": "ease_out"
        }, 2),
        
        ("Zoom In", ZoomIn, {
            "start_scale": 0.0,
            "end_scale": 1.0,
            "zoom_duration": 25,
            "easing": "bounce"
        }, 2),
        
        ("Bounce", Bounce, {
            "bounce_height": 250,
            "num_bounces": 2,
            "squash_stretch": True
        }, 3),
        
        ("Emergence", EmergenceFromStaticPoint, {
            "direction": 0,
            "emergence_speed": 4.0
        }, 3),
        
        # Distortion
        ("Warp", Warp, {
            "warp_type": "sine",
            "warp_strength": 1.2,
            "warp_frequency": 3
        }, 2),
        
        ("Wave", Wave, {
            "wave_type": "horizontal",
            "amplitude": 25,
            "frequency": 3,
            "speed": 2.0
        }, 2),
        
        # Text
        ("Typewriter", Typewriter, {
            "text": "Hello World!",
            "chars_per_second": 12,
            "show_cursor": True,
            "font_size": 72
        }, 3),
        
        # Special Effects
        ("Glitch", Glitch, {
            "glitch_intensity": 0.8,
            "glitch_frequency": 0.3,
            "color_shift": True,
            "scan_lines": True
        }, 2),
        
        ("Neon Glow", NeonGlow, {
            "glow_color": "#00FFFF",
            "glow_intensity": 1.5,
            "pulse_rate": 1.5,
            "flicker": True
        }, 2),
        
        # Motion
        ("Spin", Spin, {
            "spin_speed": 8,
            "spin_axis": "z",
            "wobble": False
        }, 2),
        
        # 3D Effect
        ("Carousel", Carousel, {
            "num_items": 4,
            "radius": 150,
            "rotation_speed": 3,
            "perspective_scale": True
        }, 3),
    ]
    
    # Add opening card
    opening_path = os.path.join(temp_dir, "opening.mp4")
    create_title_card("Animation Showcase", opening_path, duration=2)
    segments.append(opening_path)
    
    # Process each animation
    for i, (name, animation_class, params, duration) in enumerate(animations):
        print(f"\n🎯 Processing: {name}")
        print("-" * 40)
        
        # Create title card
        title_path = os.path.join(temp_dir, f"title_{i:02d}.mp4")
        create_title_card(name, title_path, duration=1)
        segments.append(title_path)
        
        # Create background for this segment
        bg_path = os.path.join(temp_dir, f"bg_{i:02d}.mp4")
        # Use dark blue background
        create_solid_background("darkblue", duration, bg_path)
        
        # Create animation
        try:
            animation_output = os.path.join(temp_dir, f"anim_{i:02d}.mp4")
            
            animation = animation_class(
                element_path=element_path,
                background_path=bg_path,
                position=(center_x, center_y),
                fps=30,
                duration=duration,
                temp_dir=os.path.join(temp_dir, f"anim_{i:02d}_work"),
                **params
            )
            
            success = animation.render(animation_output)
            
            if success and os.path.exists(animation_output):
                segments.append(animation_output)
                print(f"   ✅ {name} completed")
            else:
                print(f"   ⚠️  {name} failed - skipping")
                
        except Exception as e:
            print(f"   ❌ {name} error: {str(e)[:100]}")
    
    # Add closing card
    closing_path = os.path.join(temp_dir, "closing.mp4")
    create_title_card(f"{len(animations)} Animations Demonstrated", closing_path, duration=2)
    segments.append(closing_path)
    
    print("\n" + "=" * 60)
    print("🎬 Combining all segments into final showcase...")
    
    # Write segment list file for ffmpeg concat
    with open(segment_list_file, 'w') as f:
        for segment in segments:
            if os.path.exists(segment):
                f.write(f"file '{segment}'\n")
    
    # Combine all segments
    final_output = os.path.join(output_dir, "animation_showcase_quick.mp4")
    
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', segment_list_file,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '23',
        '-y',
        final_output
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Final showcase created: {final_output}")
        
        # Get video duration
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            final_output
        ]
        duration_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        duration = float(duration_result.stdout.strip()) if duration_result.stdout else 0
        
        print(f"📊 Total duration: {duration:.1f} seconds")
        print(f"📁 Output saved to: {output_dir}")
        
        # Open the video
        print("\n🎥 Opening showcase video...")
        subprocess.run(['open', final_output])
        
    else:
        print(f"❌ Failed to create final showcase")
        print(f"Error: {result.stderr[:500]}")

if __name__ == "__main__":
    main()