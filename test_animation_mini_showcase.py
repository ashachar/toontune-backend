#!/usr/bin/env python3
"""
Mini Animation Showcase - 5 Key Animations
Quick demonstration of animation variety
"""

import os
import sys
import subprocess
from datetime import datetime

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import selected animation classes
from utils.animations.fade_in import FadeIn
from utils.animations.bounce import Bounce
from utils.animations.wave import Wave
from utils.animations.glitch import Glitch
from utils.animations.spin import Spin

def create_solid_background(duration, output_path):
    """Create a dark blue background video"""
    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'color=c=darkblue:s=1920x1080:d={duration}:r=30',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return output_path

def add_text_overlay(input_video, text, output_path):
    """Add text overlay to video"""
    cmd = [
        'ffmpeg',
        '-i', input_video,
        '-vf', f"drawtext=text='{text}':fontcolor=white:fontsize=36:x=50:y=50:box=1:boxcolor=black@0.5:boxborderw=5",
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

def main():
    # Setup paths
    element_path = "uploads/assets/videos/do_re_mi.mov"
    
    # Check if element exists
    if not os.path.exists(element_path):
        print(f"Error: {element_path} not found!")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/animation_mini_showcase_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Center position
    center_x, center_y = 960, 540
    
    print("\n🎬 MINI ANIMATION SHOWCASE - 5 Key Animations")
    print("=" * 60)
    
    # List to store video segments
    segments = []
    
    # 5 key animations with 2 second duration each
    animations = [
        ("1. Fade In", FadeIn, {
            "center_point": (center_x, center_y),
            "fade_speed": 0.2
        }),
        
        ("2. Bounce", Bounce, {
            "bounce_height": 200,
            "num_bounces": 2,
            "squash_stretch": True,
            "bounce_duration": 50
        }),
        
        ("3. Wave", Wave, {
            "wave_type": "horizontal",
            "amplitude": 30,
            "frequency": 3,
            "speed": 3.0
        }),
        
        ("4. Glitch", Glitch, {
            "glitch_intensity": 0.8,
            "glitch_frequency": 0.5,
            "color_shift": True
        }),
        
        ("5. Spin", Spin, {
            "spin_speed": 10,
            "spin_axis": "z"
        }),
    ]
    
    # Process each animation
    for i, (name, animation_class, params) in enumerate(animations):
        print(f"\n🎯 Processing: {name}")
        
        # Create 2-second background
        bg_path = os.path.join(temp_dir, f"bg_{i}.mp4")
        create_solid_background(2, bg_path)
        
        # Create animation
        try:
            animation_output = os.path.join(temp_dir, f"anim_{i}.mp4")
            
            animation = animation_class(
                element_path=element_path,
                background_path=bg_path,
                position=(center_x, center_y),
                fps=30,
                duration=2,
                temp_dir=os.path.join(temp_dir, f"work_{i}"),
                **params
            )
            
            success = animation.render(animation_output)
            
            if success and os.path.exists(animation_output):
                # Add text label
                labeled_output = os.path.join(temp_dir, f"labeled_{i}.mp4")
                add_text_overlay(animation_output, name, labeled_output)
                segments.append(labeled_output)
                print(f"   ✅ {name} completed")
            else:
                print(f"   ⚠️  {name} failed")
                
        except Exception as e:
            print(f"   ❌ {name} error: {str(e)[:100]}")
    
    if len(segments) == 0:
        print("\n❌ No animations succeeded!")
        return
    
    print("\n" + "=" * 60)
    print("🎬 Combining segments...")
    
    # Create segment list file
    segment_list_file = os.path.join(temp_dir, "segments.txt")
    with open(segment_list_file, 'w') as f:
        for segment in segments:
            f.write(f"file '{segment}'\n")
    
    # Combine all segments
    final_output = os.path.join(output_dir, "mini_showcase.mp4")
    
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', segment_list_file,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',
        final_output
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Showcase created: {final_output}")
        print(f"📁 Total animations: {len(segments)}")
        print(f"⏱️  Duration: ~{len(segments) * 2} seconds")
        
        # Open the video
        print("\n🎥 Opening showcase video...")
        subprocess.run(['open', final_output])
        
        # Update todos
        print("\n✅ Mini showcase complete!")
        
    else:
        print(f"❌ Failed to create showcase")
        print(f"Error: {result.stderr[:500]}")

if __name__ == "__main__":
    main()