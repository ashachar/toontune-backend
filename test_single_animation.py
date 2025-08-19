#!/usr/bin/env python3
"""
Test a single animation quickly
"""

import os
import sys
import subprocess
from datetime import datetime

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.bounce import Bounce

def main():
    # Setup paths
    element_path = "uploads/assets/videos/do_re_mi.mov"
    
    if not os.path.exists(element_path):
        print(f"Error: {element_path} not found!")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/single_animation_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🎬 Testing Single Animation: Bounce")
    print("=" * 60)
    
    # Create simple blue background
    bg_path = os.path.join(output_dir, "background.mp4")
    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', 'color=c=darkblue:s=1920x1080:d=2:r=30',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',
        bg_path
    ]
    
    print("Creating background...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to create background: {result.stderr}")
        return
    
    print("Background created ✓")
    
    # Create bounce animation
    print("\nCreating bounce animation...")
    output_path = os.path.join(output_dir, "bounce_animation.mp4")
    
    try:
        animation = Bounce(
            element_path=element_path,
            background_path=bg_path,
            position=(960, 540),
            bounce_height=150,
            num_bounces=2,
            bounce_duration=50,
            squash_stretch=True,
            fps=30,
            duration=2,
            temp_dir=os.path.join(output_dir, "temp")
        )
        
        success = animation.render(output_path)
        
        if success and os.path.exists(output_path):
            print(f"✅ Animation created: {output_path}")
            
            # Get file size
            size = os.path.getsize(output_path) / 1024 / 1024
            print(f"📊 File size: {size:.2f} MB")
            
            # Open the video
            print("\n🎥 Opening animation...")
            subprocess.run(['open', output_path])
        else:
            print("❌ Animation failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()