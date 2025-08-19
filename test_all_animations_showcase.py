#!/usr/bin/env python3
"""
Comprehensive Animation Showcase Test
Shows all animations one after another using do_re_mi.mov
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all animation classes
from utils.animations.fade_in import FadeIn
from utils.animations.fade_out import FadeOut
from utils.animations.slide_in import SlideIn
from utils.animations.slide_out import SlideOut
from utils.animations.zoom_in import ZoomIn
from utils.animations.zoom_out import ZoomOut
from utils.animations.bounce import Bounce
from utils.animations.emergence_from_static_point import EmergenceFromStaticPoint
from utils.animations.submerge_to_static_point import SubmergeToStaticPoint
from utils.animations.skew import Skew
from utils.animations.stretch_squash import StretchSquash
from utils.animations.warp import Warp
from utils.animations.wave import Wave
from utils.animations.typewriter import Typewriter
from utils.animations.word_buildup import WordBuildup
from utils.animations.split_text import SplitText
from utils.animations.glitch import Glitch
from utils.animations.shatter import Shatter
from utils.animations.neon_glow import NeonGlow
from utils.animations.lens_flare import LensFlare
from utils.animations.flip import Flip
from utils.animations.spin import Spin
from utils.animations.roll import Roll
from utils.animations.carousel import Carousel
from utils.animations.depth_zoom import DepthZoom

def create_title_card(text, output_path, duration=2):
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
    output_dir = f"output/animation_showcase_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory for segments
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Center position for all animations
    center_x, center_y = 960, 540
    
    # List to store all video segments
    segments = []
    segment_list_file = os.path.join(temp_dir, "segments.txt")
    
    print("\n🎬 ANIMATION SHOWCASE - Testing All Animations")
    print("=" * 60)
    
    # Animation configurations (name, class, params, duration)
    animations = [
        # Entry/Exit Animations
        ("1. Fade In", FadeIn, {
            "center_point": (center_x, center_y),
            "fade_speed": 0.1
        }, 3),
        
        ("2. Fade Out", FadeOut, {
            "center_point": (center_x, center_y),
            "fade_speed": 0.08,
            "fade_start_frame": 30
        }, 3),
        
        ("3. Slide In", SlideIn, {
            "slide_direction": "left",
            "slide_duration": 30,
            "easing": "ease_out"
        }, 3),
        
        ("4. Slide Out", SlideOut, {
            "slide_direction": "right",
            "slide_duration": 30,
            "slide_start_frame": 30
        }, 3),
        
        ("5. Zoom In", ZoomIn, {
            "start_scale": 0.0,
            "end_scale": 1.0,
            "zoom_duration": 30,
            "easing": "bounce"
        }, 3),
        
        ("6. Zoom Out", ZoomOut, {
            "start_scale": 1.0,
            "end_scale": 0.0,
            "zoom_duration": 30,
            "zoom_start_frame": 30
        }, 3),
        
        ("7. Bounce", Bounce, {
            "bounce_height": 300,
            "num_bounces": 3,
            "squash_stretch": True
        }, 4),
        
        ("8. Emergence", EmergenceFromStaticPoint, {
            "direction": 0,
            "emergence_speed": 3.0
        }, 4),
        
        ("9. Submerge", SubmergeToStaticPoint, {
            "direction": 180,
            "submerge_speed": 3.0,
            "submerge_start_frame": 30
        }, 4),
        
        # Distortion Effects
        ("10. Skew", Skew, {
            "skew_x": 20,
            "skew_y": -10,
            "oscillate": True
        }, 3),
        
        ("11. Stretch/Squash", StretchSquash, {
            "stretch_x": 1.5,
            "stretch_y": 0.7,
            "oscillate": True,
            "preserve_volume": True
        }, 3),
        
        ("12. Warp", Warp, {
            "warp_type": "sine",
            "warp_strength": 1.0,
            "warp_frequency": 3
        }, 3),
        
        ("13. Wave", Wave, {
            "wave_type": "horizontal",
            "amplitude": 30,
            "frequency": 3,
            "speed": 2.0
        }, 3),
        
        # Text Dynamics (using text overlays)
        ("14. Typewriter", Typewriter, {
            "text": "Hello Animation World!",
            "chars_per_second": 10,
            "show_cursor": True,
            "font_size": 60
        }, 4),
        
        ("15. Word Build-up", WordBuildup, {
            "text": "Amazing Visual Effects",
            "buildup_mode": "fade",
            "word_delay": 10,
            "font_size": 60
        }, 4),
        
        ("16. Split Text", SplitText, {
            "text": "SPLIT APART",
            "split_mode": "word",
            "split_direction": "explode",
            "font_size": 72
        }, 4),
        
        # Special Effects
        ("17. Glitch", Glitch, {
            "glitch_intensity": 0.7,
            "glitch_frequency": 0.3,
            "color_shift": True,
            "scan_lines": True
        }, 3),
        
        ("18. Shatter", Shatter, {
            "num_pieces": 25,
            "shatter_point": (center_x, center_y),
            "explosion_force": 10,
            "shatter_start_frame": 30
        }, 4),
        
        ("19. Neon Glow", NeonGlow, {
            "glow_color": "#00FFFF",
            "glow_intensity": 1.5,
            "pulse_rate": 1.0,
            "flicker": True
        }, 3),
        
        ("20. Lens Flare", LensFlare, {
            "flare_type": "anamorphic",
            "flare_position": (300, 200),
            "flare_intensity": 1.2,
            "movement": True,
            "rainbow_effect": True
        }, 3),
        
        # Motion Effects
        ("21. Flip", Flip, {
            "flip_axis": "horizontal",
            "flip_duration": 30,
            "perspective": 500
        }, 3),
        
        ("22. Spin", Spin, {
            "spin_speed": 6,
            "spin_axis": "z",
            "wobble": True
        }, 3),
        
        ("23. Roll", Roll, {
            "roll_direction": "right",
            "roll_distance": 600,
            "bounce_on_land": True
        }, 4),
        
        # 3D Effects
        ("24. Carousel", Carousel, {
            "num_items": 6,
            "radius": 200,
            "rotation_speed": 2,
            "perspective_scale": True
        }, 4),
        
        ("25. Depth Zoom", DepthZoom, {
            "zoom_type": "fly_through",
            "start_depth": -5,
            "end_depth": 5,
            "depth_blur": True,
            "fog_effect": True
        }, 4),
    ]
    
    # Process each animation
    for i, (name, animation_class, params, duration) in enumerate(animations):
        print(f"\n🎯 Processing: {name}")
        print("-" * 40)
        
        # Create title card
        title_path = os.path.join(temp_dir, f"title_{i:02d}.mp4")
        create_title_card(name, title_path, duration=1.5)
        segments.append(title_path)
        
        # Create background for this segment
        bg_path = os.path.join(temp_dir, f"bg_{i:02d}.mp4")
        # Alternate background colors for variety
        bg_color = "darkblue" if i % 2 == 0 else "darkgreen"
        create_solid_background(bg_color, duration, bg_path)
        
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
                print(f"   ⚠️  {name} failed - using placeholder")
                # Create placeholder
                placeholder = os.path.join(temp_dir, f"placeholder_{i:02d}.mp4")
                create_title_card(f"{name}\n(Failed)", placeholder, duration=duration)
                segments.append(placeholder)
                
        except Exception as e:
            print(f"   ❌ {name} error: {str(e)}")
            # Create error placeholder
            error_card = os.path.join(temp_dir, f"error_{i:02d}.mp4")
            create_title_card(f"{name}\n(Error)", error_card, duration=duration)
            segments.append(error_card)
    
    # Add closing card
    closing_path = os.path.join(temp_dir, "closing.mp4")
    create_title_card("Animation Showcase Complete!\n25 Animations Demonstrated", closing_path, duration=3)
    segments.append(closing_path)
    
    print("\n" + "=" * 60)
    print("🎬 Combining all segments into final showcase...")
    
    # Write segment list file for ffmpeg concat
    with open(segment_list_file, 'w') as f:
        for segment in segments:
            f.write(f"file '{segment}'\n")
    
    # Combine all segments
    final_output = os.path.join(output_dir, "animation_showcase_complete.mp4")
    
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
        
        # Clean up temp directory (optional)
        # shutil.rmtree(temp_dir)
        
    else:
        print(f"❌ Failed to create final showcase")
        print(f"Error: {result.stderr}")

if __name__ == "__main__":
    main()