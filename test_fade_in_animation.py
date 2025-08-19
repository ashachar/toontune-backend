#!/usr/bin/env python3
"""
Test fade-in animation with sea movie and Cartographer image.
"""

import os
import sys
import subprocess
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.fade_in import FadeIn


def convert_png_to_video(png_path, output_path, duration=7, fps=30):
    """Convert static PNG to a video with specified duration."""
    cmd = [
        'ffmpeg',
        '-loop', '1',
        '-i', png_path,
        '-c:v', 'libx264',
        '-t', str(duration),
        '-pix_fmt', 'yuv420p',
        '-vf', 'fps=' + str(fps),
        '-y',
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(f"✓ Converted PNG to video: {output_path}")


def get_video_dimensions(video_path):
    """Get video dimensions using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    width, height = map(int, result.stdout.strip().split('x'))
    return width, height


def main():
    # File paths
    background_video = "./uploads/assets/videos/sea_movie.mov"
    cartographer_png = "./uploads/assets/batch_images_transparent_bg/Cartographer at Work on Map.png"
    
    # Check if files exist
    if not os.path.exists(background_video):
        print(f"Error: Background video not found: {background_video}")
        return
    
    if not os.path.exists(cartographer_png):
        print(f"Error: Cartographer image not found: {cartographer_png}")
        return
    
    print("Files found:")
    print(f"  Background: {background_video}")
    print(f"  Element: {cartographer_png}")
    
    # Get video dimensions for center calculation
    width, height = get_video_dimensions(background_video)
    center_x = width // 2
    center_y = height // 2
    print(f"  Video dimensions: {width}x{height}")
    print(f"  Center point: ({center_x}, {center_y})")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="fade_in_test_")
    print(f"\nTemp directory: {temp_dir}")
    
    try:
        # Convert PNG to video
        element_video = os.path.join(temp_dir, "cartographer.mp4")
        convert_png_to_video(cartographer_png, element_video, duration=7, fps=30)
        
        # Create fade-in animation
        print("\nCreating fade-in animation...")
        print("  Fade speed: 0.15 (fast but not fastest)")
        
        animation = FadeIn(
            element_path=element_video,
            background_path=background_video,
            position=(0, 0),  # Will be overridden by center_point
            center_point=(center_x, center_y),  # Fade in at video center
            fade_speed=0.15,  # Fast but not fastest (about 7 frames to full opacity)
            start_frame=15,  # Start fading in at frame 15 (0.5 seconds)
            animation_start_frame=15,  # Start animation at same time
            fps=30,
            duration=7.0,
            temp_dir=temp_dir,
            remove_background=False  # The PNG already has transparency
        )
        
        # Process the animation
        print("\nProcessing animation...")
        temp_output = os.path.join(temp_dir, "fade_in_output.mp4")
        success = animation.render(temp_output)
        output_path = temp_output if success else None
        
        if output_path and os.path.exists(output_path):
            # Move to a more permanent location
            output_dir = "./output/fade_in_test"
            os.makedirs(output_dir, exist_ok=True)
            
            final_output = os.path.join(output_dir, "cartographer_fade_in_sea.mp4")
            shutil.move(output_path, final_output)
            
            print(f"\n✅ SUCCESS! Animation created:")
            print(f"   {final_output}")
            
            # Open the video
            print("\nOpening video...")
            subprocess.run(['open', final_output], check=False)
            
            return final_output
        else:
            print("\n❌ Failed to create animation")
            return None
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Clean up temp directory (optional - comment out to keep for debugging)
        # if os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)
        pass


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n🎬 Animation saved to: {result}")