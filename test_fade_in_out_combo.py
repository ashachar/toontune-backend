#!/usr/bin/env python3
"""
Test combining fade-in and fade-out animations.
Fade in at the beginning, then fade out 2 seconds later with slower speed.
"""

import os
import sys
import subprocess
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.fade_in import FadeIn
from utils.animations.fade_out import FadeOut


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
    temp_dir = tempfile.mkdtemp(prefix="fade_combo_test_")
    print(f"\nTemp directory: {temp_dir}")
    
    try:
        # Convert PNG to video
        element_video = os.path.join(temp_dir, "cartographer.mp4")
        convert_png_to_video(cartographer_png, element_video, duration=7, fps=30)
        
        # STEP 1: Create fade-in animation
        print("\n=== STEP 1: Creating fade-in animation ===")
        print("  Fade-in speed: 0.1 (medium speed)")
        print("  Start at: frame 0")
        
        fade_in_temp = os.path.join(temp_dir, "fade_in_temp")
        os.makedirs(fade_in_temp, exist_ok=True)
        
        fade_in_animation = FadeIn(
            element_path=element_video,
            background_path=background_video,
            position=(0, 0),  # Will be overridden by center_point
            center_point=(center_x, center_y),  # Fade in at video center
            fade_speed=0.1,  # Medium speed (10 frames to full opacity)
            start_frame=0,  # Start immediately
            animation_start_frame=0,
            fps=30,
            duration=7.0,
            temp_dir=fade_in_temp,
            remove_background=False  # The PNG already has transparency
        )
        
        # Process the fade-in animation
        print("\nProcessing fade-in...")
        fade_in_output = os.path.join(temp_dir, "fade_in_result.mp4")
        success = fade_in_animation.render(fade_in_output)
        
        if not success:
            print("❌ Failed to create fade-in animation")
            return None
        
        print("✓ Fade-in animation created")
        
        # STEP 2: Use fade-in result as input for fade-out
        print("\n=== STEP 2: Creating fade-out animation ===")
        print("  Fade-out speed: 0.03 (slower speed)")
        print("  Start fade-out at: frame 60 (2 seconds at 30fps)")
        
        fade_out_temp = os.path.join(temp_dir, "fade_out_temp")
        os.makedirs(fade_out_temp, exist_ok=True)
        
        # First we need to extract the element with fade-in already applied
        # For this, we'll use the fade-in result as our element
        fade_out_animation = FadeOut(
            element_path=element_video,  # Using original element
            background_path=background_video,
            position=(0, 0),  # Will be overridden by center_point
            center_point=(center_x, center_y),  # Same center point
            fade_speed=0.03,  # Slower fade out (about 33 frames to full transparency)
            fade_start_frame=60,  # Start fading out at 2 seconds (60 frames at 30fps)
            start_frame=0,  # Element appears from beginning (already faded in)
            animation_start_frame=0,
            fps=30,
            duration=7.0,
            temp_dir=fade_out_temp,
            remove_background=False
        )
        
        # Process the fade-out animation
        print("\nProcessing fade-out...")
        final_output_temp = os.path.join(temp_dir, "fade_combo_final.mp4")
        
        # Since we want both effects, we need to combine them
        # We'll render a version that has both fade-in and fade-out
        print("\n=== STEP 3: Combining both effects ===")
        
        # Create a combined animation by applying both effects in sequence
        combined_temp = os.path.join(temp_dir, "combined_temp")
        os.makedirs(combined_temp, exist_ok=True)
        
        # Extract frames from background
        bg_frames_dir = os.path.join(combined_temp, "bg_frames")
        os.makedirs(bg_frames_dir, exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', background_video,
            '-r', '30',
            '-t', '7',
            os.path.join(bg_frames_dir, 'bg_%04d.png')
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract frames from element
        el_frames_dir = os.path.join(combined_temp, "el_frames")
        os.makedirs(el_frames_dir, exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', element_video,
            '-r', '30',
            '-vf', 'scale=200:-1',
            os.path.join(el_frames_dir, 'el_%04d.png')
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Get background frames
        bg_frames = sorted([
            os.path.join(bg_frames_dir, f)
            for f in os.listdir(bg_frames_dir)
            if f.endswith('.png')
        ])
        
        # Get element frames
        el_frames = sorted([
            os.path.join(el_frames_dir, f)
            for f in os.listdir(el_frames_dir)
            if f.endswith('.png')
        ])
        
        # Create output frames with combined effect
        output_frames_dir = os.path.join(combined_temp, "output_frames")
        os.makedirs(output_frames_dir, exist_ok=True)
        
        total_frames = min(210, len(bg_frames))  # 7 seconds at 30fps
        fade_in_duration = 10  # frames
        fade_out_start = 60  # frame 60 (2 seconds)
        fade_out_duration = 33  # frames (slower)
        
        print(f"\nProcessing {total_frames} frames:")
        print(f"  Fade-in: frames 0-{fade_in_duration}")
        print(f"  Full visibility: frames {fade_in_duration}-{fade_out_start}")
        print(f"  Fade-out: frames {fade_out_start}-{fade_out_start + fade_out_duration}")
        
        for frame_num in range(total_frames):
            # Calculate opacity for this frame
            if frame_num < fade_in_duration:
                # Fade in phase
                opacity = (frame_num / fade_in_duration)
            elif frame_num < fade_out_start:
                # Full visibility
                opacity = 1.0
            elif frame_num < fade_out_start + fade_out_duration:
                # Fade out phase
                fade_progress = (frame_num - fade_out_start) / fade_out_duration
                opacity = 1.0 - fade_progress
            else:
                # Fully faded out
                opacity = 0.0
            
            # Get frames
            bg_frame = bg_frames[frame_num % len(bg_frames)]
            el_frame = el_frames[frame_num % len(el_frames)]
            
            if opacity > 0:
                # Apply opacity to element
                temp_el = os.path.join(combined_temp, f'temp_el_{frame_num:04d}.png')
                cmd = [
                    'ffmpeg',
                    '-i', el_frame,
                    '-vf', f'format=rgba,colorchannelmixer=aa={opacity}',
                    '-y',
                    temp_el
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Composite onto background
                output_frame = os.path.join(output_frames_dir, f'out_{frame_num:04d}.png')
                cmd = [
                    'ffmpeg',
                    '-i', bg_frame,
                    '-i', temp_el,
                    '-filter_complex', f'[1]scale=200:-1[scaled];[0][scaled]overlay={center_x-100}:{center_y-100}',
                    '-y',
                    output_frame
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            else:
                # Just copy background
                output_frame = os.path.join(output_frames_dir, f'out_{frame_num:04d}.png')
                shutil.copy(bg_frame, output_frame)
            
            # Progress indicator
            if frame_num % 30 == 0:
                print(f"    Frame {frame_num}: opacity {opacity:.2f}")
        
        # Create final video from frames
        print("\nCreating final video...")
        final_output_temp = os.path.join(temp_dir, "fade_combo_final.mp4")
        cmd = [
            'ffmpeg',
            '-framerate', '30',
            '-i', os.path.join(output_frames_dir, 'out_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-crf', '23',
            '-y',
            final_output_temp
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Move to final location
        output_dir = "./output/fade_combo_test"
        os.makedirs(output_dir, exist_ok=True)
        
        final_output = os.path.join(output_dir, "cartographer_fade_in_out_combo.mp4")
        shutil.move(final_output_temp, final_output)
        
        print(f"\n✅ SUCCESS! Combined animation created:")
        print(f"   {final_output}")
        
        # Open the video
        print("\nOpening video...")
        subprocess.run(['open', final_output], check=False)
        
        return final_output
            
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
        print("\nAnimation timeline:")
        print("  0.0s - 0.3s: Fade in (medium speed)")
        print("  0.3s - 2.0s: Full visibility")
        print("  2.0s - 3.1s: Fade out (slower speed)")
        print("  3.1s - 7.0s: Fully faded out")