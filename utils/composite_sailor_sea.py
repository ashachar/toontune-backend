#!/usr/bin/env python3

import subprocess
import os
import sys

def composite_sailor_rising_from_sea(sailor_path, sea_path, output_path):
    """
    Composite a sailor animation rising from the middle of the sea.
    
    The sailor will:
    1. Start below the frame (hidden)
    2. Rise up from the middle of the sea
    3. Complete the salute animation while visible
    """
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # FFmpeg command to composite the sailor rising from the sea
    # The sailor animation is overlaid on the sea video with:
    # - Position: centered horizontally, animated vertically
    # - Scale: adjusted to fit nicely in the scene
    # - Timing: rises up during first 1.5 seconds, then plays animation
    
    # Build the filter complex string separately for better control
    filter_complex = (
        '[1:v]scale=200:-1[sailor];'
        '[0:v][sailor]overlay='
        'x=(W-w)/2:'
        'y=\'if(lt(t,1.5),H+50-((H+50-H*0.4)*t/1.5),H*0.4)\''
    )
    
    cmd = [
        'ffmpeg',
        '-i', sea_path,  # Background sea video
        '-i', sailor_path,  # Sailor with transparent background
        '-filter_complex', filter_complex,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-t', '5',  # 5 second output
        '-y',  # Overwrite output
        output_path
    ]
    
    print(f"Compositing sailor rising from sea...")
    print(f"Input sailor: {sailor_path}")
    print(f"Input sea: {sea_path}")
    print(f"Output: {output_path}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Successfully created composite video: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating composite video:")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Error: {e.stderr}")
        return False

def main():
    # File paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_path = "output/sailor_rising_from_sea.mp4"
    
    # Check if input files exist
    if not os.path.exists(sailor_path):
        print(f"Error: Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"Error: Sea video not found: {sea_path}")
        sys.exit(1)
    
    # Create the composite
    success = composite_sailor_rising_from_sea(sailor_path, sea_path, output_path)
    
    if success:
        print(f"\n🎥 Final video created: {output_path}")
        print("The sailor rises from the middle of the sea with a salute animation!")
    else:
        print("\n❌ Failed to create composite video")
        sys.exit(1)

if __name__ == "__main__":
    main()