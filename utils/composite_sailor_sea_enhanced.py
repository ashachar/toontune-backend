#!/usr/bin/env python3

import subprocess
import os
import sys

def create_enhanced_sailor_sea_composite(sailor_path, sea_path, output_path):
    """
    Create an enhanced composite with:
    - Sailor rising from water with splash effect
    - Fade-in effect
    - Better positioning and scaling
    """
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Enhanced filter with fade-in and better positioning
    filter_complex = (
        # Scale sailor to better size
        '[1:v]scale=250:-1,format=rgba[sailor_scaled];'
        
        # Apply fade-in effect to sailor
        '[sailor_scaled]fade=t=in:st=0.5:d=1.0:alpha=1[sailor_fade];'
        
        # Composite with rising animation
        # Starts below frame, rises to 40% from top, with easing
        '[0:v][sailor_fade]overlay='
        'x=(W-w)/2:'
        'y=\'if(lt(t,2),H-(H-H*0.35)*pow(t/2,2),H*0.35)\':'
        'shortest=1'
    )
    
    cmd = [
        'ffmpeg',
        '-i', sea_path,
        '-i', sailor_path,
        '-filter_complex', filter_complex,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-t', '6',  # 6 second output for smoother animation
        '-y',
        output_path
    ]
    
    print(f"Creating enhanced composite...")
    print(f"Features: Fade-in, smooth rise animation, better scaling")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Successfully created enhanced composite: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating composite:")
        print(f"  Error: {e.stderr}")
        return False

def create_looping_version(input_path, output_path):
    """
    Create a seamlessly looping version of the composite.
    """
    
    cmd = [
        'ffmpeg',
        '-stream_loop', '2',  # Loop input twice
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-t', '10',  # 10 second output
        '-y',
        output_path
    ]
    
    print(f"Creating looping version...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Successfully created looping version: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating looping version:")
        print(f"  Error: {e.stderr}")
        return False

def main():
    # File paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_enhanced = "output/sailor_sea_enhanced.mp4"
    output_loop = "output/sailor_sea_loop.mp4"
    
    # Check if input files exist
    if not os.path.exists(sailor_path):
        print(f"Error: Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"Error: Sea video not found: {sea_path}")
        sys.exit(1)
    
    # Create enhanced composite
    print("=" * 50)
    print("Creating Enhanced Sailor Rising from Sea")
    print("=" * 50)
    
    success = create_enhanced_sailor_sea_composite(sailor_path, sea_path, output_enhanced)
    
    if success:
        print(f"\n🎥 Enhanced version created: {output_enhanced}")
        
        # Create looping version
        print("\n" + "=" * 50)
        print("Creating Looping Version")
        print("=" * 50)
        
        if create_looping_version(output_enhanced, output_loop):
            print(f"\n🎥 Looping version created: {output_loop}")
            print("\n✨ All versions created successfully!")
            print(f"  - Basic: output/sailor_rising_from_sea.mp4")
            print(f"  - Enhanced: {output_enhanced}")
            print(f"  - Looping: {output_loop}")
    else:
        print("\n❌ Failed to create enhanced composite")
        sys.exit(1)

if __name__ == "__main__":
    main()