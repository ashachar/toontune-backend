#!/usr/bin/env python3

import subprocess
import os
import sys

def composite_sailor_with_transparency(sailor_path, sea_path, output_path):
    """
    Composite sailor with proper transparency handling.
    The WebM's alpha channel is preserved during overlay.
    """
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Filter with proper alpha channel handling
    # The key is using format=rgba to preserve alpha channel
    filter_complex = (
        # Scale sailor and ensure RGBA format is preserved
        '[1:v]scale=250:-1,format=rgba[sailor_scaled];'
        
        # Apply fade-in to the alpha channel
        '[sailor_scaled]fade=t=in:st=0.5:d=1.0:alpha=1[sailor_fade];'
        
        # Overlay with alpha blending - the key is format=auto which preserves alpha
        '[0:v][sailor_fade]overlay='
        'x=(W-w)/2:'
        'y=\'if(lt(t,2),H-(H-H*0.35)*pow(t/2,2),H*0.35)\':'
        'format=auto:alpha=premultiplied:shortest=1'
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
        '-t', '6',
        '-y',
        output_path
    ]
    
    print(f"Creating composite with proper transparency...")
    print(f"Input sailor: {sailor_path}")
    print(f"Input sea: {sea_path}")
    print(f"Output: {output_path}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Successfully created composite with transparency preserved")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating composite:")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Error: {e.stderr}")
        return False

def composite_sailor_with_chromakey_removal(sailor_path, sea_path, output_path):
    """
    Alternative approach: Convert any black background to transparent during compositing.
    This works if the WebM has a black background instead of true transparency.
    """
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use colorkey to remove black background if present
    filter_complex = (
        # Scale sailor
        '[1:v]scale=250:-1[sailor_scaled];'
        
        # Remove black background using colorkey (if WebM lost transparency)
        '[sailor_scaled]colorkey=0x000000:0.3:0.1[sailor_keyed];'
        
        # Apply fade-in
        '[sailor_keyed]fade=t=in:st=0.5:d=1.0:alpha=1[sailor_fade];'
        
        # Overlay with rising animation
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
        '-t', '6',
        '-y',
        output_path
    ]
    
    print(f"Creating composite with black removal...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Successfully created composite with black background removed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e.stderr}")
        return False

def main():
    # File paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_transparent = "output/sailor_sea_transparent.mp4"
    output_keyed = "output/sailor_sea_keyed.mp4"
    
    # Check if input files exist
    if not os.path.exists(sailor_path):
        print(f"Error: Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"Error: Sea video not found: {sea_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("METHOD 1: Preserving Original WebM Transparency")
    print("=" * 60)
    
    success1 = composite_sailor_with_transparency(sailor_path, sea_path, output_transparent)
    
    print("\n" + "=" * 60)
    print("METHOD 2: Removing Black Background with Colorkey")
    print("=" * 60)
    
    success2 = composite_sailor_with_chromakey_removal(sailor_path, sea_path, output_keyed)
    
    if success1 or success2:
        print("\n✨ Composite videos created:")
        if success1:
            print(f"  - With preserved transparency: {output_transparent}")
        if success2:
            print(f"  - With black removal: {output_keyed}")
        print("\nThe sailor should now appear without the black background!")
    else:
        print("\n❌ Failed to create composites")
        sys.exit(1)

if __name__ == "__main__":
    main()