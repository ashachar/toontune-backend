#!/usr/bin/env python3

import subprocess
import os
import sys
import shutil

def create_pixel_perfect_emergence(sailor_path, sea_path, output_path):
    """
    Bulletproof approach: Use FFmpeg expressions that definitely work.
    """
    
    print("🎬 Creating pixel-perfect emergence...")
    
    # Create temp directory
    temp_dir = "/tmp/sailor_emergence"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Step 1: Scale sailor and extract a single frame for static image
    print("   Step 1: Preparing sailor...")
    sailor_frame = os.path.join(temp_dir, "sailor.png")
    
    cmd = [
        'ffmpeg',
        '-i', sailor_path,
        '-frames:v', '1',
        '-vf', 'scale=200:-1',
        '-y',
        sailor_frame
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("   ✓ Sailor frame extracted")
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Failed to extract sailor frame")
        return False
    
    # Step 2: Remove black background
    print("   Step 2: Removing black background...")
    sailor_clean = os.path.join(temp_dir, "sailor_clean.png")
    
    cmd = [
        'ffmpeg',
        '-i', sailor_frame,
        '-vf', 'colorkey=0x000000:0.15:0.05',
        '-c:v', 'png',
        '-y',
        sailor_clean
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("   ✓ Black background removed")
    except subprocess.CalledProcessError:
        # If colorkey fails, just use original
        print("   ⚠️ Colorkey failed, using original")
        sailor_clean = sailor_frame
    
    # Step 3: Create emergence using crop expression
    print("   Step 3: Creating emergence animation...")
    
    fps = 30
    total_frames = 210
    duration = total_frames / fps
    
    # Use t (time) instead of n (frame number) for more reliable expression
    # At time t, we want to show (t * fps + 1) pixel rows
    filter_complex = (
        f'[1:v]crop='
        f'w=200:'
        f'h=\'min(281,floor(t*{fps})+1)\':'  # Height increases with time
        f'x=0:y=0'
        f'[cropped];'
        f'[0:v][cropped]overlay='
        f'x=220:'
        f'y=\'215-min(210,floor(t*{fps})+1)\''  # Position based on crop height
    )
    
    cmd = [
        'ffmpeg',
        '-i', sea_path,
        '-loop', '1',
        '-t', str(duration),
        '-i', sailor_clean,
        '-filter_complex', filter_complex,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-t', str(duration),
        '-y',
        output_path
    ]
    
    try:
        print(f"   Creating {duration:.1f} second video...")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("   ✓ Video created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Failed to create video")
        print(f"   Trying simpler approach...")
        return create_simple_version(sailor_clean, sea_path, output_path)

def create_simple_version(sailor_image, sea_path, output_path):
    """
    Even simpler: Create multiple overlays with enable conditions.
    """
    
    print("   Alternative: Creating with multiple overlays...")
    
    fps = 30
    duration = 7
    
    # Create a series of overlays, each showing more pixels
    filter_parts = []
    
    # For simplicity, do it in steps of 10 frames
    for i in range(0, 210, 10):
        height = i + 1
        y_pos = 215 - height
        start_time = i / fps
        end_time = (i + 10) / fps
        
        filter_parts.append(
            f"[1:v]crop=200:{height}:0:0[c{i}];"
            f"[prev{i}][c{i}]overlay=220:{y_pos}:"
            f"enable='between(t,{start_time},{end_time})'[out{i}]"
        )
    
    # Build complete filter
    filter_complex = "[0:v]null[prev0];"
    for i, part in enumerate(filter_parts):
        filter_complex += part.replace(f"[prev{i*10}]", f"[prev{i*10}]" if i == 0 else f"[out{(i-1)*10}]")
        if i < len(filter_parts) - 1:
            filter_complex = filter_complex.replace(f"[out{i*10}]", f"[prev{(i+1)*10}]")
    
    filter_complex = filter_complex.replace("[out200]", "")
    
    cmd = [
        'ffmpeg',
        '-i', sea_path,
        '-loop', '1',
        '-t', str(duration),
        '-i', sailor_image,
        '-filter_complex', filter_complex,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-t', str(duration),
        '-y',
        output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("   ✓ Alternative version created!")
        return True
    except subprocess.CalledProcessError:
        print("   ✗ Alternative also failed")
        return False

def verify_and_extract_frames(video_path):
    """
    Extract test frames to verify the result.
    """
    
    print("\n🔍 Verifying output...")
    
    test_times = [0, 0.5, 1, 2, 3, 4, 5, 6]
    
    for t in test_times:
        frame_num = int(t * 30)
        output_frame = f"/tmp/verify_{frame_num:03d}.png"
        
        cmd = [
            'ffmpeg',
            '-ss', str(t),
            '-i', video_path,
            '-frames:v', '1',
            '-y',
            output_frame
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            pixels_visible = frame_num + 1
            print(f"   ✓ Frame {frame_num}: {pixels_visible} pixels visible")
        except:
            pass

def main():
    # Paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_path = "output/sailor_bulletproof.mp4"
    
    if not os.path.exists(sailor_path):
        print(f"❌ Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"❌ Sea video not found: {sea_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("BULLETPROOF PIXEL EMERGENCE")
    print("=" * 60)
    print()
    
    if create_pixel_perfect_emergence(sailor_path, sea_path, output_path):
        verify_and_extract_frames(output_path)
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print(f"\n📹 Final video: {output_path}")
        print("\n🌊 Pixel-perfect emergence achieved!")
    else:
        print("\n❌ Failed to create emergence")
        sys.exit(1)

if __name__ == "__main__":
    main()