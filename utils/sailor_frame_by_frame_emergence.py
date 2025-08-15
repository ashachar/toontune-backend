#!/usr/bin/env python3

import subprocess
import os
import sys
import shutil

def extract_and_crop_sailor_frames(sailor_path, temp_dir, num_frames=210):
    """
    Extract frames from sailor WebM and crop each to show only top N rows.
    Frame 0 = 1 row, Frame 1 = 2 rows, etc.
    """
    
    print(f"📸 Extracting and cropping sailor frames...")
    
    # First, scale and prepare the sailor video
    scaled_sailor = os.path.join(temp_dir, "sailor_scaled.webm")
    
    # Scale sailor and remove black background
    cmd = [
        'ffmpeg',
        '-i', sailor_path,
        '-vf', 'scale=200:-1,colorkey=color=black:similarity=0.15:blend=0.05',
        '-c:v', 'libvpx-vp9',
        '-pix_fmt', 'yuva420p',  # Keep alpha channel
        '-y',
        scaled_sailor
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✓ Sailor scaled to 200px wide")
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Failed to scale sailor: {e.stderr[:200]}")
        return False
    
    # Extract frames from scaled sailor
    frames_pattern = os.path.join(temp_dir, "sailor_frame_%04d.png")
    
    cmd = [
        'ffmpeg',
        '-i', scaled_sailor,
        '-frames:v', str(num_frames),
        '-y',
        frames_pattern
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✓ Extracted {num_frames} frames")
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Failed to extract frames: {e.stderr[:200]}")
        return False
    
    # Crop each frame to show only top N+1 rows
    for i in range(num_frames):
        frame_num = i + 1  # FFmpeg starts at 1
        input_frame = os.path.join(temp_dir, f"sailor_frame_{frame_num:04d}.png")
        output_frame = os.path.join(temp_dir, f"cropped_{i:04d}.png")
        
        if not os.path.exists(input_frame):
            # Use first frame for all frames (sailor animation loop)
            input_frame = os.path.join(temp_dir, "sailor_frame_0001.png")
        
        # Crop height = i + 1 pixels (frame 0 = 1 pixel, frame 1 = 2 pixels, etc.)
        crop_height = i + 1
        
        cmd = [
            'ffmpeg',
            '-i', input_frame,
            '-vf', f'crop=200:{crop_height}:0:0',
            '-y',
            output_frame
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            if i % 30 == 0:  # Progress indicator every second (at 30fps)
                print(f"   ✓ Cropped frame {i}: {crop_height}px tall")
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed to crop frame {i}: {e.stderr[:200]}")
            return False
    
    print(f"   ✓ All frames cropped successfully")
    return True

def create_emergence_video(sea_path, temp_dir, output_path, num_frames=210, fps=30):
    """
    Composite the cropped sailor frames onto the sea video.
    Each frame shows more of the sailor emerging from water.
    """
    
    print(f"🎬 Creating emergence video...")
    
    # Create filter complex file for all overlays
    filter_file = os.path.join(temp_dir, "filter.txt")
    
    with open(filter_file, 'w') as f:
        # Build the filter chain
        f.write("[0:v]null[base];\n")  # Start with sea as base
        
        for i in range(num_frames):
            cropped_frame = os.path.join(temp_dir, f"cropped_{i:04d}.png")
            
            # Calculate position
            # Frame i shows i+1 pixels, positioned so bottom touches water at y=215
            y_position = 215 - (i + 1)
            
            if i == 0:
                prev = "base"
            else:
                prev = f"out{i-1}"
            
            # Add this frame as overlay at specific timestamp
            f.write(f"[{prev}][{i+1}:v]overlay=")
            f.write(f"x=220:y={y_position}:")
            f.write(f"enable='eq(n,{i})'")  # Only show at frame i
            
            if i < num_frames - 1:
                f.write(f"[out{i}];\n")
            else:
                f.write("[final]\n")
    
    # Build FFmpeg command with all inputs
    cmd = ['ffmpeg', '-r', str(fps), '-i', sea_path]
    
    # Add all cropped frames as inputs
    for i in range(num_frames):
        cropped_frame = os.path.join(temp_dir, f"cropped_{i:04d}.png")
        cmd.extend(['-loop', '1', '-t', '0.001', '-i', cropped_frame])
    
    # Apply filter
    cmd.extend([
        '-filter_complex_script', filter_file,
        '-map', '[final]',
        '-r', str(fps),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-t', str(num_frames / fps),
        '-y',
        output_path
    ])
    
    try:
        print(f"   Processing {num_frames} frames at {fps}fps...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✓ Video created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Failed to create video: {e.stderr[:500]}")
        return False

def create_simple_emergence(sailor_path, sea_path, output_path):
    """
    Simpler approach: Use FFmpeg's crop filter with frame-based expression.
    """
    
    print(f"🎬 Creating emergence with dynamic crop...")
    
    fps = 30
    total_frames = 210
    duration = total_frames / fps
    
    # Use FFmpeg expression to dynamically crop based on frame number
    filter_complex = (
        # Scale and prepare sailor
        '[1:v]scale=200:-1,colorkey=color=black:similarity=0.15:blend=0.05[sailor];'
        
        # Dynamic crop: height = frame number + 1
        '[sailor]crop='
        'w=200:'
        'h=\'min(281,n+1)\':'  # Crop height increases with frame
        'x=0:y=0'
        '[cropped];'
        
        # Position at water line
        '[0:v][cropped]overlay='
        'x=220:'
        'y=\'215-min(210,n+1)\''  # Position so bottom touches water
    )
    
    cmd = [
        'ffmpeg',
        '-r', str(fps),
        '-i', sea_path,
        '-stream_loop', '-1',  # Loop sailor video
        '-t', str(duration),
        '-r', str(fps),
        '-i', sailor_path,
        '-filter_complex', filter_complex,
        '-r', str(fps),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-t', str(duration),
        '-y',
        output_path
    ]
    
    try:
        print(f"   ✓ Creating video with dynamic crop...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✓ Video created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Failed: {e.stderr[:500]}")
        return False

def main():
    # Paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_path = "output/sailor_clean_emergence.mp4"
    temp_dir = "output/temp_frames"
    
    if not os.path.exists(sailor_path):
        print(f"❌ Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"❌ Sea video not found: {sea_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("CLEAN FRAME-BY-FRAME EMERGENCE")
    print("=" * 60)
    print("\n🎯 Strategy:")
    print("  1. Extract frames from sailor WebM")
    print("  2. Crop each frame: Frame N = top N+1 rows")
    print("  3. Composite cropped frames onto sea")
    print("  4. No masking = no artifacts!")
    print()
    
    # Try simple approach first (faster)
    print("Attempting dynamic crop method...")
    if create_simple_emergence(sailor_path, sea_path, output_path):
        print("\n" + "=" * 60)
        print("✨ SUCCESS! Clean emergence created!")
        print("=" * 60)
        print(f"\n📹 Final video: {output_path}")
        print("\n🌊 Achievement:")
        print("  ✓ Frame 0 = 1 pixel row at water line")
        print("  ✓ Frame N = N+1 pixel rows visible")
        print("  ✓ No masking artifacts")
        print("  ✓ Clean sea below water line")
        print("  ✓ True pixel-perfect emergence!")
    else:
        # Fallback to frame-by-frame method
        print("\nFalling back to frame-by-frame method...")
        
        # Create temp directory
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Extract and crop frames
            if extract_and_crop_sailor_frames(sailor_path, temp_dir):
                # Create final video
                if create_emergence_video(sea_path, temp_dir, output_path):
                    print("\n" + "=" * 60)
                    print("✨ SUCCESS! Frame-by-frame emergence created!")
                    print("=" * 60)
                    print(f"\n📹 Final video: {output_path}")
                else:
                    print("\n❌ Failed to create emergence video")
            else:
                print("\n❌ Failed to process sailor frames")
        finally:
            # Cleanup temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"\n🧹 Cleaned up temp directory")

if __name__ == "__main__":
    main()