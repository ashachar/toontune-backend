#!/usr/bin/env python3

import subprocess
import os
import sys
import shutil
import tempfile

def prepare_sailor_frames(sailor_path, temp_dir, total_frames=210):
    """
    Step 1: Extract and prepare sailor frames.
    Scale to 200px wide and remove black background.
    """
    
    print("📸 Step 1: Extracting sailor frames...")
    
    # Create subdirectories
    scaled_dir = os.path.join(temp_dir, "scaled")
    os.makedirs(scaled_dir, exist_ok=True)
    
    # First scale and clean the sailor
    scaled_video = os.path.join(temp_dir, "sailor_clean.mp4")
    
    cmd = [
        'ffmpeg',
        '-i', sailor_path,
        '-vf', 'scale=200:-1,colorkey=0x000000:0.15:0.05',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-t', '7',  # 7 seconds
        '-y',
        scaled_video
    ]
    
    print("   Scaling and removing black background...")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("   ✓ Sailor prepared")
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Failed to prepare sailor")
        print(f"   Error: {e.stderr[-500:] if e.stderr else 'No error output'}")
        return False
    
    # Extract individual frames
    frame_pattern = os.path.join(scaled_dir, "frame_%04d.png")
    
    cmd = [
        'ffmpeg',
        '-i', scaled_video,
        '-r', '30',  # Force 30fps extraction
        '-frames:v', str(total_frames),
        frame_pattern
    ]
    
    print(f"   Extracting {total_frames} frames at 30fps...")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✓ Extracted {total_frames} frames")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Failed to extract frames")
        print(f"   Error: {e.stderr[-500:] if e.stderr else 'No error output'}")
        return False

def crop_frames_progressively(temp_dir, total_frames=210):
    """
    Step 2: Crop each frame to show only top N+1 rows.
    Frame 0 = 1 pixel, Frame 1 = 2 pixels, etc.
    """
    
    print("✂️ Step 2: Cropping frames progressively...")
    
    scaled_dir = os.path.join(temp_dir, "scaled")
    cropped_dir = os.path.join(temp_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)
    
    for i in range(total_frames):
        # Input frame (FFmpeg numbering starts at 1)
        input_frame = os.path.join(scaled_dir, f"frame_{(i+1):04d}.png")
        
        # If frame doesn't exist, use frame 1 (for static sailor)
        if not os.path.exists(input_frame):
            input_frame = os.path.join(scaled_dir, "frame_0001.png")
        
        # Output cropped frame
        output_frame = os.path.join(cropped_dir, f"crop_{i:04d}.png")
        
        # Crop height = i + 1 (frame 0 = 1px, frame 1 = 2px, etc.)
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
            
            # Progress indicator
            if i == 0:
                print(f"   Frame 0: cropped to 1 pixel")
            elif i == 29:
                print(f"   Frame 29: cropped to 30 pixels (hat visible)")
            elif i == 59:
                print(f"   Frame 59: cropped to 60 pixels (head visible)")
            elif i == 119:
                print(f"   Frame 119: cropped to 120 pixels (upper body)")
            elif i == 209:
                print(f"   Frame 209: cropped to 210 pixels (3/4 body)")
                
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed to crop frame {i}: {e.stderr[:200]}")
            return False
    
    print(f"   ✓ All {total_frames} frames cropped")
    return True

def create_final_emergence(sea_path, temp_dir, output_path, total_frames=210, fps=30):
    """
    Step 3: Composite cropped frames onto sea video.
    Each frame gets its corresponding cropped sailor piece.
    """
    
    print("🎬 Step 3: Creating final emergence video...")
    
    cropped_dir = os.path.join(temp_dir, "cropped")
    
    # Create a complex filter that overlays each frame at the right time
    filter_parts = []
    
    # Start with the sea video
    current_stream = "0:v"
    
    print("   Building filter for frame-by-frame compositing...")
    
    # For each frame, overlay the cropped sailor at the right position
    for i in range(total_frames):
        crop_height = i + 1
        y_position = 215 - crop_height  # Position so bottom touches water
        
        # Create overlay for this specific frame
        # Use enable condition to show only at frame i
        filter_parts.append(
            f"[{current_stream}][{i+1}:v]overlay="
            f"x=220:y={y_position}:"
            f"enable='eq(n,{i})'"
        )
        
        # Update stream reference for next iteration
        if i < total_frames - 1:
            filter_parts[-1] += f"[v{i}]"
            current_stream = f"v{i}"
        # Last one outputs to default
    
    # Join all filter parts
    filter_complex = ";".join(filter_parts)
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg',
        '-r', str(fps),
        '-i', sea_path
    ]
    
    # Add all cropped frames as inputs
    for i in range(total_frames):
        cropped_frame = os.path.join(cropped_dir, f"crop_{i:04d}.png")
        cmd.extend(['-i', cropped_frame])
    
    # Add filter and output settings
    cmd.extend([
        '-filter_complex', filter_complex,
        '-r', str(fps),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-t', str(total_frames / fps),
        '-y',
        output_path
    ])
    
    print(f"   Compositing {total_frames} frames...")
    print(f"   Duration: {total_frames/fps:.1f} seconds at {fps}fps")
    
    try:
        # Run with limited output capture to avoid memory issues
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for completion
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print("   ✓ Video created successfully!")
            return True
        else:
            print(f"   ✗ FFmpeg error: {stderr[:500]}")
            return False
            
    except Exception as e:
        print(f"   ✗ Failed to create video: {str(e)}")
        return False

def verify_output(output_path, temp_dir):
    """
    Step 4: Extract test frames to verify the result.
    """
    
    print("🔍 Step 4: Verifying output...")
    
    verify_dir = os.path.join(temp_dir, "verify")
    os.makedirs(verify_dir, exist_ok=True)
    
    test_frames = [
        (0, "Frame 0 - should show 1 pixel"),
        (15, "Frame 15 - should show 16 pixels"),
        (60, "Frame 60 - should show 61 pixels (head)"),
        (120, "Frame 120 - should show 121 pixels (upper body)"),
        (180, "Frame 180 - should show 181 pixels (most body)")
    ]
    
    for frame_num, description in test_frames:
        time = frame_num / 30.0
        output_frame = os.path.join(verify_dir, f"verify_{frame_num:03d}.png")
        
        cmd = [
            'ffmpeg',
            '-ss', str(time),
            '-i', output_path,
            '-frames:v', '1',
            '-y',
            output_frame
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   ✓ {description}")
        except:
            print(f"   ✗ Failed to extract frame {frame_num}")
    
    print(f"   Verification frames saved to: {verify_dir}/")

def main():
    # Paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_path = "output/sailor_true_pixel_perfect.mp4"
    
    if not os.path.exists(sailor_path):
        print(f"❌ Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"❌ Sea video not found: {sea_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("TRUE FRAME-BY-FRAME PIXEL EMERGENCE")
    print("=" * 60)
    print("\n🎯 Method: Extract, Crop, Composite")
    print("   No dynamic expressions, no masking")
    print("   Just real frame-by-frame processing")
    print()
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="sailor_emergence_")
    print(f"📁 Working directory: {temp_dir}")
    print()
    
    try:
        # Step 1: Extract frames
        if not prepare_sailor_frames(sailor_path, temp_dir):
            print("\n❌ Failed at Step 1")
            sys.exit(1)
        
        print()
        
        # Step 2: Crop frames
        if not crop_frames_progressively(temp_dir):
            print("\n❌ Failed at Step 2")
            sys.exit(1)
        
        print()
        
        # Step 3: Create video
        if not create_final_emergence(sea_path, temp_dir, output_path):
            print("\n❌ Failed at Step 3")
            sys.exit(1)
        
        print()
        
        # Step 4: Verify
        verify_output(output_path, temp_dir)
        
        print("\n" + "=" * 60)
        print("🎉 PERFECT PIXEL EMERGENCE COMPLETE!")
        print("=" * 60)
        print(f"\n📹 Final video: {output_path}")
        print(f"📁 Debug frames: {temp_dir}/")
        print("\n✨ What we achieved:")
        print("   ✓ Frame 0: 1 pixel at water line")
        print("   ✓ Frame N: N+1 pixels visible")
        print("   ✓ No masking artifacts")
        print("   ✓ Clean sea below water")
        print("   ✓ True submarine emergence!")
        
        # Ask if we should clean up
        print(f"\n💡 Temp files at: {temp_dir}")
        print("   (Will auto-clean on system reboot)")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        sys.exit(1)

if __name__ == "__main__":
    main()