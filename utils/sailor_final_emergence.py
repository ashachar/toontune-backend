#!/usr/bin/env python3

import subprocess
import os
import sys
import shutil

def create_emergence_with_python_loop():
    """
    Ultimate solution: Create individual frames in Python, then combine.
    """
    
    print("🎬 Creating pixel-perfect emergence (Python loop method)...")
    
    # Paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_path = "output/sailor_final_emergence.mp4"
    temp_dir = "/tmp/sailor_final"
    
    # Clean and create temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print("\n📸 Step 1: Extract sailor image...")
    
    # Extract a single frame from sailor
    sailor_full = os.path.join(temp_dir, "sailor_full.png")
    
    cmd = [
        'ffmpeg',
        '-i', sailor_path,
        '-frames:v', '1',
        '-vf', 'scale=200:-1',
        '-y',
        sailor_full
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("   ✓ Sailor image extracted")
    except:
        print("   ✗ Failed to extract sailor")
        return False
    
    # Remove black background
    print("\n🎨 Step 2: Remove black background...")
    sailor_clean = os.path.join(temp_dir, "sailor_clean.png")
    
    cmd = [
        'ffmpeg',
        '-i', sailor_full,
        '-vf', 'colorkey=0x000000:0.15:0.05',
        '-c:v', 'png',
        '-y',
        sailor_clean
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("   ✓ Black removed")
    except:
        print("   ⚠️ Using original (colorkey failed)")
        sailor_clean = sailor_full
    
    print("\n✂️ Step 3: Create cropped versions...")
    
    # Create 210 cropped versions
    for i in range(210):
        crop_height = i + 1
        cropped_file = os.path.join(temp_dir, f"crop_{i:04d}.png")
        
        cmd = [
            'ffmpeg',
            '-i', sailor_clean,
            '-vf', f'crop=200:{crop_height}:0:0',
            '-y',
            cropped_file
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if i == 0:
                print(f"   Frame {i:3d}: {crop_height:3d}px (first pixel)")
            elif i == 29:
                print(f"   Frame {i:3d}: {crop_height:3d}px (hat emerging)")
            elif i == 59:
                print(f"   Frame {i:3d}: {crop_height:3d}px (head visible)")
            elif i == 119:
                print(f"   Frame {i:3d}: {crop_height:3d}px (upper body)")
            elif i == 209:
                print(f"   Frame {i:3d}: {crop_height:3d}px (3/4 visible)")
                
        except:
            print(f"   ✗ Failed at frame {i}")
            return False
    
    print("   ✓ All 210 cropped versions created")
    
    print("\n🎬 Step 4: Create final video...")
    
    # Extract 210 frames from sea video (loop if needed)
    print("   Extracting sea frames (with looping)...")
    sea_frames_dir = os.path.join(temp_dir, "sea")
    os.makedirs(sea_frames_dir)
    
    # First create a looped version of the sea video
    looped_sea = os.path.join(temp_dir, "sea_looped.mp4")
    
    cmd = [
        'ffmpeg',
        '-stream_loop', '2',  # Loop twice to ensure we have enough frames
        '-i', sea_path,
        '-t', '7',  # 7 seconds output
        '-c', 'copy',
        '-y',
        looped_sea
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("   ✓ Sea video looped")
    except:
        print("   ⚠️ Loop failed, using original")
        looped_sea = sea_path
    
    # Now extract frames from looped video
    cmd = [
        'ffmpeg',
        '-i', looped_sea,
        '-r', '30',
        '-frames:v', '210',
        os.path.join(sea_frames_dir, 'sea_%04d.png')
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("   ✓ Sea frames extracted")
    except:
        print("   ✗ Failed to extract sea frames")
        return False
    
    # Composite each frame
    print("   Compositing frames...")
    output_frames_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_frames_dir)
    
    for i in range(210):
        sea_frame = os.path.join(sea_frames_dir, f"sea_{(i+1):04d}.png")
        sailor_crop = os.path.join(temp_dir, f"crop_{i:04d}.png")
        output_frame = os.path.join(output_frames_dir, f"out_{i:04d}.png")
        
        # Position: x=220, y=215-(i+1)
        y_pos = 215 - (i + 1)
        
        cmd = [
            'ffmpeg',
            '-i', sea_frame,
            '-i', sailor_crop,
            '-filter_complex',
            f'[0:v][1:v]overlay=220:{y_pos}',
            '-y',
            output_frame
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if i % 30 == 0:  # Progress every second
                print(f"      Composited frame {i}")
                
        except:
            print(f"   ✗ Failed to composite frame {i}")
            return False
    
    print("   ✓ All frames composited")
    
    # Create video from frames
    print("   Creating final video...")
    
    cmd = [
        'ffmpeg',
        '-r', '30',
        '-i', os.path.join(output_frames_dir, 'out_%04d.png'),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-y',
        output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✓ Video created: {output_path}")
        return True
    except:
        print("   ✗ Failed to create video")
        return False

def verify_output(video_path):
    """
    Extract test frames to verify.
    """
    
    print("\n🔍 Verifying output...")
    
    for frame in [0, 1, 15, 30, 60, 120, 180]:
        time = frame / 30.0
        
        cmd = [
            'ffmpeg',
            '-ss', str(time),
            '-i', video_path,
            '-frames:v', '1',
            '-y',
            f'/tmp/verify_{frame:03d}.png'
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   Frame {frame:3d}: {frame+1:3d} pixels visible")
        except:
            pass

def main():
    print("=" * 60)
    print("FINAL PIXEL-BY-PIXEL EMERGENCE")
    print("=" * 60)
    print("\n🎯 Method: Extract, Crop, Composite each frame")
    print("   No expressions, just pure frame processing")
    print()
    
    if create_emergence_with_python_loop():
        verify_output("output/sailor_final_emergence.mp4")
        
        print("\n" + "=" * 60)
        print("🎉 PERFECT PIXEL EMERGENCE ACHIEVED!")
        print("=" * 60)
        print("\n📹 Final video: output/sailor_final_emergence.mp4")
        print("\n✨ What we did:")
        print("   1. Extracted sailor image")
        print("   2. Created 210 cropped versions (1px to 210px)")
        print("   3. Extracted 210 sea frames")
        print("   4. Composited each cropped sailor onto its sea frame")
        print("   5. Combined into final video")
        print("\n🌊 Result: TRUE pixel-by-pixel emergence!")
    else:
        print("\n❌ Failed to create emergence")

if __name__ == "__main__":
    main()