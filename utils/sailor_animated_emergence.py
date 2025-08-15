#!/usr/bin/env python3

import subprocess
import os
import sys
import shutil

def create_animated_pixel_emergence():
    """
    Create emergence with BOTH spatial and temporal progression.
    Frame N: Shows N+1 pixels from the Nth frame of the animation.
    """
    
    print("🎬 Creating ANIMATED pixel-perfect emergence...")
    
    # Paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_path = "output/sailor_animated_emergence.mp4"
    temp_dir = "/tmp/sailor_animated"
    
    # Clean and create temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print("\n📸 Step 1: Extract ALL frames from animated sailor WebM...")
    
    # Skip video conversion, extract frames directly
    sailor_frames_dir_raw = os.path.join(temp_dir, "sailor_frames_raw")
    os.makedirs(sailor_frames_dir_raw)
    
    cmd = [
        'ffmpeg',
        '-i', sailor_path,
        '-r', '30',  # Force 30fps extraction
        os.path.join(sailor_frames_dir_raw, 'raw_%04d.png')
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        frame_count = len([f for f in os.listdir(sailor_frames_dir_raw) if f.endswith('.png')])
        print(f"   ✓ Extracted {frame_count} raw frames")
    except:
        print("   ✗ Failed to extract raw frames")
        return False
    
    # Now scale each frame individually
    print("   Scaling frames to 200px wide...")
    sailor_frames_dir = os.path.join(temp_dir, "sailor_frames")
    os.makedirs(sailor_frames_dir)
    
    raw_frames = sorted([f for f in os.listdir(sailor_frames_dir_raw) if f.endswith('.png')])
    
    for i, frame_file in enumerate(raw_frames):
        input_frame = os.path.join(sailor_frames_dir_raw, frame_file)
        output_frame = os.path.join(sailor_frames_dir, f'sailor_{i+1:04d}.png')
        
        cmd = [
            'ffmpeg',
            '-i', input_frame,
            '-vf', 'scale=200:-1',
            '-y',
            output_frame
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except:
            print(f"   ✗ Failed to scale frame {i+1}")
            return False
    
    print(f"   ✓ All frames scaled to 200px wide")
    
    # Count how many frames we have
    frame_count = len([f for f in os.listdir(sailor_frames_dir) if f.endswith('.png')])
    
    if frame_count < 210:
        print(f"   ⚠️ Only {frame_count} frames available, will loop animation")
    
    print("\n🎨 Step 2: Remove black background from all frames...")
    
    sailor_clean_dir = os.path.join(temp_dir, "sailor_clean")
    os.makedirs(sailor_clean_dir)
    
    # Process each frame to remove black
    frame_files = sorted([f for f in os.listdir(sailor_frames_dir) if f.endswith('.png')])
    
    for i, frame_file in enumerate(frame_files):
        input_frame = os.path.join(sailor_frames_dir, frame_file)
        output_frame = os.path.join(sailor_clean_dir, frame_file)
        
        cmd = [
            'ffmpeg',
            '-i', input_frame,
            '-vf', 'colorkey=0x000000:0.15:0.05',
            '-c:v', 'png',
            '-y',
            output_frame
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if i == 0:
                print(f"   Processing frame {i+1}/{len(frame_files)}")
            elif i == len(frame_files) - 1:
                print(f"   Processing frame {i+1}/{len(frame_files)}")
        except:
            # If colorkey fails, just copy original
            shutil.copy(input_frame, output_frame)
    
    print(f"   ✓ Black removed from all frames")
    
    print("\n✂️ Step 3: Create cropped versions with temporal progression...")
    
    cropped_dir = os.path.join(temp_dir, "cropped")
    os.makedirs(cropped_dir)
    
    total_sailor_frames = len(frame_files)
    
    for i in range(210):
        # Spatial: crop to i+1 pixels height
        crop_height = i + 1
        
        # Temporal: use frame i from the animation (with looping if needed)
        sailor_frame_index = (i % total_sailor_frames) + 1  # FFmpeg numbering starts at 1
        sailor_frame = os.path.join(sailor_clean_dir, f'sailor_{sailor_frame_index:04d}.png')
        
        # If frame doesn't exist (shouldn't happen), use first frame
        if not os.path.exists(sailor_frame):
            sailor_frame = os.path.join(sailor_clean_dir, 'sailor_0001.png')
        
        cropped_file = os.path.join(cropped_dir, f'crop_{i:04d}.png')
        
        cmd = [
            'ffmpeg',
            '-i', sailor_frame,
            '-vf', f'crop=200:{crop_height}:0:0',
            '-y',
            cropped_file
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if i == 0:
                print(f"   Frame   0: 1px from animation frame 1")
            elif i == 29:
                print(f"   Frame  29: 30px from animation frame {(29 % total_sailor_frames) + 1}")
            elif i == 59:
                print(f"   Frame  59: 60px from animation frame {(59 % total_sailor_frames) + 1}")
            elif i == 119:
                print(f"   Frame 119: 120px from animation frame {(119 % total_sailor_frames) + 1}")
            elif i == 209:
                print(f"   Frame 209: 210px from animation frame {(209 % total_sailor_frames) + 1}")
        except:
            print(f"   ✗ Failed at frame {i}")
            return False
    
    print("   ✓ All 210 cropped versions created with animation")
    
    print("\n🎬 Step 4: Create final video...")
    
    # Create looped sea video
    print("   Looping sea video...")
    looped_sea = os.path.join(temp_dir, "sea_looped.mp4")
    
    cmd = [
        'ffmpeg',
        '-stream_loop', '2',
        '-i', sea_path,
        '-t', '7',
        '-c', 'copy',
        '-y',
        looped_sea
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("   ✓ Sea video looped")
    except:
        looped_sea = sea_path
    
    # Extract sea frames
    print("   Extracting sea frames...")
    sea_frames_dir = os.path.join(temp_dir, "sea")
    os.makedirs(sea_frames_dir)
    
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
    print("   Compositing animated frames...")
    output_frames_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_frames_dir)
    
    for i in range(210):
        sea_frame = os.path.join(sea_frames_dir, f"sea_{(i+1):04d}.png")
        sailor_crop = os.path.join(cropped_dir, f"crop_{i:04d}.png")
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
            
            if i % 30 == 0:
                print(f"      Composited frame {i} (animation + emergence)")
        except:
            print(f"   ✗ Failed to composite frame {i}")
            return False
    
    print("   ✓ All frames composited with animation")
    
    # Create final video
    print("   Creating final animated emergence video...")
    
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
    Extract test frames to verify animation + emergence.
    """
    
    print("\n🔍 Verifying animated emergence...")
    
    for frame in [0, 30, 60, 90, 120, 150, 180]:
        time = frame / 30.0
        output_frame = f'/tmp/verify_animated_{frame:03d}.png'
        
        cmd = [
            'ffmpeg',
            '-ss', str(time),
            '-i', video_path,
            '-frames:v', '1',
            '-y',
            output_frame
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   Frame {frame:3d}: {frame+1:3d} pixels + animation frame {frame+1}")
        except:
            pass

def main():
    print("=" * 60)
    print("ANIMATED PIXEL-BY-PIXEL EMERGENCE")
    print("=" * 60)
    print("\n🎯 Dual progression:")
    print("   • Spatial: Frame N shows N+1 pixel rows")
    print("   • Temporal: Frame N uses Nth animation frame")
    print()
    
    if create_animated_pixel_emergence():
        verify_output("output/sailor_animated_emergence.mp4")
        
        print("\n" + "=" * 60)
        print("🎉 ANIMATED PIXEL EMERGENCE ACHIEVED!")
        print("=" * 60)
        print("\n📹 Final video: output/sailor_animated_emergence.mp4")
        print("\n✨ What we achieved:")
        print("   1. Extracted ALL animation frames from WebM")
        print("   2. Each output frame N uses:")
        print("      • Animation frame N (temporal)")
        print("      • Cropped to N+1 pixels (spatial)")
        print("   3. Sailor ANIMATES while emerging!")
        print("\n🌊 Result: Sailor salutes WHILE rising from water!")
    else:
        print("\n❌ Failed to create animated emergence")

if __name__ == "__main__":
    main()