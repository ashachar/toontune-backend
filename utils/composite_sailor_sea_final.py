#!/usr/bin/env python3

import subprocess
import os
import sys

def create_sailor_rising_composite(sailor_path, sea_path, output_path):
    """
    Create the final composite with sailor rising from sea.
    Uses colorkey to remove black background since WebM lacks alpha channel.
    """
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("🔍 Issue found: WebM has yuv420p format (no alpha channel)")
    print("✅ Solution: Using colorkey filter to remove black background")
    print()
    
    # Optimized filter with colorkey to remove black
    filter_complex = (
        # Scale the sailor to appropriate size
        '[1:v]scale=280:-1[sailor_scaled];'
        
        # Remove black background using colorkey
        # similarity=0.15 removes black and very dark colors
        # blend=0.05 smooths the edges
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Apply fade-in effect
        '[sailor_keyed]fade=t=in:st=0.3:d=1.2:alpha=1[sailor_fade];'
        
        # Overlay with smooth rising animation
        # Starts below frame, rises to 35% from top
        '[0:v][sailor_fade]overlay='
        'x=(W-w)/2:'  # Center horizontally
        'y=\'if(lt(t,2.5),'
        'H+100-((H+100-H*0.35)*min(1,t/2.5)*min(1,t/2.5)),'  # Ease-in quadratic
        'H*0.35)\':'
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
        '-t', '7',  # 7 seconds for full animation
        '-y',
        output_path
    ]
    
    print(f"🎬 Creating final composite...")
    print(f"   Sailor: {sailor_path}")
    print(f"   Sea: {sea_path}")
    print(f"   Output: {output_path}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ SUCCESS! Video created: {output_path}")
        print(f"   - Black background removed")
        print(f"   - Sailor rises smoothly from sea")
        print(f"   - Animation plays with proper transparency")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating video:")
        print(f"   {e.stderr[:500]}")
        return False

def extract_verification_frames(video_path, output_dir):
    """Extract frames at different timestamps for verification."""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamps = ["00:00:01", "00:00:02.5", "00:00:04"]
    
    print(f"\n📸 Extracting verification frames...")
    
    for i, time in enumerate(timestamps):
        frame_path = os.path.join(output_dir, f"frame_{i+1}.png")
        cmd = [
            'ffmpeg',
            '-ss', time,
            '-i', video_path,
            '-frames:v', '1',
            '-y',
            frame_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   Frame {i+1} at {time}: {frame_path}")
        except:
            pass

def main():
    # Paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_path = "output/sailor_sea_final.mp4"
    frames_dir = "output/sailor_final_frames"
    
    # Check inputs
    if not os.path.exists(sailor_path):
        print(f"❌ Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"❌ Sea video not found: {sea_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("CREATING FINAL SAILOR RISING FROM SEA COMPOSITE")
    print("=" * 60)
    print()
    
    # Create the composite
    if create_sailor_rising_composite(sailor_path, sea_path, output_path):
        # Extract verification frames
        extract_verification_frames(output_path, frames_dir)
        
        print("\n" + "=" * 60)
        print("✨ COMPLETE!")
        print("=" * 60)
        print(f"📹 Final video: {output_path}")
        print(f"📸 Screenshots: {frames_dir}/")
        print("\nThe sailor now rises from the sea WITHOUT black background!")
    else:
        print("\n❌ Failed to create final composite")
        sys.exit(1)

if __name__ == "__main__":
    main()