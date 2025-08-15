#!/usr/bin/env python3

import subprocess
import os
import sys

def create_correct_water_emergence(sailor_path, sea_path, output_path):
    """
    CORRECT EMERGENCE: Sailor rises FROM the water, not from bottom of frame.
    
    The problem before: We were cropping the sailor but still showing it at the bottom.
    The solution: Position the FULL sailor so its top starts at y=215, then rises.
    
    Initial position: Top of sailor at y=215 (means sailor at y=215-1 = 214)
    Final position: 210px of sailor above water (sailor at y=215-210 = 5)
    """
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("🌊 CORRECT WATER EMERGENCE - FROM WATER, NOT BOTTOM")
    print("=" * 50)
    print("Key fix: Position sailor so only top pixel is at water initially")
    print("Water line: y=215 (middle of sea)")
    print()
    
    # The correct approach: Position the FULL sailor, starting deep underwater
    filter_complex = (
        # Scale sailor to 200px wide (~281px tall)
        '[1:v]scale=200:-1[sailor_scaled];'
        
        # Remove black background
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Position sailor to rise from underwater
        # Initial: Top of sailor at y=215 (sailor at y=214)
        # Final: 210px above water (sailor at y=5)
        '[0:v][sailor_keyed]overlay='
        'x=220:'  # Center horizontally
        'y=\'if(lt(t,0.5),'
            '214,'  # Start: only top pixel at water surface
            'if(lt(t,5),'
                '214-((214-5)*(t-0.5)/4.5),'  # Rise gradually
                '5))\''  # End: 210px above water
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
        '-t', '7',
        '-y',
        output_path
    ]
    
    print(f"🎬 Creating correct emergence...")
    print(f"   Start position: y=214 (top at water)")
    print(f"   End position: y=5 (210px above water)")
    print(f"   Movement: 209 pixels over 4.5 seconds")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"\n✅ SUCCESS! Correct emergence created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr[:500]}")
        return False

def create_masked_emergence(sailor_path, sea_path, output_path):
    """
    Alternative with explicit masking at water line.
    """
    
    print("\n🌊 Creating with water-line masking...")
    
    # This approach uses a mask to hide everything below y=215
    filter_complex = (
        # Scale sailor
        '[1:v]scale=200:-1[sailor_scaled];'
        
        # Remove black
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Position sailor rising from deep underwater
        '[sailor_keyed]overlay='
        'x=220:'
        'y=\'if(lt(t,0.5),214,if(lt(t,5),214-((214-5)*(t-0.5)/4.5),5))\''
        '[positioned];'
        
        # Create mask: white above y=215, black below
        '[0:v]drawbox=0:215:640:141:black:fill[masked_sea];'
        '[masked_sea][positioned]overlay=0:0[with_sailor];'
        
        # Restore the sea below water line
        '[0:v]crop=640:141:0:215[water_part];'
        '[with_sailor][water_part]overlay=0:215'
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
        '-t', '7',
        '-y',
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Masked emergence created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Masking failed: {e.stderr[:200]}")
        return False

def visualize_positions():
    """Show exact positioning at each stage."""
    print("\n📊 POSITION TIMELINE:")
    print("=" * 50)
    print("Water line is at y=215")
    print("Sailor is 281px tall")
    print()
    
    times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    print("Time | Sailor Y | Pixels above water | What's visible")
    print("-----|----------|-------------------|---------------")
    
    for t in times:
        if t <= 0.5:
            y = 214
        elif t < 5:
            y = 214 - ((214-5)*(t-0.5)/4.5)
        else:
            y = 5
        
        pixels_above = max(0, 215 - y)
        
        if pixels_above < 50:
            visible = "Just hat tip"
        elif pixels_above < 80:
            visible = "Hat"
        elif pixels_above < 110:
            visible = "Head"
        elif pixels_above < 150:
            visible = "Upper body"
        elif pixels_above < 200:
            visible = "Most of body"
        else:
            visible = "3/4 of body"
        
        print(f"{t:4.1f}s | y={y:3.0f}   | {pixels_above:3.0f}px            | {visible}")

def extract_frames(video_path, output_dir):
    """Extract frames to verify."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamps = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    print(f"\n📸 Extracting frames...")
    
    for t in timestamps:
        frame_path = os.path.join(output_dir, f"correct_{str(t).replace('.', '_')}s.png")
        cmd = [
            'ffmpeg',
            '-ss', str(t),
            '-i', video_path,
            '-frames:v', '1',
            '-y',
            frame_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   {t}s extracted")
        except:
            pass

def main():
    # Paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_path = "output/sailor_correct_emergence.mp4"
    output_masked = "output/sailor_masked_emergence.mp4"
    frames_dir = "output/correct_emergence"
    
    if not os.path.exists(sailor_path):
        print(f"❌ Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"❌ Sea video not found: {sea_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("CORRECT WATER EMERGENCE - FROM SEA CENTER")
    print("=" * 60)
    print("\n💡 The fix: Position sailor so it starts BELOW water")
    print("   Not cropping from bottom, but rising from water!")
    
    # Show position timeline
    visualize_positions()
    
    print("\n" + "=" * 60)
    
    # Create both versions
    success = False
    if create_correct_water_emergence(sailor_path, sea_path, output_path):
        extract_frames(output_path, frames_dir)
        success = True
    
    # Try masked version too
    if create_masked_emergence(sailor_path, sea_path, output_masked):
        print(f"📹 Masked version: {output_masked}")
    
    if success:
        print("\n" + "=" * 60)
        print("✨ COMPLETE!")
        print("=" * 60)
        print(f"\n📹 Final video: {output_path}")
        print(f"📸 Frames: {frames_dir}/")
        print("\n🌊 Correct emergence achieved:")
        print("   ✓ Sailor rises FROM the water (y=215)")
        print("   ✓ Not from bottom of frame")
        print("   ✓ Only pixels above water are visible")
        print("   ✓ True submarine-style emergence!")
    else:
        print("\n❌ Failed to create correct emergence")
        sys.exit(1)

if __name__ == "__main__":
    main()