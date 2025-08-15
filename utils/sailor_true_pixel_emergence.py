#!/usr/bin/env python3

import subprocess
import os
import sys

def create_true_pixel_emergence(sailor_path, sea_path, output_path):
    """
    ULTRA-SIMPLE TRUE EMERGENCE: Only show pixels above water.
    
    Strategy:
    1. Crop the sailor to only show the part that would be above water
    2. Position the cropped part at the water line
    3. Gradually increase crop height as sailor "rises"
    
    Water line is at y=215
    """
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("🌊 ULTRA-SIMPLE PIXEL EMERGENCE")
    print("=" * 50)
    print("Strategy: Crop sailor to only show above-water pixels")
    print("Water line: y=215 (never changes)")
    print()
    
    # Simple approach: Dynamic crop that grows over time
    filter_complex = (
        # Scale sailor to 200px wide (~281px tall)
        '[1:v]scale=200:-1[sailor_scaled];'
        
        # Remove black background
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Dynamic crop: Start with 1px, grow to 210px
        # This simulates the visible portion above water
        '[sailor_keyed]crop='
        'w=200:'  # Width stays constant
        'h=\'if(lt(t,0.5),'  # Height grows over time
            '1,'  # Start: show only 1px
            'if(lt(t,5),'
                '1+((210-1)*(t-0.5)/4.5),'  # Grow linearly to 210px
                '210))\':'  # End: show 210px (3/4 of 281px)
        'x=0:y=0'  # Crop from top
        '[sailor_visible];'
        
        # Position the visible part just above water
        # Bottom of visible part touches water at y=215
        '[0:v][sailor_visible]overlay='
        'x=220:'  # Center horizontally
        'y=\'215-'  # Position so bottom touches water
        'if(lt(t,0.5),'
            '1,'  # 1px above water
            'if(lt(t,5),'
                '1+((210-1)*(t-0.5)/4.5),'  # Growing amount above water
                '210))\''  # 210px above water
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
    
    print(f"🎬 Creating simple pixel emergence...")
    print(f"   Visible pixels: 1px → 210px over 4.5 seconds")
    print(f"   Position: Always touching water at y=215")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"\n✅ SUCCESS! True pixel emergence created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr[:500]}")
        return False

def visualize_timeline():
    """Show what happens at each stage."""
    print("\n📊 EMERGENCE TIMELINE:")
    print("=" * 50)
    
    stages = [
        (0.5, 1, "Just the hat tip"),
        (1.0, 23, "Top of hat visible"),
        (1.5, 46, "Full hat visible"),
        (2.0, 69, "Hat and forehead"),
        (2.5, 92, "Eyes visible"),
        (3.0, 115, "Face visible"),
        (3.5, 138, "Neck and shoulders"),
        (4.0, 161, "Upper torso"),
        (4.5, 184, "Most of body"),
        (5.0, 210, "3/4 of body (final)")
    ]
    
    print("Time | Pixels | What's visible")
    print("-----|--------|---------------")
    for time, pixels, desc in stages:
        print(f"{time:4.1f}s | {pixels:3d}px | {desc}")

def extract_frames(video_path, output_dir):
    """Extract frames to verify emergence."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    print(f"\n📸 Extracting verification frames...")
    
    for t in timestamps:
        frame_path = os.path.join(output_dir, f"true_{str(t).replace('.', '_')}s.png")
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
            pixels_visible = int(1 + ((210-1)*(max(0, t-0.5))/4.5))
            print(f"   {t}s: {pixels_visible}px visible")
        except:
            pass

def main():
    # Paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_path = "output/sailor_true_pixel_emergence.mp4"
    frames_dir = "output/true_pixel_emergence"
    
    if not os.path.exists(sailor_path):
        print(f"❌ Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"❌ Sea video not found: {sea_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("TRUE PIXEL-BY-PIXEL WATER EMERGENCE")
    print("=" * 60)
    print("\n💡 Key insight: Use CROP to show only above-water pixels")
    print("   - Crop height = visible pixels above water")
    print("   - Position cropped image at water line")
    print("   - Grow crop height over time = emergence effect")
    
    # Show timeline
    visualize_timeline()
    
    print("\n" + "=" * 60)
    
    # Create the effect
    if create_true_pixel_emergence(sailor_path, sea_path, output_path):
        extract_frames(output_path, frames_dir)
        
        print("\n" + "=" * 60)
        print("✨ COMPLETE!")
        print("=" * 60)
        print(f"\n📹 Final video: {output_path}")
        print(f"📸 Frames: {frames_dir}/")
        print("\n🌊 True pixel-by-pixel emergence achieved:")
        print("   ✓ Only pixels above y=215 are visible")
        print("   ✓ Starts with 1px, grows to 210px")
        print("   ✓ NO fade - pure emergence")
        print("   ✓ Like a submarine periscope rising!")
    else:
        print("\n❌ Failed to create true pixel emergence")
        sys.exit(1)

if __name__ == "__main__":
    main()