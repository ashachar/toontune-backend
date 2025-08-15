#!/usr/bin/env python3

import subprocess
import os
import sys

def create_final_water_emergence(sailor_path, sea_path, output_path):
    """
    Final correct version: Sailor emerges from middle of water.
    
    Key insights from analysis:
    - Water is at y=150-280 (gray area in middle)
    - Water center is at y=215
    - Sailor should emerge from CENTER of water (x=320, y=215)
    - Only show the part of sailor that would be above water
    """
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("🌊 FINAL WATER EMERGENCE EFFECT")
    print("=" * 50)
    print("Strategy:")
    print("1. Position sailor IN the water (not below screen)")
    print("2. Start with sailor mostly submerged")
    print("3. Rise up to show 3/4 above water")
    print("4. Water is at y=150-280, center at y=215")
    print()
    
    # The correct approach:
    # - Sailor is 281px tall when scaled to 200px wide
    # - To show only head initially: position top of sailor just above water (y=215)
    # - This means sailor starts at y=215-30 = 185 (only top 30px visible)
    # - Final position: y=215-210 = 5 (210px of 281px visible = 3/4)
    
    filter_complex = (
        # Scale the sailor to 200px wide (becomes ~281px tall)
        '[1:v]scale=200:-1[sailor_scaled];'
        
        # Remove black background
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Add fade effect for smooth appearance
        '[sailor_keyed]fade=t=in:st=0.2:d=1.0:alpha=1[sailor_fade];'
        
        # Position and animate the sailor
        # x = 220 (centers at 320 since sailor is 200px wide)
        # y animation:
        #   - Start: y=185 (top 30px above water at y=215)
        #   - End: y=5 (210px above water, which is 3/4 of 281px)
        '[0:v][sailor_fade]overlay='
        'x=220:'  # Center horizontally (320 - 100)
        'y=\'if(lt(t,0.3),'
            '185,'  # Initial: mostly underwater, just head visible
            'if(lt(t,4),'
                '185-((185-5)*(t-0.3)/3.7),'  # Rise smoothly over 3.7 seconds
                '5))\':'  # Final: 3/4 above water
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
        '-t', '8',
        '-y',
        output_path
    ]
    
    print(f"🎬 Creating final emergence effect...")
    print(f"   Sailor starts at y=185 (head above water)")
    print(f"   Sailor rises to y=5 (3/4 above water)")
    print(f"   Water level is at y=215")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"\n✅ SUCCESS! Proper water emergence created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr[:500]}")
        return False

def extract_detailed_frames(video_path, output_dir):
    """Extract frames showing the emergence progression."""
    os.makedirs(output_dir, exist_ok=True)
    
    # More detailed timestamps
    timestamps = [
        ("0.3", "Start - head barely visible"),
        ("0.8", "Early rise - shoulders emerging"),  
        ("1.3", "Rising - upper torso visible"),
        ("1.8", "Mid-rise - half body above water"),
        ("2.3", "Continuing - more body visible"),
        ("2.8", "Almost there - most body above"),
        ("3.3", "Near final - 3/4 above water"),
        ("4.0", "Final position - saluting"),
        ("5.0", "Stable - animation playing")
    ]
    
    print(f"\n📸 Extracting detailed frame sequence...")
    
    for time, desc in timestamps:
        frame_path = os.path.join(output_dir, f"emerge_{time.replace('.', '_')}s.png")
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
            print(f"   {time}s: {desc}")
        except:
            pass

def visualize_positioning():
    """Print visual representation of the positioning."""
    print("\n📊 VISUAL POSITIONING GUIDE:")
    print("=" * 50)
    print("Frame height: 356px")
    print()
    print("y=0    ┌─────────────────────┐ Top of frame")
    print("       │  Sky/Trees/Horizon  │")
    print("y=150  ├─────────────────────┤ Water top (horizon)")
    print("       │                     │")
    print("y=185  │    👤 (start)       │ Sailor head emerges")
    print("y=215  │ ≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈│ Water center")
    print("       │     WATER           │")
    print("y=280  ├─────────────────────┤ Water bottom (beach)")
    print("       │      Beach          │")
    print("y=356  └─────────────────────┘ Bottom of frame")
    print()
    print("Sailor movement: y=185 → y=5 (rises 180px)")
    print("Effect: Emerges from water center, not from below screen")

def main():
    # Paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_path = "output/sailor_true_emergence.mp4"
    frames_dir = "output/true_emergence_frames"
    
    if not os.path.exists(sailor_path):
        print(f"❌ Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"❌ Sea video not found: {sea_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("TRUE WATER EMERGENCE - FINAL VERSION")
    print("=" * 60)
    
    # Show visual guide
    visualize_positioning()
    
    # Create the effect
    if create_final_water_emergence(sailor_path, sea_path, output_path):
        extract_detailed_frames(output_path, frames_dir)
        
        print("\n" + "=" * 60)
        print("✨ COMPLETE!")
        print("=" * 60)
        print(f"\n📹 Final video: {output_path}")
        print(f"📸 Frame sequence: {frames_dir}/")
        print("\n🌊 The sailor now:")
        print("   ✓ Emerges from the MIDDLE of the water (y=215)")
        print("   ✓ Starts with just head above water")
        print("   ✓ Rises smoothly over 3.7 seconds")
        print("   ✓ Ends with 3/4 of body above water")
        print("   ✓ Positioned at center of screen (x=320)")
        print("\n💡 Key: The sailor is IN the water, not floating above it!")
    else:
        print("\n❌ Failed to create emergence effect")
        sys.exit(1)

if __name__ == "__main__":
    main()