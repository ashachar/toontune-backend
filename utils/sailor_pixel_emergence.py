#!/usr/bin/env python3

import subprocess
import os
import sys

def create_pixel_by_pixel_emergence(sailor_path, sea_path, output_path):
    """
    TRUE WATER EMERGENCE: Pixel-by-pixel revelation.
    
    Key concept:
    - Water level is at y=215 (never changes)
    - Only pixels ABOVE y=215 are ever visible
    - Sailor starts deep underwater and rises
    - Each pixel the sailor rises reveals one more pixel of the sailor
    
    Math:
    - Sailor is ~281px tall when scaled to 200px wide
    - To show only top 1px: sailor top must be at y=214 (one pixel above water)
    - To show top 50px: sailor top must be at y=165 (50px above water)
    - To show 210px (3/4): sailor top must be at y=5 (210px above water)
    """
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("🌊 TRUE PIXEL-BY-PIXEL WATER EMERGENCE")
    print("=" * 50)
    print("Concept: Only pixels above y=215 (water) are visible")
    print("Effect: Sailor rises, revealing more pixels from top down")
    print("NO fade - just pure emergence")
    print()
    
    # The key is to mask everything below y=215
    # We'll position the sailor and use a crop mask
    
    filter_complex = (
        # Scale the sailor (NO fade!)
        '[1:v]scale=200:-1[sailor_scaled];'
        
        # Remove black background
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Position sailor to rise from deep underwater to above water
        # Start: y=214 (only 1px above water)
        # End: y=5 (210px above water)
        '[sailor_keyed]'
        'overlay=x=220:'
        'y=\'if(lt(t,0.5),'
            '214,'  # Start with just top pixel visible
            'if(lt(t,4.5),'
                '214-((214-5)*min(1,(t-0.5)/4)),'  # Linear rise over 4 seconds
                '5))\''
        '[sailor_positioned];'
        
        # Create a hard mask at water level y=215
        # This cuts off everything below the water
        '[0:v][sailor_positioned]overlay=x=0:y=0[with_sailor];'
        
        # Draw a box to cover everything below water level
        # This ensures nothing below y=215 is visible
        '[with_sailor]'
        'drawbox=x=0:y=215:w=640:h=141:color=black@0:t=fill'
        '[masked];'
        
        # Overlay the original sea back on top of the black box
        # But only the part below water
        '[0:v]crop=640:141:0:215[bottom_sea];'
        '[masked][bottom_sea]overlay=0:215'
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
    
    print(f"🎬 Creating pixel-perfect emergence...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Success! Pixel-by-pixel emergence created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Complex masking failed, trying alternative approach...")
        return create_crop_based_emergence(sailor_path, sea_path, output_path)

def create_crop_based_emergence(sailor_path, sea_path, output_path):
    """
    Alternative: Use dynamic cropping to reveal sailor pixel by pixel.
    """
    
    print("\n🌊 Using crop-based pixel emergence...")
    
    # This approach crops the sailor dynamically
    # Only the part that would be above water is shown
    
    filter_complex = (
        # Scale the sailor (NO fade!)
        '[1:v]scale=200:-1[sailor_scaled];'
        
        # Remove black background
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Dynamic crop that reveals more over time
        # Start showing 1px, end showing 210px
        '[sailor_keyed]crop='
        'w=200:'
        'h=\'min(281,if(lt(t,0.5),1,if(lt(t,4.5),1+((210-1)*(t-0.5)/4),210)))\':'
        'x=0:y=0'
        '[sailor_cropped];'
        
        # Position the cropped sailor at water level
        # The top of the visible portion stays at water level
        '[0:v][sailor_cropped]overlay='
        'x=220:'
        'y=\'215-min(281,if(lt(t,0.5),1,if(lt(t,4.5),1+((210-1)*(t-0.5)/4),210)))\''
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
        print(f"✅ Success! Crop-based emergence created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Crop approach failed, using masking method...")
        return create_mask_emergence(sailor_path, sea_path, output_path)

def create_mask_emergence(sailor_path, sea_path, output_path):
    """
    Final approach: Use masking to hide underwater portion.
    """
    
    print("\n🌊 Using mask-based emergence...")
    
    # Position sailor to rise, mask everything below water
    filter_complex = (
        # Scale sailor
        '[1:v]scale=200:-1[sailor_scaled];'
        
        # Remove black
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Create a mask that hides everything below y=215
        # White above water, black below water
        'color=white:s=640x215:d=7[top_mask];'
        'color=black:s=640x141:d=7[bottom_mask];'
        '[top_mask][bottom_mask]vstack[full_mask];'
        
        # Position sailor to rise from underwater
        # Start at y=280 (completely hidden)
        # Rise to y=5 (210px visible)
        '[0:v][sailor_keyed]overlay='
        'x=220:'
        'y=\'if(lt(t,0.5),280,if(lt(t,4.5),280-((280-5)*(t-0.5)/4),5))\''
        '[scene];'
        
        # Apply the mask to hide underwater portion
        '[scene][full_mask]blend=all_mode=multiply'
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
        print(f"✅ Success! Mask-based emergence created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr[:500]}")
        return False

def extract_emergence_frames(video_path, output_dir):
    """Extract frames showing pixel-by-pixel emergence."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamps = [
        ("0.5", "Start - first pixels"),
        ("1.0", "Few pixels visible"),
        ("1.5", "Head emerging"),
        ("2.0", "More head visible"),
        ("2.5", "Shoulders appearing"),
        ("3.0", "Upper body emerging"),
        ("3.5", "Half body visible"),
        ("4.0", "Most body above water"),
        ("4.5", "Final - 3/4 visible")
    ]
    
    print(f"\n📸 Extracting pixel emergence frames...")
    
    for time, desc in timestamps:
        frame_path = os.path.join(output_dir, f"pixel_{time.replace('.', '_')}s.png")
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

def visualize_concept():
    """Show the concept visually."""
    print("\n📊 PIXEL-BY-PIXEL EMERGENCE CONCEPT:")
    print("=" * 50)
    print("Water level NEVER moves - always at y=215")
    print("Only pixels ABOVE water are visible")
    print()
    print("Time 0.5s: Sailor at y=214")
    print("  ┌─────────────┐")
    print("  │     Sky     │")
    print("  ├─────────────┤ y=215 (water)")
    print("  │≈≈≈≈≈👤≈≈≈≈≈≈│ <- Just tip visible")
    print("  │   Water     │")
    print("  └─────────────┘")
    print()
    print("Time 2.0s: Sailor at y=165")
    print("  ┌─────────────┐")
    print("  │     Sky     │")
    print("  │      👤      │ <- Head visible")
    print("  ├─────────────┤ y=215 (water)")
    print("  │≈≈≈≈≈[↓]≈≈≈≈≈│ <- Body hidden")
    print("  │   Water     │")
    print("  └─────────────┘")
    print()
    print("Time 4.5s: Sailor at y=5")
    print("  ┌─────────────┐")
    print("  │      👤      │ <- 3/4 visible")
    print("  │      │       │")
    print("  ├─────────────┤ y=215 (water)")
    print("  │≈≈≈≈[legs]≈≈≈│ <- Only legs hidden")
    print("  └─────────────┘")

def main():
    # Paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_path = "output/sailor_pixel_emergence.mp4"
    frames_dir = "output/pixel_emergence"
    
    if not os.path.exists(sailor_path):
        print(f"❌ Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"❌ Sea video not found: {sea_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("PIXEL-BY-PIXEL TRUE WATER EMERGENCE")
    print("=" * 60)
    
    # Show concept
    visualize_concept()
    
    # Create the effect
    if create_pixel_by_pixel_emergence(sailor_path, sea_path, output_path):
        extract_emergence_frames(output_path, frames_dir)
        
        print("\n" + "=" * 60)
        print("✨ COMPLETE!")
        print("=" * 60)
        print(f"\n📹 Final video: {output_path}")
        print(f"📸 Frames: {frames_dir}/")
        print("\n🌊 True emergence achieved:")
        print("   ✓ NO fade effect")
        print("   ✓ Only pixels above water (y=215) are visible")
        print("   ✓ Sailor rises, revealing more pixels from top down")
        print("   ✓ Like actually emerging from water!")
    else:
        print("\n❌ Failed to create pixel emergence")
        sys.exit(1)

if __name__ == "__main__":
    main()