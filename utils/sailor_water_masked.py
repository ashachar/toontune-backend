#!/usr/bin/env python3

import subprocess
import os
import sys

def create_water_masked_emergence(sailor_path, sea_path, output_path):
    """
    Create sailor emergence with proper water masking.
    The sailor's body below water level should be hidden.
    
    Water analysis:
    - Water spans y=150 to y=280
    - We need to crop/mask the sailor to only show what's above water
    """
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("🌊 Creating water-masked emergence...")
    print("   The sailor will be partially hidden by water")
    print("   Only the part above water line will be visible")
    print()
    
    # Use drawbox to create a mask effect
    # We'll overlay the sailor, then draw boxes to hide the underwater part
    
    filter_complex = (
        # Scale the sailor
        '[1:v]scale=200:-1[sailor_scaled];'
        
        # Remove black background
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Apply fade
        '[sailor_keyed]fade=t=in:st=0.3:d=1.0:alpha=1[sailor_fade];'
        
        # Position sailor and animate rising
        # Start position: mostly underwater (y=400, below visible area)
        # End position: 3/4 above water (y=70)
        '[0:v][sailor_fade]overlay='
        'x=220:'  # Center horizontally
        'y=\'if(lt(t,0.5),'
            '400,'  # Start below frame (hidden)
            'if(lt(t,4),'
                '400-((400-70)*(t-0.5)/3.5),'  # Rise up
                '70))\''  # Final position
        '[with_sailor];'
        
        # Draw water over the sailor to hide underwater portion
        # This creates the illusion of being IN the water
        # Water region is from y=280 to bottom of frame
        '[with_sailor]drawbox='
        'x=0:y=280:w=640:h=76:'  # Cover beach area
        'color=0x9B8A70@1.0:t=fill'  # Beach color
        '[final]'
    )
    
    cmd = [
        'ffmpeg',
        '-i', sea_path,
        '-i', sailor_path,
        '-filter_complex', filter_complex,
        '-map', '[final]',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-t', '8',
        '-y',
        output_path
    ]
    
    print(f"🎬 Rendering with water masking...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Success! Water-masked emergence created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Drawbox approach failed, trying crop method...")
        return create_cropped_emergence(sailor_path, sea_path, output_path)

def create_cropped_emergence(sailor_path, sea_path, output_path):
    """
    Alternative: Use crop to show only the visible part of sailor.
    """
    
    print("\n🌊 Using crop-based emergence...")
    
    # This approach crops the sailor dynamically to simulate emergence
    filter_complex = (
        # Scale the sailor
        '[1:v]scale=200:-1[sailor_scaled];'
        
        # Remove black background
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Split for processing
        '[sailor_keyed]split[sailor1][sailor2];'
        
        # Create a dynamic crop that reveals more of the sailor over time
        # Start showing only top 50px, end showing 210px (3/4 of 281px)
        '[sailor1]crop='
        'w=200:'
        'h=\'if(lt(t,0.5),50,if(lt(t,4),50+((210-50)*(t-0.5)/3.5),210))\':'
        'x=0:y=0[sailor_cropped];'
        
        # Apply fade to cropped sailor
        '[sailor_cropped]fade=t=in:st=0.3:d=1.0:alpha=1[sailor_fade];'
        
        # Overlay at water level
        # Position so the visible part appears to rise from water
        '[0:v][sailor_fade]overlay='
        'x=220:'  # Center horizontally
        'y=\'if(lt(t,0.5),'
            '165,'  # Start at water level minus initial visible height
            'if(lt(t,4),'
                '165-((165-70)*(t-0.5)/3.5),'  # Rise up
                '70))\':'  # Final position
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
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Success! Cropped emergence created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Crop failed too. Using simple rise with water overlay...")
        return create_water_overlay_emergence(sailor_path, sea_path, output_path)

def create_water_overlay_emergence(sailor_path, sea_path, output_path):
    """
    Simplest approach: Overlay water on top of sailor.
    """
    
    print("\n🌊 Using water overlay method...")
    
    # Position sailor to rise, then overlay sea water on top
    filter_complex = (
        # First, create the sailor overlay
        '[1:v]scale=200:-1[sailor_scaled];'
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        '[sailor_keyed]fade=t=in:st=0.3:d=1.0:alpha=1[sailor_fade];'
        
        # Position sailor to rise from below
        '[0:v][sailor_fade]overlay='
        'x=220:'
        'y=\'if(lt(t,0.5),250,if(lt(t,4),250-((250-70)*(t-0.5)/3.5),70))\''
        '[scene];'
        
        # Extract water region from original sea and overlay it
        # This creates the illusion of sailor being IN water
        '[0:v]crop=640:130:0:150[water_region];'
        '[scene][water_region]overlay=0:150:enable=\'lt(t,3.5)\''
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
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Success! Water overlay emergence created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr[:500]}")
        return False

def extract_frames(video_path, output_dir):
    """Extract verification frames."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamps = ["0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.5"]
    
    print(f"\n📸 Extracting frames...")
    
    for time in timestamps:
        frame_path = os.path.join(output_dir, f"masked_{time}s.png")
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
            print(f"   Frame at {time}s saved")
        except:
            pass

def main():
    # Paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_path = "output/sailor_water_masked_final.mp4"
    frames_dir = "output/masked_emergence"
    
    if not os.path.exists(sailor_path):
        print(f"❌ Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"❌ Sea video not found: {sea_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("SAILOR EMERGING WITH WATER MASKING")
    print("=" * 60)
    print("\n📍 Approach:")
    print("   1. Sailor rises from below water level")
    print("   2. Water masks/hides the underwater portion")
    print("   3. Creates illusion of emerging FROM water")
    print()
    
    if create_water_masked_emergence(sailor_path, sea_path, output_path):
        extract_frames(output_path, frames_dir)
        
        print("\n" + "=" * 60)
        print("✨ COMPLETE!")
        print("=" * 60)
        print(f"\n📹 Final video: {output_path}")
        print(f"📸 Frames: {frames_dir}/")
        print("\n The sailor now emerges from the water with proper masking!")
    else:
        print("\n❌ Failed to create masked emergence")
        sys.exit(1)

if __name__ == "__main__":
    main()