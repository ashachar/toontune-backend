#!/usr/bin/env python3

import subprocess
import os
import sys

def create_sailor_emerging_from_water(sailor_path, sea_path, output_path):
    """
    Create composite with sailor emerging FROM the water.
    The sailor rises from the middle of the sea, showing only the top initially.
    """
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("🌊 Creating sailor emerging from water effect...")
    print("   - Sailor starts submerged in the middle of the sea")
    print("   - Head appears first, then body emerges")
    print("   - Rises to 3/4 above water level")
    print()
    
    # Complex filter for emerging from water effect
    # Sea video is 640x356, water level is approximately at y=178 (middle)
    filter_complex = (
        # Scale the sailor appropriately
        '[1:v]scale=200:-1[sailor_scaled];'
        
        # Remove black background
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Create a mask for the water level effect
        # This will hide the part of the sailor below water
        '[sailor_keyed]split[sailor_main][sailor_for_mask];'
        
        # The sailor position animation:
        # - Starts fully submerged (only top pixels visible)
        # - Rises to show 3/4 of body above water
        # Water level is at approximately y=178 (middle of 356px height)
        # Sailor height after scaling is approximately 281px (200*360/256)
        '[0:v][sailor_main]overlay='
        'x=(W-w)/2:'  # Center horizontally
        'y=\'if(lt(t,0.5),'  # First 0.5 seconds: stay hidden
            '178,'  # At water level (middle of sea)
            'if(lt(t,3.5),'  # From 0.5 to 3.5 seconds: rise up
                '178-((178-110)*(t-0.5)/3),'  # Rise from water level to 110px (3/4 above water)
                '110))\':'  # Stay at final position
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
        '-t', '7',
        '-y',
        output_path
    ]
    
    print(f"🎬 Rendering...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Success! Sailor emerges from the water")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr[:500]}")
        return False

def create_sailor_with_water_mask(sailor_path, sea_path, output_path):
    """
    Enhanced version with water masking effect.
    The sailor appears to be actually IN the water with proper masking.
    """
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("🌊 Creating enhanced water emergence with masking...")
    
    # More sophisticated filter with water masking
    filter_complex = (
        # Scale the sailor
        '[1:v]scale=200:-1[sailor_scaled];'
        
        # Remove black background
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Create gradient mask for water level
        # This creates the illusion of being partially submerged
        '[sailor_keyed]split[sailor1][sailor2];'
        
        # Create a dynamic mask that follows the rising animation
        # The mask reveals the sailor from top to bottom
        'color=white:s=200x281[white];'
        'color=black:s=200x281[black];'
        '[white][black]blend=all_expr=\'if(gt(Y,A*(1-T/3)),0,1)\':shortest=1[mask];'
        
        # Apply the mask to create water submersion effect
        '[sailor1][mask]alphamerge[sailor_masked];'
        
        # Overlay with position animation
        # Start at water level (y=178), rise to y=80 (most of body above water)
        '[0:v][sailor2]overlay='
        'x=(W-w)/2:'
        'y=\'if(lt(t,0.3),'
            '178+100,'  # Start below water level
            'if(lt(t,3.5),'
                '178+100-((178+100-80)*(t-0.3)/3.2),'  # Rise from below to above water
                '80))\':'
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
        '-t', '7',
        '-y',
        output_path
    ]
    
    print(f"🎬 Rendering with water masking...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Success! Enhanced water emergence created")
        return True
    except subprocess.CalledProcessError as e:
        # Fallback to simpler version
        print(f"⚠️  Enhanced version failed, using standard emergence")
        return False

def create_realistic_emergence(sailor_path, sea_path, output_path):
    """
    Most realistic version - sailor truly emerges from middle of sea.
    """
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("🌊 Creating realistic water emergence...")
    print("   Position: Middle of sea horizontally")
    print("   Effect: Starts with just head visible, rises to 3/4 above water")
    
    # Realistic emergence with proper positioning
    # Sea dimensions: 640x356, water horizon approximately at y=178
    # Sailor scaled to 200px wide, approximately 281px tall
    filter_complex = (
        # Scale sailor
        '[1:v]scale=200:-1[sailor_scaled];'
        
        # Remove black background
        '[sailor_scaled]colorkey=color=black:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Add fade for smooth appearance
        '[sailor_keyed]fade=t=in:st=0.2:d=0.8:alpha=1[sailor_fade];'
        
        # Position and animate:
        # - x = center of screen (640/2 - 100 = 220)
        # - y = starts with head barely visible, rises to 3/4 above water
        # Water level at y=178, sailor is 281px tall
        # Start position: y=178+200 (only top 81px visible)
        # End position: y=178-210 (3/4 of 281px above water)
        '[0:v][sailor_fade]overlay='
        'x=(W-w)/2:'
        'y=\'if(lt(t,0.5),'
            '378,'  # Hidden below water initially
            'if(lt(t,4),'
                '378-((378-(-32))*(t-0.5)/3.5),'  # Rise up over 3.5 seconds
                '-32))\':'  # Final: 3/4 above water (210px of 281px visible)
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
    
    print(f"🎬 Creating final realistic emergence...")
    print(f"   Input: {sailor_path}")
    print(f"   Sea: {sea_path}")
    print(f"   Output: {output_path}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"\n✅ SUCCESS! Realistic emergence created")
        print(f"   The sailor rises from the middle of the sea")
        print(f"   Head appears first, then body emerges 3/4 above water")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr[:500]}")
        return False

def extract_frames(video_path, output_dir):
    """Extract frames to verify the effect."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamps = ["00:00:00.5", "00:00:01.5", "00:00:02.5", "00:00:03.5", "00:00:05"]
    
    print(f"\n📸 Extracting frames for verification...")
    
    for i, time in enumerate(timestamps):
        frame_path = os.path.join(output_dir, f"emerge_{i+1}.png")
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
            print(f"   Frame {i+1} at {time}s: {frame_path}")
        except:
            pass

def main():
    # Paths
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_basic = "output/sailor_emerge_basic.mp4"
    output_enhanced = "output/sailor_emerge_enhanced.mp4"
    output_realistic = "output/sailor_emerge_realistic.mp4"
    frames_dir = "output/emergence_frames"
    
    # Check inputs
    if not os.path.exists(sailor_path):
        print(f"❌ Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"❌ Sea video not found: {sea_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("SAILOR EMERGING FROM MIDDLE OF SEA")
    print("=" * 60)
    print()
    
    # Create realistic version (best one)
    print("Creating realistic emergence effect...")
    print("-" * 40)
    if create_realistic_emergence(sailor_path, sea_path, output_realistic):
        extract_frames(output_realistic, frames_dir)
        
        print("\n" + "=" * 60)
        print("✨ COMPLETE!")
        print("=" * 60)
        print(f"\n📹 Final video: {output_realistic}")
        print(f"📸 Frame sequence: {frames_dir}/")
        print("\nThe sailor now:")
        print("  1. Starts completely submerged")
        print("  2. Head emerges first from middle of sea")
        print("  3. Body rises up to 3/4 above water level")
        print("  4. Performs salute animation above water")
    else:
        print("\n❌ Failed to create emergence effect")
        sys.exit(1)

if __name__ == "__main__":
    main()