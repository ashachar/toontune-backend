#!/usr/bin/env python3

import subprocess
import os
import sys

def check_video_info(video_path):
    """Check video codec and pixel format information."""
    print(f"\n📊 Checking video info for: {video_path}")
    
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,pix_fmt,width,height',
        '-of', 'default=noprint_wrappers=1',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None

def extract_frame_with_transparency(video_path, output_path, time="00:00:01"):
    """Extract a frame preserving alpha channel."""
    print(f"\n🖼️ Extracting frame at {time} with transparency...")
    
    cmd = [
        'ffmpeg',
        '-ss', time,
        '-i', video_path,
        '-frames:v', '1',
        '-vf', 'format=rgba',
        '-y',
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Frame extracted to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e.stderr}")
        return False

def test_overlay_methods(sailor_path, sea_path, output_dir):
    """Test different overlay methods to preserve transparency."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    methods = [
        {
            "name": "method1_alpha_premult",
            "filter": (
                '[1:v]format=rgba,scale=250:-1[sailor];'
                '[0:v][sailor]overlay=x=(W-w)/2:y=H*0.4:format=rgb'
            ),
            "desc": "Using format=rgb in overlay"
        },
        {
            "name": "method2_alphaextract",
            "filter": (
                '[1:v]scale=250:-1,format=rgba,split[sailor][alpha];'
                '[alpha]alphaextract[mask];'
                '[0:v][sailor]overlay=x=(W-w)/2:y=H*0.4[tmp];'
                '[tmp][mask]alphamerge'
            ),
            "desc": "Using alphaextract and alphamerge"
        },
        {
            "name": "method3_chromakey",
            "filter": (
                '[1:v]scale=250:-1[sailor];'
                '[sailor]chromakey=black:0.01:0.0[keyed];'
                '[0:v][keyed]overlay=x=(W-w)/2:y=H*0.4'
            ),
            "desc": "Using chromakey to remove black"
        },
        {
            "name": "method4_colorkey_aggressive",
            "filter": (
                '[1:v]scale=250:-1[sailor];'
                '[sailor]colorkey=0x000000:0.1:0.2[keyed];'
                '[0:v][keyed]overlay=x=(W-w)/2:y=H*0.4'
            ),
            "desc": "Using aggressive colorkey"
        },
        {
            "name": "method5_direct_alpha",
            "filter": (
                '[1:v]scale=250:-1[sailor];'
                '[0:v][sailor]overlay=x=(W-w)/2:y=H*0.4:format=auto'
            ),
            "desc": "Direct overlay with format=auto"
        }
    ]
    
    results = []
    
    for method in methods:
        output_path = os.path.join(output_dir, f"{method['name']}.mp4")
        print(f"\n🔧 Testing {method['name']}: {method['desc']}")
        
        cmd = [
            'ffmpeg',
            '-i', sea_path,
            '-i', sailor_path,
            '-filter_complex', method['filter'],
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-t', '3',  # Just 3 seconds for testing
            '-y',
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Success: {output_path}")
            
            # Extract a frame from the result for inspection
            frame_path = os.path.join(output_dir, f"{method['name']}_frame.png")
            extract_frame_with_transparency(output_path, frame_path, "00:00:01")
            
            results.append((method['name'], True, output_path))
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {e.stderr[:200]}")
            results.append((method['name'], False, None))
    
    return results

def create_best_composite(sailor_path, sea_path, output_path):
    """Create the best working composite based on testing."""
    print(f"\n🎬 Creating final composite with best method...")
    
    # Method that should work best - using colorkey to remove black
    filter_complex = (
        # First scale the sailor
        '[1:v]scale=250:-1[sailor_scaled];'
        
        # Remove black background aggressively
        '[sailor_scaled]colorkey=color=0x000000:similarity=0.15:blend=0.05[sailor_keyed];'
        
        # Add fade in
        '[sailor_keyed]fade=t=in:st=0.5:d=1.0:alpha=1[sailor_fade];'
        
        # Overlay with rising animation
        '[0:v][sailor_fade]overlay='
        'x=(W-w)/2:'
        'y=\'if(lt(t,2),H-(H-H*0.35)*pow(t/2,2),H*0.35)\''
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
        '-t', '6',
        '-y',
        output_path
    ]
    
    try:
        print("Running FFmpeg with colorkey filter...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Created final composite: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e.stderr}")
        return False

def main():
    sailor_path = "output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm"
    sea_path = "backend/uploads/assets/videos/sea_small_segmented_fixed.mp4"
    output_dir = "output/transparency_debug"
    final_output = "output/sailor_sea_fixed.mp4"
    
    if not os.path.exists(sailor_path):
        print(f"Error: Sailor file not found: {sailor_path}")
        sys.exit(1)
    
    if not os.path.exists(sea_path):
        print(f"Error: Sea video not found: {sea_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("DEBUGGING TRANSPARENCY ISSUE")
    print("=" * 60)
    
    # Step 1: Check video info
    print("\n📋 STEP 1: Checking input videos...")
    check_video_info(sailor_path)
    check_video_info(sea_path)
    
    # Step 2: Extract frames to verify transparency
    print("\n📋 STEP 2: Extracting frames...")
    os.makedirs(output_dir, exist_ok=True)
    extract_frame_with_transparency(sailor_path, f"{output_dir}/sailor_frame.png")
    
    # Step 3: Test different overlay methods
    print("\n📋 STEP 3: Testing overlay methods...")
    results = test_overlay_methods(sailor_path, sea_path, output_dir)
    
    print("\n📊 TEST RESULTS:")
    print("-" * 40)
    for name, success, path in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}: {'Success' if success else 'Failed'}")
        if path:
            print(f"   Output: {path}")
    
    # Step 4: Create final composite with best method
    print("\n📋 STEP 4: Creating final composite...")
    if create_best_composite(sailor_path, sea_path, final_output):
        print(f"\n✨ SUCCESS! Final video created: {final_output}")
        print("The sailor should now appear without black background!")
        
        # Extract final frame for verification
        final_frame = f"{output_dir}/final_frame.png"
        extract_frame_with_transparency(final_output, final_frame, "00:00:02")
        print(f"📸 Screenshot saved: {final_frame}")
    else:
        print("\n❌ Failed to create final composite")

if __name__ == "__main__":
    main()