#!/usr/bin/env python3
"""
Correctly apply stock backgrounds - fixing mask order and chromakey color detection.
"""

import subprocess
from pathlib import Path


def apply_background_with_inverted_mask(original_video, stock_video, mask_video, output_path, start_time, duration):
    """
    Apply background using mask - but INVERT it if needed.
    The RVM mask might have white=background, black=foreground (opposite of what we expect).
    """
    
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(stock_video),    # [0] Stock background
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(original_video),                       # [1] Original video
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(mask_video),                           # [2] Alpha mask
        "-filter_complex",
        # Scale everything
        "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
        "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
        "[2:v]scale=1280:720,setpts=PTS-STARTPTS,format=gray[mask_raw];"
        # Try INVERTING the mask - if white was background, make it black
        "[mask_raw]negate[mask];"
        # Now apply maskedmerge with inverted mask
        "[bg][fg][mask]maskedmerge[out]",
        "-map", "[out]",
        "-map", "1:a?",
        "-shortest",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr[-500:]}")
        return False
    return True


def apply_chromakey_properly(green_video, stock_video, output_path, start_time, duration):
    """
    Apply chromakey with correct green detection.
    Use 'green' keyword instead of hex value for better detection.
    """
    
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(stock_video),    # [0] Background
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(green_video),                          # [1] Green screen
        "-filter_complex",
        "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
        "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
        # Use 'green' keyword with moderate threshold
        "[fg]chromakey=green:0.08:0.05[keyed];"
        # Overlay the keyed foreground on background
        "[bg][keyed]overlay=shortest=1[out]",
        "-map", "[out]",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr[-500:]}")
        return False
    return True


def test_mask_orientation(original_video, mask_video, output_dir):
    """
    Test the mask to see what white and black represent.
    """
    print("Testing mask orientation...")
    
    # Extract a frame from mask
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mask_video),
        "-vframes", "1",
        "-vf", "scale=1280:720,format=gray",
        str(output_dir / "test_mask.png")
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Create test with solid colors to see mask behavior
    test_white = output_dir / "test_white_bg.mp4"
    test_black = output_dir / "test_black_bg.mp4"
    
    # Test 1: White background with regular mask
    cmd1 = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=white:size=1280x720:duration=2",
        "-i", str(original_video), "-t", "2",
        "-i", str(mask_video), "-t", "2",
        "-filter_complex",
        "[0:v][1:v][2:v]maskedmerge[out]",
        "-map", "[out]",
        str(test_white)
    ]
    subprocess.run(cmd1, capture_output=True)
    
    # Test 2: Black background with inverted mask
    cmd2 = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=black:size=1280x720:duration=2",
        "-i", str(original_video), "-t", "2",
        "-i", str(mask_video), "-t", "2",
        "-filter_complex",
        "[2:v]negate[mask];"
        "[0:v][1:v][mask]maskedmerge[out]",
        "-map", "[out]",
        str(test_black)
    ]
    subprocess.run(cmd2, capture_output=True)
    
    print(f"  Created test_white_bg.mp4 - if speaker visible on white, mask is correct")
    print(f"  Created test_black_bg.mp4 - if speaker visible on black, mask needs inversion")


def main():
    """Apply stock backgrounds with correct mask orientation and chromakey."""
    
    # Paths
    project_folder = Path("uploads/assets/videos/ai_math1")
    original_video = project_folder / "ai_math1_final.mp4"
    mask_video = project_folder / "ai_math1_rvm_mask_5s_024078685789.mp4"
    green_video = project_folder / "ai_math1_rvm_green_5s_024078685789.mp4"
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # First test mask orientation
    test_mask_orientation(original_video, mask_video, output_dir)
    
    # Stock videos
    stock_segments = [
        {
            "start": 0.5,
            "duration": 1.2,
            "stock_video": "ai_math1_background_7_2_9_5_zwvgcNhxei.mp4",
            "description": "Math - Drawing cube"
        },
        {
            "start": 1.7,
            "duration": 1.3,
            "stock_video": "ai_math1_background_27_9_31_6_6l4lJ9gGfk.mp4",
            "description": "AI - AI art"
        },
        {
            "start": 3.0,
            "duration": 1.0,
            "stock_video": "ai_math1_background_112_7_127_2_isS82K91sI.mp4",
            "description": "Data - Notebook"
        },
        {
            "start": 4.0,
            "duration": 1.0,
            "stock_video": "ai_math1_background_145_2_152_0_pZm2kl1dD3.mp4",
            "description": "Trends - Crypto"
        }
    ]
    
    print("\nApplying stock backgrounds with CORRECTED methods...\n")
    
    processed_inverted = []
    processed_chromakey = []
    
    for i, seg in enumerate(stock_segments):
        print(f"[{i+1}/{len(stock_segments)}] {seg['description']}")
        
        stock_path = project_folder / seg['stock_video']
        if not stock_path.exists():
            print(f"  ⚠️  Stock video not found")
            continue
        
        # Method 1: Inverted mask
        output_inverted = output_dir / f"correct_inverted_segment_{i}.mp4"
        print(f"  Applying with INVERTED mask...")
        
        success = apply_background_with_inverted_mask(
            original_video, stock_path, mask_video, output_inverted,
            seg['start'], seg['duration']
        )
        
        if success:
            processed_inverted.append(output_inverted)
            print(f"  ✓ Inverted mask complete")
        
        # Method 2: Proper chromakey
        output_chromakey = output_dir / f"correct_chromakey_segment_{i}.mp4"
        print(f"  Applying with proper chromakey...")
        
        success = apply_chromakey_properly(
            green_video, stock_path, output_chromakey,
            seg['start'], seg['duration']
        )
        
        if success:
            processed_chromakey.append(output_chromakey)
            print(f"  ✓ Chromakey complete\n")
    
    # Create finals
    if processed_inverted:
        print("Creating final (inverted mask)...")
        concat_list = output_dir / "correct_concat_inverted.txt"
        with open(concat_list, 'w') as f:
            for path in processed_inverted:
                f.write(f"file '{path.absolute()}'\n")
        
        final_inverted = output_dir / "ai_math1_CORRECT_inverted_mask.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(final_inverted)
        ], check=True, capture_output=True)
        print(f"✅ Inverted mask version: {final_inverted}")
    
    if processed_chromakey:
        print("Creating final (chromakey)...")
        concat_list = output_dir / "correct_concat_chromakey.txt"
        with open(concat_list, 'w') as f:
            for path in processed_chromakey:
                f.write(f"file '{path.absolute()}'\n")
        
        final_chromakey = output_dir / "ai_math1_CORRECT_chromakey.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(final_chromakey)
        ], check=True, capture_output=True)
        print(f"✅ Chromakey version: {final_chromakey}")
    
    # Open results
    if processed_chromakey:
        subprocess.run(["open", str(final_chromakey)])
    if processed_inverted:
        subprocess.run(["open", str(final_inverted)])
    
    print("\n📌 Check which version works correctly:")
    print("   - Inverted mask: Should show speaker with stock background")
    print("   - Chromakey: Should remove green and show stock background")


if __name__ == "__main__":
    main()