#!/usr/bin/env python3
"""
Debug exactly what the mask contains and test direct mask application.
"""

import subprocess
from pathlib import Path


def analyze_mask(mask_video):
    """Extract frames and analyze mask values to understand what's white/black."""
    print("🔍 Analyzing mask to understand white/black values...\n")
    
    output_dir = Path("outputs")
    
    # Extract first frame of mask
    mask_frame = output_dir / "mask_frame_raw.png"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mask_video),
        "-vframes", "1",
        str(mask_frame)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✓ Extracted raw mask frame: {mask_frame}")
    
    # Create inverted version
    mask_inverted = output_dir / "mask_frame_inverted.png"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mask_video),
        "-vframes", "1",
        "-vf", "negate",
        str(mask_inverted)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✓ Created inverted mask: {mask_inverted}")
    
    # Get histogram to see pixel distribution
    print("\n📊 Checking pixel distribution in mask...")
    cmd = [
        "ffmpeg", "-i", str(mask_video),
        "-vframes", "1",
        "-vf", "histogram",
        str(output_dir / "mask_histogram.png")
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✓ Created histogram: outputs/mask_histogram.png")


def test_simple_mask_application(original_video, mask_video, output_dir):
    """Test the simplest possible mask application to verify it works."""
    print("\n🧪 Testing simple mask applications...\n")
    
    # Test 1: Replace background with solid color (red)
    output1 = output_dir / "test_red_background.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=red:size=1280x720:duration=2",  # Red background
        "-i", str(original_video), "-t", "2",                       # Original video
        "-i", str(mask_video), "-t", "2",                          # Mask
        "-filter_complex",
        "[0:v][1:v][2:v]maskedmerge[out]",
        "-map", "[out]",
        str(output1)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Test 1: Red background (no inversion)")
    else:
        print("✗ Test 1 failed")
    
    # Test 2: Same but with inverted mask
    output2 = output_dir / "test_red_background_inverted.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=red:size=1280x720:duration=2",
        "-i", str(original_video), "-t", "2",
        "-i", str(mask_video), "-t", "2",
        "-filter_complex",
        "[2:v]negate[mask_inv];"
        "[0:v][1:v][mask_inv]maskedmerge[out]",
        "-map", "[out]",
        str(output2)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Test 2: Red background (inverted mask)")
    else:
        print("✗ Test 2 failed")
    
    # Test 3: Show only the mask itself
    output3 = output_dir / "test_mask_only.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mask_video),
        "-t", "2",
        "-vf", "scale=1280:720",
        str(output3)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print("✓ Test 3: Mask visualization")
    
    # Test 4: Use select filter instead of maskedmerge
    output4 = output_dir / "test_select_method.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=blue:size=1280x720:duration=2",
        "-i", str(original_video), "-t", "2",
        "-i", str(mask_video), "-t", "2",
        "-filter_complex",
        "[2:v]format=gray,negate[mask];"
        "[1:v][mask]alphamerge[fg_alpha];"
        "[0:v][fg_alpha]overlay[out]",
        "-map", "[out]",
        str(output4)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Test 4: Blue background (alphamerge method)")
    else:
        print("✗ Test 4 failed")


def main():
    # Paths
    project_folder = Path("uploads/assets/videos/ai_math1")
    original_video = project_folder / "ai_math1_final.mp4"
    mask_video = project_folder / "ai_math1_rvm_mask_5s_024078685789.mp4"
    output_dir = Path("outputs")
    
    # Analyze mask
    analyze_mask(mask_video)
    
    # Test simple applications
    test_simple_mask_application(original_video, mask_video, output_dir)
    
    print("\n📌 Check these files:")
    print("  1. mask_frame_raw.png - Original mask")
    print("  2. mask_frame_inverted.png - Inverted mask")
    print("  3. test_red_background.mp4 - Should show red where background is")
    print("  4. test_red_background_inverted.mp4 - With inverted mask")
    print("  5. test_mask_only.mp4 - Just the mask itself")
    print("  6. test_select_method.mp4 - Alternative method")
    print("\nOne of these should work correctly!")


if __name__ == "__main__":
    main()