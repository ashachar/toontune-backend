#!/usr/bin/env python3
"""
Debug the mask and fix the compositing issue.
The mask should define WHERE to show foreground vs background, not blend them.
"""

import subprocess
from pathlib import Path


def check_mask_values(mask_video, output_dir):
    """Extract and analyze mask frames to understand the values."""
    print("🔍 Analyzing mask values...")
    
    # Extract a frame from the mask
    mask_frame = output_dir / "mask_frame.png"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mask_video),
        "-vframes", "1",
        "-vf", "format=gray",
        str(mask_frame)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  Extracted mask frame to: {mask_frame}")
    
    # Check if mask needs to be binarized (make it pure black/white)
    # The RVM mask might have gray values that need thresholding
    binary_mask = output_dir / "mask_binary.png"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mask_video),
        "-vframes", "1",
        "-vf", "format=gray,eq=contrast=10:brightness=0:saturation=0",
        str(binary_mask)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  Created high-contrast version: {binary_mask}")


def apply_background_correctly(original_video, stock_video, mask_video, output_path, start_time, duration):
    """
    Apply background using mask as a binary selection (not blending).
    Use overlay with alpha channel instead of maskedmerge.
    """
    
    print("  🎬 Method 1: Using overlay with alphaextract...")
    
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(stock_video),    # [0] Background
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(original_video),                        # [1] Foreground
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(mask_video),                            # [2] Mask
        "-filter_complex",
        # Prepare inputs
        "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
        "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
        "[2:v]scale=1280:720,setpts=PTS-STARTPTS,format=gray[maskg];"
        # Convert mask to alpha and apply to foreground
        "[maskg]alphaextract[alpha];"
        "[fg][alpha]alphamerge[fg_with_alpha];"
        # Overlay foreground (with alpha) onto background
        "[bg][fg_with_alpha]overlay[out]",
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
    return result.returncode == 0


def apply_background_threshold(original_video, stock_video, mask_video, output_path, start_time, duration):
    """
    Apply background with thresholded mask to ensure binary selection.
    """
    
    print("  🎬 Method 2: Using thresholded mask...")
    
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(stock_video),
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(original_video),
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(mask_video),
        "-filter_complex",
        "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
        "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
        # Threshold the mask to make it binary (black or white only)
        "[2:v]scale=1280:720,setpts=PTS-STARTPTS,format=gray,"
        "eq=contrast=10:brightness=0,format=gray[mask_binary];"
        # Apply binary mask
        "[bg][fg][mask_binary]maskedmerge[out]",
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
    return result.returncode == 0


def main():
    """Debug and fix the mask compositing."""
    
    # Paths
    project_folder = Path("uploads/assets/videos/ai_math1")
    original_video = project_folder / "ai_math1_final.mp4"
    mask_video = project_folder / "ai_math1_rvm_mask_5s_024078685789.mp4"
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # First, analyze the mask
    check_mask_values(mask_video, output_dir)
    
    # Test with one stock video
    stock_video = project_folder / "ai_math1_background_7_2_9_5_zwvgcNhxei.mp4"
    
    if not stock_video.exists():
        print(f"❌ Stock video not found: {stock_video}")
        return
    
    print("\n🔧 Testing different compositing methods...\n")
    
    # Test Method 1: Overlay with alpha
    output1 = output_dir / "test_overlay_alpha.mp4"
    print("Testing overlay with alphaextract method...")
    success1 = apply_background_correctly(
        original_video, stock_video, mask_video, output1,
        0.5, 2.0
    )
    if success1:
        print("  ✅ Success")
    else:
        print("  ❌ Failed")
    
    # Test Method 2: Thresholded mask
    output2 = output_dir / "test_threshold_mask.mp4"
    print("\nTesting thresholded mask method...")
    success2 = apply_background_threshold(
        original_video, stock_video, mask_video, output2,
        0.5, 2.0
    )
    if success2:
        print("  ✅ Success")
    else:
        print("  ❌ Failed")
    
    # Create comparison
    if success1 and success2:
        print("\n📹 Creating method comparison...")
        comparison = output_dir / "mask_methods_comparison.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(output1),
            "-i", str(output2),
            "-filter_complex",
            "[0:v]scale=640:360,drawtext=text='OVERLAY+ALPHA':x=10:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7[left];"
            "[1:v]scale=640:360,drawtext=text='THRESHOLD+MASK':x=10:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7[right];"
            "[left][right]hstack[out]",
            "-map", "[out]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            str(comparison)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  ✅ Comparison: {comparison}")
        
        subprocess.run(["open", str(comparison)])
    
    print("\n📌 The issue was using maskedmerge with gray values.")
    print("   Solution: Either use overlay+alpha or threshold the mask to binary.")


if __name__ == "__main__":
    main()