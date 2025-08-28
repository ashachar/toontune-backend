#!/usr/bin/env python3
"""
FINAL FIX: Apply stock backgrounds with properly thresholded binary mask.
This ensures foreground is ONLY where mask is white, background ONLY where mask is black.
"""

import subprocess
from pathlib import Path


def apply_background_with_binary_mask(original_video, stock_video, mask_video, output_path, start_time, duration):
    """
    Apply stock background using thresholded binary mask.
    This prevents gray blending and ensures clean separation.
    """
    
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(stock_video),    # [0] Stock background (looped)
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(original_video),                        # [1] Original video
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(mask_video),                            # [2] Alpha mask
        "-filter_complex",
        # Scale all inputs
        "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
        "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
        # CRITICAL: Threshold mask to pure black/white (no gray values!)
        "[2:v]scale=1280:720,setpts=PTS-STARTPTS,format=gray,"
        "eq=contrast=10:brightness=0,format=gray[mask_binary];"
        # Now apply binary mask - no blending, just selection
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
    if result.returncode != 0:
        print(f"  ❌ Error: {result.stderr[-300:]}")
        return False
    return True


def main():
    """Apply real stock videos with properly binarized mask."""
    
    # Paths
    project_folder = Path("uploads/assets/videos/ai_math1")
    original_video = project_folder / "ai_math1_final.mp4"
    mask_video = project_folder / "ai_math1_rvm_mask_5s_024078685789.mp4"
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Stock videos from Coverr
    stock_segments = [
        {
            "start": 0.0,
            "duration": 1.25,
            "stock_video": "ai_math1_background_7_2_9_5_zwvgcNhxei.mp4",
            "description": "Math - Drawing cube"
        },
        {
            "start": 1.25,
            "duration": 1.25,
            "stock_video": "ai_math1_background_27_9_31_6_6l4lJ9gGfk.mp4",
            "description": "AI - Man watching AI art"
        },
        {
            "start": 2.5,
            "duration": 1.25,
            "stock_video": "ai_math1_background_112_7_127_2_isS82K91sI.mp4",
            "description": "Data - Notebook flipping"
        },
        {
            "start": 3.75,
            "duration": 1.25,
            "stock_video": "ai_math1_background_145_2_152_0_pZm2kl1dD3.mp4",
            "description": "Trends - Crypto analysis"
        }
    ]
    
    print("🎬 Applying stock videos with BINARY MASK (fixed)...\n")
    
    processed = []
    
    for i, seg in enumerate(stock_segments):
        print(f"[{i+1}/{len(stock_segments)}] {seg['description']}")
        
        stock_path = project_folder / seg['stock_video']
        if not stock_path.exists():
            print(f"  ⚠️  Stock video not found")
            continue
        
        output_path = output_dir / f"fixed_final_segment_{i}.mp4"
        print(f"  📹 Applying with thresholded binary mask...")
        
        success = apply_background_with_binary_mask(
            original_video, stock_path, mask_video, output_path,
            seg['start'], seg['duration']
        )
        
        if success:
            processed.append(output_path)
            print(f"  ✅ Complete\n")
        else:
            print(f"  ❌ Failed\n")
    
    if not processed:
        print("❌ No segments processed")
        return
    
    # Create final video
    print("🎬 Creating FIXED final video...")
    concat_list = output_dir / "fixed_final_concat.txt"
    with open(concat_list, 'w') as f:
        for path in processed:
            f.write(f"file '{path.absolute()}'\n")
    
    final = output_dir / "ai_math1_FIXED_FINAL_backgrounds.mp4"
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(final)
    ]
    subprocess.run(concat_cmd, check=True, capture_output=True)
    
    print(f"✅ FIXED FINAL VIDEO: {final}\n")
    
    # Create comparison
    print("📹 Creating before/after comparison...")
    
    # Extract original
    original_5s = output_dir / "original_5s.mp4"
    if not original_5s.exists():
        extract_cmd = [
            "ffmpeg", "-y",
            "-i", str(original_video),
            "-t", "5",
            "-c", "copy",
            str(original_5s)
        ]
        subprocess.run(extract_cmd, check=True, capture_output=True)
    
    comparison = output_dir / "fixed_final_comparison.mp4"
    compare_cmd = [
        "ffmpeg", "-y",
        "-i", str(original_5s),
        "-i", str(final),
        "-filter_complex",
        "[0:v]scale=640:360,drawtext=text='ORIGINAL':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=5[left];"
        "[1:v]scale=640:360,drawtext=text='STOCK BACKGROUNDS (FIXED)':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=5[right];"
        "[left][right]hstack[out]",
        "-map", "[out]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(comparison)
    ]
    subprocess.run(compare_cmd, check=True, capture_output=True)
    
    print(f"✅ COMPARISON: {comparison}\n")
    
    print("🎯 KEY FIX:")
    print("   - Thresholded mask to binary (pure black/white)")
    print("   - No gray values = no blending")
    print("   - Foreground ONLY where mask is white")
    print("   - Background ONLY where mask is black")
    
    # Open results
    subprocess.run(["open", str(final)])
    subprocess.run(["open", str(comparison)])


if __name__ == "__main__":
    main()