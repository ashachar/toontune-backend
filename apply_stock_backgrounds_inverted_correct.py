#!/usr/bin/env python3
"""
CORRECT FIX: Invert the mask (white was background, not foreground!)
Also ensure color is preserved (not grayscale output).
"""

import subprocess
from pathlib import Path


def apply_background_with_inverted_mask(original_video, stock_video, mask_video, output_path, start_time, duration):
    """
    Apply stock background with INVERTED mask.
    The RVM mask has: white=background, black=foreground (opposite of expected)
    """
    
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(stock_video),    # [0] Stock background
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(original_video),                        # [1] Original video
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(mask_video),                            # [2] Mask
        "-filter_complex",
        # Scale all inputs
        "[0:v]scale=1280:720,setpts=PTS-STARTPTS,format=yuv420p[bg];"
        "[1:v]scale=1280:720,setpts=PTS-STARTPTS,format=yuv420p[fg];"
        # INVERT the mask because white=background in RVM output
        "[2:v]scale=1280:720,setpts=PTS-STARTPTS,format=gray,"
        "negate,"  # INVERT: now white=foreground, black=background
        "eq=contrast=10:brightness=0[mask_inv];"
        # Apply inverted mask
        "[bg][fg][mask_inv]maskedmerge,format=yuv420p[out]",
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
    """Apply stock videos with correctly inverted mask."""
    
    # Paths
    project_folder = Path("uploads/assets/videos/ai_math1")
    original_video = project_folder / "ai_math1_final.mp4"
    mask_video = project_folder / "ai_math1_rvm_mask_5s_024078685789.mp4"
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Stock videos
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
    
    print("🎬 Applying stock videos with INVERTED mask (correct orientation)...\n")
    
    processed = []
    
    for i, seg in enumerate(stock_segments):
        print(f"[{i+1}/{len(stock_segments)}] {seg['description']}")
        
        stock_path = project_folder / seg['stock_video']
        if not stock_path.exists():
            print(f"  ⚠️  Stock video not found")
            continue
        
        output_path = output_dir / f"inverted_correct_segment_{i}.mp4"
        print(f"  📹 Applying with inverted mask (white=fg, black=bg)...")
        
        success = apply_background_with_inverted_mask(
            original_video, stock_path, mask_video, output_path,
            seg['start'], seg['duration']
        )
        
        if success:
            processed.append(output_path)
            print(f"  ✅ Complete\n")
    
    if not processed:
        return
    
    # Create final
    print("🎬 Creating CORRECT final video...")
    concat_list = output_dir / "inverted_concat.txt"
    with open(concat_list, 'w') as f:
        for path in processed:
            f.write(f"file '{path.absolute()}'\n")
    
    final = output_dir / "ai_math1_CORRECT_INVERTED_backgrounds.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(final)
    ], check=True, capture_output=True)
    
    print(f"✅ CORRECT VIDEO: {final}\n")
    
    print("🎯 FIX APPLIED:")
    print("   - INVERTED the mask (negate filter)")
    print("   - Now: white=foreground, black=background")
    print("   - Preserved color with format=yuv420p")
    print("   - Speaker in front, stock videos in back")
    
    subprocess.run(["open", str(final)])


if __name__ == "__main__":
    main()