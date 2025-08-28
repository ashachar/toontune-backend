#!/usr/bin/env python3
"""
Final correct implementation: Apply stock backgrounds using RVM alpha mask.
Always uses the mask for clean, accurate compositing without green edges.
"""

import subprocess
from pathlib import Path


def apply_background_with_mask(original_video, stock_video, mask_video, output_path, start_time, duration):
    """
    Apply stock background using the RVM alpha mask.
    Mask: white (255) = foreground to keep, black (0) = background to replace.
    """
    
    cmd = [
        "ffmpeg", "-y",
        # Inputs
        "-stream_loop", "-1", "-i", str(stock_video),    # [0] Stock background (looped)
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(original_video),                        # [1] Original video segment
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(mask_video),                            # [2] Alpha mask segment
        "-filter_complex",
        # Scale all to same size and sync timing
        "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
        "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
        "[2:v]scale=1280:720,setpts=PTS-STARTPTS,format=gray[mask];"
        # Apply maskedmerge: bg where mask is black, fg where mask is white
        "[bg][fg][mask]maskedmerge[out]",
        "-map", "[out]",
        "-map", "1:a?",  # Keep audio from original
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
    """Apply real Coverr stock videos using the alpha mask method."""
    
    # Paths
    project_folder = Path("uploads/assets/videos/ai_math1")
    original_video = project_folder / "ai_math1_final.mp4"
    mask_video = project_folder / "ai_math1_rvm_mask_5s_024078685789.mp4"
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Stock videos downloaded from Coverr
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
    
    print("🎬 Applying REAL stock videos with ALPHA MASK (correct method)...\n")
    
    processed = []
    
    for i, seg in enumerate(stock_segments):
        print(f"[{i+1}/{len(stock_segments)}] {seg['description']}")
        
        stock_path = project_folder / seg['stock_video']
        if not stock_path.exists():
            print(f"  ⚠️  Stock video not found: {stock_path.name}")
            continue
        
        # Apply using alpha mask
        output_path = output_dir / f"final_segment_{i}.mp4"
        print(f"  📹 Stock: {stock_path.name}")
        print(f"  🎭 Applying alpha mask compositing...")
        
        success = apply_background_with_mask(
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
    print("🎬 Creating final video with proper masking...")
    concat_list = output_dir / "final_concat.txt"
    with open(concat_list, 'w') as f:
        for path in processed:
            f.write(f"file '{path.absolute()}'\n")
    
    final = output_dir / "ai_math1_FINAL_stock_backgrounds.mp4"
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
    
    print(f"✅ FINAL VIDEO: {final}\n")
    
    # Create side-by-side comparison with original
    print("📹 Creating comparison video...")
    
    # Extract 5 seconds from original
    original_5s = output_dir / "original_5s.mp4"
    extract_cmd = [
        "ffmpeg", "-y",
        "-i", str(original_video),
        "-t", "5",
        "-c", "copy",
        str(original_5s)
    ]
    subprocess.run(extract_cmd, check=True, capture_output=True)
    
    # Side-by-side
    comparison = output_dir / "final_comparison.mp4"
    compare_cmd = [
        "ffmpeg", "-y",
        "-i", str(original_5s),
        "-i", str(final),
        "-filter_complex",
        "[0:v]scale=640:360,drawtext=text='ORIGINAL':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=5[left];"
        "[1:v]scale=640:360,drawtext=text='WITH STOCK BACKGROUNDS':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=5[right];"
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
    
    print("📌 Summary:")
    print("   - Used RVM alpha mask for precise compositing")
    print("   - No chromakey = no green edges")
    print("   - Stock videos appear ONLY in background")
    print("   - Foreground (speaker) remains fully intact")
    
    # Open results
    subprocess.run(["open", str(final)])
    subprocess.run(["open", str(comparison)])


if __name__ == "__main__":
    main()