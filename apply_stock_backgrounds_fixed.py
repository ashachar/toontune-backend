#!/usr/bin/env python3
"""
Apply stock backgrounds correctly - only replacing background, not affecting foreground.
Uses the RVM mask directly for proper alpha compositing.
"""

import subprocess
from pathlib import Path


def apply_background_with_alpha_mask(original_video, stock_video, mask_video, output_path, start_time, duration):
    """
    Use the RVM alpha mask to properly composite foreground over stock background.
    The mask defines: white (255) = foreground to keep, black (0) = background to replace.
    """
    
    cmd = [
        "ffmpeg", "-y",
        # Input streams
        "-stream_loop", "-1", "-i", str(stock_video),    # [0] Stock background (looped)
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(original_video),                       # [1] Original video segment
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(mask_video),                           # [2] Alpha mask
        "-filter_complex",
        # Scale everything to same size
        "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
        "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
        "[2:v]scale=1280:720,setpts=PTS-STARTPTS,format=gray[mask];"
        # Use mask to blend: where mask is white, show foreground; where black, show background
        "[bg][fg][mask]maskedmerge[out]",
        "-map", "[out]",
        "-map", "1:a?",  # Audio from original
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


def apply_background_with_green_screen_improved(green_video, stock_video, output_path, start_time, duration):
    """
    Alternative: Use green screen but with more conservative chromakey settings.
    """
    
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(stock_video),    # Background
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(green_video),                          # Green screen
        "-filter_complex",
        "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
        "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
        # More conservative chromakey - only remove pure green
        "[fg]chromakey=0x00FF00:0.05:0.05[keyed];"  # Tighter similarity and blend
        # Overlay without despill to preserve original colors
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


def main():
    """Apply stock backgrounds correctly using alpha mask."""
    
    # Paths
    project_folder = Path("uploads/assets/videos/ai_math1")
    original_video = project_folder / "ai_math1_final.mp4"
    mask_video = project_folder / "ai_math1_rvm_mask_5s_024078685789.mp4"
    green_video = project_folder / "ai_math1_rvm_green_5s_024078685789.mp4"
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Stock videos to use
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
            "description": "AI - AI art video"
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
    
    print("Applying stock backgrounds CORRECTLY (preserving foreground)...\n")
    print("Using ALPHA MASK method for proper compositing\n")
    
    processed = []
    processed_green = []
    
    for i, seg in enumerate(stock_segments):
        print(f"[{i+1}/{len(stock_segments)}] {seg['description']}")
        
        stock_path = project_folder / seg['stock_video']
        if not stock_path.exists():
            print(f"  ⚠️  Stock video not found")
            continue
        
        # Method 1: Use alpha mask (most accurate)
        output_mask = output_dir / f"fixed_mask_segment_{i}.mp4"
        print(f"  Applying with alpha mask...")
        
        success = apply_background_with_alpha_mask(
            original_video, stock_path, mask_video, output_mask,
            seg['start'], seg['duration']
        )
        
        if success:
            processed.append(output_mask)
            print(f"  ✓ Mask method complete")
        
        # Method 2: Conservative chromakey (for comparison)
        output_green = output_dir / f"fixed_green_segment_{i}.mp4"
        print(f"  Applying with conservative chromakey...")
        
        success = apply_background_with_green_screen_improved(
            green_video, stock_path, output_green,
            seg['start'], seg['duration']
        )
        
        if success:
            processed_green.append(output_green)
            print(f"  ✓ Chromakey method complete\n")
    
    # Create final videos
    if processed:
        print("Creating final video (mask method)...")
        concat_list = output_dir / "fixed_concat_mask.txt"
        with open(concat_list, 'w') as f:
            for path in processed:
                f.write(f"file '{path.absolute()}'\n")
        
        final_mask = output_dir / "ai_math1_FIXED_stock_backgrounds.mp4"
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(final_mask)
        ]
        subprocess.run(concat_cmd, check=True, capture_output=True)
        print(f"✅ Fixed version (mask): {final_mask}\n")
    
    if processed_green:
        print("Creating final video (chromakey method)...")
        concat_list_green = output_dir / "fixed_concat_green.txt"
        with open(concat_list_green, 'w') as f:
            for path in processed_green:
                f.write(f"file '{path.absolute()}'\n")
        
        final_green = output_dir / "ai_math1_FIXED_chromakey_backgrounds.mp4"
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list_green),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(final_green)
        ]
        subprocess.run(concat_cmd, check=True, capture_output=True)
        print(f"✅ Fixed version (chromakey): {final_green}\n")
    
    # Create comparison
    if processed and processed_green:
        print("Creating method comparison...")
        
        comparison = output_dir / "mask_vs_chromakey_comparison.mp4"
        compare_cmd = [
            "ffmpeg", "-y",
            "-i", str(final_mask),
            "-i", str(final_green),
            "-filter_complex",
            "[0:v]scale=640:360,drawtext=text='MASK METHOD':x=10:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7[left];"
            "[1:v]scale=640:360,drawtext=text='CHROMAKEY METHOD':x=10:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7[right];"
            "[left][right]hstack[out]",
            "-map", "[out]",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(comparison)
        ]
        subprocess.run(compare_cmd, check=True, capture_output=True)
        print(f"✅ Method comparison: {comparison}\n")
        
        # Open results
        subprocess.run(["open", str(final_mask)])
        subprocess.run(["open", str(comparison)])
    
    print("📌 The MASK method should preserve the foreground perfectly")
    print("📌 The CHROMAKEY method uses conservative settings to avoid transparency")


if __name__ == "__main__":
    main()