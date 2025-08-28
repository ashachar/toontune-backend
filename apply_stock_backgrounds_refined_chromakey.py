#!/usr/bin/env python3
"""
Final refined version with more selective chromakey to preserve the blue ball.
"""

import subprocess
from pathlib import Path


def apply_refined_chromakey(green_video, stock_video, output_path, start_time, duration):
    """
    Use more selective chromakey to avoid removing blue/cyan from foreground.
    """
    
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(stock_video),    # Background
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(green_video),                          # Green screen
        "-filter_complex",
        "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
        "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
        # More selective chromakey - tighter thresholds
        # Lower similarity (0.08) = only very green pixels removed
        # Lower blend (0.04) = sharper edges, less feathering
        "[fg]chromakey=green:0.08:0.04[keyed];"
        # Lighter despill to preserve original colors
        "[keyed]despill=type=green:mix=0.2:expand=0[clean];"
        # Overlay
        "[bg][clean]overlay=shortest=1[out]",
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
        print(f"  Error: {result.stderr[-200:]}")
        return False
    return True


def test_different_thresholds(green_video, stock_video, output_dir):
    """Test various chromakey thresholds to find optimal settings."""
    
    print("🧪 Testing different chromakey thresholds...\n")
    
    test_configs = [
        {"similarity": 0.05, "blend": 0.02, "name": "very_tight"},
        {"similarity": 0.08, "blend": 0.04, "name": "tight"},
        {"similarity": 0.10, "blend": 0.05, "name": "moderate"},
        {"similarity": 0.12, "blend": 0.06, "name": "loose"},
    ]
    
    for config in test_configs:
        output = output_dir / f"test_chromakey_{config['name']}.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(stock_video),
            "-i", str(green_video), "-t", "2",
            "-filter_complex",
            f"[0:v]scale=1280:720[bg];"
            f"[1:v]scale=1280:720[fg];"
            f"[fg]chromakey=green:{config['similarity']}:{config['blend']}[keyed];"
            f"[bg][keyed]overlay[out]",
            "-map", "[out]",
            str(output)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {config['name']}: similarity={config['similarity']}, blend={config['blend']}")
        else:
            print(f"✗ {config['name']} failed")
    
    print("\nCheck which threshold preserves the blue ball best!")


def main():
    """Apply stock backgrounds with refined chromakey settings."""
    
    project_folder = Path("uploads/assets/videos/ai_math1")
    green_video = project_folder / "ai_math1_rvm_green_5s_024078685789.mp4"
    
    output_dir = Path("outputs")
    
    # First test different thresholds with one stock video
    test_stock = project_folder / "ai_math1_background_7_2_9_5_zwvgcNhxei.mp4"
    if test_stock.exists():
        test_different_thresholds(green_video, test_stock, output_dir)
    
    # Stock videos
    stock_segments = [
        {
            "start": 0.0,
            "duration": 1.25,
            "stock_video": "ai_math1_background_7_2_9_5_zwvgcNhxei.mp4",
            "description": "Math"
        },
        {
            "start": 1.25,
            "duration": 1.25,
            "stock_video": "ai_math1_background_27_9_31_6_6l4lJ9gGfk.mp4",
            "description": "AI"
        },
        {
            "start": 2.5,
            "duration": 1.25,
            "stock_video": "ai_math1_background_112_7_127_2_isS82K91sI.mp4",
            "description": "Data"
        },
        {
            "start": 3.75,
            "duration": 1.25,
            "stock_video": "ai_math1_background_145_2_152_0_pZm2kl1dD3.mp4",
            "description": "Trends"
        }
    ]
    
    print("\n🎬 Applying with REFINED chromakey (preserving blue ball)...\n")
    
    processed = []
    
    for i, seg in enumerate(stock_segments):
        print(f"[{i+1}/4] {seg['description']}")
        
        stock_path = project_folder / seg['stock_video']
        if not stock_path.exists():
            continue
        
        output_path = output_dir / f"refined_segment_{i}.mp4"
        
        success = apply_refined_chromakey(
            green_video, stock_path, output_path,
            seg['start'], seg['duration']
        )
        
        if success:
            processed.append(output_path)
            print(f"  ✅ Complete\n")
    
    if processed:
        # Create final
        concat_list = output_dir / "refined_concat.txt"
        with open(concat_list, 'w') as f:
            for path in processed:
                f.write(f"file '{path.absolute()}'\n")
        
        final = output_dir / "ai_math1_REFINED_FINAL.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(final)
        ], check=True, capture_output=True)
        
        print(f"✅ FINAL (Refined): {final}")
        print("\n📌 Used tighter chromakey thresholds to preserve blue/cyan objects")
        
        subprocess.run(["open", str(final)])


if __name__ == "__main__":
    main()