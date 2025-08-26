#!/usr/bin/env python3
"""
Integration module for using video encoder with animation pipeline.
This ensures all animation outputs are QuickTime-compatible.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_encoder import VideoEncoder


def encode_animation_output(input_path: str, quality: str = "high") -> str:
    """
    Encode animation output for guaranteed QuickTime compatibility.
    
    Args:
        input_path: Path to animation output video
        quality: "standard" or "high" quality preset
        
    Returns:
        Path to properly encoded video
    """
    encoder = VideoEncoder(verbose=True)
    
    input_path = Path(input_path)
    output_path = input_path.parent / f"{input_path.stem}_final.mp4"
    
    print(f"[ANIMATION_ENCODER] Ensuring QuickTime compatibility for {input_path.name}...")
    
    try:
        final_path = encoder.encode_for_quicktime(
            str(input_path),
            str(output_path),
            quality=quality,
            custom_settings={
                "crf": "18" if quality == "high" else "23",
                "preset": "slow" if quality == "high" else "medium"
            }
        )
        
        print(f"[ANIMATION_ENCODER] ✅ Final video ready: {final_path}")
        return final_path
        
    except Exception as e:
        print(f"[ANIMATION_ENCODER] ❌ Encoding failed: {e}")
        print(f"[ANIMATION_ENCODER] Returning original file: {input_path}")
        return str(input_path)


def batch_encode_animations(directory: str, pattern: str = "*_hq.mp4") -> list:
    """
    Batch encode all animation outputs in a directory.
    
    Args:
        directory: Directory containing animation videos
        pattern: File pattern to match
        
    Returns:
        List of encoded video paths
    """
    encoder = VideoEncoder(verbose=True)
    
    return encoder.batch_encode(
        directory,
        output_dir=directory,
        pattern=pattern,
        quality="high"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Encode animation outputs for QuickTime compatibility"
    )
    parser.add_argument("input", help="Input video or directory")
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument("--pattern", default="*_hq.mp4", help="Pattern for batch mode")
    parser.add_argument("--quality", choices=["standard", "high"], default="high")
    
    args = parser.parse_args()
    
    if args.batch:
        encoded = batch_encode_animations(args.input, args.pattern)
        print(f"\n✅ Encoded {len(encoded)} animation videos")
    else:
        output = encode_animation_output(args.input, args.quality)
        print(f"\n✅ Final video: {output}")