#!/usr/bin/env python3
"""
Utility script for video encoding - designed to be called by Claude subagents.
Provides simple, reliable video encoding with QuickTime compatibility.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import json


def encode_video(input_path, output_path=None, quality="standard", verbose=True):
    """
    Encode a video for QuickTime compatibility.
    
    Args:
        input_path: Path to input video
        output_path: Optional output path (auto-generated if None)
        quality: "standard", "high", or "compatible"
        verbose: Print progress messages
    
    Returns:
        Path to encoded video or None if failed
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        if verbose:
            print(f"❌ Error: Input file not found: {input_path}")
        return None
    
    # Generate output path if not provided
    if output_path is None:
        suffix = "_quicktime" if quality == "standard" else f"_{quality}"
        output_path = input_path.parent / f"{input_path.stem}{suffix}.mp4"
    else:
        output_path = Path(output_path)
    
    # Select encoding parameters based on quality
    if quality == "high":
        params = [
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.0"
        ]
    elif quality == "compatible":
        params = [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        ]
    else:  # standard
        params = [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.0"
        ]
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        *params,
        "-movflags", "+faststart",
        "-y",  # Overwrite output
        str(output_path)
    ]
    
    if verbose:
        print(f"🎬 Encoding {input_path.name} ({quality} quality)...")
    
    try:
        # Run FFmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0 and output_path.exists():
            if verbose:
                print(f"✅ Success: {output_path}")
            return str(output_path)
        else:
            if verbose:
                print(f"⚠️  Encoding failed, trying compatible mode...")
            
            # Try fallback encoding
            if quality != "compatible":
                return encode_video(input_path, output_path, "compatible", verbose)
            else:
                if verbose:
                    print(f"❌ Failed to encode video")
                return None
                
    except Exception as e:
        if verbose:
            print(f"❌ Error: {e}")
        return None


def verify_video(video_path, verbose=True):
    """
    Verify a video file's encoding and compatibility.
    
    Returns:
        Dict with encoding info or None if invalid
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        return None
    
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,profile",
            "-of", "json",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        if info.get("streams"):
            stream = info["streams"][0]
            
            # Check compatibility
            is_compatible = (
                stream.get("codec_name") == "h264" and
                stream.get("pix_fmt") in ["yuv420p", "yuv422p", "yuv444p"] and
                stream.get("width", 0) % 2 == 0 and
                stream.get("height", 0) % 2 == 0
            )
            
            if verbose:
                print(f"\n📊 Video Info for {video_path.name}:")
                print(f"  Codec: {stream.get('codec_name', 'unknown')}")
                print(f"  Pixel Format: {stream.get('pix_fmt', 'unknown')}")
                print(f"  Dimensions: {stream.get('width')}x{stream.get('height')}")
                print(f"  Profile: {stream.get('profile', 'unknown')}")
                print(f"  QuickTime Compatible: {'✅ Yes' if is_compatible else '❌ No'}")
            
            return {
                "codec": stream.get("codec_name"),
                "pix_fmt": stream.get("pix_fmt"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "profile": stream.get("profile"),
                "compatible": is_compatible
            }
            
    except Exception as e:
        if verbose:
            print(f"❌ Could not verify video: {e}")
        return None


def batch_encode(directory, pattern="*.mp4", quality="standard", verbose=True):
    """
    Batch encode all videos matching pattern in directory.
    
    Returns:
        List of successfully encoded video paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        if verbose:
            print(f"❌ Directory not found: {directory}")
        return []
    
    videos = list(directory.glob(pattern))
    
    if verbose:
        print(f"🎬 Found {len(videos)} videos to encode")
    
    encoded = []
    
    for i, video in enumerate(videos, 1):
        if verbose:
            print(f"\n[{i}/{len(videos)}] Processing {video.name}...")
        
        output = encode_video(video, quality=quality, verbose=verbose)
        
        if output:
            encoded.append(output)
    
    if verbose:
        print(f"\n✅ Batch encoding complete: {len(encoded)}/{len(videos)} successful")
    
    return encoded


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Video encoder utility for QuickTime compatibility"
    )
    
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("-o", "--output", help="Output path (optional)")
    parser.add_argument(
        "-q", "--quality",
        choices=["standard", "high", "compatible"],
        default="standard",
        help="Encoding quality preset"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch process directory"
    )
    parser.add_argument(
        "--pattern",
        default="*.mp4",
        help="File pattern for batch mode"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify video encoding instead of encoding"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if args.verify:
        # Verify mode
        info = verify_video(args.input, verbose)
        if info and not info["compatible"]:
            sys.exit(1)  # Non-zero exit if not compatible
    elif args.batch:
        # Batch mode
        encoded = batch_encode(
            args.input,
            args.pattern,
            args.quality,
            verbose
        )
        if verbose and encoded:
            print("\nEncoded files:")
            for path in encoded:
                print(f"  • {path}")
    else:
        # Single file mode
        output = encode_video(
            args.input,
            args.output,
            args.quality,
            verbose
        )
        if not output:
            sys.exit(1)  # Non-zero exit on failure


if __name__ == "__main__":
    main()