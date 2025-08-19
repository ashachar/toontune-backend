#!/usr/bin/env python3
"""
Video trimming utility for cutting videos at specific timestamps.
"""

import argparse
import subprocess
import os
from pathlib import Path
import sys


def get_video_duration(input_path):
    """Get the duration of a video in seconds."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting video duration: {e}")
    return None


def trim_video(input_path, output_path=None, start_time=None, end_time=None, duration=None):
    """
    Trim a video file.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to output video (optional, auto-generated if not provided)
        start_time (float): Start time in seconds (default: 0)
        end_time (float): End time in seconds (optional)
        duration (float): Duration in seconds from start_time (optional)
    
    Returns:
        str: Path to the trimmed video file
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    # Get video info
    total_duration = get_video_duration(str(input_path))
    if total_duration:
        print(f"📹 Input video duration: {total_duration:.2f}s")
    
    # Handle time parameters
    if start_time is None:
        start_time = 0
    
    # Build output filename if not provided
    if output_path is None:
        stem = input_path.stem
        suffix = input_path.suffix
        
        if end_time is not None:
            time_suffix = f"_{start_time}s_to_{end_time}s"
        elif duration is not None:
            time_suffix = f"_{start_time}s_for_{duration}s"
        else:
            time_suffix = f"_from_{start_time}s"
        
        output_path = input_path.parent / f"{stem}{time_suffix}_trimmed{suffix}"
    
    output_path = Path(output_path)
    
    # Build FFmpeg command
    cmd = ['ffmpeg', '-i', str(input_path)]
    
    # Add start time
    if start_time > 0:
        cmd.extend(['-ss', str(start_time)])
    
    # Add duration or end time
    if duration is not None:
        cmd.extend(['-t', str(duration)])
    elif end_time is not None:
        if end_time <= start_time:
            raise ValueError(f"End time ({end_time}) must be greater than start time ({start_time})")
        duration = end_time - start_time
        cmd.extend(['-t', str(duration)])
    
    # Copy codecs to avoid re-encoding when possible
    cmd.extend(['-c', 'copy'])
    
    # Add output file (overwrite if exists)
    cmd.extend(['-y', str(output_path)])
    
    print(f"✂️  Trimming video...")
    print(f"   Start: {start_time}s")
    if end_time:
        print(f"   End: {end_time}s")
    elif duration:
        print(f"   Duration: {duration}s")
    else:
        print(f"   End: (until end of video)")
    
    # Run FFmpeg
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ FFmpeg error: {result.stderr}")
            return None
        
        # Get output file size
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            
            # Get new duration
            new_duration = get_video_duration(str(output_path))
            
            print(f"\n✅ Success!")
            print(f"   Output: {output_path}")
            print(f"   Size: {size_mb:.2f} MB")
            if new_duration:
                print(f"   Duration: {new_duration:.2f}s")
            
            return str(output_path)
    
    except Exception as e:
        print(f"❌ Error trimming video: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Trim video files at specific timestamps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Trim from second 5 to end
  python trim_video.py video.mp4 --start 5
  
  # Trim from second 5 to second 30
  python trim_video.py video.mp4 --start 5 --end 30
  
  # Trim first 10 seconds (from 0 to 10)
  python trim_video.py video.mp4 --end 10
  
  # Extract 20 seconds starting from second 5
  python trim_video.py video.mp4 --start 5 --duration 20
  
  # Specify custom output path
  python trim_video.py video.mp4 --start 5 --output trimmed.mp4
        """
    )
    
    parser.add_argument('input', help='Input video file')
    parser.add_argument('-o', '--output', help='Output video file (auto-generated if not specified)')
    parser.add_argument('-s', '--start', type=float, default=0, 
                       help='Start time in seconds (default: 0)')
    parser.add_argument('-e', '--end', type=float, 
                       help='End time in seconds')
    parser.add_argument('-d', '--duration', type=float,
                       help='Duration in seconds from start time')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.end is not None and args.duration is not None:
        print("❌ Error: Cannot specify both --end and --duration")
        sys.exit(1)
    
    # Trim the video
    result = trim_video(
        input_path=args.input,
        output_path=args.output,
        start_time=args.start,
        end_time=args.end,
        duration=args.duration
    )
    
    if result is None:
        sys.exit(1)


if __name__ == '__main__':
    main()