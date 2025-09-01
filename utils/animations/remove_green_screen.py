#!/usr/bin/env python3
"""
Remove green screen from video and replace with black background for full opacity
"""

import subprocess
import os


def remove_green_screen(input_video: str, output_video: str):
    """
    Remove green screen from video and replace with black background.
    This ensures the character is fully opaque.
    """
    
    print(f"Removing green screen from: {input_video}")
    
    # Get video properties
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'csv=s=x:p=0',
        input_video
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    parts = result.stdout.strip().split('x')
    width = int(parts[0])
    height = int(parts[1])
    fps_str = parts[2]
    
    # Remove green screen and composite on black background
    # MUST composite on black to ensure full opacity
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi', '-i', f'color=black:s={width}x{height}',
        '-i', input_video,
        '-filter_complex',
        '[1:v]chromakey=green:0.15:0.1[keyed];[0:v][keyed]overlay=shortest=1,format=yuv420p',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        output_video
    ]
    
    print("Applying chromakey filter to remove green...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Chromakey failed, trying colorkey with specific green #7BFA9D")
        # Try with specific Runway green color
        cmd = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-vf', 'colorkey=0x7BFA9D:0.3:0.1,format=yuv420p',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            output_video
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error removing green screen: {result.stderr}")
            return False
    
    print(f"Green screen removed successfully: {output_video}")
    return True