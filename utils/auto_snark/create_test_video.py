#!/usr/bin/env python3
"""
Create a simple test video with audio for testing the snark narrator
"""

import subprocess
import os

def create_test_video():
    """Create a 70-second test video with audio using FFmpeg"""
    
    output_file = "test_video.mp4"
    
    # Create a test video with:
    # - Color bars pattern
    # - Sine wave audio
    # - 70 seconds duration
    # - 30 fps
    # - 1280x720 resolution
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=70:size=1280x720:rate=30",  # Video source
        "-f", "lavfi", "-i", "sine=frequency=440:duration=70",              # Audio source
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",                # Video encoding
        "-c:a", "aac", "-b:a", "128k",                                     # Audio encoding
        "-pix_fmt", "yuv420p",                                             # Pixel format
        "-movflags", "+faststart",                                         # Web optimization
        output_file
    ]
    
    print(f"🎬 Creating test video: {output_file}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Test video created successfully: {output_file}")
        # Get file size
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"📊 File size: {size_mb:.2f} MB")
        return output_file
    else:
        print(f"❌ Error creating test video:")
        print(result.stderr)
        return None

if __name__ == "__main__":
    create_test_video()