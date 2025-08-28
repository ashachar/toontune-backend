#!/usr/bin/env python3
"""
Create zoom out effect on background-replaced video.
Uses cached RVM output for efficient processing.
"""

import subprocess
from pathlib import Path
import tempfile


def create_zoom_out_video(input_video, background_video, rvm_green_screen, output_path, duration=5.0):
    """
    Create a video with zoom out effect starting immediately.
    
    Args:
        input_video: Original video path (for audio)
        background_video: Background video path
        rvm_green_screen: Cached RVM green screen output
        output_path: Output video path
        duration: Duration in seconds
    """
    print("=" * 60)
    print("Creating Zoom Out Effect with Background Replacement")
    print("=" * 60)
    
    print(f"Input: {input_video}")
    print(f"Background: {background_video}")
    print(f"RVM Green Screen: {rvm_green_screen}")
    print(f"Duration: {duration}s")
    
    # First, create the composited video with zoom effect
    # Zoom starts at 1.5x and goes to 1.0x over the duration
    print("\n🎬 Creating zoom out effect...")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(background_video),
        "-i", str(rvm_green_screen),
        "-filter_complex",
        # Scale background to HD
        "[0:v]scale=1280:720[bg];"
        # Apply chromakey to remove green from RVM output with better tolerance
        "[1:v]chromakey=green:0.10:0.08[ckout];"
        # Apply despill to remove green edges
        "[ckout]despill=type=green:mix=0.5:expand=0.0[despill];"
        # Composite foreground on background
        "[bg][despill]overlay=0:0:shortest=1[composite];"
        # Apply zoom out effect on the playing video
        # Starts at 10x zoom (1000%) and rapidly transitions to 1x (100%) within 0.2 seconds
        # d=1 means process each frame individually (allows video to play)
        f"[composite]zoompan=z='if(lte(in,5),10-9*in/5,1)':"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d=1:"
        f"s=1280x720:"
        f"fps=25[out]",
        "-map", "[out]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-t", str(duration),
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in zoom effect creation: {result.stderr}")
        return None
    
    print(f"✓ Zoom effect applied")
    
    # Add audio from original video
    print("\n🔊 Adding audio from original...")
    final_output = output_path.parent / f"{output_path.stem}_with_audio.mp4"
    
    cmd_audio = [
        "ffmpeg", "-y",
        "-i", str(output_path),
        "-i", str(input_video),
        "-c:v", "copy",
        "-map", "0:v",
        "-map", "1:a?",
        "-shortest",
        "-t", str(duration),
        str(final_output)
    ]
    
    result = subprocess.run(cmd_audio, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: Audio addition failed (video might not have audio)")
        return output_path
    
    print(f"✓ Audio added")
    
    # Remove intermediate file
    output_path.unlink()
    
    return final_output


def main():
    """Create zoom out effect on ai_math1 with background replacement."""
    
    # Setup paths
    input_video = Path("uploads/assets/videos/ai_math1.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    rvm_green_screen = Path("uploads/assets/videos/ai_math1/ai_math1_rvm_green_5s_024078685789.mp4")
    output_video = Path("outputs/ai_math1_zoomout.mp4")
    
    # Validate inputs
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return
    
    if not background_video.exists():
        print(f"Error: Background video not found: {background_video}")
        return
    
    if not rvm_green_screen.exists():
        print(f"Error: RVM green screen not found: {rvm_green_screen}")
        print("Run cached_rvm.py first to generate the green screen output")
        return
    
    # Create zoom out effect
    final_output = create_zoom_out_video(
        input_video,
        background_video,
        rvm_green_screen,
        output_video,
        duration=5.0
    )
    
    if final_output and final_output.exists():
        print(f"\n✅ Success! Output saved to: {final_output}")
        print("\nOpening result...")
        subprocess.run(["open", str(final_output)])
    else:
        print("\n❌ Failed to create zoom out video")


if __name__ == "__main__":
    main()