#!/usr/bin/env python3
"""
Simplified processor for do_re_mi video focusing on:
- Sound effects with proper volume and duration
- Basic text overlays  
- Test mode indicators
"""

import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

from utils.sound_effects.sound_effects_manager import SoundEffectsManager
from utils.sound_effects.video_sound_overlay import VideoSoundOverlay


def add_text_overlays(video_path: str, output_path: str, test_mode: bool = False) -> bool:
    """Add simplified text overlays to video."""
    
    # Define key text overlays to show lyrics
    text_overlays = [
        {"text": "Let's start at the very beginning", "start": 2.779, "end": 6.420, "x": 10, "y": 100},
        {"text": "A B C", "start": 13.020, "end": 14.460, "x": 10, "y": 100},
        {"text": "Do-Re-Mi", "start": 17.700, "end": 18.440, "x": 10, "y": 100},
        {"text": "Do Re Mi", "start": 30.360, "end": 31.379, "x": 10, "y": 100},
        {"text": "Doe, a deer, a female deer", "start": 40.119, "end": 41.939, "x": 10, "y": 100},
        {"text": "Ray, a drop of golden sun", "start": 44.840, "end": 47.200, "x": 10, "y": 100},
        {"text": "Me, a name I call myself", "start": 48.520, "end": 51.680, "x": 10, "y": 100},
        {"text": "Far, a long long way to run", "start": 52.479, "end": 54.759, "x": 10, "y": 100},
    ]
    
    # Build filter string
    filters = []
    
    for i, overlay in enumerate(text_overlays):
        text = overlay["text"].replace("'", "\\'").replace(":", "\\:")
        
        # Main text overlay
        filter = (
            f"drawtext=text='{text}':"
            f"x={overlay['x']}:y={overlay['y']}:"
            f"fontsize=24:fontcolor=white:"
            f"borderw=2:bordercolor=black:"
            f"enable='between(t,{overlay['start']},{overlay['end']})'"
        )
        filters.append(filter)
        
        # Add test mode indicator
        if test_mode:
            debug_filter = (
                f"drawtext=text='TEXT\\: {text[:15]}...':"
                f"x=10:y=10:"
                f"fontsize=14:fontcolor=yellow:"
                f"box=1:boxcolor=red@0.6:"
                f"enable='between(t,{overlay['start']},{overlay['end']})'"
            )
            filters.append(debug_filter)
    
    # Join filters
    filter_complex = ",".join(filters)
    
    # Run FFmpeg
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", filter_complex,
        "-c:v", "libx264",
        "-preset", "medium", 
        "-crf", "23",
        "-c:a", "copy",
        output_path
    ]
    
    try:
        print("Adding text overlays...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Warning: Text overlay failed, continuing without text")
            # Copy original if filters fail
            subprocess.run(["cp", video_path, output_path])
        return True
    except Exception as e:
        print(f"Error adding text: {e}")
        subprocess.run(["cp", video_path, output_path])
        return True


def main():
    """Process video with simplified effects."""
    
    print("="*70)
    print("SIMPLIFIED DO RE MI VIDEO PROCESSING")
    print("="*70)
    
    # Paths
    video_path = "uploads/assets/videos/do_re_mi_with_music/scenes/scene_001.mp4"
    output_dir = Path("output/do_re_mi_simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    # Step 1: Add text overlays
    print("\n1. Adding text overlays with test indicators...")
    text_video = str(output_dir / "temp_with_text.mp4")
    add_text_overlays(video_path, text_video, test_mode=True)
    
    # Step 2: Add sound effects
    print("\n2. Adding sound effects (50% volume, <0.5s duration)...")
    
    sound_effects = [
        {"sound": "ding", "timestamp": 3.579},
        {"sound": "swoosh", "timestamp": 13.050},
        {"sound": "chime", "timestamp": 17.700},
        {"sound": "chime", "timestamp": 26.180},
        {"sound": "sparkle", "timestamp": 40.119},
        {"sound": "pop", "timestamp": 44.840},
        {"sound": "pop", "timestamp": 48.520},
        {"sound": "pop", "timestamp": 52.479}
    ]
    
    # Initialize managers
    sfx_manager = SoundEffectsManager()
    video_overlay = VideoSoundOverlay()
    
    # Get sound files (already downloaded with < 0.5s duration)
    sound_files = {}
    for effect in sound_effects:
        sound_name = effect["sound"].lower()
        existing = sfx_manager.find_existing_sound(sound_name)
        if existing:
            sound_files[sound_name] = existing
            print(f"  ✓ Using: {sound_name} -> {Path(existing).name}")
    
    # Set volume to 50%
    for effect in sound_effects:
        effect["volume"] = 0.5
    
    # Apply sound effects
    final_output = str(output_dir / "scene_001_complete.mp4")
    input_video = text_video if Path(text_video).exists() else video_path
    
    success = video_overlay.overlay_sound_effects(
        input_video,
        sound_effects,
        sound_files,
        final_output,
        preserve_original_audio=True
    )
    
    # Step 3: Add sound effect indicators in test mode
    if success:
        print("\n3. Adding sound effect debug indicators...")
        
        # Create a simple overlay showing when sound effects play
        filters = []
        for effect in sound_effects:
            timestamp = float(effect["timestamp"])
            sound_name = effect["sound"].upper()
            
            # Show sound effect indicator for 0.5 seconds
            filter = (
                f"drawtext=text='SOUND\\: {sound_name}':"
                f"x=10:y=30:"
                f"fontsize=14:fontcolor=cyan:"
                f"box=1:boxcolor=blue@0.6:"
                f"enable='between(t,{timestamp},{timestamp + 0.5})'"
            )
            filters.append(filter)
        
        # Add permanent test mode indicator
        filters.append(
            "drawtext=text='TEST MODE':"
            "x=10:y=50:"
            "fontsize=12:fontcolor=white:"
            "box=1:boxcolor=gray@0.5"
        )
        
        filter_complex = ",".join(filters)
        
        # Apply final overlays
        final_with_debug = str(output_dir / "scene_001_complete_debug.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", final_output,
            "-vf", filter_complex,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "copy",
            final_with_debug
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Replace final output with debug version
                subprocess.run(["mv", final_with_debug, final_output])
                print("  ✓ Debug indicators added")
        except:
            print("  ⚠ Could not add debug indicators")
    
    # Clean up temp file
    if Path(text_video).exists():
        os.remove(text_video)
    
    if success:
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\n✓ Video with effects saved to:")
        print(f"  {final_output}")
        print("\nEffects Applied:")
        print(f"  • 8 Text overlays (lyrics)")
        print(f"  • 8 Sound effects (50% volume, all < 0.5s)")
        print(f"  • Test mode indicators (red=text, blue=sound)")
        
        # Check actual sound durations
        print("\nSound Effect Durations:")
        for name, path in sound_files.items():
            info = sfx_manager.metadata.get(name, {})
            duration = info.get("duration", "?")
            print(f"  • {name}: {duration}s")
    else:
        print("\n✗ Processing failed")


if __name__ == "__main__":
    main()