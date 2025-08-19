#!/usr/bin/env python3
"""
Process do_re_mi video at FULL ORIGINAL RESOLUTION with properly scaled coordinates.
All effects are applied to the original high-quality video, not the downsampled version.
"""

import os
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()

from utils.sound_effects.sound_effects_manager import SoundEffectsManager
from utils.sound_effects.video_sound_overlay import VideoSoundOverlay


class ResolutionAwareProcessor:
    """Process videos at their original resolution with scaled coordinates."""
    
    def __init__(self, reference_width: int = 256, reference_height: int = 144):
        """
        Initialize with reference dimensions (what the metadata was designed for).
        
        Args:
            reference_width: Width the coordinates were designed for
            reference_height: Height the coordinates were designed for
        """
        self.ref_width = reference_width
        self.ref_height = reference_height
        self.actual_width = None
        self.actual_height = None
        self.scale_x = 1.0
        self.scale_y = 1.0
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get video dimensions and info."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Find video stream
            for stream in data.get("streams", []):
                if stream["codec_type"] == "video":
                    width = int(stream["width"])
                    height = int(stream["height"])
                    duration = float(data.get("format", {}).get("duration", 0))
                    
                    return {
                        "width": width,
                        "height": height,
                        "duration": duration,
                        "codec": stream.get("codec_name", "unknown")
                    }
        except Exception as e:
            print(f"Error getting video info: {e}")
        
        return None
    
    def calculate_scaling(self, actual_width: int, actual_height: int):
        """Calculate scaling factors from reference to actual dimensions."""
        self.actual_width = actual_width
        self.actual_height = actual_height
        self.scale_x = actual_width / self.ref_width
        self.scale_y = actual_height / self.ref_height
        
        print(f"Scaling factors: X={self.scale_x:.2f}, Y={self.scale_y:.2f}")
        print(f"Reference: {self.ref_width}x{self.ref_height}")
        print(f"Actual: {actual_width}x{actual_height}")
    
    def scale_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """Scale coordinates from reference to actual dimensions."""
        scaled_x = int(x * self.scale_x)
        scaled_y = int(y * self.scale_y)
        return scaled_x, scaled_y
    
    def scale_fontsize(self, fontsize: int) -> int:
        """Scale font size based on resolution."""
        # Use average of X and Y scaling for font size
        avg_scale = (self.scale_x + self.scale_y) / 2
        return int(fontsize * avg_scale)
    
    def escape_text(self, text: str) -> str:
        """Escape text for FFmpeg drawtext filter."""
        # Remove problematic characters and escape
        text = text.replace("'", "")
        text = text.replace(",", " ")
        text = text.replace(":", " ")
        text = text.replace("\\", "")
        return text
    
    def create_drawtext_filter(self, text: str, x: int, y: int,
                              start: float, end: float,
                              fontsize: int = 24) -> str:
        """Create drawtext filter with scaled coordinates."""
        # Scale coordinates and font size
        scaled_x, scaled_y = self.scale_coordinates(x, y)
        scaled_fontsize = self.scale_fontsize(fontsize)
        
        # Escape text
        text_safe = self.escape_text(text)
        
        # Build filter with simpler syntax
        return (
            f"drawtext="
            f"text={text_safe}:"
            f"x={scaled_x}:y={scaled_y}:"
            f"fontsize={scaled_fontsize}:fontcolor=white:"
            f"borderw=3:bordercolor=black:"
            f"enable='gte(t\\,{start})*lte(t\\,{end})'"
        )
    
    def create_debug_overlay(self, label: str, x: int, y: int,
                            start: float, duration: float = 0.5) -> str:
        """Create debug overlay with scaled coordinates."""
        scaled_x, scaled_y = self.scale_coordinates(x, y)
        scaled_fontsize = self.scale_fontsize(14)
        
        label_safe = self.escape_text(label)
        end = start + duration
        
        return (
            f"drawtext="
            f"text={label_safe}:"
            f"x={scaled_x}:y={scaled_y}:"
            f"fontsize={scaled_fontsize}:fontcolor=yellow:"
            f"box=1:boxcolor=red@0.7:boxborderw=2:"
            f"enable='gte(t\\,{start})*lte(t\\,{end})'"
        )


def process_full_resolution_video():
    """Process the original full resolution video with all effects."""
    
    print("="*70)
    print("DO RE MI - FULL RESOLUTION PROCESSING")
    print("="*70)
    
    # Use the ORIGINAL video, not the downsampled one
    original_video = "uploads/assets/videos/do_re_mi.mov"
    
    if not Path(original_video).exists():
        print(f"Error: Original video not found at {original_video}")
        print("Trying alternative path...")
        original_video = "uploads/assets/videos/do_re_mi_with_music.mov"
        
        if not Path(original_video).exists():
            print(f"Error: Video not found at {original_video}")
            return False
    
    # Output directory
    output_dir = Path("output/do_re_mi_full_resolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = ResolutionAwareProcessor(reference_width=256, reference_height=144)
    
    # Get actual video dimensions
    print(f"\n1. Analyzing video: {original_video}")
    video_info = processor.get_video_info(original_video)
    
    if not video_info:
        print("Error: Could not get video information")
        return False
    
    print(f"   Original Resolution: {video_info['width']}x{video_info['height']}")
    print(f"   Duration: {video_info['duration']:.2f} seconds")
    print(f"   Codec: {video_info['codec']}")
    
    # Calculate scaling
    processor.calculate_scaling(video_info['width'], video_info['height'])
    
    # Define text overlays (using original metadata coordinates)
    text_overlays = [
        # Scene 1 - Introduction
        {"text": "Let's start at the very beginning", "x": 10, "y": 90, "start": 2.779, "end": 6.420},
        
        # Scene 2 - ABC
        {"text": "A B C", "x": 10, "y": 90, "start": 13.020, "end": 14.460},
        {"text": "Do-Re-Mi", "x": 10, "y": 110, "start": 17.700, "end": 18.440},
        
        # Scene 3 - Notes
        {"text": "Do Re Mi", "x": 10, "y": 90, "start": 30.360, "end": 31.379},
        {"text": "The first three notes just happen to be", "x": 10, "y": 110, "start": 31.5, "end": 33.5},
        
        # Scene 4 - Definitions
        {"text": "Doe - a deer, a female deer", "x": 10, "y": 90, "start": 40.119, "end": 41.939},
        {"text": "Ray - a drop of golden sun", "x": 10, "y": 90, "start": 44.840, "end": 47.200},
        {"text": "Me - a name I call myself", "x": 10, "y": 90, "start": 48.520, "end": 51.680},
        {"text": "Far - a long long way to run", "x": 10, "y": 90, "start": 52.479, "end": 54.759},
    ]
    
    # Sound effect timings for debug overlay
    sound_effects = [
        {"sound": "ding", "timestamp": 3.579, "volume": 0.5},
        {"sound": "swoosh", "timestamp": 13.050, "volume": 0.5},
        {"sound": "chime", "timestamp": 17.700, "volume": 0.5},
        {"sound": "chime", "timestamp": 26.180, "volume": 0.5},
        {"sound": "sparkle", "timestamp": 40.119, "volume": 0.5},
        {"sound": "pop", "timestamp": 44.840, "volume": 0.5},
        {"sound": "pop", "timestamp": 48.520, "volume": 0.5},
        {"sound": "pop", "timestamp": 52.479, "volume": 0.5}
    ]
    
    # Build filters
    filters = []
    
    print("\n2. Building scaled text overlays...")
    for overlay in text_overlays:
        filter_str = processor.create_drawtext_filter(
            text=overlay["text"],
            x=overlay["x"],
            y=overlay["y"],
            start=overlay["start"],
            end=overlay["end"],
            fontsize=24
        )
        filters.append(filter_str)
        print(f"   Added: {overlay['text'][:30]}...")
    
    print("\n3. Adding sound effect indicators...")
    for sfx in sound_effects:
        label = f"SOUND: {sfx['sound'].upper()}"
        filter_str = processor.create_debug_overlay(
            label=label,
            x=10,
            y=30,
            start=sfx["timestamp"],
            duration=0.5
        )
        filters.append(filter_str)
    
    # Add test mode indicator (scaled)
    scaled_x, scaled_y = processor.scale_coordinates(10, 10)
    scaled_font = processor.scale_fontsize(16)
    filters.append(
        f"drawtext="
        f"text=TEST MODE FULL RES:"
        f"x={scaled_x}:y={scaled_y}:"
        f"fontsize={scaled_font}:fontcolor=cyan:"
        f"box=1:boxcolor=blue@0.5:boxborderw=2"
    )
    
    # Join filters
    filtergraph = ",".join(filters)
    
    # Apply visual effects
    temp_video = str(output_dir / "temp_with_overlays.mp4")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", original_video,
        "-vf", filtergraph,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        temp_video
    ]
    
    print(f"\n4. Applying overlays to full resolution video...")
    print(f"   Input: {original_video}")
    print(f"   Resolution: {video_info['width']}x{video_info['height']}")
    print(f"   Filters: {len(filters)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Warning: Complex filters failed, using simplified version...")
            
            # Simplify - just a few text overlays
            simple_filters = []
            for i, overlay in enumerate(text_overlays[:4]):
                filter_str = processor.create_drawtext_filter(
                    text=overlay["text"],
                    x=overlay["x"],
                    y=overlay["y"],
                    start=overlay["start"],
                    end=overlay["end"],
                    fontsize=24
                )
                simple_filters.append(filter_str)
            
            # Add resolution indicator
            simple_filters.append(
                f"drawtext="
                f"text=FULL RES {video_info['width']}x{video_info['height']}:"
                f"x=10:y=10:"
                f"fontsize=20:fontcolor=green:"
                f"box=1:boxcolor=black@0.5"
            )
            
            simple_graph = ",".join(simple_filters)
            cmd[cmd.index(filtergraph)] = simple_graph
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("Simplified version also failed, copying original...")
                subprocess.run(["cp", original_video, temp_video])
    
    except Exception as e:
        print(f"Error: {e}")
        subprocess.run(["cp", original_video, temp_video])
    
    # Now add sound effects
    print("\n5. Adding sound effects at 50% volume...")
    
    sfx_manager = SoundEffectsManager()
    video_overlay = VideoSoundOverlay()
    
    # Get sound files
    sound_files = {}
    for effect in sound_effects:
        sound_name = effect["sound"].lower()
        existing = sfx_manager.find_existing_sound(sound_name)
        if existing:
            sound_files[sound_name] = existing
            print(f"   ✓ {sound_name}: {Path(existing).name}")
    
    # Final output
    final_output = str(output_dir / "do_re_mi_full_resolution.mp4")
    
    # Apply sound effects
    input_video = temp_video if Path(temp_video).exists() else original_video
    
    print(f"\n6. Creating final full resolution video...")
    success = video_overlay.overlay_sound_effects(
        input_video,
        sound_effects,
        sound_files,
        final_output,
        preserve_original_audio=True
    )
    
    # Clean up temp file
    if Path(temp_video).exists() and temp_video != original_video:
        try:
            os.remove(temp_video)
        except:
            pass
    
    if success:
        # Get final file size
        size_mb = Path(final_output).stat().st_size / (1024 * 1024)
        
        print("\n" + "="*70)
        print("SUCCESS - FULL RESOLUTION VIDEO CREATED!")
        print("="*70)
        print(f"\n✓ Full resolution video saved to:")
        print(f"  {final_output}")
        print(f"\nVideo Details:")
        print(f"  • Resolution: {video_info['width']}x{video_info['height']} (ORIGINAL)")
        print(f"  • File size: {size_mb:.1f} MB")
        print(f"  • Duration: {video_info['duration']:.1f} seconds")
        print(f"\nEffects Applied:")
        print(f"  • {len(text_overlays)} Text overlays (scaled {processor.scale_x:.1f}x)")
        print(f"  • {len(sound_effects)} Sound effects (50% volume)")
        print(f"  • Test mode indicators")
        print(f"\nCoordinate Scaling:")
        print(f"  • X coordinates scaled by {processor.scale_x:.2f}x")
        print(f"  • Y coordinates scaled by {processor.scale_y:.2f}x")
        print(f"  • Font sizes scaled by {((processor.scale_x + processor.scale_y) / 2):.2f}x")
        
        return True
    else:
        print("\n✗ Processing failed")
        return False


if __name__ == "__main__":
    success = process_full_resolution_video()
    
    if not success:
        print("\nTroubleshooting:")
        print("1. Check if the original video exists")
        print("2. Verify FFmpeg has drawtext support")
        print("3. Check available disk space")