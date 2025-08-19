#!/usr/bin/env python3
"""
Video Sound Effects Overlay Script.
Embeds sound effects into videos at specified timestamps using FFmpeg.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import tempfile
import shutil


@dataclass
class SoundEffectEvent:
    """Represents a sound effect event in the video."""
    sound_name: str
    timestamp: float
    filepath: str
    volume: float = 0.5  # Default to 50% volume for sound effects


class VideoSoundOverlay:
    """Handles overlaying sound effects onto videos."""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Initialize Video Sound Overlay.
        
        Args:
            ffmpeg_path: Path to ffmpeg executable
        """
        self.ffmpeg_path = ffmpeg_path
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            subprocess.run([self.ffmpeg_path, "-version"], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(f"FFmpeg not found at: {self.ffmpeg_path}")
    
    def create_filter_complex(self, sound_events: List[SoundEffectEvent], 
                            video_duration: Optional[float] = None) -> str:
        """
        Create FFmpeg filter_complex string for overlaying multiple sound effects.
        
        Args:
            sound_events: List of sound effect events
            video_duration: Duration of the video (optional)
        
        Returns:
            FFmpeg filter_complex string
        """
        filters = []
        inputs = []
        
        # Start with base audio from video (or silent if no audio)
        base_audio = "[0:a]"
        
        # If video has no audio track, create silent audio
        if video_duration:
            filters.append(f"anullsrc=duration={video_duration}:sample_rate=44100[base]")
            base_audio = "[base]"
        
        # Process each sound effect
        for i, event in enumerate(sound_events, 1):
            # Apply volume adjustment and delay
            filter_parts = []
            
            # Volume adjustment if needed
            if event.volume != 1.0:
                filter_parts.append(f"volume={event.volume}")
            
            # Add delay to position the sound at the right timestamp
            delay_ms = int(event.timestamp * 1000)
            filter_parts.append(f"adelay={delay_ms}|{delay_ms}")
            
            # Pad to ensure the audio extends through the video
            if video_duration:
                filter_parts.append(f"apad=whole_dur={video_duration}")
            
            # Combine filter parts
            if filter_parts:
                filter_str = ",".join(filter_parts)
                filters.append(f"[{i}:a]{filter_str}[sfx{i}]")
                inputs.append(f"[sfx{i}]")
            else:
                inputs.append(f"[{i}:a]")
        
        # Mix all audio streams together
        if inputs:
            # Add base audio to inputs
            all_inputs = [base_audio] + inputs
            mix_inputs = "".join(all_inputs)
            filters.append(f"{mix_inputs}amix=inputs={len(all_inputs)}:duration=longest[outa]")
        else:
            filters.append(f"{base_audio}acopy[outa]")
        
        return ";".join(filters)
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get video information using ffprobe.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with video information
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Extract key information
            duration = float(info.get("format", {}).get("duration", 0))
            has_audio = any(s["codec_type"] == "audio" for s in info.get("streams", []))
            
            return {
                "duration": duration,
                "has_audio": has_audio,
                "format": info.get("format", {}),
                "streams": info.get("streams", [])
            }
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not get video info: {e}")
            return {"duration": None, "has_audio": False}
    
    def overlay_sound_effects(self, 
                             video_path: str,
                             sound_effects: List[Dict],
                             sound_files: Dict[str, str],
                             output_path: str,
                             preserve_original_audio: bool = True) -> bool:
        """
        Overlay sound effects onto a video.
        
        Args:
            video_path: Path to input video
            sound_effects: List of dicts with 'sound' and 'timestamp' keys
            sound_files: Dictionary mapping sound names to file paths
            output_path: Path for output video
            preserve_original_audio: Whether to keep original audio track
        
        Returns:
            True if successful, False otherwise
        """
        # Check input video exists
        if not Path(video_path).exists():
            print(f"Error: Video not found: {video_path}")
            return False
        
        # Get video information
        video_info = self.get_video_info(video_path)
        duration = video_info.get("duration")
        has_audio = video_info.get("has_audio")
        
        print(f"Video duration: {duration}s, Has audio: {has_audio}")
        
        # Prepare sound effect events
        events = []
        missing_sounds = []
        
        for effect in sound_effects:
            sound_name = effect.get("sound", "").lower()
            timestamp = float(effect.get("timestamp", 0))
            volume = float(effect.get("volume", 1.0))
            
            if sound_name in sound_files:
                filepath = sound_files[sound_name]
                if Path(filepath).exists():
                    events.append(SoundEffectEvent(
                        sound_name=sound_name,
                        timestamp=timestamp,
                        filepath=filepath,
                        volume=volume
                    ))
                else:
                    missing_sounds.append(sound_name)
            else:
                missing_sounds.append(sound_name)
        
        if missing_sounds:
            print(f"Warning: Missing sound files for: {missing_sounds}")
        
        if not events:
            print("No sound effects to overlay")
            return False
        
        print(f"Overlaying {len(events)} sound effects...")
        
        # Build FFmpeg command
        cmd = [self.ffmpeg_path, "-y"]  # Overwrite output
        
        # Input video
        cmd.extend(["-i", video_path])
        
        # Input sound effect files
        for event in events:
            cmd.extend(["-i", event.filepath])
        
        # Create filter complex
        if not has_audio and not preserve_original_audio:
            # Video has no audio, create silent base
            filter_complex = self.create_filter_complex(events, duration)
        else:
            filter_complex = self.create_filter_complex(events)
        
        cmd.extend(["-filter_complex", filter_complex])
        
        # Map video and audio streams
        cmd.extend(["-map", "0:v"])  # Video from original
        cmd.extend(["-map", "[outa]"])  # Audio from filter
        
        # Output settings for compatibility
        cmd.extend([
            "-c:v", "libx264",  # H.264 video codec
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",  # AAC audio codec
            "-b:a", "192k",
            "-ar", "44100",  # Standard sample rate
            "-pix_fmt", "yuv420p",  # Compatibility pixel format
            "-movflags", "+faststart",  # Web optimization
            output_path
        ])
        
        # Execute FFmpeg
        try:
            print(f"Running FFmpeg...")
            print(f"Command: {' '.join(cmd[:20])}...")  # Show partial command
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False
            
            # Verify output
            if Path(output_path).exists():
                size = Path(output_path).stat().st_size
                print(f"✓ Output created: {output_path} ({size/1024/1024:.2f} MB)")
                return True
            else:
                print("Error: Output file not created")
                return False
                
        except Exception as e:
            print(f"Error running FFmpeg: {e}")
            return False
    
    def batch_overlay(self,
                     video_sound_mapping: List[Tuple[str, List[Dict], Dict[str, str]]],
                     output_dir: str) -> List[str]:
        """
        Batch process multiple videos with sound effects.
        
        Args:
            video_sound_mapping: List of tuples (video_path, sound_effects, sound_files)
            output_dir: Directory for output videos
        
        Returns:
            List of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for video_path, sound_effects, sound_files in video_sound_mapping:
            video_name = Path(video_path).stem
            output_path = str(output_dir / f"{video_name}_with_sfx.mp4")
            
            print(f"\nProcessing: {video_name}")
            success = self.overlay_sound_effects(
                video_path, sound_effects, sound_files, output_path
            )
            
            if success:
                results.append(output_path)
        
        return results


def main():
    """Test the video sound overlay functionality."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_sound_overlay.py <video_path> [output_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output_with_sfx.mp4"
    
    # Test sound effects (from the provided data)
    test_effects = [
        {"sound": "ding", "timestamp": 3.579},
        {"sound": "swoosh", "timestamp": 13.050},
        {"sound": "chime", "timestamp": 17.700},
        {"sound": "sparkle", "timestamp": 40.119},
        {"sound": "pop", "timestamp": 44.840},
    ]
    
    # Mock sound files mapping (would come from SoundEffectsManager)
    test_sound_files = {
        "ding": "sound_effects/downloaded/ding.mp3",
        "swoosh": "sound_effects/downloaded/swoosh.mp3",
        "chime": "sound_effects/downloaded/chime.mp3",
        "sparkle": "sound_effects/downloaded/sparkle.mp3",
        "pop": "sound_effects/downloaded/pop.mp3",
    }
    
    # Create overlay instance
    overlay = VideoSoundOverlay()
    
    # Process video
    success = overlay.overlay_sound_effects(
        video_path,
        test_effects,
        test_sound_files,
        output_path
    )
    
    if success:
        print(f"\n✓ Video with sound effects saved to: {output_path}")
    else:
        print("\n✗ Failed to create video with sound effects")


if __name__ == "__main__":
    main()