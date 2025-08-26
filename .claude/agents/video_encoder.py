#!/usr/bin/env python3
"""
Claude Agent: Video Encoder
Purpose: Ensure all videos are properly encoded for QuickTime compatibility
Author: Claude
Created: 2024

This agent handles all video encoding to ensure QuickTime compatibility.
It automatically applies the correct encoding settings and handles various input formats.
"""

import subprocess
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import tempfile
import shutil

class VideoEncoder:
    """
    A robust video encoder that ensures QuickTime compatibility.
    """
    
    # QuickTime-compatible encoding settings
    QUICKTIME_SETTINGS = {
        "codec": "libx264",
        "preset": "medium",
        "crf": "23",
        "pix_fmt": "yuv420p",  # Most compatible pixel format
        "profile": "baseline",   # Maximum compatibility
        "level": "3.0",
        "movflags": "+faststart",  # Optimize for streaming
        "audio_codec": "aac",  # If audio exists
        "audio_bitrate": "128k"
    }
    
    # High-quality settings for advanced use
    HQ_SETTINGS = {
        "codec": "libx264",
        "preset": "slow",
        "crf": "18",
        "pix_fmt": "yuv420p",
        "profile": "high",
        "level": "4.0",
        "movflags": "+faststart"
    }
    
    def __init__(self, verbose: bool = True):
        """Initialize the video encoder."""
        self.verbose = verbose
        self.ffmpeg_path = self._find_ffmpeg()
        
    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable."""
        if shutil.which("ffmpeg"):
            return "ffmpeg"
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    
    def _log(self, message: str):
        """Log messages if verbose mode is enabled."""
        if self.verbose:
            print(f"[VIDEO_ENCODER] {message}")
    
    def encode_for_quicktime(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        quality: str = "standard",
        preserve_transparency: bool = False,
        custom_settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Encode a video for QuickTime compatibility.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video (auto-generated if None)
            quality: "standard" or "high" quality preset
            preserve_transparency: If True, attempts to preserve alpha channel
            custom_settings: Override specific encoding settings
            
        Returns:
            Path to the encoded video
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_quicktime.mp4"
        else:
            output_path = Path(output_path)
        
        self._log(f"Encoding {input_path.name} for QuickTime...")
        
        # Select base settings
        settings = self.QUICKTIME_SETTINGS.copy() if quality == "standard" else self.HQ_SETTINGS.copy()
        
        # Apply custom settings
        if custom_settings:
            settings.update(custom_settings)
        
        # Build FFmpeg command
        cmd = [
            self.ffmpeg_path,
            "-i", str(input_path),
            "-c:v", settings["codec"],
            "-preset", settings["preset"],
            "-crf", settings["crf"],
        ]
        
        # Add pixel format (handle transparency if requested)
        if preserve_transparency and self._supports_transparency(input_path):
            cmd.extend(["-pix_fmt", "yuva420p"])  # Try to preserve alpha
            self._log("Attempting to preserve transparency...")
        else:
            cmd.extend(["-pix_fmt", settings["pix_fmt"]])
        
        # Add profile and level for compatibility
        if "profile" in settings:
            cmd.extend(["-profile:v", settings["profile"]])
        if "level" in settings:
            cmd.extend(["-level", settings["level"]])
        
        # Add movflags for streaming
        cmd.extend(["-movflags", settings["movflags"]])
        
        # Handle audio if present
        if self._has_audio(input_path):
            cmd.extend([
                "-c:a", settings.get("audio_codec", "aac"),
                "-b:a", settings.get("audio_bitrate", "128k")
            ])
        
        # Add output path and overwrite flag
        cmd.extend(["-y", str(output_path)])
        
        # Execute encoding
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                self._log(f"Warning: FFmpeg reported issues: {result.stderr[-500:]}")
                
                # Try fallback encoding with maximum compatibility
                self._log("Attempting fallback encoding with maximum compatibility...")
                return self._fallback_encode(input_path, output_path)
            
            self._log(f"✅ Successfully encoded: {output_path}")
            
            # Verify the output
            if not self._verify_output(output_path):
                self._log("Warning: Output verification failed, attempting re-encode...")
                return self._fallback_encode(input_path, output_path)
                
            return str(output_path)
            
        except Exception as e:
            self._log(f"Error during encoding: {e}")
            raise
    
    def _fallback_encode(self, input_path: Path, output_path: Path) -> str:
        """
        Fallback encoding with maximum compatibility settings.
        """
        self._log("Using maximum compatibility settings...")
        
        cmd = [
            self.ffmpeg_path,
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-movflags", "+faststart",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure even dimensions
            "-y", str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to encode video: {result.stderr}")
        
        self._log(f"✅ Fallback encoding successful: {output_path}")
        return str(output_path)
    
    def _has_audio(self, video_path: Path) -> bool:
        """Check if video has audio stream."""
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-hide_banner"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return "Audio:" in result.stderr
    
    def _supports_transparency(self, video_path: Path) -> bool:
        """Check if video has alpha channel."""
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-hide_banner"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return "yuva" in result.stderr.lower() or "rgba" in result.stderr.lower()
    
    def _verify_output(self, output_path: Path) -> bool:
        """Verify the output video is valid."""
        if not output_path.exists():
            return False
        
        cmd = [
            self.ffmpeg_path,
            "-i", str(output_path),
            "-f", "null",
            "-"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def batch_encode(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        pattern: str = "*.mp4",
        **kwargs
    ) -> list:
        """
        Batch encode multiple videos.
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory for output videos (same as input if None)
            pattern: File pattern to match (e.g., "*.mp4", "*.avi")
            **kwargs: Additional arguments for encode_for_quicktime
            
        Returns:
            List of encoded video paths
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir is None:
            output_dir = input_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        encoded_videos = []
        video_files = list(input_dir.glob(pattern))
        
        self._log(f"Found {len(video_files)} videos to encode")
        
        for i, video_file in enumerate(video_files, 1):
            self._log(f"Processing {i}/{len(video_files)}: {video_file.name}")
            
            output_path = output_dir / f"{video_file.stem}_quicktime.mp4"
            
            try:
                encoded_path = self.encode_for_quicktime(
                    str(video_file),
                    str(output_path),
                    **kwargs
                )
                encoded_videos.append(encoded_path)
            except Exception as e:
                self._log(f"Failed to encode {video_file.name}: {e}")
        
        self._log(f"✅ Batch encoding complete: {len(encoded_videos)}/{len(video_files)} successful")
        return encoded_videos


def main():
    """
    Command-line interface for the video encoder.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Encode videos for QuickTime compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode a single video
  python video_encoder.py input.mp4
  
  # Encode with high quality
  python video_encoder.py input.mp4 --quality high
  
  # Batch encode all MP4s in a directory
  python video_encoder.py --batch input_dir/ --pattern "*.mp4"
  
  # Specify output path
  python video_encoder.py input.mp4 -o output_quicktime.mp4
        """
    )
    
    parser.add_argument("input", help="Input video file or directory (for batch mode)")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("-q", "--quality", choices=["standard", "high"], default="standard",
                       help="Encoding quality preset")
    parser.add_argument("--batch", action="store_true",
                       help="Enable batch processing mode")
    parser.add_argument("--pattern", default="*.mp4",
                       help="File pattern for batch mode (default: *.mp4)")
    parser.add_argument("--preserve-transparency", action="store_true",
                       help="Attempt to preserve alpha channel")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Initialize encoder
    encoder = VideoEncoder(verbose=not args.quiet)
    
    try:
        if args.batch:
            # Batch mode
            encoded = encoder.batch_encode(
                args.input,
                args.output,
                args.pattern,
                quality=args.quality,
                preserve_transparency=args.preserve_transparency
            )
            print(f"\n✅ Encoded {len(encoded)} videos successfully")
            for path in encoded:
                print(f"  • {path}")
        else:
            # Single file mode
            output = encoder.encode_for_quicktime(
                args.input,
                args.output,
                quality=args.quality,
                preserve_transparency=args.preserve_transparency
            )
            print(f"\n✅ Encoded video saved to: {output}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()