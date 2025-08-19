#!/usr/bin/env python3
"""
Video Downsampling Utility
Reduces video resolution to save space and processing time
Supports extreme downsampling to as low as 32x32 pixels
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Tuple


class VideoDownsampler:
    """Downsample videos to lower resolutions"""
    
    # Preset resolutions (width x height)
    PRESETS = {
        'micro': (32, 32),      # Absolute minimum - barely recognizable
        'tiny': (64, 64),       # Very small but shapes visible
        'mini': (128, 128),     # Small but usable
        'small': (256, 256),    # Reasonable for previews
        'medium': (512, 512),   # Good balance
        'hd': (1280, 720),      # Standard HD
        'fhd': (1920, 1080),    # Full HD
    }
    
    def __init__(self, 
                 input_path: str,
                 output_path: Optional[str] = None,
                 resolution: Optional[Tuple[int, int]] = None,
                 preset: str = 'small',
                 maintain_aspect: bool = True,
                 fps: Optional[int] = None,
                 bitrate: Optional[str] = None,
                 codec: str = 'libx264'):
        """
        Initialize video downsampler
        
        Args:
            input_path: Path to input video
            output_path: Path for output (auto-generated if None)
            resolution: Target resolution as (width, height) tuple
            preset: Preset size ('micro', 'tiny', 'mini', 'small', 'medium') - default 'small'
            maintain_aspect: Maintain aspect ratio
            fps: Target frame rate (None keeps original)
            bitrate: Target bitrate (e.g., '50k', '200k')
            codec: Video codec to use
        """
        self.input_path = input_path
        self.output_path = output_path
        self.resolution = resolution or self.PRESETS.get(preset, self.PRESETS['tiny'])
        self.maintain_aspect = maintain_aspect
        self.fps = fps
        self.bitrate = bitrate
        self.codec = codec
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Auto-generate output path if not provided
        if not self.output_path:
            input_path_obj = Path(input_path)
            input_name = input_path_obj.stem
            input_ext = input_path_obj.suffix
            input_dir = input_path_obj.parent
            width, height = self.resolution
            self.output_path = str(input_dir / f"{input_name}_{width}x{height}_downsampled{input_ext}")
    
    def get_video_info(self) -> dict:
        """Get information about the input video"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-show_format',
            self.input_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            info = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            for stream in info.get('streams', []):
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if video_stream:
                return {
                    'width': int(video_stream.get('width', 0)),
                    'height': int(video_stream.get('height', 0)),
                    'fps': eval(video_stream.get('r_frame_rate', '30/1')),
                    'duration': float(info['format'].get('duration', 0)),
                    'bitrate': info['format'].get('bit_rate', 'N/A'),
                    'codec': video_stream.get('codec_name', 'unknown')
                }
        except:
            return {}
        
        return {}
    
    def calculate_scale_filter(self) -> str:
        """Calculate the scale filter string for ffmpeg"""
        target_width, target_height = self.resolution
        
        if self.maintain_aspect:
            # Scale to fit within target dimensions while maintaining aspect ratio
            # -2 ensures even dimensions
            return f"scale='min({target_width},iw)':min'({target_height},ih)':force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2"
        else:
            # Stretch to exact dimensions
            return f"scale={target_width}:{target_height}"
    
    def downsample(self, verbose: bool = True) -> bool:
        """
        Perform the downsampling
        
        Args:
            verbose: Print progress information
        
        Returns:
            Success status
        """
        if verbose:
            info = self.get_video_info()
            if info:
                print(f"📹 Input Video Info:")
                print(f"   Resolution: {info['width']}x{info['height']}")
                print(f"   FPS: {info['fps']:.2f}")
                print(f"   Duration: {info['duration']:.2f}s")
                print(f"   Bitrate: {info['bitrate']}")
                print(f"   Codec: {info['codec']}")
                print()
            
            print(f"🎯 Target Settings:")
            print(f"   Resolution: {self.resolution[0]}x{self.resolution[1]}")
            if self.fps:
                print(f"   FPS: {self.fps}")
            if self.bitrate:
                print(f"   Bitrate: {self.bitrate}")
            print(f"   Codec: {self.codec}")
            print(f"   Maintain Aspect: {self.maintain_aspect}")
            print()
        
        # Build ffmpeg command
        cmd = ['ffmpeg', '-i', self.input_path]
        
        # Video filters
        filters = [self.calculate_scale_filter()]
        
        # Add FPS filter if specified
        if self.fps:
            filters.append(f"fps={self.fps}")
        
        # Apply filters
        if filters:
            cmd.extend(['-vf', ','.join(filters)])
        
        # Video codec
        cmd.extend(['-c:v', self.codec])
        
        # Bitrate if specified
        if self.bitrate:
            cmd.extend(['-b:v', self.bitrate])
        else:
            # Auto-calculate low bitrate based on resolution
            width, height = self.resolution
            pixels = width * height
            if pixels <= 1024:  # 32x32
                auto_bitrate = '10k'
            elif pixels <= 4096:  # 64x64
                auto_bitrate = '30k'
            elif pixels <= 16384:  # 128x128
                auto_bitrate = '100k'
            elif pixels <= 65536:  # 256x256
                auto_bitrate = '300k'
            else:
                auto_bitrate = '500k'
            cmd.extend(['-b:v', auto_bitrate])
        
        # Audio settings - reduce quality for smaller files
        cmd.extend(['-c:a', 'aac', '-b:a', '32k', '-ac', '1'])  # Mono, low bitrate
        
        # Pixel format for compatibility
        cmd.extend(['-pix_fmt', 'yuv420p'])
        
        # Output file
        cmd.extend(['-y', self.output_path])
        
        if verbose:
            print("🔄 Downsampling video...")
            print(f"   Command: {' '.join(cmd[:6])}...")
        
        # Execute ffmpeg
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if verbose:
                # Get output file size
                output_size = os.path.getsize(self.output_path) / (1024 * 1024)  # MB
                input_size = os.path.getsize(self.input_path) / (1024 * 1024)  # MB
                reduction = (1 - output_size/input_size) * 100
                
                print(f"\n✅ Success!")
                print(f"   Output: {self.output_path}")
                print(f"   Size: {output_size:.2f} MB (reduced by {reduction:.1f}%)")
                
                # Get actual output resolution
                out_info = self.get_video_info_for_file(self.output_path)
                if out_info:
                    print(f"   Final Resolution: {out_info['width']}x{out_info['height']}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"\n❌ Error downsampling video:")
                print(f"   {e.stderr[:500] if e.stderr else 'Unknown error'}")
            return False
    
    def get_video_info_for_file(self, filepath: str) -> dict:
        """Get video info for any file"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            filepath
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            info = json.loads(result.stdout)
            stream = info['streams'][0] if info.get('streams') else {}
            return {
                'width': stream.get('width', 0),
                'height': stream.get('height', 0)
            }
        except:
            return {}


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Downsample videos to lower resolutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preset sizes:
  micro  : 32x32   - Absolute minimum
  tiny   : 64x64   - Default, very small
  mini   : 128x128 - Small but usable
  small  : 256x256 - Good for previews
  medium : 512x512 - Balanced size

Examples:
  %(prog)s video.mp4                    # Use default 'small' (256x256)
  %(prog)s video.mp4 --preset tiny      # Use 64x64 for extreme reduction
  %(prog)s video.mp4 --size 100 100     # Custom 100x100
  %(prog)s video.mp4 --fps 15           # Also reduce framerate
  %(prog)s video.mp4 --bitrate 50k      # Set specific bitrate
        """
    )
    
    parser.add_argument('input', help='Input video file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-p', '--preset', 
                       choices=VideoDownsampler.PRESETS.keys(),
                       default='small',
                       help='Size preset (default: small)')
    parser.add_argument('-s', '--size', nargs=2, type=int, metavar=('W', 'H'),
                       help='Custom resolution (width height)')
    parser.add_argument('-f', '--fps', type=int, 
                       help='Target frame rate')
    parser.add_argument('-b', '--bitrate',
                       help='Target bitrate (e.g., 50k, 200k)')
    parser.add_argument('--no-aspect', action='store_true',
                       help='Don\'t maintain aspect ratio')
    parser.add_argument('-c', '--codec', default='libx264',
                       help='Video codec (default: libx264)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Quiet mode')
    
    args = parser.parse_args()
    
    # Determine resolution
    if args.size:
        resolution = tuple(args.size)
    else:
        resolution = None  # Use preset
    
    # Create downsampler
    downsampler = VideoDownsampler(
        input_path=args.input,
        output_path=args.output,
        resolution=resolution,
        preset=args.preset,
        maintain_aspect=not args.no_aspect,
        fps=args.fps,
        bitrate=args.bitrate,
        codec=args.codec
    )
    
    # Perform downsampling
    success = downsampler.downsample(verbose=not args.quiet)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()