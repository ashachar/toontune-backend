#!/usr/bin/env python3
"""
Cached Robust Video Matting using Replicate.
Stores and reuses masks/green screen videos to avoid reprocessing.
"""

import os
import json
import hashlib
import subprocess
import tempfile
from pathlib import Path
import shutil
import replicate
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class CachedRobustVideoMatting:
    """Manages cached background removal using Replicate's RVM."""
    
    def __init__(self):
        """Initialize the cached RVM processor."""
        self.api_token = os.getenv('REPLICATE_API_TOKEN')
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN not found in .env")
    
    def get_video_hash(self, video_path, duration=None):
        """
        Get a hash for video identification.
        
        Args:
            video_path: Path to video file
            duration: Duration in seconds (None for full video)
            
        Returns:
            Hash string for cache identification
        """
        # Get file stats
        stats = os.stat(video_path)
        hash_input = f"{video_path.name}_{stats.st_size}_{stats.st_mtime}"
        if duration:
            hash_input += f"_{duration}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def get_cache_path(self, video_path, duration=None):
        """
        Get cache directory and file paths for a video.
        
        Args:
            video_path: Path to video file
            duration: Duration in seconds
            
        Returns:
            Dictionary with cache paths
        """
        video_path = Path(video_path)
        video_name = video_path.stem
        
        # Create project folder if needed
        project_folder = video_path.parent / video_name
        project_folder.mkdir(exist_ok=True)
        
        # Generate cache filename
        video_hash = self.get_video_hash(video_path, duration)
        duration_str = f"_{int(duration)}s" if duration else "_full"
        
        cache_info = {
            'project_folder': project_folder,
            'green_screen': project_folder / f"{video_name}_rvm_green{duration_str}_{video_hash}.mp4",
            'mask': project_folder / f"{video_name}_rvm_mask{duration_str}_{video_hash}.mp4",
            'metadata': project_folder / f"{video_name}_rvm_meta{duration_str}_{video_hash}.json"
        }
        
        return cache_info
    
    def check_cache(self, video_path, duration=None):
        """
        Check if cached RVM output exists.
        
        Args:
            video_path: Path to video file
            duration: Duration in seconds
            
        Returns:
            Cache info if exists, None otherwise
        """
        cache_info = self.get_cache_path(video_path, duration)
        
        # Check if green screen output exists
        if cache_info['green_screen'].exists():
            print(f"✓ Found cached RVM output: {cache_info['green_screen']}")
            
            # Load metadata if available
            if cache_info['metadata'].exists():
                with open(cache_info['metadata'], 'r') as f:
                    metadata = json.load(f)
                    print(f"  Created: {metadata.get('created_at', 'unknown')}")
                    print(f"  Model: {metadata.get('model', 'unknown')}")
            
            return cache_info
        
        return None
    
    def process_with_replicate(self, video_path, duration=None):
        """
        Process video with Replicate's RVM and cache the result.
        
        Args:
            video_path: Path to video file
            duration: Duration in seconds
            
        Returns:
            Path to cached green screen output
        """
        video_path = Path(video_path)
        cache_info = self.get_cache_path(video_path, duration)
        
        print(f"\n🚀 Processing with Replicate's Robust Video Matting...")
        print(f"Input: {video_path}")
        print(f"Duration: {duration if duration else 'full'} seconds")
        
        # Create temporary segment if duration specified
        if duration:
            temp_input = Path(tempfile.mktemp(suffix='.mp4'))
            print(f"Extracting {duration} second segment...")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                str(temp_input)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            processing_input = temp_input
        else:
            processing_input = video_path
        
        # Process with Replicate
        print("Uploading to Replicate...")
        with open(processing_input, 'rb') as f:
            output = replicate.run(
                "arielreplicate/robust_video_matting:73d2128a371922d5d1abf0712a1d974be0e4e2358cc1218e4e34714767232bac",
                input={"input_video": f}
            )
        
        output_url = str(output)
        print(f"✓ Processing complete: {output_url}")
        
        # Download result
        print("Downloading green screen output...")
        response = requests.get(output_url, stream=True)
        response.raise_for_status()
        
        # Save to cache
        cache_info['project_folder'].mkdir(exist_ok=True)
        with open(cache_info['green_screen'], 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Saved to cache: {cache_info['green_screen']}")
        
        # Generate mask from green screen
        print("Generating mask...")
        self.generate_mask(cache_info['green_screen'], cache_info['mask'])
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'model': 'robust_video_matting',
            'replicate_url': output_url,
            'input_video': str(video_path),
            'duration': duration,
            'video_hash': self.get_video_hash(video_path, duration)
        }
        
        with open(cache_info['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Cleanup temp file
        if duration and temp_input.exists():
            temp_input.unlink()
        
        return cache_info['green_screen']
    
    def generate_mask(self, green_screen_path, mask_output_path):
        """
        Generate a mask from green screen video.
        
        Args:
            green_screen_path: Path to green screen video
            mask_output_path: Path to save mask video
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", str(green_screen_path),
            "-vf", "chromakey=green:0.1:0.1,format=gray",
            "-c:v", "libx264", "-crf", "0",
            str(mask_output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Mask saved: {mask_output_path}")
    
    def get_rvm_output(self, video_path, duration=None):
        """
        Get RVM output, using cache if available.
        
        Args:
            video_path: Path to video file
            duration: Duration in seconds
            
        Returns:
            Path to green screen output
        """
        video_path = Path(video_path)
        
        # Check cache first
        cache_info = self.check_cache(video_path, duration)
        
        if cache_info:
            print("Using cached RVM output - no API call needed!")
            return cache_info['green_screen']
        
        # Process if not cached
        print("No cache found - processing with Replicate...")
        return self.process_with_replicate(video_path, duration)
    
    def composite_with_background(self, video_path, background_video, output_path, duration=None):
        """
        Composite video with new background using cached RVM.
        
        Args:
            video_path: Path to original video
            background_video: Path to background video
            output_path: Path to save output
            duration: Duration in seconds
        """
        # Get RVM output (cached or fresh)
        green_screen = self.get_rvm_output(video_path, duration)
        
        print(f"\nCompositing with background...")
        print(f"Background: {background_video}")
        
        # Composite using FFmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", str(background_video),
            "-i", str(green_screen),
            "-filter_complex",
            "[0:v]scale=1280:720[bg];"
            "[1:v]chromakey=0x00ff00:0.3:0.1[ckout];"
            "[bg][ckout]overlay=0:0:shortest=1[out]",
            "-map", "[out]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True)
        print(f"✓ Output saved: {output_path}")
        
        # Add audio from original
        video_path = Path(video_path)
        final_output = output_path.parent / f"{output_path.stem}_with_audio.mp4"
        
        cmd_audio = [
            "ffmpeg", "-y",
            "-i", str(output_path),
            "-i", str(video_path),
            "-c:v", "copy",
            "-map", "0:v",
            "-map", "1:a?",
            "-shortest",
            str(final_output)
        ]
        
        subprocess.run(cmd_audio, check=True, capture_output=True)
        print(f"✓ Final output with audio: {final_output}")
        
        return final_output


def main():
    """Test cached RVM system."""
    
    # Setup
    processor = CachedRobustVideoMatting()
    
    input_video = Path("uploads/assets/videos/ai_math1.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_cached_rvm.mp4")
    
    if not input_video.exists():
        print(f"Error: Input not found: {input_video}")
        return
    
    if not background_video.exists():
        print(f"Error: Background not found: {background_video}")
        return
    
    print("=" * 60)
    print("Cached Robust Video Matting System")
    print("=" * 60)
    
    # Process with caching (5 second test)
    final_output = processor.composite_with_background(
        input_video,
        background_video,
        output_video,
        duration=5.0
    )
    
    # Open result
    if final_output.exists():
        print(f"\nOpening result...")
        subprocess.run(["open", str(final_output)])


if __name__ == "__main__":
    main()