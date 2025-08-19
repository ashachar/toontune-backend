"""
Step 1: Video Downsampling
===========================

Downsamples the full video for transcript generation.
"""

import subprocess
from pathlib import Path


class DownsampleStep:
    """Handles video downsampling operations."""
    
    def __init__(self, pipeline_state, dirs, config):
        """Initialize with pipeline references."""
        self.pipeline_state = pipeline_state
        self.dirs = dirs
        self.config = config
    
    def run(self, video_path, video_name):
        """Downsample the full video for transcript generation."""
        print("\n" + "-"*60)
        print("STEP 1: DOWNSAMPLING VIDEO")
        print("-"*60)
        
        downsampled_path = self.dirs['base'] / f"{video_name}_downsampled.mp4"
        
        if downsampled_path.exists():
            print(f"  ✓ Downsampled video already exists: {downsampled_path}")
        else:
            # Use the downsample utility
            cmd = [
                "python", "utils/downsample/video_downsample.py",
                str(video_path),
                "--output", str(downsampled_path),
                "--preset", "small"  # 256x256
            ]
            
            if not self.config.dry_run:
                print(f"  Downsampling to 256x256...")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  ✓ Downsampled video saved to: {downsampled_path}")
                else:
                    print(f"  ✗ Downsampling failed: {result.stderr}")
            else:
                print(f"  [DRY RUN] Would downsample to: {downsampled_path}")
        
        self.pipeline_state['downsampled_video'] = str(downsampled_path)
        self.pipeline_state['steps_completed'].append('downsample_video')