#!/usr/bin/env python3
"""
Background Replacement Effect

Replaces the background of a video with another video using segmentation.
Uses SAM2 for foreground extraction and FFmpeg for compositing.
"""

import os
import sys
import cv2
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import json

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from utils.video.background.coverr_manager import CoverrManager
from utils.video.segmentation.segment_extractor import extract_foreground_mask


class BackgroundReplacer:
    """Replaces video background with another video."""
    
    def __init__(self, demo_mode=False):
        self.coverr_manager = CoverrManager(demo_mode=demo_mode)
        self.temp_dir = tempfile.mkdtemp()
    
    def get_video_info(self, video_path: Path) -> Tuple[int, int, float, int]:
        """
        Get video information.
        
        Returns:
            Tuple of (width, height, fps, frame_count)
        """
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return width, height, fps, frame_count
    
    def extract_foreground_masks(self, video_path: Path, output_dir: Path,
                                sample_rate: int = 5) -> Path:
        """
        Extract foreground masks for video frames.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save masks
            sample_rate: Process every Nth frame
            
        Returns:
            Path to masks directory
        """
        print("Extracting foreground masks...")
        
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                # Extract foreground mask for this frame
                mask = extract_foreground_mask(frame)
                
                # Save mask
                mask_path = masks_dir / f"mask_{frame_idx:06d}.png"
                cv2.imwrite(str(mask_path), mask)
                
                print(f"Processed frame {frame_idx}")
            
            frame_idx += 1
        
        cap.release()
        return masks_dir
    
    def create_mask_video(self, video_path: Path, output_path: Path) -> Path:
        """
        Create a video of foreground masks.
        
        Args:
            video_path: Path to input video
            output_path: Path to save mask video
            
        Returns:
            Path to mask video
        """
        print("Creating mask video...")
        
        width, height, fps, frame_count = self.get_video_info(video_path)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract foreground mask
            mask = extract_foreground_mask(frame)
            
            # Convert mask to 3-channel
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            out.write(mask_rgb)
            
            if frame_idx % 30 == 0:
                print(f"Processing frame {frame_idx}/{frame_count}")
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        return output_path
    
    def replace_background_ffmpeg(self, video_path: Path, background_path: Path,
                                 output_path: Path, mask_video_path: Optional[Path] = None):
        """
        Replace background using FFmpeg with mask video or chromakey.
        
        Args:
            video_path: Path to input video
            background_path: Path to background video
            output_path: Path to save output
            mask_video_path: Optional path to mask video
        """
        print("Compositing with FFmpeg...")
        
        if mask_video_path and mask_video_path.exists():
            # Use mask video for compositing
            cmd = [
                "ffmpeg", "-y",
                "-i", str(background_path),  # Background video
                "-i", str(video_path),        # Foreground video
                "-i", str(mask_video_path),   # Mask video
                "-filter_complex", 
                "[0:v]scale=1920:1080[bg];"
                "[1:v]scale=1920:1080[fg];"
                "[2:v]scale=1920:1080,format=gray[mask];"
                "[bg][fg][mask]maskedmerge[out]",
                "-map", "[out]",
                "-map", "1:a?",  # Use audio from original video if present
                "-c:v", "libx264",
                "-preset", "fast", 
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path)
            ]
        else:
            # Fallback: Use chromakey (requires green screen in original)
            cmd = [
                "ffmpeg", "-y",
                "-i", str(background_path),
                "-i", str(video_path),
                "-filter_complex",
                "[0:v]scale=1920:1080[bg];"
                "[1:v]scale=1920:1080,chromakey=green:0.10:0.08[fg];"
                "[bg][fg]overlay[out]",
                "-map", "[out]",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23", 
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path)
            ]
        
        subprocess.run(cmd, check=True)
        print(f"Output saved to: {output_path}")
    
    def process_video(self, video_path: Path, project_name: str,
                     start_time: float = 0, end_time: float = 5,
                     use_mask: bool = True) -> Path:
        """
        Replace video background with Coverr video.
        
        Args:
            video_path: Path to input video
            project_name: Project name for caching
            start_time: Background video start time
            end_time: Background video end time
            use_mask: Whether to use mask extraction (slower but better)
            
        Returns:
            Path to output video
        """
        # Get transcript path
        project_folder = Path(f"uploads/assets/videos/{project_name}")
        transcript_path = project_folder / f"{project_name}_whisper_transcript.txt"
        
        if not transcript_path.exists():
            print(f"Warning: No transcript found at {transcript_path}")
            # Create dummy transcript
            transcript_path = Path(self.temp_dir) / "dummy_transcript.txt"
            transcript_path.write_text("AI mathematics technology science")
        
        # Get background video
        print("Getting background video...")
        background_path = self.coverr_manager.get_background_for_video(
            project_name, transcript_path, start_time, end_time
        )
        
        # Setup output path
        output_name = f"{project_name}_replaced_bg_{int(start_time)}_{int(end_time)}.mp4"
        output_path = project_folder / output_name
        
        # Create mask video if requested
        mask_video_path = None
        if use_mask:
            print("Creating foreground mask video...")
            mask_video_path = Path(self.temp_dir) / "mask_video.mp4"
            self.create_mask_video(video_path, mask_video_path)
        
        # Replace background
        self.replace_background_ffmpeg(video_path, background_path, 
                                      output_path, mask_video_path)
        
        return output_path


def main():
    """Apply background replacement to ai_math1.mp4"""
    
    # Setup paths
    project_name = "ai_math1"
    
    # First extract a 5-second segment to process
    full_video_path = Path(f"uploads/assets/videos/{project_name}.mp4")
    segment_path = Path(f"uploads/assets/videos/{project_name}_segment_5sec.mp4")
    
    if not full_video_path.exists():
        print(f"Error: Video not found at {full_video_path}")
        return
    
    # Extract 5-second segment if not exists
    if not segment_path.exists():
        print("Extracting 5-second segment...")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(full_video_path),
            "-t", "5",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            str(segment_path)
        ]
        subprocess.run(cmd, check=True)
    
    # Create replacer
    replacer = BackgroundReplacer(demo_mode=False)
    
    # Process video segment
    output_path = replacer.process_video(
        segment_path,
        project_name,
        start_time=0,
        end_time=5,
        use_mask=True  # Must use mask extraction for non-green screen videos
    )
    
    print(f"\nBackground replacement complete!")
    print(f"Output: {output_path}")
    
    # Open the video
    if sys.platform == "darwin":  # macOS
        subprocess.run(["open", str(output_path)])
    elif sys.platform == "linux":
        subprocess.run(["xdg-open", str(output_path)])
    else:  # Windows
        os.startfile(str(output_path))


if __name__ == "__main__":
    main()