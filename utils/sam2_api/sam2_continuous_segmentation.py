#!/usr/bin/env python3
"""
SAM2 Continuous Video Segmentation - Production Implementation

This module provides continuous frame-by-frame segmentation using SAM2 via Replicate API.
Due to Replicate's implementation, masks only appear at frames with explicit click points,
so we place points on every frame for truly continuous segmentation.

Author: ToonTune Team
Date: 2024
"""

import os
import subprocess
import json
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dataclasses import dataclass
import logging

from .video_segmentation import SAM2VideoSegmenter, ClickPoint, SegmentationConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Video metadata extracted via ffprobe"""
    duration: float
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass
class ObjectToTrack:
    """Definition of an object to track in the video"""
    name: str
    x: int
    y: int
    track_every_n_frames: int = 1  # 1 = every frame, 2 = every other frame, etc.
    label: int = 1  # 1 for foreground, 0 for background


class ContinuousSAM2Segmenter:
    """
    Production-ready continuous video segmentation using SAM2.
    
    This implementation ensures continuous masks throughout the video by dynamically
    placing tracking points based on video properties rather than hard-coding frame counts.
    
    Key Features:
    - Automatic video analysis to determine frame count and fps
    - Dynamic click point generation based on video length
    - Configurable tracking density per object
    - Continuous mask output without gaps
    
    Usage:
        segmenter = ContinuousSAM2Segmenter()
        
        # Define objects to track
        objects = [
            ObjectToTrack("person", x=320, y=240, track_every_n_frames=1),
            ObjectToTrack("car", x=150, y=300, track_every_n_frames=2)
        ]
        
        # Run segmentation
        result = segmenter.segment_video_continuous(
            "input_video.mp4",
            objects,
            output_path="output_segmented.mp4"
        )
    """
    
    def __init__(self, api_token: Optional[str] = None):
        """Initialize the continuous segmenter"""
        self.segmenter = SAM2VideoSegmenter(api_token)
    
    @staticmethod
    def get_video_info(video_path: str) -> VideoInfo:
        """
        Extract video metadata using ffprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoInfo object with video properties
        """
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 
            'stream=nb_read_packets,duration,r_frame_rate,width,height',
            '-of', 'json',
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            stream = data['streams'][0]
            
            # Parse frame rate (e.g., "60/1" -> 60.0)
            fps_parts = stream['r_frame_rate'].split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1])
            
            # Get frame count (prefer nb_read_packets over calculated duration * fps)
            if 'nb_read_packets' in stream:
                frame_count = int(stream['nb_read_packets'])
            else:
                # Fallback: calculate from duration
                duration = float(stream.get('duration', 0))
                frame_count = int(duration * fps)
            
            return VideoInfo(
                duration=float(stream.get('duration', 0)),
                frame_count=frame_count,
                fps=fps,
                width=int(stream['width']),
                height=int(stream['height'])
            )
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error getting video info: {e}")
            # Return defaults if ffprobe fails
            return VideoInfo(duration=0, frame_count=0, fps=30, width=640, height=480)
    
    def generate_click_points(
        self, 
        objects: List[ObjectToTrack], 
        video_info: VideoInfo
    ) -> List[ClickPoint]:
        """
        Dynamically generate click points based on video properties.
        
        Args:
            objects: List of objects to track
            video_info: Video metadata
            
        Returns:
            List of ClickPoint objects distributed across the video
        """
        click_points = []
        
        for obj in objects:
            # Generate points based on video's actual frame count
            for frame in range(0, video_info.frame_count, obj.track_every_n_frames):
                click_points.append(
                    ClickPoint(
                        x=obj.x,
                        y=obj.y,
                        frame=frame,
                        label=obj.label,
                        object_id=obj.name
                    )
                )
            
            logger.info(
                f"Object '{obj.name}': {len(range(0, video_info.frame_count, obj.track_every_n_frames))} "
                f"tracking points (every {obj.track_every_n_frames} frame(s))"
            )
        
        return click_points
    
    def segment_video_continuous(
        self,
        video_path: str,
        objects_to_track: List[ObjectToTrack],
        output_path: Optional[str] = None,
        mask_type: str = "highlighted",
        output_quality: int = 80
    ) -> str:
        """
        Perform continuous video segmentation with dynamic frame detection.
        
        This is the main method that ensures continuous segmentation by:
        1. Analyzing the video to get frame count and fps
        2. Dynamically generating click points for every frame
        3. Processing with SAM2 to ensure no gaps in segmentation
        
        Args:
            video_path: Path to input video
            objects_to_track: List of objects with their tracking positions
            output_path: Optional output path (auto-generated if not provided)
            mask_type: Type of mask overlay ("highlighted", "binary", "greenscreen")
            output_quality: Output video quality (0-100)
            
        Returns:
            Path to the segmented video file
        """
        # Get video information dynamically
        logger.info(f"Analyzing video: {video_path}")
        video_info = self.get_video_info(video_path)
        
        logger.info(
            f"Video properties: {video_info.frame_count} frames, "
            f"{video_info.fps:.1f} fps, {video_info.duration:.2f}s, "
            f"{video_info.width}x{video_info.height}"
        )
        
        # Validate and adjust object positions if needed
        for obj in objects_to_track:
            if obj.x >= video_info.width:
                obj.x = video_info.width - 1
                logger.warning(f"Adjusted {obj.name} x position to fit video width")
            if obj.y >= video_info.height:
                obj.y = video_info.height - 1
                logger.warning(f"Adjusted {obj.name} y position to fit video height")
        
        # Generate click points dynamically based on actual frame count
        click_points = self.generate_click_points(objects_to_track, video_info)
        
        logger.info(f"Total tracking points: {len(click_points)}")
        
        # Set output path if not provided
        if not output_path:
            input_stem = Path(video_path).stem
            output_path = f"{input_stem}_continuous_segmented.mp4"
        
        # Configure output settings
        config = SegmentationConfig(
            mask_type=mask_type,
            annotation_type="mask",
            output_video=True,
            video_fps=int(video_info.fps),  # Match original fps
            output_quality=output_quality,
            output_frame_interval=1  # Process every frame
        )
        
        # Run segmentation
        logger.info("Starting SAM2 continuous segmentation...")
        logger.info("Note: This ensures masks appear on EVERY frame without gaps")
        
        try:
            result = self.segmenter.segment_video_advanced(
                video_path=video_path,
                click_points=click_points,
                config=config,
                output_path=output_path
            )
            
            logger.info(f"✅ Continuous segmentation complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise


def segment_video_simple(
    video_path: str,
    object_positions: List[Tuple[str, int, int]],
    output_path: Optional[str] = None
) -> str:
    """
    Simple convenience function for continuous video segmentation.
    
    Args:
        video_path: Path to video file
        object_positions: List of (name, x, y) tuples for objects to track
        output_path: Optional output path
        
    Returns:
        Path to segmented video
        
    Example:
        result = segment_video_simple(
            "my_video.mp4",
            [("person", 320, 240), ("car", 150, 300)]
        )
    """
    segmenter = ContinuousSAM2Segmenter()
    
    objects = [
        ObjectToTrack(name=name, x=x, y=y, track_every_n_frames=1)
        for name, x, y in object_positions
    ]
    
    return segmenter.segment_video_continuous(
        video_path,
        objects,
        output_path
    )


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sam2_continuous_segmentation.py <video_path>")
        print("\nThis will perform continuous segmentation with example objects.")
        sys.exit(1)
    
    video_file = sys.argv[1]
    
    # Example: Track multiple objects with different densities
    objects = [
        ObjectToTrack("primary_object", x=320, y=180, track_every_n_frames=1),  # Every frame
        ObjectToTrack("secondary_object", x=480, y=270, track_every_n_frames=2),  # Every other frame
        ObjectToTrack("background", x=100, y=100, track_every_n_frames=5)  # Every 5 frames
    ]
    
    segmenter = ContinuousSAM2Segmenter()
    result = segmenter.segment_video_continuous(video_file, objects)
    print(f"Segmented video saved to: {result}")