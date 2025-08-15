#!/usr/bin/env python3
"""
SAM2 Video Segmentation Utility using Replicate API
Provides easy-to-use interface for segmenting objects in videos using Meta's SAM 2 model.
"""

import os
import json
import time
import requests
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import replicate
from dataclasses import dataclass, field


@dataclass
class ClickPoint:
    """Represents a click point for SAM2 segmentation"""
    x: int
    y: int
    frame: int = 0
    label: int = 1  # 1 for foreground, 0 for background
    object_id: str = ""


@dataclass
class SegmentationConfig:
    """Configuration for SAM2 video segmentation"""
    mask_type: str = "highlighted"  # binary, highlighted, or greenscreen
    annotation_type: str = "mask"  # mask, bbox, or both
    output_video: bool = True  # True for video, False for image sequence
    video_fps: int = 30
    output_format: str = "webp"  # webp, png, jpg (for image sequence)
    output_quality: int = 80  # 0-100 for jpg/webp
    output_frame_interval: int = 1  # Output every Nth frame


class SAM2VideoSegmenter:
    """
    A wrapper class for Meta's SAM 2 video segmentation model on Replicate.
    
    Usage:
        segmenter = SAM2VideoSegmenter()
        
        # Simple segmentation with automatic object detection
        result = segmenter.segment_video(
            "path/to/video.mp4",
            click_points=[(100, 100, 0), (200, 200, 10)]
        )
        
        # Advanced segmentation with multiple objects
        points = [
            ClickPoint(100, 100, 0, 1, "dog"),
            ClickPoint(200, 200, 0, 1, "dog"),
            ClickPoint(300, 300, 10, 1, "cat")
        ]
        result = segmenter.segment_video_advanced(
            "path/to/video.mp4",
            points,
            config=SegmentationConfig(mask_type="highlighted")
        )
    """
    
    MODEL_VERSION = "meta/sam-2-video:33432afdfc06a10da6b4018932893d39b0159f838b6d11dd1236dff85cc5ec1d"
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the SAM2 video segmenter.
        
        Args:
            api_token: Replicate API token. If not provided, uses REPLICATE_API_TOKEN env var.
        """
        if api_token:
            os.environ["REPLICATE_API_TOKEN"] = api_token
        elif not os.environ.get("REPLICATE_API_TOKEN"):
            raise ValueError(
                "REPLICATE_API_TOKEN environment variable not set. "
                "Please set it or pass api_token parameter."
            )
        
        self.client = replicate
    
    def segment_video(
        self,
        video_path: str,
        click_points: List[Tuple[int, int, int]],
        object_ids: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
        mask_type: str = "highlighted",
        output_video: bool = True,
        output_path: Optional[str] = None
    ) -> str:
        """
        Simple interface for video segmentation.
        
        Args:
            video_path: Path to input video file
            click_points: List of (x, y, frame) tuples for click coordinates
            object_ids: Optional list of object IDs for each click point
            labels: Optional list of labels (1=foreground, 0=background) for each point
            mask_type: Type of mask output (binary, highlighted, greenscreen)
            output_video: If True, outputs video; if False, outputs image sequence
            output_path: Optional path to save the output file
            
        Returns:
            Path to the output file or URL of the result
        """
        points = []
        for i, (x, y, frame) in enumerate(click_points):
            point = ClickPoint(
                x=x,
                y=y,
                frame=frame,
                label=labels[i] if labels and i < len(labels) else 1,
                object_id=object_ids[i] if object_ids and i < len(object_ids) else f"object_{i+1}"
            )
            points.append(point)
        
        config = SegmentationConfig(
            mask_type=mask_type,
            output_video=output_video
        )
        
        return self.segment_video_advanced(video_path, points, config, output_path)
    
    def segment_video_advanced(
        self,
        video_path: str,
        click_points: List[ClickPoint],
        config: Optional[SegmentationConfig] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Advanced interface for video segmentation with full control.
        
        Args:
            video_path: Path to input video file
            click_points: List of ClickPoint objects defining segmentation targets
            config: SegmentationConfig object with output settings
            output_path: Optional path to save the output file
            
        Returns:
            Path to the output file or URL of the result
        """
        if config is None:
            config = SegmentationConfig()
        
        # Prepare input parameters
        coordinates = ",".join([f"[{p.x},{p.y}]" for p in click_points])
        labels = ",".join([str(p.label) for p in click_points])
        frames = ",".join([str(p.frame) for p in click_points])
        object_ids = ",".join([p.object_id or f"object_{i+1}" for i, p in enumerate(click_points)])
        
        # Run the model
        print(f"Running SAM2 segmentation on {video_path}...")
        print(f"Click coordinates: {coordinates}")
        print(f"Object IDs: {object_ids}")
        
        # Prepare input parameters
        input_params = {
            "click_coordinates": coordinates,
            "click_labels": labels,
            "click_frames": frames,
            "click_object_ids": object_ids,
            "mask_type": config.mask_type,
            "annotation_type": config.annotation_type,
            "output_video": config.output_video,
            "video_fps": config.video_fps,
            "output_format": config.output_format,
            "output_quality": config.output_quality,
            "output_frame_interval": config.output_frame_interval
        }
        
        try:
            # Handle video file upload
            if Path(video_path).exists():
                with open(video_path, "rb") as video_file:
                    input_params["input_video"] = video_file
                    output = self.client.run(
                        self.MODEL_VERSION,
                        input=input_params
                    )
            else:
                # Assume it's a URL
                input_params["input_video"] = video_path
                output = self.client.run(
                    self.MODEL_VERSION,
                    input=input_params
                )
            
            # Handle output - it might be a generator or list
            if hasattr(output, '__iter__') and not isinstance(output, str):
                # Convert generator to list if needed
                output_list = list(output)
                if output_list and len(output_list) > 0:
                    result_url = output_list[0] if isinstance(output_list[0], str) else str(output_list[0])
                else:
                    result_url = str(output)
            else:
                result_url = str(output)
            
            print(f"Segmentation complete! Output: {result_url}")
            
            # Download output if output_path is specified
            if output_path and result_url.startswith("http"):
                print(f"Downloading result to {output_path}...")
                response = requests.get(result_url)
                response.raise_for_status()
                
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"Saved to {output_path}")
                return output_path
            
            return result_url
            
        except Exception as e:
            print(f"Error during segmentation: {e}")
            raise
    
    def auto_detect_objects(
        self,
        video_path: str,
        num_objects: int = 5,
        frame_sample_rate: int = 30
    ) -> List[ClickPoint]:
        """
        Automatically detect objects in video for segmentation.
        This is a placeholder for future implementation that could use
        object detection models to automatically identify click points.
        
        Args:
            video_path: Path to input video
            num_objects: Number of objects to detect
            frame_sample_rate: Sample every Nth frame for detection
            
        Returns:
            List of ClickPoint objects for detected objects
        """
        # This would integrate with an object detection model
        # For now, returns example points
        print(f"Auto-detection not yet implemented. Using example points.")
        return [
            ClickPoint(100, 100, 0, 1, "object_1"),
            ClickPoint(200, 200, 0, 1, "object_2")
        ]


def main():
    """Example usage of the SAM2VideoSegmenter"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM2 Video Segmentation Tool")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--points",
        help="Click points as 'x,y,frame' separated by semicolons (e.g., '100,100,0;200,200,10')",
        default="100,100,0"
    )
    parser.add_argument(
        "--object-ids",
        help="Object IDs separated by commas (e.g., 'dog,cat')",
        default=""
    )
    parser.add_argument(
        "--labels",
        help="Labels (1=foreground, 0=background) separated by commas",
        default=""
    )
    parser.add_argument(
        "--mask-type",
        choices=["binary", "highlighted", "greenscreen"],
        default="highlighted",
        help="Type of mask output"
    )
    parser.add_argument(
        "--output",
        help="Output file path",
        default=None
    )
    parser.add_argument(
        "--output-video",
        action="store_true",
        default=True,
        help="Output as video (default) instead of image sequence"
    )
    
    args = parser.parse_args()
    
    # Parse click points
    click_points = []
    for point_str in args.points.split(";"):
        parts = point_str.split(",")
        if len(parts) >= 2:
            x, y = int(parts[0]), int(parts[1])
            frame = int(parts[2]) if len(parts) > 2 else 0
            click_points.append((x, y, frame))
    
    # Parse object IDs and labels
    object_ids = args.object_ids.split(",") if args.object_ids else None
    labels = [int(l) for l in args.labels.split(",")] if args.labels else None
    
    # Initialize segmenter
    segmenter = SAM2VideoSegmenter()
    
    # Run segmentation
    result = segmenter.segment_video(
        args.video,
        click_points,
        object_ids=object_ids,
        labels=labels,
        mask_type=args.mask_type,
        output_video=args.output_video,
        output_path=args.output
    )
    
    print(f"Result: {result}")


if __name__ == "__main__":
    main()