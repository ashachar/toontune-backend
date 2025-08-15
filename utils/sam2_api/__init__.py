"""
SAM2 Video Segmentation API

This module provides video segmentation capabilities using Meta's SAM2 model via Replicate API.
For continuous frame-by-frame segmentation without gaps.
"""

from .video_segmentation import SAM2VideoSegmenter, ClickPoint, SegmentationConfig
from .sam2_continuous_segmentation import (
    ContinuousSAM2Segmenter,
    ObjectToTrack,
    VideoInfo,
    segment_video_simple
)

__all__ = [
    'SAM2VideoSegmenter',
    'ClickPoint',
    'SegmentationConfig',
    'ContinuousSAM2Segmenter',
    'ObjectToTrack',
    'VideoInfo',
    'segment_video_simple'
]