"""
Word-Level Text Animation Pipeline with Fog Dissolve
Golden standard implementation from session development

This package provides a modular word-level animation pipeline that:
- Uses AI-powered transcript enrichment for intelligent phrasing
- Implements face-aware text placement to avoid covering faces
- Uses stripe-based layout with optimal positioning
- Provides importance-based visual styling
- Tracks words individually throughout animation
- Maintains fixed positions during fog dissolve effects
- Renders clean fog dissolve transitions
"""

from .models import WordObject
from .masking import ForegroundMaskExtractor
from .word_factory import WordFactory
from .rendering import WordRenderer
from .frame_processor import FrameProcessor
from .transcript_handler import TranscriptHandler
from .video_generator import VideoGenerator
from .scene_processor import SceneProcessor
from .pipeline import WordLevelPipeline

# Main function for creating word-level videos
from .main import create_word_level_video

__all__ = [
    'WordObject',
    'ForegroundMaskExtractor', 
    'WordFactory',
    'WordRenderer',
    'FrameProcessor',
    'TranscriptHandler',
    'VideoGenerator',
    'SceneProcessor',
    'WordLevelPipeline',
    'create_word_level_video'
]