"""
Pipeline Configuration Module
==============================

Defines the configuration dataclass for the unified video pipeline.
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    video_path: str
    output_base_dir: str = "uploads/assets/videos"
    downsample_width: int = 256
    dry_run: bool = False  # Only affects LLM inference
    downsample_video: bool = True
    generate_transcripts: bool = True
    split_scenes: bool = True
    generate_prompts: bool = True
    run_inference: bool = True
    edit_videos: bool = True
    target_scene_duration: int = 60  # Target scene duration in seconds
    test_mode: bool = False  # When enabled, shows effect labels on video
    generate_karaoke: bool = False  # Generate karaoke-style captions
    karaoke_style: str = "continuous"  # Style: "continuous" (no flicker) or "simple"
    embed_phrases: bool = True  # Embed key phrases as text overlays
    embed_cartoons: bool = True  # Embed cartoon characters as animations