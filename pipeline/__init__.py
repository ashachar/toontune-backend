"""
Unified Video Processing Pipeline Package
=========================================

A modular pipeline for processing videos with:
- Transcript generation
- Scene splitting
- Prompt generation
- LLM inference
- Video editing with effects
- Karaoke caption generation
"""

from .core.config import PipelineConfig
from .core.pipeline import UnifiedVideoPipeline

__all__ = ['PipelineConfig', 'UnifiedVideoPipeline']