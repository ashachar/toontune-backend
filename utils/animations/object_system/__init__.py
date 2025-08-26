"""
Object-based animation system with state management.

This system separates:
1. Object state (is_behind, position, etc.)
2. Animation logic (motion, dissolve, etc.)
3. Post-processing (occlusion, effects, etc.)
"""

from .scene_object import SceneObject, LetterObject
from .render_pipeline import RenderPipeline
from .post_processor import PostProcessor, OcclusionProcessor

__all__ = [
    'SceneObject',
    'LetterObject', 
    'RenderPipeline',
    'PostProcessor',
    'OcclusionProcessor'
]