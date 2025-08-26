"""Auto-comment pipeline package."""

from .main_pipeline import EndToEndCommentPipeline
from .models import Comment, SilenceGap

__all__ = [
    "EndToEndCommentPipeline",
    "Comment",
    "SilenceGap"
]