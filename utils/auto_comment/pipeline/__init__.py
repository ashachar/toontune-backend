"""Auto-comment pipeline package."""

from .main_pipeline import EndToEndCommentPipeline
from .models import Snark, SilenceGap

__all__ = [
    "EndToEndCommentPipeline",
    "Snark",
    "SilenceGap"
]