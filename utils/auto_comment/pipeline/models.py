"""Data models for the comment pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SilenceGap:
    """Represents a silence gap in the audio."""
    start: float
    end: float
    duration: float


@dataclass
class Comment:
    """Represents a comment to be inserted."""
    text: str
    time: float
    emotion: str
    context: str
    gap_duration: float
    audio_path: Optional[str] = None