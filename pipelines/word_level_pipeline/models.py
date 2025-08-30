"""
Data models for word-level pipeline
"""

from dataclasses import dataclass
import random


@dataclass
class WordObject:
    """Individual word with persistent position and state"""
    text: str
    x: int  # Fixed X position (never changes)
    y: int  # Fixed Y position (never changes)
    width: int
    height: int
    start_time: float
    end_time: float
    rise_duration: float
    from_below: bool  # Direction for this word's sentence
    is_behind: bool  # Whether text should render behind foreground
    font_size: int = 48  # Font size for proper rendering
    scene_index: int = 0  # Which scene this word belongs to (0-based)
    color: tuple = (255, 255, 255)  # RGB color for the text
    # Fog parameters (randomized once, then fixed)
    blur_x: float = 1.0
    blur_y: float = 1.0
    fog_density: float = 1.0
    dissolve_speed: float = 1.0

    def __post_init__(self):
        """Initialize randomized fog parameters if not set"""
        if self.blur_x == 1.0:
            self.blur_x = random.uniform(0.8, 1.2)
        if self.blur_y == 1.0:
            self.blur_y = random.uniform(0.8, 1.2)
        if self.fog_density == 1.0:
            self.fog_density = random.uniform(0.8, 1.2)
        if self.dissolve_speed == 1.0:
            self.dissolve_speed = random.uniform(0.9, 1.1)