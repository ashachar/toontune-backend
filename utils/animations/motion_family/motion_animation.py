"""
Motion-based text animations
"""

import numpy as np
import cv2
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_text_animation import BaseTextAnimation, AnimationConfig, EasingType


class SlideInAnimation(BaseTextAnimation):
    """Slide in from specified direction"""
    
    def __init__(self, config: AnimationConfig,
                 direction: str = "left",  # left, right, top, bottom
                 distance: int = 100,
                 fade_with_slide: bool = True):
        super().__init__(config)
        self.direction = direction
        self.distance = distance
        self.fade_with_slide = fade_with_slide
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply slide animation"""
        progress = self.get_progress(frame_number, fps)
        
        # Calculate position offset
        if self.direction == "left":
            offset_x = -self.distance * (1 - progress)
            offset_y = 0
        elif self.direction == "right":
            offset_x = self.distance * (1 - progress)
            offset_y = 0
        elif self.direction == "top":
            offset_x = 0
            offset_y = -self.distance * (1 - progress)
        elif self.direction == "bottom":
            offset_x = 0
            offset_y = self.distance * (1 - progress)
        else:
            offset_x = offset_y = 0
        
        # Calculate current position
        current_position = (
            int(self.config.position[0] + offset_x),
            int(self.config.position[1] + offset_y)
        )
        
        # Optional fade effect
        opacity = progress if self.fade_with_slide else 1.0
        
        return self.draw_text_with_shadow(
            frame, self.config.text, current_position,
            self.config.font_color, opacity
        )


class FloatUpAnimation(BaseTextAnimation):
    """Text floats upward while fading in"""
    
    def __init__(self, config: AnimationConfig,
                 float_distance: int = 30,
                 start_opacity: float = 0.0):
        super().__init__(config)
        self.float_distance = float_distance
        self.start_opacity = start_opacity
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply float up animation"""
        progress = self.get_progress(frame_number, fps)
        
        # Calculate vertical offset (starts below, moves up)
        offset_y = self.float_distance * (1 - progress)
        
        # Calculate position
        current_position = (
            self.config.position[0],
            int(self.config.position[1] + offset_y)
        )
        
        # Calculate opacity
        opacity = self.start_opacity + (1.0 - self.start_opacity) * progress
        
        return self.draw_text_with_shadow(
            frame, self.config.text, current_position,
            self.config.font_color, opacity
        )


class BounceInAnimation(BaseTextAnimation):
    """Slide with elastic bounce effect"""
    
    def __init__(self, config: AnimationConfig,
                 direction: str = "bottom",
                 distance: int = 150,
                 overshoot: float = 1.2,
                 bounce_count: int = 2):
        super().__init__(config)
        self.direction = direction
        self.distance = distance
        self.overshoot = overshoot
        self.bounce_count = bounce_count
        # Force elastic easing for bounce effect
        self.config.easing = EasingType.ELASTIC
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply bounce animation"""
        raw_progress = (frame_number / fps) * 1000 / self.config.duration_ms
        raw_progress = min(1.0, max(0.0, raw_progress))
        
        # Custom bounce calculation
        if raw_progress < 0.6:
            # Initial approach
            progress = self.ease_value(raw_progress / 0.6, EasingType.EASE_OUT)
            progress = progress * self.overshoot
        else:
            # Bounce back
            bounce_progress = (raw_progress - 0.6) / 0.4
            damping = 1 - (bounce_progress * 0.2)
            bounce = np.sin(bounce_progress * np.pi * self.bounce_count) * damping
            progress = self.overshoot - (self.overshoot - 1.0) * bounce_progress - bounce * 0.1
        
        # Calculate position offset
        if self.direction == "bottom":
            offset_x = 0
            offset_y = self.distance * (1 - progress)
        elif self.direction == "top":
            offset_x = 0
            offset_y = -self.distance * (1 - progress)
        elif self.direction == "left":
            offset_x = -self.distance * (1 - progress)
            offset_y = 0
        elif self.direction == "right":
            offset_x = self.distance * (1 - progress)
            offset_y = 0
        else:
            offset_x = offset_y = 0
        
        # Calculate current position
        current_position = (
            int(self.config.position[0] + offset_x),
            int(self.config.position[1] + offset_y)
        )
        
        # Fade in during first 30% of animation
        opacity = min(1.0, raw_progress / 0.3)
        
        return self.draw_text_with_shadow(
            frame, self.config.text, current_position,
            self.config.font_color, opacity
        )