"""
Base class for all text animations
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum


class EasingType(Enum):
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    ELASTIC = "elastic"
    BOUNCE = "bounce"


@dataclass
class AnimationConfig:
    """Configuration for text animation"""
    text: str
    duration_ms: int
    position: Tuple[int, int]
    font_size: int = 48
    font_color: Tuple[int, int, int] = (255, 255, 255)
    font_thickness: int = 2
    easing: EasingType = EasingType.LINEAR
    shadow: bool = True
    shadow_offset: Tuple[int, int] = (2, 2)
    shadow_color: Tuple[int, int, int] = (0, 0, 0)


class BaseTextAnimation(ABC):
    """Abstract base class for text animations"""
    
    def __init__(self, config: AnimationConfig):
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self._text_size = None
        self._baseline = None
        
    def get_text_dimensions(self) -> Tuple[int, int]:
        """Get text width and height"""
        if self._text_size is None:
            self._text_size, self._baseline = cv2.getTextSize(
                self.config.text,
                self.font,
                self.config.font_size / 30,  # Scale factor
                self.config.font_thickness
            )
        return self._text_size
    
    def ease_value(self, t: float, easing: EasingType) -> float:
        """Apply easing function to normalized time value"""
        if easing == EasingType.LINEAR:
            return t
        elif easing == EasingType.EASE_IN:
            return t * t
        elif easing == EasingType.EASE_OUT:
            return 1 - (1 - t) ** 2
        elif easing == EasingType.EASE_IN_OUT:
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - ((2 * (1 - t)) ** 2) / 2
        elif easing == EasingType.ELASTIC:
            if t == 0 or t == 1:
                return t
            p = 0.3
            s = p / 4
            return -(2 ** (10 * (t - 1))) * np.sin((t - 1 - s) * (2 * np.pi) / p)
        elif easing == EasingType.BOUNCE:
            if t < 0.363636:
                return 7.5625 * t * t
            elif t < 0.727273:
                t = t - 0.545454
                return 7.5625 * t * t + 0.75
            elif t < 0.909091:
                t = t - 0.818182
                return 7.5625 * t * t + 0.9375
            else:
                t = t - 0.954545
                return 7.5625 * t * t + 0.984375
        return t
    
    def draw_text_with_shadow(self, frame: np.ndarray, text: str, 
                            position: Tuple[int, int], 
                            color: Tuple[int, int, int],
                            opacity: float = 1.0) -> np.ndarray:
        """Draw text with optional shadow"""
        overlay = frame.copy()
        
        # Draw shadow if enabled
        if self.config.shadow and opacity > 0:
            shadow_pos = (
                position[0] + self.config.shadow_offset[0],
                position[1] + self.config.shadow_offset[1]
            )
            cv2.putText(
                overlay, text, shadow_pos, self.font,
                self.config.font_size / 30, self.config.shadow_color,
                self.config.font_thickness, cv2.LINE_AA
            )
        
        # Draw main text
        if opacity > 0:
            cv2.putText(
                overlay, text, position, self.font,
                self.config.font_size / 30, color,
                self.config.font_thickness, cv2.LINE_AA
            )
        
        # Apply opacity
        if opacity < 1.0:
            frame = cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0)
        else:
            frame = overlay
            
        return frame
    
    @abstractmethod
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply animation to a single frame"""
        pass
    
    def get_progress(self, frame_number: int, fps: float) -> float:
        """Get animation progress (0.0 to 1.0) for current frame"""
        current_time_ms = (frame_number / fps) * 1000
        progress = min(1.0, max(0.0, current_time_ms / self.config.duration_ms))
        return self.ease_value(progress, self.config.easing)