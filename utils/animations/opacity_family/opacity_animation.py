"""
Opacity-based text animations
"""

import numpy as np
import cv2
from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_text_animation import BaseTextAnimation, AnimationConfig


class SimpleFadeAnimation(BaseTextAnimation):
    """Simple fade in/out animation"""
    
    def __init__(self, config: AnimationConfig, 
                 start_opacity: float = 0.0,
                 end_opacity: float = 1.0):
        super().__init__(config)
        self.start_opacity = start_opacity
        self.end_opacity = end_opacity
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply fade animation to frame"""
        progress = self.get_progress(frame_number, fps)
        opacity = self.start_opacity + (self.end_opacity - self.start_opacity) * progress
        
        return self.draw_text_with_shadow(
            frame, self.config.text, self.config.position,
            self.config.font_color, opacity
        )


class BlurFadeAnimation(BaseTextAnimation):
    """Fade with blur effect"""
    
    def __init__(self, config: AnimationConfig,
                 start_blur: int = 10,
                 end_blur: int = 0,
                 start_opacity: float = 0.0,
                 end_opacity: float = 1.0):
        super().__init__(config)
        self.start_blur = start_blur
        self.end_blur = end_blur
        self.start_opacity = start_opacity
        self.end_opacity = end_opacity
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply blur fade animation"""
        progress = self.get_progress(frame_number, fps)
        
        # Calculate current blur and opacity
        blur = int(self.start_blur + (self.end_blur - self.start_blur) * progress)
        opacity = self.start_opacity + (self.end_opacity - self.start_opacity) * progress
        
        # Draw text
        frame_with_text = self.draw_text_with_shadow(
            frame, self.config.text, self.config.position,
            self.config.font_color, opacity
        )
        
        # Apply blur if needed
        if blur > 0:
            kernel_size = blur * 2 + 1
            frame_with_text = cv2.GaussianBlur(frame_with_text, (kernel_size, kernel_size), blur)
        
        return frame_with_text


class GlowFadeAnimation(BaseTextAnimation):
    """Fade with glowing outline effect"""
    
    def __init__(self, config: AnimationConfig,
                 glow_radius: int = 5,
                 glow_intensity: float = 1.5,
                 pulse_count: int = 0,
                 start_opacity: float = 0.0,
                 end_opacity: float = 1.0):
        super().__init__(config)
        self.glow_radius = glow_radius
        self.glow_intensity = glow_intensity
        self.pulse_count = pulse_count
        self.start_opacity = start_opacity
        self.end_opacity = end_opacity
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply glow fade animation"""
        progress = self.get_progress(frame_number, fps)
        opacity = self.start_opacity + (self.end_opacity - self.start_opacity) * progress
        
        # Calculate pulse effect if enabled
        glow_multiplier = 1.0
        if self.pulse_count > 0:
            pulse_phase = (progress * self.pulse_count * 2 * np.pi)
            glow_multiplier = 0.5 + 0.5 * np.sin(pulse_phase)
        
        # Create glow effect
        overlay = np.zeros_like(frame)
        
        # Draw multiple layers for glow
        for i in range(self.glow_radius, 0, -1):
            glow_opacity = (opacity * glow_multiplier * self.glow_intensity * 
                          (i / self.glow_radius) * 0.3)
            glow_color = tuple(int(c * 1.2) for c in self.config.font_color)
            glow_color = tuple(min(255, c) for c in glow_color)
            
            cv2.putText(
                overlay, self.config.text, self.config.position, self.font,
                self.config.font_size / 30, glow_color,
                self.config.font_thickness + i * 2, cv2.LINE_AA
            )
        
        # Blend glow with frame
        frame = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)
        
        # Draw main text on top
        return self.draw_text_with_shadow(
            frame, self.config.text, self.config.position,
            self.config.font_color, opacity
        )