"""
Compound animations that combine multiple effects
"""

import numpy as np
import cv2
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_text_animation import BaseTextAnimation, AnimationConfig, EasingType

# Import specific animations
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'opacity_family'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'motion_family'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scale_family'))

from opacity_animation import SimpleFadeAnimation, BlurFadeAnimation
from motion_animation import SlideInAnimation, FloatUpAnimation
from scale_animation import ZoomInAnimation


class FadeSlideAnimation(BaseTextAnimation):
    """Combines fade and slide animations"""
    
    def __init__(self, config: AnimationConfig,
                 slide_direction: str = "bottom",
                 slide_distance: int = 50,
                 start_opacity: float = 0.0):
        super().__init__(config)
        self.slide_direction = slide_direction
        self.slide_distance = slide_distance
        self.start_opacity = start_opacity
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply combined fade and slide"""
        progress = self.get_progress(frame_number, fps)
        
        # Calculate slide offset
        if self.slide_direction == "bottom":
            offset_x = 0
            offset_y = self.slide_distance * (1 - progress)
        elif self.slide_direction == "top":
            offset_x = 0
            offset_y = -self.slide_distance * (1 - progress)
        elif self.slide_direction == "left":
            offset_x = -self.slide_distance * (1 - progress)
            offset_y = 0
        elif self.slide_direction == "right":
            offset_x = self.slide_distance * (1 - progress)
            offset_y = 0
        else:
            offset_x = offset_y = 0
        
        # Calculate position
        current_position = (
            int(self.config.position[0] + offset_x),
            int(self.config.position[1] + offset_y)
        )
        
        # Calculate opacity
        opacity = self.start_opacity + (1.0 - self.start_opacity) * progress
        
        return self.draw_text_with_shadow(
            frame, self.config.text, current_position,
            self.config.font_color, opacity
        )


class ScaleBlurAnimation(BaseTextAnimation):
    """Combines scale and blur animations"""
    
    def __init__(self, config: AnimationConfig,
                 start_scale: float = 1.5,
                 start_blur: int = 15):
        super().__init__(config)
        self.start_scale = start_scale
        self.start_blur = start_blur
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply combined scale and blur"""
        progress = self.get_progress(frame_number, fps)
        
        # Calculate current scale and blur
        scale = self.start_scale + (1.0 - self.start_scale) * progress
        blur = int(self.start_blur * (1 - progress))
        
        # Calculate scaled font size
        scaled_font_size = (self.config.font_size / 30) * scale
        
        # Get text dimensions
        text_size, baseline = cv2.getTextSize(
            self.config.text, self.font, scaled_font_size,
            self.config.font_thickness
        )
        
        # Adjust position to keep centered
        original_width, _ = self.get_text_dimensions()
        offset_x = (text_size[0] - original_width) / 2
        
        adjusted_position = (
            int(self.config.position[0] - offset_x),
            self.config.position[1]
        )
        
        # Create temporary overlay
        overlay = frame.copy()
        
        # Draw scaled text
        cv2.putText(
            overlay, self.config.text, adjusted_position, self.font,
            scaled_font_size, self.config.font_color,
            self.config.font_thickness, cv2.LINE_AA
        )
        
        # Apply blur if needed
        if blur > 0:
            kernel_size = blur * 2 + 1
            # Extract region around text for blur
            y1 = max(0, adjusted_position[1] - text_size[1] - 20)
            y2 = min(frame.shape[0], adjusted_position[1] + 20)
            x1 = max(0, adjusted_position[0] - 20)
            x2 = min(frame.shape[1], adjusted_position[0] + text_size[0] + 20)
            
            overlay[y1:y2, x1:x2] = cv2.GaussianBlur(
                overlay[y1:y2, x1:x2], (kernel_size, kernel_size), blur
            )
        
        # Apply opacity fade
        opacity = progress
        frame = cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0)
        
        return frame


class MultiLayerAnimation(BaseTextAnimation):
    """Orchestrates multiple animations in parallel or sequence"""
    
    def __init__(self, config: AnimationConfig,
                 animations: List[Dict[str, Any]],
                 timing_mode: str = "parallel"):  # parallel or sequential
        super().__init__(config)
        self.animations = animations
        self.timing_mode = timing_mode
        self._animation_instances = None
    
    def create_animation_instances(self):
        """Create instances of specified animations"""
        if self._animation_instances is not None:
            return
        
        self._animation_instances = []
        
        for anim_config in self.animations:
            anim_type = anim_config.get("type")
            params = anim_config.get("params", {})
            
            # Create appropriate animation instance
            if anim_type == "fade":
                anim = SimpleFadeAnimation(self.config, **params)
            elif anim_type == "slide":
                anim = SlideInAnimation(self.config, **params)
            elif anim_type == "zoom":
                anim = ZoomInAnimation(self.config, **params)
            elif anim_type == "float":
                anim = FloatUpAnimation(self.config, **params)
            elif anim_type == "blur_fade":
                anim = BlurFadeAnimation(self.config, **params)
            else:
                continue
            
            self._animation_instances.append({
                "animation": anim,
                "start_time": anim_config.get("start_time", 0),
                "duration": anim_config.get("duration", self.config.duration_ms)
            })
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply multiple animations"""
        self.create_animation_instances()
        
        current_time_ms = (frame_number / fps) * 1000
        result_frame = frame.copy()
        
        if self.timing_mode == "parallel":
            # Apply all animations simultaneously
            overlay_frames = []
            
            for anim_data in self._animation_instances:
                anim = anim_data["animation"]
                # Create a copy for this animation
                temp_frame = frame.copy()
                temp_frame = anim.apply_frame(temp_frame, frame_number, fps)
                overlay_frames.append(temp_frame)
            
            # Blend all frames together
            if overlay_frames:
                result_frame = overlay_frames[0]
                for overlay in overlay_frames[1:]:
                    result_frame = cv2.addWeighted(result_frame, 0.5, overlay, 0.5, 0)
        
        elif self.timing_mode == "sequential":
            # Apply animations in sequence
            for anim_data in self._animation_instances:
                anim = anim_data["animation"]
                start_time = anim_data["start_time"]
                duration = anim_data["duration"]
                
                # Check if this animation should be active
                if start_time <= current_time_ms < start_time + duration:
                    # Calculate local frame number for this animation
                    local_time = current_time_ms - start_time
                    local_frame = int(local_time * fps / 1000)
                    
                    # Apply this animation
                    result_frame = anim.apply_frame(result_frame, local_frame, fps)
        
        return result_frame