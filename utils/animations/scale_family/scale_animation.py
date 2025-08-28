"""
Scale and 3D transformation text animations
"""

import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_text_animation import BaseTextAnimation, AnimationConfig


class ZoomInAnimation(BaseTextAnimation):
    """Text zooms in from larger scale"""
    
    def __init__(self, config: AnimationConfig,
                 start_scale: float = 1.5,
                 end_scale: float = 1.0,
                 fade_with_zoom: bool = True):
        super().__init__(config)
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.fade_with_zoom = fade_with_zoom
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply zoom animation"""
        progress = self.get_progress(frame_number, fps)
        
        # Calculate current scale
        scale = self.start_scale + (self.end_scale - self.start_scale) * progress
        
        # Calculate scaled font size
        scaled_font_size = (self.config.font_size / 30) * scale
        
        # Get text dimensions at current scale
        text_size, baseline = cv2.getTextSize(
            self.config.text, self.font, scaled_font_size, 
            self.config.font_thickness
        )
        
        # Adjust position to keep text centered during zoom
        text_width, text_height = text_size
        original_width, original_height = self.get_text_dimensions()
        
        offset_x = (text_width - original_width) / 2
        offset_y = (text_height - original_height) / 2
        
        adjusted_position = (
            int(self.config.position[0] - offset_x),
            int(self.config.position[1] + offset_y)
        )
        
        # Optional fade effect
        opacity = progress if self.fade_with_zoom else 1.0
        
        # Draw scaled text
        overlay = frame.copy()
        if self.config.shadow and opacity > 0:
            shadow_pos = (
                adjusted_position[0] + self.config.shadow_offset[0],
                adjusted_position[1] + self.config.shadow_offset[1]
            )
            cv2.putText(
                overlay, self.config.text, shadow_pos, self.font,
                scaled_font_size, self.config.shadow_color,
                self.config.font_thickness, cv2.LINE_AA
            )
        
        cv2.putText(
            overlay, self.config.text, adjusted_position, self.font,
            scaled_font_size, self.config.font_color,
            self.config.font_thickness, cv2.LINE_AA
        )
        
        # Apply opacity
        if opacity < 1.0:
            frame = cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0)
        else:
            frame = overlay
        
        return frame


class Rotate3DAnimation(BaseTextAnimation):
    """3D rotation animation"""
    
    def __init__(self, config: AnimationConfig,
                 rotation_axis: str = "Y",  # X, Y, or Z
                 start_rotation: float = 90,
                 end_rotation: float = 0,
                 perspective: float = 1000):
        super().__init__(config)
        self.rotation_axis = rotation_axis
        self.start_rotation = start_rotation
        self.end_rotation = end_rotation
        self.perspective = perspective
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply 3D rotation animation"""
        progress = self.get_progress(frame_number, fps)
        
        # Calculate current rotation angle
        angle_deg = self.start_rotation + (self.end_rotation - self.start_rotation) * progress
        angle_rad = np.radians(angle_deg)
        
        # Create text image
        text_width, text_height = self.get_text_dimensions()
        text_img = np.zeros((text_height + 20, text_width + 20, 4), dtype=np.uint8)
        
        # Draw text on temporary image
        cv2.putText(
            text_img, self.config.text, (10, text_height + 5),
            self.font, self.config.font_size / 30, 
            (*self.config.font_color, 255),
            self.config.font_thickness, cv2.LINE_AA
        )
        
        # Apply 3D transformation
        h, w = text_img.shape[:2]
        
        # Create rotation matrix based on axis
        if self.rotation_axis == "Y":
            # Rotation around Y-axis (horizontal flip effect)
            pts_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            
            # Apply perspective transformation
            scale_x = abs(np.cos(angle_rad))
            if scale_x < 0.01:
                scale_x = 0.01
            
            center_x = w / 2
            pts_dst = np.float32([
                [center_x - center_x * scale_x, 0],
                [center_x + center_x * scale_x, 0],
                [center_x + center_x * scale_x, h],
                [center_x - center_x * scale_x, h]
            ])
        elif self.rotation_axis == "X":
            # Rotation around X-axis (vertical flip effect)
            pts_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            
            scale_y = abs(np.cos(angle_rad))
            if scale_y < 0.01:
                scale_y = 0.01
            
            center_y = h / 2
            pts_dst = np.float32([
                [0, center_y - center_y * scale_y],
                [w, center_y - center_y * scale_y],
                [w, center_y + center_y * scale_y],
                [0, center_y + center_y * scale_y]
            ])
        else:  # Z-axis
            # Rotation around Z-axis (2D rotation)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            text_img[:, :, :3] = cv2.warpAffine(text_img[:, :, :3], M, (w, h))
            text_img[:, :, 3] = cv2.warpAffine(text_img[:, :, 3], M, (w, h))
            pts_src = pts_dst = None
        
        if pts_src is not None and pts_dst is not None:
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            text_img[:, :, :3] = cv2.warpPerspective(text_img[:, :, :3], M, (w, h))
            text_img[:, :, 3] = cv2.warpPerspective(text_img[:, :, 3], M, (w, h))
        
        # Calculate opacity based on rotation (fade when edge-on)
        visibility = abs(np.cos(angle_rad))
        opacity = max(0.0, min(1.0, visibility * 2))
        
        # Blend text image with frame
        x, y = self.config.position
        x = max(0, min(x, frame.shape[1] - w))
        y = max(0, min(y, frame.shape[0] - h))
        
        roi = frame[y:y+h, x:x+w]
        alpha = text_img[:, :, 3] / 255.0 * opacity
        
        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha) + text_img[:, :, c] * alpha
        
        frame[y:y+h, x:x+w] = roi
        
        return frame