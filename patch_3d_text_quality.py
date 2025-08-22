#!/usr/bin/env python3
"""
Patch the original Text3DBehindSegment class to fix quality and center-shrinking issues.
"""

import os
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Monkey-patch the original class
from utils.animations import text_3d_behind_segment
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2


# Store original methods
original_render_3d_text = text_3d_behind_segment.Text3DBehindSegment.render_3d_text
original_generate_frame = text_3d_behind_segment.Text3DBehindSegment.generate_frame


def patched_render_3d_text(
    self,
    text: str, 
    font: ImageFont.FreeTypeFont,
    scale: float = 1.0,
    alpha: float = 1.0,
    apply_perspective: bool = True
) -> Image.Image:
    """
    Patched render_3d_text with improved anti-aliasing.
    """
    # Supersampling factor for anti-aliasing
    supersample = 2
    
    # Calculate scaled font size at higher resolution
    scaled_font_size = int(self.font_size * scale * supersample)
    if self.font_path and os.path.exists(self.font_path):
        scaled_font = ImageFont.truetype(self.font_path, scaled_font_size)
    else:
        try:
            scaled_font = ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", 
                scaled_font_size
            )
        except:
            scaled_font = ImageFont.load_default()
    
    # Get text dimensions
    temp_img = Image.new('RGBA', (self.resolution[0] * supersample, self.resolution[1] * supersample), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=scaled_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Create larger canvas for 3D effect
    depth_offset_scaled = int(self.depth_offset * scale * supersample)
    canvas_width = text_width + self.depth_layers * depth_offset_scaled * 3
    canvas_height = text_height + self.depth_layers * depth_offset_scaled * 3
    
    # Create 3D text image
    text_3d = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_3d)
    
    # Calculate base position (centered in canvas)
    base_x = (canvas_width - text_width) // 2
    base_y = (canvas_height - text_height) // 2
    
    # Draw depth layers (back to front)
    for i in range(self.depth_layers, 0, -1):
        offset_x = base_x + i * depth_offset_scaled
        offset_y = base_y - i * depth_offset_scaled
        
        # Smoother color interpolation
        t = (self.depth_layers - i) / max(self.depth_layers - 1, 1)
        t = t * t * (3.0 - 2.0 * t)  # Smoothstep
        
        color_r = int(self.depth_color[0] * (1 - t) + self.text_color[0] * t * 0.8)
        color_g = int(self.depth_color[1] * (1 - t) + self.text_color[1] * t * 0.8)
        color_b = int(self.depth_color[2] * (1 - t) + self.text_color[2] * t * 0.8)
        layer_alpha = int(255 * alpha)
        
        draw.text(
            (offset_x, offset_y),
            text,
            font=scaled_font,
            fill=(color_r, color_g, color_b, layer_alpha)
        )
    
    # Draw front face with outline
    # Better anti-aliased outline
    outline_width_scaled = self.outline_width * supersample
    for radius in [outline_width_scaled * 1.5, outline_width_scaled]:
        for angle in range(0, 360, 45):
            ox = int(radius * np.cos(np.radians(angle)))
            oy = int(radius * np.sin(np.radians(angle)))
            draw.text(
                (base_x + ox, base_y + oy),
                text,
                font=scaled_font,
                fill=(0, 0, 0, int(150 * alpha))
            )
    
    # Draw main text face
    draw.text(
        (base_x, base_y),
        text,
        font=scaled_font,
        fill=(*self.text_color, int(255 * alpha))
    )
    
    # Apply perspective if requested
    if apply_perspective and self.perspective_angle > 0:
        text_3d_np = np.array(text_3d)
        h, w = text_3d_np.shape[:2]
        
        angle_rad = np.radians(self.perspective_angle)
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        perspective_offset = int(h * np.tan(angle_rad) * 0.2)
        dst_pts = np.float32([
            [perspective_offset, 0],
            [w - perspective_offset, 0],
            [w, h],
            [0, h]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        text_3d_np = cv2.warpPerspective(
            text_3d_np, 
            matrix, 
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_TRANSPARENT
        )
        
        text_3d = Image.fromarray(text_3d_np)
    
    # Add shadow
    shadow = text_3d.copy()
    shadow_np = np.array(shadow)
    shadow_np[:, :, :3] = 0
    shadow_np[:, :, 3] = (shadow_np[:, :, 3] * 0.3).astype(np.uint8)
    shadow = Image.fromarray(shadow_np)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=supersample*2))
    
    # Composite shadow
    final_img = Image.new('RGBA', text_3d.size, (0, 0, 0, 0))
    shadow_offset_scaled = int(self.shadow_offset * scale * supersample)
    final_img.paste(shadow, (shadow_offset_scaled, shadow_offset_scaled), shadow)
    final_img = Image.alpha_composite(final_img, text_3d)
    
    # Apply slight blur before downsampling for anti-aliasing
    final_img = final_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Downsample to target resolution
    target_width = final_img.width // supersample
    target_height = final_img.height // supersample
    final_img = final_img.resize(
        (target_width, target_height), 
        Image.Resampling.LANCZOS
    )
    
    return final_img


def patched_generate_frame(self, frame_number: int, background: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Patched generate_frame with proper center-shrinking.
    """
    # Create base frame
    if background is not None:
        frame = background.copy()
        # Ensure RGBA format
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
        elif frame.shape[2] == 3:  # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    else:
        frame = np.ones((*self.resolution[::-1], 4), dtype=np.uint8) * 255
        frame[:, :, :3] = 0  # Black background
    
    # Determine animation phase
    if frame_number < self.phase1_frames:
        # Phase 1: Shrinking (foreground)
        phase_progress = frame_number / max(self.phase1_frames - 1, 1)
        # Smoothstep for smoother animation
        t = phase_progress * phase_progress * (3.0 - 2.0 * phase_progress)
        scale = self.start_scale + (self.end_scale - self.start_scale) * t
        alpha = 1.0
        is_behind = False
        
    elif frame_number < self.phase1_frames + self.phase2_frames:
        # Phase 2: Moving behind (transition)
        phase_progress = (frame_number - self.phase1_frames) / max(self.phase2_frames - 1, 1)
        scale = self.end_scale
        k = 3.0
        alpha = 1.0 - 0.5 * (1 - np.exp(-k * phase_progress)) / (1 - np.exp(-k))
        is_behind = phase_progress > 0.5
        
    else:
        # Phase 3: Stable behind
        scale = self.end_scale
        alpha = 0.5
        is_behind = True
    
    # Render 3D text
    text_img = self.render_3d_text(
        self.text, 
        self.font, 
        scale=scale, 
        alpha=alpha,
        apply_perspective=(frame_number >= self.phase1_frames)  # Apply perspective after shrink
    )
    
    # Convert to numpy array
    text_np = np.array(text_img)
    
    # CRITICAL: Center the text properly
    text_h, text_w = text_np.shape[:2]
    
    # Always center at the same point
    center_x, center_y = self.center_position
    pos_x = center_x - text_w // 2
    pos_y = center_y - text_h // 2
    
    # Ensure within bounds
    pos_x = max(0, min(pos_x, self.resolution[0] - text_w))
    pos_y = max(0, min(pos_y, self.resolution[1] - text_h))
    
    # Apply text to frame
    if is_behind and self.segment_mask is not None:
        # Apply masking for behind effect
        for y in range(text_h):
            for x in range(text_w):
                world_y = pos_y + y
                world_x = pos_x + x
                
                if 0 <= world_y < self.resolution[1] and 0 <= world_x < self.resolution[0]:
                    if text_np[y, x, 3] > 0:  # Non-transparent pixel
                        if self.segment_mask[world_y, world_x] < 128:  # Background area
                            # Composite text pixel onto frame
                            alpha_text = text_np[y, x, 3] / 255.0
                            frame[world_y, world_x, :3] = (
                                frame[world_y, world_x, :3] * (1 - alpha_text) +
                                text_np[y, x, :3] * alpha_text
                            ).astype(np.uint8)
    else:
        # Foreground - composite normally
        for y in range(text_h):
            for x in range(text_w):
                world_y = pos_y + y
                world_x = pos_x + x
                
                if 0 <= world_y < self.resolution[1] and 0 <= world_x < self.resolution[0]:
                    if text_np[y, x, 3] > 0:  # Non-transparent pixel
                        alpha_text = text_np[y, x, 3] / 255.0
                        frame[world_y, world_x, :3] = (
                            frame[world_y, world_x, :3] * (1 - alpha_text) +
                            text_np[y, x, :3] * alpha_text
                        ).astype(np.uint8)
    
    # Return RGB (remove alpha channel for video)
    return frame[:, :, :3] if frame.shape[2] == 4 else frame


# Apply patches
text_3d_behind_segment.Text3DBehindSegment.render_3d_text = patched_render_3d_text
text_3d_behind_segment.Text3DBehindSegment.generate_frame = patched_generate_frame

print("✅ Text3DBehindSegment class patched successfully!")
print("Improvements applied:")
print("  • 2x supersampling for anti-aliasing")
print("  • Smoother gradients and shadows")
print("  • Proper center-point scaling")
print("  • Better perspective interpolation")
print("  • Fixed frame compositing")