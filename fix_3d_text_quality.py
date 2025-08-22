#!/usr/bin/env python3
"""
Fix for 3D text quality and center-shrinking issues.
"""

import os
import numpy as np
from typing import Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

# Import the original class
from utils.animations.text_3d_behind_segment import Text3DBehindSegment


def create_improved_3d_text_class():
    """Create an improved version of Text3DBehindSegment with better quality and center-shrinking."""
    
    class Text3DBehindSegmentImproved(Text3DBehindSegment):
        """Improved 3D text animation with anti-aliasing and proper center-shrinking."""
        
        def __init__(self, *args, **kwargs):
            # Add supersampling factor for anti-aliasing
            self.supersample_factor = kwargs.pop('supersample_factor', 3)
            super().__init__(*args, **kwargs)
        
        def render_3d_text(
            self, 
            text: str, 
            font: ImageFont.FreeTypeFont,
            scale: float = 1.0,
            alpha: float = 1.0,
            apply_perspective: bool = True
        ) -> Image.Image:
            """
            Render 3D text with improved anti-aliasing and quality.
            """
            # Render at higher resolution for anti-aliasing
            supersample = self.supersample_factor
            
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
                    # Use a better default font with anti-aliasing
                    from PIL import ImageFont
                    scaled_font = ImageFont.load_default()
            
            # Get text dimensions
            temp_img = Image.new('RGBA', (self.resolution[0] * supersample, self.resolution[1] * supersample), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), text, font=scaled_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Create larger canvas for 3D effect (at higher resolution)
            depth_offset_scaled = int(self.depth_offset * scale * supersample)
            canvas_width = text_width + self.depth_layers * depth_offset_scaled * 2
            canvas_height = text_height + self.depth_layers * depth_offset_scaled * 2
            
            # Create 3D text image with anti-aliasing
            text_3d = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_3d)
            
            # Calculate base position (centered in canvas)
            base_x = (canvas_width - text_width) // 2
            base_y = (canvas_height - text_height) // 2
            
            # Draw depth layers with smoother gradients
            for i in range(self.depth_layers, 0, -1):
                # Calculate offset for this layer
                offset_x = base_x + i * depth_offset_scaled
                offset_y = base_y - i * depth_offset_scaled
                
                # Smoother color interpolation
                t = (self.depth_layers - i) / max(self.depth_layers - 1, 1)
                # Use cubic interpolation for smoother gradient
                t = t * t * (3.0 - 2.0 * t)  # Smoothstep function
                
                color_r = int(self.depth_color[0] * (1 - t) + self.text_color[0] * t * 0.8)
                color_g = int(self.depth_color[1] * (1 - t) + self.text_color[1] * t * 0.8)
                color_b = int(self.depth_color[2] * (1 - t) + self.text_color[2] * t * 0.8)
                layer_alpha = int(255 * alpha)
                
                # Draw this depth layer
                draw.text(
                    (offset_x, offset_y),
                    text,
                    font=scaled_font,
                    fill=(color_r, color_g, color_b, layer_alpha),
                    stroke_width=0  # Disable stroke for cleaner rendering
                )
            
            # Draw front face with better anti-aliased outline
            front_x = base_x
            front_y = base_y
            
            # Create a separate image for the outline with anti-aliasing
            outline_img = Image.new('RGBA', text_3d.size, (0, 0, 0, 0))
            outline_draw = ImageDraw.Draw(outline_img)
            
            # Draw multiple outline layers for smoother edges
            outline_width_scaled = self.outline_width * supersample
            for radius in range(outline_width_scaled, 0, -1):
                alpha_factor = 1.0 - (radius - 1) / outline_width_scaled * 0.5
                for angle in range(0, 360, 30):  # Sample points around circle
                    ox = int(radius * np.cos(np.radians(angle)))
                    oy = int(radius * np.sin(np.radians(angle)))
                    outline_draw.text(
                        (front_x + ox, front_y + oy),
                        text,
                        font=scaled_font,
                        fill=(0, 0, 0, int(100 * alpha * alpha_factor))
                    )
            
            # Apply slight blur to outline for anti-aliasing
            outline_img = outline_img.filter(ImageFilter.GaussianBlur(radius=supersample//2))
            
            # Composite outline onto main image
            text_3d = Image.alpha_composite(text_3d, outline_img)
            
            # Draw main text face
            main_text = Image.new('RGBA', text_3d.size, (0, 0, 0, 0))
            main_draw = ImageDraw.Draw(main_text)
            main_draw.text(
                (front_x, front_y),
                text,
                font=scaled_font,
                fill=(*self.text_color, int(255 * alpha))
            )
            
            # Composite main text
            text_3d = Image.alpha_composite(text_3d, main_text)
            
            # Apply perspective transformation if requested
            if apply_perspective and self.perspective_angle > 0:
                text_3d_np = np.array(text_3d)
                h, w = text_3d_np.shape[:2]
                
                angle_rad = np.radians(self.perspective_angle)
                
                # Source points (rectangle)
                src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                
                # Destination points (trapezoid for perspective)
                perspective_offset = int(h * np.tan(angle_rad) * 0.2)
                dst_pts = np.float32([
                    [perspective_offset, 0],
                    [w - perspective_offset, 0],
                    [w, h],
                    [0, h]
                ])
                
                # Get perspective transform matrix
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                # Apply perspective transform with better interpolation
                text_3d_np = cv2.warpPerspective(
                    text_3d_np, 
                    matrix, 
                    (w, h),
                    flags=cv2.INTER_CUBIC,  # Use cubic interpolation
                    borderMode=cv2.BORDER_TRANSPARENT
                )
                
                text_3d = Image.fromarray(text_3d_np)
            
            # Add high-quality shadow
            shadow = Image.new('RGBA', text_3d.size, (0, 0, 0, 0))
            shadow_text = text_3d.copy()
            shadow_text_np = np.array(shadow_text)
            shadow_text_np[:, :, :3] = 0  # Make shadow black
            shadow_text_np[:, :, 3] = (shadow_text_np[:, :, 3] * 0.4).astype(np.uint8)
            shadow = Image.fromarray(shadow_text_np)
            
            # Apply multi-stage blur for softer shadow
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=supersample*2))
            
            # Composite shadow with offset
            final_img = Image.new('RGBA', text_3d.size, (0, 0, 0, 0))
            shadow_offset_scaled = int(self.shadow_offset * scale * supersample)
            final_img.paste(shadow, (shadow_offset_scaled, shadow_offset_scaled), shadow)
            final_img = Image.alpha_composite(final_img, text_3d)
            
            # Downsample to target resolution with high-quality anti-aliasing
            target_width = final_img.width // supersample
            target_height = final_img.height // supersample
            
            # Apply slight blur before downsampling for better anti-aliasing
            final_img = final_img.filter(ImageFilter.GaussianBlur(radius=supersample/4))
            
            # Use LANCZOS resampling for best quality
            final_img = final_img.resize(
                (target_width, target_height), 
                Image.Resampling.LANCZOS
            )
            
            return final_img
        
        def generate_frame(self, frame_number: int, background: Optional[np.ndarray] = None) -> np.ndarray:
            """Generate a frame with proper center-shrinking."""
            
            # Create base frame
            if background is not None:
                frame = background.copy()
                if frame.shape[2] == 3:
                    frame = np.concatenate([frame, np.ones((*frame.shape[:2], 1), dtype=np.uint8) * 255], axis=2)
            else:
                frame = np.zeros((*self.resolution[::-1], 4), dtype=np.uint8)
                frame[:, :, 3] = 255
            
            # Determine animation phase
            if frame_number < self.phase1_frames:
                # Phase 1: Shrinking (foreground)
                phase = "shrink"
                phase_progress = frame_number / max(self.phase1_frames - 1, 1)
                # Use ease-in-out for smoother animation
                t = phase_progress
                t = t * t * (3.0 - 2.0 * t)  # Smoothstep
                scale = self.start_scale + (self.end_scale - self.start_scale) * t
                alpha = 1.0
                is_behind = False
                
            elif frame_number < self.phase1_frames + self.phase2_frames:
                # Phase 2: Moving behind (transition)
                phase = "transition"
                phase_progress = (frame_number - self.phase1_frames) / max(self.phase2_frames - 1, 1)
                scale = self.end_scale
                k = 3.0
                alpha = 1.0 - 0.5 * (1 - np.exp(-k * phase_progress)) / (1 - np.exp(-k))
                is_behind = phase_progress > 0.5
                
            else:
                # Phase 3: Stable behind
                phase = "stable"
                scale = self.end_scale
                alpha = 0.5
                is_behind = True
            
            # Render 3D text with improved quality
            text_img = self.render_3d_text(
                self.text, 
                self.font, 
                scale=scale, 
                alpha=alpha,
                apply_perspective=(phase != "shrink")
            )
            
            # Convert to numpy array
            text_np = np.array(text_img)
            
            # CRITICAL FIX: Calculate position to keep text centered while scaling
            text_h, text_w = text_np.shape[:2]
            
            # The text should always be centered at the same point
            # Calculate offset to maintain center position during scaling
            center_x, center_y = self.center_position
            
            # Position text so its center aligns with the target center
            pos_x = center_x - text_w // 2
            pos_y = center_y - text_h // 2
            
            # Ensure position is within frame bounds
            pos_x = max(0, min(pos_x, self.resolution[0] - text_w))
            pos_y = max(0, min(pos_y, self.resolution[1] - text_h))
            
            # Create text layer
            text_layer = np.zeros_like(frame)
            
            # Calculate the region where text will be placed
            y1 = pos_y
            y2 = min(pos_y + text_h, self.resolution[1])
            x1 = pos_x
            x2 = min(pos_x + text_w, self.resolution[0])
            
            # Calculate the corresponding region in the text image
            ty1 = 0
            ty2 = y2 - y1
            tx1 = 0
            tx2 = x2 - x1
            
            # Place text in the layer
            if ty2 > ty1 and tx2 > tx1:
                text_layer[y1:y2, x1:x2] = text_np[ty1:ty2, tx1:tx2]
            
            if is_behind:
                # Apply segment mask to hide text behind subject
                mask_region = self.segment_mask[y1:y2, x1:x2]
                text_alpha = text_layer[y1:y2, x1:x2, 3].astype(float)
                text_alpha = text_alpha * (1 - mask_region / 255.0)
                text_layer[y1:y2, x1:x2, 3] = text_alpha.astype(np.uint8)
            
            # Composite text layer onto frame
            frame_pil = Image.fromarray(frame)
            text_pil = Image.fromarray(text_layer)
            result = Image.alpha_composite(frame_pil, text_pil)
            
            return np.array(result)
    
    return Text3DBehindSegmentImproved


# Export the improved class
Text3DBehindSegmentImproved = create_improved_3d_text_class()


if __name__ == "__main__":
    import os
    print("Improved Text3DBehindSegment class created with:")
    print("  ✓ 3x supersampling for anti-aliasing")
    print("  ✓ High-quality LANCZOS resampling")
    print("  ✓ Smooth gradient interpolation")
    print("  ✓ Multi-layer soft shadows")
    print("  ✓ Proper center-point scaling")
    print("  ✓ Cubic interpolation for perspective transform")