"""
Text 3D Behind Segment animation.
3D text with depth moves from foreground to background behind a segmented object.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter
import cv2

class Text3DBehindSegment:
    """
    Animation where 3D text with depth moves from foreground to background behind a segmented object.
    
    This creates a dolly zoom / parallax-like effect where 3D text appears to move
    through 3D space, going from in front of the subject to behind them.
    The text has depth/extrusion effects to appear three-dimensional.
    
    Additional Parameters:
    ---------------------
    text : str
        Text to animate (default 'START')
    segment_mask : Union[np.ndarray, str]
        Either a numpy array of the segment mask or path to mask image
        (Alpha 255 -> subject (foreground), 0 -> background)
    font_size : int
        Base font size (default 150)
    font_path : str
        Path to font file (optional, uses system font if None)
    text_color : Tuple[int, int, int]
        RGB color for text face (default (255, 220, 0) - golden yellow)
    depth_color : Tuple[int, int, int]
        RGB color for text depth/sides (default (180, 150, 0) - darker yellow)
    depth_layers : int
        Number of depth layers to create 3D effect (default 8)
    depth_offset : int
        Pixel offset between depth layers (default 2)
    start_scale : float
        Initial scale factor for text (default 2.0)
    end_scale : float
        Final scale factor for text (default 1.0)
    phase1_duration : float
        Duration of shrinking phase in seconds (default 1.0)
    phase2_duration : float
        Duration of moving-behind phase in seconds (default 0.67)
    phase3_duration : float
        Duration of stable-behind phase in seconds (default 1.33)
    center_position : Optional[Tuple[int, int]]
        Center position for text (default: frame center)
    shadow_offset : int
        Shadow offset in pixels (default 5)
    outline_width : int
        Outline width in pixels (default 3)
    perspective_angle : float
        Angle for 3D perspective effect in degrees (default 25)
    """
    
    def __init__(
        self,
        duration: float = 3.0,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        text: str = 'START',
        segment_mask: Optional[Union[np.ndarray, str]] = None,
        font_size: int = 150,
        font_path: Optional[str] = None,
        text_color: Tuple[int, int, int] = (255, 220, 0),
        depth_color: Tuple[int, int, int] = (180, 150, 0),
        depth_layers: int = 8,
        depth_offset: int = 2,
        start_scale: float = 2.0,
        end_scale: float = 1.0,
        phase1_duration: float = 1.0,
        phase2_duration: float = 0.67,
        phase3_duration: float = 1.33,
        center_position: Optional[Tuple[int, int]] = None,
        shadow_offset: int = 5,
        outline_width: int = 3,
        perspective_angle: float = 25,
        **kwargs
    ):
        self.duration = duration
        self.fps = fps
        self.resolution = resolution
        self.total_frames = int(fps * duration)
        
        self.text = text
        self.font_size = font_size
        self.font_path = font_path
        self.text_color = text_color
        self.depth_color = depth_color
        self.depth_layers = depth_layers
        self.depth_offset = depth_offset
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.phase1_duration = phase1_duration
        self.phase2_duration = phase2_duration
        self.phase3_duration = phase3_duration
        self.center_position = center_position or (resolution[0] // 2, resolution[1] // 2)
        self.shadow_offset = shadow_offset
        self.outline_width = outline_width
        self.perspective_angle = perspective_angle
        
        # Load or create segment mask
        if segment_mask is None:
            # Create default mask (everything is background)
            self.segment_mask = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)
        elif isinstance(segment_mask, str):
            # Load from file
            mask_img = Image.open(segment_mask).convert('L')
            mask_img = mask_img.resize(resolution, Image.Resampling.LANCZOS)
            self.segment_mask = np.array(mask_img)
        else:
            # Use provided numpy array
            self.segment_mask = segment_mask
            if self.segment_mask.shape[:2] != (resolution[1], resolution[0]):
                # Resize if needed
                self.segment_mask = cv2.resize(
                    self.segment_mask, 
                    resolution, 
                    interpolation=cv2.INTER_LINEAR
                )
        
        # Ensure mask is binary-ish (threshold at 128)
        self.segment_mask = (self.segment_mask > 128).astype(np.uint8) * 255
        
        # Create font
        if self.font_path and os.path.exists(self.font_path):
            self.font = ImageFont.truetype(self.font_path, self.font_size)
        else:
            try:
                self.font = ImageFont.truetype(
                    "/System/Library/Fonts/Helvetica.ttc", 
                    self.font_size
                )
            except:
                self.font = ImageFont.load_default()
        
        # Calculate text bounds for centering
        temp_img = Image.new('RGBA', resolution, (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), self.text, font=self.font)
        self.text_width = bbox[2] - bbox[0]
        self.text_height = bbox[3] - bbox[1]
        
        # Calculate phase frame counts
        self.phase1_frames = int(self.phase1_duration * self.fps)
        self.phase2_frames = int(self.phase2_duration * self.fps)
        self.phase3_frames = int(self.phase3_duration * self.fps)
        
        # Ensure we don't exceed total frames
        total_phase_frames = self.phase1_frames + self.phase2_frames + self.phase3_frames
        if total_phase_frames > self.total_frames:
            # Scale down proportionally
            scale_factor = self.total_frames / total_phase_frames
            self.phase1_frames = int(self.phase1_frames * scale_factor)
            self.phase2_frames = int(self.phase2_frames * scale_factor)
            self.phase3_frames = self.total_frames - self.phase1_frames - self.phase2_frames
    
    def render_3d_text(
        self, 
        text: str, 
        font: ImageFont.FreeTypeFont,
        scale: float = 1.0,
        alpha: float = 1.0,
        apply_perspective: bool = True
    ) -> Image.Image:
        """
        Render 3D text with depth effect.
        
        Parameters:
        ----------
        text : str
            Text to render
        font : ImageFont
            Font to use
        scale : float
            Scale factor for text
        alpha : float
            Opacity (0-1)
        apply_perspective : bool
            Whether to apply perspective transformation
            
        Returns:
        -------
        Image.Image
            RGBA image with 3D text
        """
        # Calculate scaled font size
        scaled_font_size = int(self.font_size * scale)
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
        temp_img = Image.new('RGBA', self.resolution, (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=scaled_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create larger canvas for 3D effect
        canvas_width = text_width + self.depth_layers * self.depth_offset * 2
        canvas_height = text_height + self.depth_layers * self.depth_offset * 2
        
        # Create 3D text image
        text_3d = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_3d)
        
        # Calculate base position (centered in canvas)
        base_x = (canvas_width - text_width) // 2
        base_y = (canvas_height - text_height) // 2
        
        # Draw depth layers (back to front)
        depth_offset_scaled = int(self.depth_offset * scale)
        for i in range(self.depth_layers, 0, -1):
            # Calculate offset for this layer
            offset_x = base_x + i * depth_offset_scaled
            offset_y = base_y - i * depth_offset_scaled  # Negative for upward depth
            
            # Interpolate color between depth color and text color
            t = (self.depth_layers - i) / max(self.depth_layers - 1, 1)
            color_r = int(self.depth_color[0] * (1 - t) + self.text_color[0] * t * 0.7)
            color_g = int(self.depth_color[1] * (1 - t) + self.text_color[1] * t * 0.7)
            color_b = int(self.depth_color[2] * (1 - t) + self.text_color[2] * t * 0.7)
            layer_alpha = int(255 * alpha)
            
            # Draw this depth layer
            draw.text(
                (offset_x, offset_y),
                text,
                font=scaled_font,
                fill=(color_r, color_g, color_b, layer_alpha)
            )
        
        # Draw front face with outline
        front_x = base_x
        front_y = base_y
        
        # Draw outline
        for ox in range(-self.outline_width, self.outline_width + 1):
            for oy in range(-self.outline_width, self.outline_width + 1):
                if ox != 0 or oy != 0:
                    draw.text(
                        (front_x + ox, front_y + oy),
                        text,
                        font=scaled_font,
                        fill=(0, 0, 0, int(255 * alpha))
                    )
        
        # Draw main text face
        draw.text(
            (front_x, front_y),
            text,
            font=scaled_font,
            fill=(*self.text_color, int(255 * alpha))
        )
        
        # Apply perspective transformation if requested
        if apply_perspective and self.perspective_angle > 0:
            # Convert to numpy array for perspective transform
            text_3d_np = np.array(text_3d)
            h, w = text_3d_np.shape[:2]
            
            # Calculate perspective transformation matrix
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
            
            # Apply perspective transform
            text_3d_np = cv2.warpPerspective(
                text_3d_np, 
                matrix, 
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT
            )
            
            text_3d = Image.fromarray(text_3d_np)
        
        # Add shadow
        shadow = Image.new('RGBA', text_3d.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        
        # Create shadow from the 3D text
        shadow_text = text_3d.copy()
        shadow_text_np = np.array(shadow_text)
        # Make shadow black but preserve alpha
        shadow_text_np[:, :, :3] = 0  # Set RGB to black
        shadow_text_np[:, :, 3] = (shadow_text_np[:, :, 3] * 0.5).astype(np.uint8)  # Reduce alpha
        shadow = Image.fromarray(shadow_text_np)
        
        # Blur shadow
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Composite shadow with offset
        final_img = Image.new('RGBA', text_3d.size, (0, 0, 0, 0))
        shadow_offset_scaled = int(self.shadow_offset * scale)
        final_img.paste(shadow, (shadow_offset_scaled, shadow_offset_scaled), shadow)
        final_img = Image.alpha_composite(final_img, text_3d)
        
        return final_img
    
    def generate_frame(self, frame_number: int, background: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a single frame of the 3D text animation."""
        
        # Create base frame
        if background is not None:
            frame = background.copy()
            if frame.shape[2] == 3:
                # Add alpha channel
                frame = np.concatenate([frame, np.ones((*frame.shape[:2], 1), dtype=np.uint8) * 255], axis=2)
        else:
            frame = np.zeros((*self.resolution[::-1], 4), dtype=np.uint8)
            frame[:, :, 3] = 255  # Full opacity
        
        # Determine animation phase
        if frame_number < self.phase1_frames:
            # Phase 1: Shrinking (foreground)
            phase = "shrink"
            phase_progress = frame_number / max(self.phase1_frames - 1, 1)
            scale = self.start_scale + (self.end_scale - self.start_scale) * phase_progress
            alpha = 1.0  # Full opacity in foreground
            is_behind = False
            
        elif frame_number < self.phase1_frames + self.phase2_frames:
            # Phase 2: Moving behind (transition)
            phase = "transition"
            phase_progress = (frame_number - self.phase1_frames) / max(self.phase2_frames - 1, 1)
            scale = self.end_scale
            # Exponential fade for more natural transition
            k = 3.0  # Exponential factor
            alpha = 1.0 - 0.5 * (1 - np.exp(-k * phase_progress)) / (1 - np.exp(-k))
            is_behind = phase_progress > 0.5  # Switch to behind halfway through
            
        else:
            # Phase 3: Stable behind
            phase = "stable"
            scale = self.end_scale
            alpha = 0.5  # Half opacity when behind
            is_behind = True
        
        # Render 3D text
        text_img = self.render_3d_text(
            self.text, 
            self.font, 
            scale=scale, 
            alpha=alpha,
            apply_perspective=(phase != "shrink")  # Apply perspective after shrink phase
        )
        
        # Convert to numpy array
        text_np = np.array(text_img)
        
        # Calculate position to center the text
        text_h, text_w = text_np.shape[:2]
        pos_x = self.center_position[0] - text_w // 2
        pos_y = self.center_position[1] - text_h // 2
        
        # Ensure position is within frame bounds
        pos_x = max(0, min(pos_x, self.resolution[0] - text_w))
        pos_y = max(0, min(pos_y, self.resolution[1] - text_h))
        
        # Create text layer with correct size
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
        text_layer[y1:y2, x1:x2] = text_np[ty1:ty2, tx1:tx2]
        
        if is_behind:
            # Apply segment mask to hide text behind subject
            
            # Get mask for the text region
            mask_region = self.segment_mask[y1:y2, x1:x2]
            
            # Where mask is 255 (foreground), make text transparent
            text_alpha = text_layer[y1:y2, x1:x2, 3].astype(float)
            text_alpha = text_alpha * (1 - mask_region / 255.0)
            text_layer[y1:y2, x1:x2, 3] = text_alpha.astype(np.uint8)
        
        # Composite text layer onto frame
        frame_pil = Image.fromarray(frame)
        text_pil = Image.fromarray(text_layer)
        result = Image.alpha_composite(frame_pil, text_pil)
        
        return np.array(result)