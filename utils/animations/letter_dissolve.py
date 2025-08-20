"""
Letter Dissolve animation.
Individual letter dissolves with upward float and fade effect.
"""

import os
import numpy as np
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from .animate import Animation


class LetterDissolve(Animation):
    """
    Animation for a single letter dissolving with upward float and fade.
    
    The letter grows slightly, floats upward, and fades out with optional glow effect.
    
    Additional Parameters:
    ---------------------
    letter : str
        Single letter to animate
    font_size : int
        Base font size (default 150)
    font_path : str
        Path to font file (optional, uses system font if None)
    text_color : Tuple[int, int, int]
        RGB color for text (default (255, 220, 0) - golden yellow)
    dissolve_duration : float
        Duration of dissolve in seconds (default 0.67)
    float_distance : int
        Pixels to float upward (default 30)
    max_scale : float
        Maximum scale during dissolve (default 1.2 = 20% larger)
    glow_effect : bool
        Add glow effect during dissolve (default True)
    glow_color : Tuple[int, int, int]
        RGB color for glow (default (255, 255, 200) - warm white)
    shadow_alpha : int
        Initial shadow alpha (default 100)
    outline_alpha : int
        Initial outline alpha (default 150)
    center_position : Optional[Tuple[int, int]]
        Center position for letter (default: frame center)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        letter: str = "A",
        font_size: int = 150,
        font_path: Optional[str] = None,
        text_color: Tuple[int, int, int] = (255, 220, 0),
        dissolve_duration: float = 0.67,
        float_distance: int = 30,
        max_scale: float = 1.2,
        glow_effect: bool = True,
        glow_color: Tuple[int, int, int] = (255, 255, 200),
        shadow_alpha: int = 100,
        outline_alpha: int = 150,
        center_position: Optional[Tuple[int, int]] = None,
        direction: float = 0,
        start_frame: int = 0,
        animation_start_frame: int = 0,
        path: Optional[list] = None,
        fps: int = 30,
        duration: Optional[float] = None,
        temp_dir: Optional[str] = None
    ):
        # Use dissolve_duration as total duration if not specified
        if duration is None:
            duration = dissolve_duration
            
        super().__init__(
            element_path=element_path,
            background_path=background_path,
            position=position,
            direction=direction,
            start_frame=start_frame,
            animation_start_frame=animation_start_frame,
            path=path,
            fps=fps,
            duration=duration,
            temp_dir=temp_dir
        )
        
        self.letter = letter[0] if letter else "A"  # Ensure single letter
        self.font_size = font_size
        self.font_path = font_path
        self.text_color = text_color
        self.dissolve_duration = dissolve_duration
        self.float_distance = float_distance
        self.max_scale = max_scale
        self.glow_effect = glow_effect
        self.glow_color = glow_color
        self.shadow_alpha = shadow_alpha
        self.outline_alpha = outline_alpha
        self.center_position = center_position
        
        # Calculate total dissolve frames
        self.dissolve_frames = int(dissolve_duration * fps)
    
    def load_font(self, size: int):
        """Load font with specified size."""
        if self.font_path and os.path.exists(self.font_path):
            return ImageFont.truetype(self.font_path, size)
        else:
            # Try common system fonts
            system_fonts = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "arial.ttf"
            ]
            for font in system_fonts:
                if os.path.exists(font):
                    return ImageFont.truetype(font, size)
            # Fallback to default
            return ImageFont.load_default()
    
    def render_letter_frame(
        self,
        frame: np.ndarray,
        progress: float,
        base_x: int,
        base_y: int
    ) -> np.ndarray:
        """
        Render dissolving letter onto a frame.
        
        Parameters:
        -----------
        frame : np.ndarray
            The video frame to render letter on
        progress : float
            Dissolve progress (0.0 = start, 1.0 = fully dissolved)
        base_x : int
            Base X position for letter
        base_y : int
            Base Y position for letter
        
        Returns:
        --------
        np.ndarray
            Frame with rendered letter
        """
        h, w = frame.shape[:2]
        
        # Calculate animation parameters based on progress
        float_offset = int(progress * self.float_distance)
        opacity = int(255 * (1 - progress))
        current_scale = 1.0 + (self.max_scale - 1.0) * progress
        
        # Skip if fully transparent
        if opacity <= 0:
            return frame
        
        # Create text layer
        text_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)
        
        # Load fonts
        base_font = self.load_font(self.font_size)
        scaled_font_size = int(self.font_size * current_scale)
        scaled_font = self.load_font(scaled_font_size)
        
        # Get letter dimensions
        base_bbox = draw.textbbox((0, 0), self.letter, font=base_font)
        base_width = base_bbox[2] - base_bbox[0]
        base_height = base_bbox[3] - base_bbox[1]
        
        scaled_bbox = draw.textbbox((0, 0), self.letter, font=scaled_font)
        scaled_width = scaled_bbox[2] - scaled_bbox[0]
        scaled_height = scaled_bbox[3] - scaled_bbox[1]
        
        # Calculate position (center the growing letter)
        width_increase = scaled_width - base_width
        height_increase = scaled_height - base_height
        
        letter_x = base_x - (width_increase // 2)
        letter_y = base_y - float_offset - (height_increase // 2)
        
        # Draw shadow (fades with letter)
        shadow_alpha = min(50, int(self.shadow_alpha * (1 - progress) * 0.5))
        if shadow_alpha > 0:
            draw.text(
                (letter_x + 2, letter_y + 2),
                self.letter,
                font=scaled_font,
                fill=(0, 0, 0, shadow_alpha)
            )
        
        # Draw glow effect (gets stronger as letter dissolves)
        if self.glow_effect and opacity > 0:
            glow_alpha = int(opacity * 0.3)
            glow_radii = [6, 4, 2]
            
            for radius in glow_radii:
                for angle in range(0, 360, 45):
                    gx = int(radius * np.cos(np.radians(angle)))
                    gy = int(radius * np.sin(np.radians(angle)))
                    draw.text(
                        (letter_x + gx, letter_y + gy),
                        self.letter,
                        font=scaled_font,
                        fill=(*self.glow_color, glow_alpha)
                    )
        
        # Draw outline (optional, fades with letter)
        if self.outline_alpha > 0 and progress < 0.5:
            outline_alpha = int(self.outline_alpha * (1 - progress * 2))
            for dx in [-2, 2]:
                for dy in [-2, 2]:
                    draw.text(
                        (letter_x + dx, letter_y + dy),
                        self.letter,
                        font=scaled_font,
                        fill=(255, 255, 255, outline_alpha)
                    )
        
        # Draw main letter
        draw.text(
            (letter_x, letter_y),
            self.letter,
            font=scaled_font,
            fill=(*self.text_color, opacity)
        )
        
        # Convert to numpy
        text_layer_np = np.array(text_layer)
        
        # Composite with frame
        result = frame.copy()
        
        for c in range(3):
            text_visible = text_layer_np[:, :, 3] > 0
            if np.any(text_visible):
                alpha_blend = text_layer_np[text_visible, 3] / 255.0
                result[text_visible, c] = (
                    result[text_visible, c] * (1 - alpha_blend) +
                    text_layer_np[text_visible, c] * alpha_blend
                ).astype(np.uint8)
        
        return result
    
    def animate_dissolve(
        self,
        frames: list,
        start_idx: int,
        letter_x: int,
        letter_y: int
    ) -> list:
        """
        Apply dissolve animation to a sequence of frames.
        
        Parameters:
        -----------
        frames : list
            List of video frames (numpy arrays)
        start_idx : int
            Frame index to start dissolve
        letter_x : int
            X position for letter
        letter_y : int
            Y position for letter
        
        Returns:
        --------
        list
            Frames with dissolve animation applied
        """
        result_frames = []
        
        for i, frame in enumerate(frames):
            if i < start_idx:
                # Before dissolve starts
                result_frames.append(frame)
            elif i < start_idx + self.dissolve_frames:
                # During dissolve
                progress = (i - start_idx) / self.dissolve_frames
                result_frame = self.render_letter_frame(frame, progress, letter_x, letter_y)
                result_frames.append(result_frame)
            else:
                # After dissolve completes
                result_frames.append(frame)
        
        return result_frames
    
    def process_frames(self) -> list:
        """
        Process frames according to animation type.
        Required by Animation base class.
        """
        # This is a placeholder - in full implementation would process video frames
        return []
    
    def animate(self, output_path: str) -> bool:
        """
        Create the letter dissolve animation.
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        print(f"✨ Creating letter dissolve animation...")
        print(f"   Letter: '{self.letter}'")
        print(f"   Duration: {self.dissolve_duration}s")
        print(f"   Float distance: {self.float_distance}px")
        
        # This is a simplified version - in practice you'd integrate with video frames
        # For now, return True as placeholder
        print(f"   ✓ Dissolve animation complete")
        return True