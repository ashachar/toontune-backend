"""
Base class for 3D text animations with individual letter control
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from enum import Enum


class Animation3DType(Enum):
    """Types of 3D animations"""
    OPACITY_3D = "opacity_3d"
    MOTION_3D = "motion_3d" 
    SCALE_3D = "scale_3d"
    ROTATION_3D = "rotation_3d"
    PROGRESSIVE_3D = "progressive_3d"
    COMPOUND_3D = "compound_3d"


@dataclass
class Letter3D:
    """Represents a single letter in 3D space"""
    character: str
    index: int
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [rx, ry, rz] in radians
    scale: np.ndarray     # [sx, sy, sz]
    opacity: float
    color: Tuple[int, int, int]
    depth_color: Optional[Tuple[int, int, int]] = None
    glow_intensity: float = 0.0
    blur_amount: float = 0.0
    
    # Cached rendering
    _rendered_sprite: Optional[np.ndarray] = field(default=None, init=False)
    _sprite_bounds: Optional[Tuple[int, int, int, int]] = field(default=None, init=False)


@dataclass
class Animation3DConfig:
    """Configuration for 3D text animations"""
    text: str
    duration_ms: int
    position: Tuple[int, int, int]  # 3D position
    font_size: int = 60
    font_color: Tuple[int, int, int] = (255, 255, 255)
    depth_color: Tuple[int, int, int] = (180, 180, 180)
    font_thickness: int = 4  # Increased for more substantial text
    easing: str = "ease_in_out"
    
    # 3D specific parameters
    depth_layers: int = 8
    depth_offset: int = 3
    perspective_distance: float = 1000.0
    
    # Letter spacing
    letter_spacing: float = 1.2  # Multiplier for letter width
    
    # Stagger options
    stagger_ms: int = 50  # Delay between letters
    stagger_type: str = "sequential"  # sequential, center_out, random
    
    # Effects
    enable_shadows: bool = True
    shadow_distance: int = 5
    shadow_opacity: float = 0.5
    
    enable_glow: bool = False
    glow_radius: int = 5
    glow_color: Optional[Tuple[int, int, int]] = None
    
    # Font
    font_path: Optional[str] = None


class Base3DTextAnimation(ABC):
    """Abstract base class for 3D text animations"""
    
    def __init__(self, config: Animation3DConfig):
        self.config = config
        self.letters: List[Letter3D] = []
        self.original_positions: List[np.ndarray] = []
        
        # Initialize font
        if config.font_path:
            self.font = ImageFont.truetype(config.font_path, config.font_size)
        else:
            # Try to use a default font
            try:
                self.font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", config.font_size)
            except:
                self.font = ImageFont.load_default()
        
        # Initialize letters
        self._initialize_letters()
        
        # Cache for rendered frames
        self._frame_cache: Dict[int, np.ndarray] = {}
    
    def _initialize_letters(self):
        """Initialize individual letter objects"""
        self.letters = []
        self.original_positions = []
        
        # Create a temporary image to measure text
        temp_img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        
        # Calculate total text width
        total_width = 0
        letter_widths = []
        
        for char in self.config.text:
            bbox = draw.textbbox((0, 0), char, font=self.font)
            char_width = bbox[2] - bbox[0]
            letter_widths.append(char_width)
            total_width += char_width * self.config.letter_spacing
        
        # Starting position (centered)
        start_x = self.config.position[0] - total_width / 2
        current_x = start_x
        
        # Create letter objects
        for i, char in enumerate(self.config.text):
            if char == ' ':
                # Handle spaces
                current_x += letter_widths[i] * self.config.letter_spacing
                continue
            
            position = np.array([
                current_x + letter_widths[i] / 2,
                self.config.position[1],
                self.config.position[2]
            ], dtype=np.float32)
            
            letter = Letter3D(
                character=char,
                index=i,
                position=position.copy(),
                rotation=np.zeros(3, dtype=np.float32),
                scale=np.ones(3, dtype=np.float32),
                opacity=1.0,
                color=self.config.font_color,
                depth_color=self.config.depth_color
            )
            
            self.letters.append(letter)
            self.original_positions.append(position.copy())
            
            current_x += letter_widths[i] * self.config.letter_spacing
    
    def render_letter_3d(self, letter: Letter3D) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Render a single letter with 3D effects
        Returns: (sprite_image, (x, y) position on screen)
        """
        # Create sprite if not cached
        if letter._rendered_sprite is None:
            self._create_letter_sprite(letter)
        
        # Apply 3D transformations
        transformed_sprite = self._apply_3d_transform(letter)
        
        # Calculate screen position using perspective projection
        screen_pos = self._project_to_screen(letter.position)
        
        return transformed_sprite, screen_pos
    
    def _create_letter_sprite(self, letter: Letter3D):
        """Create the base sprite for a letter"""
        # Calculate sprite size with padding for effects
        padding = 20  # Extra space for shadows, glow, etc.
        
        # Create image with letter
        img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a solid shadow/depth layer first
        if self.config.depth_layers > 1:
            # Draw one solid dark layer for depth
            shadow_offset = self.config.depth_offset * 3
            shadow_color = (60, 60, 60, 255)  # Full opacity shadow - opacity applied during compositing
            draw.text(
                (100 + shadow_offset, 100 + 2),
                letter.character,
                fill=shadow_color,
                font=self.font,
                anchor='mm',
                stroke_width=1,
                stroke_fill=shadow_color
            )
        
        # Draw main letter at FULL OPACITY - we'll apply opacity during compositing
        # This fixes the caching issue where sprites were stuck at initial low opacity
        draw.text(
            (100, 100),
            letter.character,
            fill=(*letter.color, 255),  # Always full opacity in sprite
            font=self.font,
            anchor='mm',
            stroke_width=2,
            stroke_fill=(*letter.color, 255)  # Always full opacity in sprite
        )
        
        # Convert to numpy array
        letter._rendered_sprite = np.array(img)
        
        # Find actual bounds
        alpha = letter._rendered_sprite[:, :, 3]
        rows = np.any(alpha, axis=1)
        cols = np.any(alpha, axis=0)
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        
        if len(row_indices) > 0 and len(col_indices) > 0:
            rmin, rmax = row_indices[[0, -1]]
            cmin, cmax = col_indices[[0, -1]]
            letter._sprite_bounds = (rmin, rmax, cmin, cmax)
        else:
            # Default bounds if sprite is empty
            letter._sprite_bounds = (0, 199, 0, 199)
    
    def _apply_3d_transform(self, letter: Letter3D) -> np.ndarray:
        """Apply 3D transformations to letter sprite"""
        if letter._rendered_sprite is None:
            self._create_letter_sprite(letter)
        
        sprite = letter._rendered_sprite.copy()
        
        # Apply scale with proper interpolation to avoid artifacts
        if not np.array_equal(letter.scale, [1.0, 1.0, 1.0]):
            h, w = sprite.shape[:2]
            new_w = max(1, int(w * letter.scale[0]))  # Ensure minimum size of 1
            new_h = max(1, int(h * letter.scale[1]))  # Ensure minimum size of 1
            
            # Use INTER_AREA for downscaling (reduces artifacts), INTER_LINEAR for upscaling
            if letter.scale[0] < 1.0 or letter.scale[1] < 1.0:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_LINEAR
            
            # Resize with alpha premultiplication to avoid color bleeding
            if sprite.shape[2] == 4:
                # Premultiply alpha to avoid artifacts
                alpha = sprite[:, :, 3:4] / 255.0
                sprite_premult = sprite.copy()
                sprite_premult[:, :, :3] = sprite[:, :, :3] * alpha
                
                # Resize
                resized = cv2.resize(sprite_premult, (new_w, new_h), interpolation=interpolation)
                
                # Unpremultiply alpha
                alpha_resized = resized[:, :, 3:4] / 255.0
                alpha_resized[alpha_resized == 0] = 1  # Avoid division by zero
                resized[:, :, :3] = resized[:, :, :3] / alpha_resized
                resized[:, :, :3] = np.clip(resized[:, :, :3], 0, 255)
                
                sprite = resized
            else:
                sprite = cv2.resize(sprite, (new_w, new_h), interpolation=interpolation)
        
        # Apply rotation (simplified 2D rotation for now, can be extended to full 3D)
        if letter.rotation[2] != 0:  # Z-axis rotation
            h, w = sprite.shape[:2]
            center = (w // 2, h // 2)
            angle_deg = np.degrees(letter.rotation[2])
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            sprite = cv2.warpAffine(sprite, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        
        # Apply blur if needed
        if letter.blur_amount > 0:
            kernel_size = int(letter.blur_amount * 2) * 2 + 1
            sprite[:, :, :3] = cv2.GaussianBlur(sprite[:, :, :3], (kernel_size, kernel_size), letter.blur_amount)
        
        # Apply glow if needed
        if letter.glow_intensity > 0 and self.config.enable_glow:
            sprite = self._add_glow(sprite, letter.glow_intensity)
        
        # Apply opacity
        sprite[:, :, 3] = (sprite[:, :, 3] * letter.opacity).astype(np.uint8)
        
        return sprite
    
    def _project_to_screen(self, position_3d: np.ndarray) -> Tuple[int, int]:
        """Project 3D position to 2D screen coordinates using perspective projection"""
        x, y, z = position_3d
        
        # Simple perspective projection
        perspective_scale = self.config.perspective_distance / (self.config.perspective_distance + z)
        
        screen_x = int(x * perspective_scale)
        screen_y = int(y * perspective_scale)
        
        return (screen_x, screen_y)
    
    def _add_glow(self, sprite: np.ndarray, intensity: float) -> np.ndarray:
        """Add glow effect to sprite"""
        # Create glow layer
        glow = sprite.copy()
        
        # Expand and blur for glow
        kernel_size = self.config.glow_radius * 2 + 1
        glow[:, :, :3] = cv2.GaussianBlur(glow[:, :, :3], (kernel_size, kernel_size), self.config.glow_radius)
        
        # Blend with original
        alpha = intensity
        result = cv2.addWeighted(sprite, 1.0, glow, alpha, 0)
        
        return result
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply animation to a single frame"""
        # Check cache first
        if frame_number in self._frame_cache:
            cached = self._frame_cache[frame_number]
            return cv2.addWeighted(frame, 1.0, cached, 1.0, 0)
        
        # Calculate animation progress
        progress = self.get_progress(frame_number, fps)
        
        # Update letter states based on animation
        self.update_letters(progress, frame_number, fps)
        
        # Sort letters by Z depth (back to front)
        sorted_letters = sorted(self.letters, key=lambda l: l.position[2], reverse=True)
        
        # Render each letter
        result = frame.copy()
        
        for letter in sorted_letters:
            if letter.opacity <= 0:
                continue
            
            # Render letter
            sprite, screen_pos = self.render_letter_3d(letter)
            
            # Composite onto frame
            result = self._composite_sprite(result, sprite, screen_pos)
        
        # Cache rendered overlay
        overlay = result - frame
        self._frame_cache[frame_number] = overlay
        
        return result
    
    def _composite_sprite(self, frame: np.ndarray, sprite: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """Composite a sprite onto the frame at the given position"""
        x, y = position
        h, w = sprite.shape[:2]
        
        # Calculate sprite position (centered)
        x1 = x - w // 2
        y1 = y - h // 2
        x2 = x1 + w
        y2 = y1 + h
        
        # Clip to frame bounds
        frame_h, frame_w = frame.shape[:2]
        
        x1_clipped = max(0, x1)
        y1_clipped = max(0, y1)
        x2_clipped = min(frame_w, x2)
        y2_clipped = min(frame_h, y2)
        
        if x2_clipped <= x1_clipped or y2_clipped <= y1_clipped:
            return frame
        
        # Calculate sprite region
        sprite_x1 = x1_clipped - x1
        sprite_y1 = y1_clipped - y1
        sprite_x2 = sprite_x1 + (x2_clipped - x1_clipped)
        sprite_y2 = sprite_y1 + (y2_clipped - y1_clipped)
        
        # Extract regions
        frame_region = frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped]
        sprite_region = sprite[sprite_y1:sprite_y2, sprite_x1:sprite_x2]
        
        # Alpha blending
        if sprite_region.shape[2] == 4:
            alpha = sprite_region[:, :, 3:4] / 255.0
            blended = frame_region * (1 - alpha) + sprite_region[:, :, :3] * alpha
            frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped] = blended.astype(np.uint8)
        else:
            frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped] = sprite_region
        
        return frame
    
    def get_progress(self, frame_number: int, fps: float) -> float:
        """Get animation progress (0.0 to 1.0) for current frame"""
        current_time_ms = (frame_number / fps) * 1000
        progress = min(1.0, max(0.0, current_time_ms / self.config.duration_ms))
        return self.ease_value(progress)
    
    def ease_value(self, t: float) -> float:
        """Apply easing function to normalized time value"""
        if self.config.easing == "linear":
            return t
        elif self.config.easing == "ease_in":
            return t * t
        elif self.config.easing == "ease_out":
            return 1 - (1 - t) ** 2
        elif self.config.easing == "ease_in_out":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - ((2 * (1 - t)) ** 2) / 2
        elif self.config.easing == "elastic":
            if t == 0 or t == 1:
                return t
            p = 0.3
            s = p / 4
            return -(2 ** (10 * (t - 1))) * np.sin((t - 1 - s) * (2 * np.pi) / p)
        elif self.config.easing == "bounce":
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
    
    def get_letter_progress(self, letter_index: int, global_progress: float) -> float:
        """Get progress for a specific letter considering stagger"""
        if self.config.stagger_ms == 0:
            return global_progress
        
        # Calculate stagger delay for this letter
        stagger_fraction = self.config.stagger_ms / self.config.duration_ms
        
        if self.config.stagger_type == "sequential":
            delay = letter_index * stagger_fraction
        elif self.config.stagger_type == "center_out":
            center = len(self.letters) / 2
            distance = abs(letter_index - center)
            delay = distance * stagger_fraction
        elif self.config.stagger_type == "random":
            # Use deterministic "random" based on index
            delay = ((letter_index * 137) % 100) / 100.0 * stagger_fraction
        else:
            delay = 0
        
        # Adjust progress for this letter
        letter_progress = (global_progress - delay) / (1 - delay * len(self.letters) / (len(self.letters) + 1))
        return max(0.0, min(1.0, letter_progress))
    
    @abstractmethod
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letter states based on animation progress"""
        pass