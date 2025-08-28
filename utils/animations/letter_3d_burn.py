#!/usr/bin/env python3
"""
3D letter burn animation where each letter burns individually.

Following the exact same structure as letter_3d_dissolve.py but with burning effects.
Letters are rendered in 3D with depth layers, then burn with fire, smoke, and char effects.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, List
import random
import os


class Letter3DBurn:
    """3D letter-by-letter burn animation with photorealistic effects."""
    
    def __init__(
        self,
        duration: float = 3.0,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        text: str = "BURN",
        font_size: int = 120,
        text_color: Tuple[int, int, int] = (255, 220, 0),
        depth_color: Tuple[int, int, int] = (200, 170, 0),
        depth_layers: int = 8,
        depth_offset: int = 3,
        initial_scale: float = 0.9,
        initial_position: Optional[Tuple[int, int]] = None,
        burn_duration: float = 0.8,
        burn_stagger: float = 0.15,
        burn_color: Tuple[int, int, int] = (255, 100, 0),
        char_color: Tuple[int, int, int] = (40, 30, 20),
        segment_mask: Optional[np.ndarray] = None,
        is_behind: bool = False,
        supersample_factor: int = 2,
        font_path: Optional[str] = None,
        debug: bool = False,
    ):
        self.duration = duration
        self.fps = fps
        self.total_frames = int(round(duration * fps))
        self.resolution = resolution
        self.width, self.height = resolution
        self.text = text
        self.font_size = font_size
        self.text_color = text_color
        self.depth_color = depth_color
        self.depth_layers = depth_layers
        self.depth_offset = depth_offset
        self.initial_scale = initial_scale
        self.initial_position = initial_position or (resolution[0] // 2, resolution[1] // 2)
        self.burn_duration = burn_duration
        self.burn_stagger = burn_stagger
        self.burn_color = burn_color
        self.char_color = char_color
        self.segment_mask = segment_mask
        self.is_behind = is_behind
        self.supersample_factor = supersample_factor
        self.font_path = font_path
        self.debug = debug
        
        # Create 3D letter sprites
        self.letter_sprites = self._create_3d_letter_sprites()
        
        # Calculate burn timings
        self.burn_schedules = self._calculate_burn_schedules()
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get font with fallback options."""
        if self.font_path and os.path.exists(self.font_path):
            return ImageFont.truetype(self.font_path, size)
        
        # Try common font paths
        font_options = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Avenir.ttc",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        
        for font_path in font_options:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        
        # Fallback to default
        return ImageFont.load_default()
    
    def _create_3d_letter_sprites(self) -> List[dict]:
        """Create 3D rendered sprites for each letter."""
        sprites = []
        
        # Calculate supersampled size
        ss_size = self.font_size * self.supersample_factor
        font = self._get_font(ss_size)
        
        # Calculate letter positions
        test_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        test_draw = ImageDraw.Draw(test_img)
        
        letter_widths = []
        for letter in self.text:
            bbox = test_draw.textbbox((0, 0), letter, font=font)
            letter_widths.append(bbox[2] - bbox[0])
        
        # Create sprites for each letter
        spacing = int(ss_size * 0.1)
        current_x = 0
        
        for i, letter in enumerate(self.text):
            # Create 3D letter with depth layers
            letter_img = self._render_3d_letter(letter, font, ss_size)
            
            # Downscale to target size
            target_size = (
                letter_img.width // self.supersample_factor,
                letter_img.height // self.supersample_factor
            )
            letter_img = letter_img.resize(target_size, Image.LANCZOS)
            
            # Calculate position
            letter_x = current_x
            current_x += letter_widths[i] // self.supersample_factor + spacing // self.supersample_factor
            
            sprites.append({
                'letter': letter,
                'image': letter_img,
                'position': letter_x,
                'width': letter_img.width,
                'height': letter_img.height,
            })
        
        return sprites
    
    def _render_3d_letter(self, letter: str, font: ImageFont.FreeTypeFont, size: int) -> Image.Image:
        """Render a single letter with 3D depth effect."""
        # Calculate letter size
        test_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        test_draw = ImageDraw.Draw(test_img)
        bbox = test_draw.textbbox((0, 0), letter, font=font)
        width = bbox[2] - bbox[0] + self.depth_offset * self.depth_layers * 2
        height = bbox[3] - bbox[1] + self.depth_offset * self.depth_layers * 2
        
        # Create image
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw depth layers
        for i in range(self.depth_layers - 1, -1, -1):
            offset = self.depth_offset * i
            layer_color = self.depth_color if i > 0 else self.text_color
            alpha = 255
            color_with_alpha = (*layer_color, alpha)
            
            x = offset + self.depth_offset * self.depth_layers
            y = offset + self.depth_offset * self.depth_layers
            
            draw.text((x, y), letter, font=font, fill=color_with_alpha)
        
        return img
    
    def _calculate_burn_schedules(self) -> List[dict]:
        """Calculate burn timing for each letter."""
        schedules = []
        
        for i, sprite in enumerate(self.letter_sprites):
            start_time = i * self.burn_stagger
            end_time = start_time + self.burn_duration
            
            start_frame = int(start_time * self.fps)
            end_frame = int(end_time * self.fps)
            
            schedules.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'burn_duration_frames': end_frame - start_frame
            })
        
        return schedules
    
    def _apply_burn_effect(self, letter_img: np.ndarray, burn_progress: float, 
                          letter_index: int) -> np.ndarray:
        """Apply burning effect to a letter."""
        if burn_progress <= 0:
            return letter_img
        
        h, w = letter_img.shape[:2]
        result = letter_img.copy()
        
        # Create mask from alpha channel
        if letter_img.shape[2] == 4:
            alpha = letter_img[:, :, 3]
            mask = alpha > 0
        else:
            gray = cv2.cvtColor(letter_img, cv2.COLOR_BGR2GRAY)
            mask = gray > 0
        
        if burn_progress < 0.3:
            # Ignition phase - edges start glowing
            edge_intensity = burn_progress / 0.3
            
            # Find edges
            edges = cv2.Canny((mask * 255).astype(np.uint8), 100, 200)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            glow_mask = cv2.dilate(edges, kernel, iterations=2)
            
            # Add orange glow
            glow_color = np.array(self.burn_color, dtype=np.float32)
            for c in range(3):
                result[:, :, c] = np.where(
                    glow_mask > 0,
                    np.clip(result[:, :, c].astype(np.float32) + glow_color[c] * edge_intensity, 0, 255),
                    result[:, :, c]
                )
        
        elif burn_progress < 0.7:
            # Active burning - fire spreads inward
            burn_intensity = (burn_progress - 0.3) / 0.4
            
            # Create burn progression from edges
            dist_transform = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
            max_dist = dist_transform.max()
            if max_dist > 0:
                burn_threshold = max_dist * (1 - burn_intensity)
                burning_mask = (dist_transform < burn_threshold) & mask
                
                # Apply fire color gradient
                for c in range(3):
                    fire_gradient = self.burn_color[c] * (1 - burn_intensity) + self.char_color[c] * burn_intensity
                    result[:, :, c] = np.where(
                        burning_mask,
                        fire_gradient,
                        result[:, :, c]
                    )
                
                # Add flickering
                noise = np.random.rand(h, w) * 30 * burn_intensity
                result[:, :, :3] = np.clip(result[:, :, :3] + noise[:, :, np.newaxis], 0, 255)
        
        else:
            # Charring and ashing - letter turns to ash
            char_progress = (burn_progress - 0.7) / 0.3
            
            # Fade to char/ash
            for c in range(3):
                result[:, :, c] = np.where(
                    mask,
                    result[:, :, c] * (1 - char_progress) + self.char_color[c] * char_progress,
                    result[:, :, c]
                )
            
            # Reduce alpha for ashing effect
            if result.shape[2] == 4:
                result[:, :, 3] = (result[:, :, 3] * (1 - char_progress * 0.8)).astype(np.uint8)
        
        # Add smoke particles
        if burn_progress > 0.2:
            num_particles = int(10 * burn_progress)
            for _ in range(num_particles):
                px = np.random.randint(0, w)
                py = np.random.randint(0, h // 2)
                if mask[py, px]:
                    smoke_y = max(0, py - int(20 * burn_progress))
                    cv2.circle(result, (px, smoke_y), 3, (100, 100, 100, 100), -1)
        
        return result.astype(np.uint8)
    
    def generate_frame(self, frame_num: int, background: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a single frame of the animation."""
        if background is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            frame = background.copy()
        
        # Apply occlusion mask if behind segment
        if self.is_behind and self.segment_mask is not None:
            # Recalculate mask every frame for dynamic scenes
            occlusion_mask = self.segment_mask
        else:
            occlusion_mask = None
        
        # Calculate total width for centering
        total_width = sum(s['width'] for s in self.letter_sprites) + \
                     int(self.font_size * 0.1) * (len(self.letter_sprites) - 1)
        start_x = (self.width - total_width) // 2
        y_position = (self.height - self.letter_sprites[0]['height']) // 2
        
        # Render each letter
        for i, sprite in enumerate(self.letter_sprites):
            schedule = self.burn_schedules[i]
            
            # Calculate burn progress
            if frame_num < schedule['start_frame']:
                burn_progress = 0.0
            elif frame_num >= schedule['end_frame']:
                burn_progress = 1.0
            else:
                burn_progress = (frame_num - schedule['start_frame']) / schedule['burn_duration_frames']
            
            # Skip fully burned letters
            if burn_progress >= 1.0:
                continue
            
            # Convert PIL image to numpy array
            letter_array = np.array(sprite['image'])
            
            # Apply burn effect
            burned_letter = self._apply_burn_effect(letter_array, burn_progress, i)
            
            # Calculate position
            x = start_x + sprite['position']
            
            # Composite onto frame
            self._composite_letter(frame, burned_letter, x, y_position, occlusion_mask)
        
        return frame
    
    def _composite_letter(self, frame: np.ndarray, letter: np.ndarray, 
                         x: int, y: int, occlusion_mask: Optional[np.ndarray]) -> None:
        """Composite a letter onto the frame with optional occlusion."""
        h, w = letter.shape[:2]
        
        # Calculate bounds
        y1 = max(0, y)
        y2 = min(frame.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(frame.shape[1], x + w)
        
        # Calculate letter region bounds
        ly1 = max(0, -y)
        ly2 = ly1 + (y2 - y1)
        lx1 = max(0, -x)
        lx2 = lx1 + (x2 - x1)
        
        if y2 <= y1 or x2 <= x1:
            return
        
        # Extract regions
        frame_region = frame[y1:y2, x1:x2]
        letter_region = letter[ly1:ly2, lx1:lx2]
        
        # Handle alpha channel
        if letter_region.shape[2] == 4:
            alpha = letter_region[:, :, 3:4] / 255.0
            
            # Apply occlusion if needed
            if occlusion_mask is not None:
                mask_region = occlusion_mask[y1:y2, x1:x2]
                if mask_region.shape == alpha.squeeze().shape:
                    alpha = alpha * (1 - mask_region[:, :, np.newaxis])
            
            # Blend
            frame_region[:] = frame_region * (1 - alpha) + letter_region[:, :, :3] * alpha
        else:
            # No alpha channel - use simple copy
            if occlusion_mask is not None:
                mask_region = occlusion_mask[y1:y2, x1:x2]
                visible_mask = mask_region == 0
                frame_region[visible_mask] = letter_region[visible_mask]
            else:
                frame_region[:] = letter_region