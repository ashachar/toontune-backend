#!/usr/bin/env python3
"""
Refactored Text3DMotion that renders letters individually but moves them as a unified group.
This enables perfect handoff to Letter3DDissolve without position jumps.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass


@dataclass
class LetterSprite:
    """Individual letter sprite with its 3D rendering and position."""
    char: str
    sprite_3d: Optional[Image.Image]
    position: Tuple[int, int]   # paste top-left
    width: int
    height: int
    anchor: Tuple[int, int]     # FRONT-FACE top-left inside sprite
    base_position: Optional[Tuple[int, int]] = None  # Store original position for handoff


@dataclass
class MotionState:
    """State captured at the end of motion animation for handoff to next animation."""
    scale: float
    position: Tuple[int, int]           # Top-left position used during final composite
    text_size: Tuple[int, int]          # Rendered sprite size
    center_position: Tuple[int, int]    # Intended front-face center
    is_behind: bool                      # Whether text is behind subject at end of motion
    alpha: float                         # Current alpha/opacity value at end of motion
    letter_sprites: Optional[List[LetterSprite]] = None  # Individual letter sprites for handoff


class Text3DMotion:
    """
    3D text animation with individual letter rendering but unified group movement.
    """

    def __init__(
        self,
        duration: float = 1.0,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        text: str = "HELLO",
        segment_mask: Optional[np.ndarray] = None,
        font_size: int = 120,
        text_color: Tuple[int, int, int] = (255, 220, 0),
        depth_color: Tuple[int, int, int] = (200, 170, 0),
        depth_layers: int = 8,
        depth_offset: int = 3,
        start_scale: float = 2.0,
        end_scale: float = 1.0,
        final_scale: float = 0.9,
        start_position: Optional[Tuple[int, int]] = None,
        end_position: Optional[Tuple[int, int]] = None,
        shrink_duration: float = 0.8,
        settle_duration: float = 0.2,
        final_alpha: float = 0.3,
        shadow_offset: int = 5,
        outline_width: int = 2,
        perspective_angle: float = 0,
        supersample_factor: int = 8,  # Much higher for smooth edges
        glow_effect: bool = True,
        font_path: Optional[str] = None,
        debug: bool = False,
    ):
        self.duration = duration
        self.fps = fps
        self.total_frames = int(duration * fps)
        self.resolution = resolution
        self.text = text
        self.segment_mask = segment_mask
        self.font_size = font_size
        self.text_color = text_color
        self.depth_color = depth_color
        self.depth_layers = depth_layers
        self.depth_offset = depth_offset
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.final_scale = final_scale
        self.start_position = start_position or (resolution[0] // 2, resolution[1] // 2)
        self.end_position = end_position or (resolution[0] // 2, resolution[1] // 2)
        self.shrink_duration = shrink_duration
        self.settle_duration = settle_duration
        self.shrink_frames = int(shrink_duration * fps)
        self.settle_frames = int(settle_duration * fps)
        self.final_alpha = max(0.0, min(1.0, final_alpha))
        self.shadow_offset = shadow_offset
        self.outline_width = outline_width
        self.perspective_angle = perspective_angle
        self.supersample_factor = supersample_factor
        self.glow_effect = glow_effect
        self.font_path = font_path
        self.debug = debug

        # Letter sprites - rendered once and reused
        self.letter_sprites: List[LetterSprite] = []
        self._prepare_letter_sprites()
        
        # Final state for handoff
        self._final_state: Optional[MotionState] = None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get font at specified size - prioritize vector fonts."""
        candidates = []
        if self.font_path:
            candidates.append(self.font_path)
        
        # Common font paths
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\arial.ttf",
        ])
        
        for p in candidates:
            try:
                if p and os.path.isfile(p):
                    return ImageFont.truetype(p, size)
            except Exception:
                continue
        
        return ImageFont.load_default()

    def _render_3d_letter(
        self, letter: str, scale: float, alpha: float, depth_scale: float
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """Render a single 3D letter with depth layers and smooth antialiasing."""
        # Use EXTREMELY high supersampling for perfectly smooth edges
        actual_supersample = 20  # Force 20x supersampling for maximum smoothness
        font_px = int(self.font_size * scale * actual_supersample)
        font = self._get_font(font_px)

        # Get letter bounds
        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), letter, font=font)
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

        # Add margin for depth layers
        margin = int(self.depth_offset * self.depth_layers * actual_supersample)
        width = bbox_w + 2 * margin
        height = bbox_h + 2 * margin

        # Create canvas with antialiasing mode - use L mode for each layer first
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        
        # Draw depth layers (back to front) with individual antialiasing
        from PIL import ImageFilter
        for i in range(self.depth_layers - 1, -1, -1):
            # Create a grayscale layer for perfect antialiasing
            layer = Image.new("L", (width, height), 0)
            layer_draw = ImageDraw.Draw(layer)
            
            depth_alpha = int(alpha * 255 * (0.3 + 0.7 * (1 - i / self.depth_layers)))
            offset = int(i * self.depth_offset * depth_scale * actual_supersample)

            if i == 0:
                color_rgb = self.text_color
            else:
                factor = 0.7 - (i / self.depth_layers) * 0.4
                color_rgb = tuple(int(c * factor) for c in self.depth_color)

            x = margin - bbox[0] + offset
            # BASELINE ALIGNMENT FIX:
            # PIL's textbbox returns (left, top, right, bottom) relative to baseline at (0,0)
            # Most letters have negative top (above baseline) and small positive bottom (descenders)
            # To align bottoms: we want y_draw + bbox[3] = same for all letters
            # So: y_draw = desired_bottom - bbox[3]
            desired_bottom = height - margin  # Put text near bottom of canvas
            y = desired_bottom - bbox[3] + offset
            
            # Draw with maximum antialiasing in grayscale
            layer_draw.text((x, y), letter, font=font, fill=255)
            
            # Apply slight blur to the layer for smoother edges
            layer = layer.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Convert to RGBA and apply color
            layer_rgba = Image.new("RGBA", (width, height), (*color_rgb, 0))
            layer_rgba.putalpha(layer.point(lambda x: int(x * depth_alpha / 255)))
            
            # Composite onto main canvas
            canvas = Image.alpha_composite(canvas, layer_rgba)

        # Apply stronger Gaussian blur for ultra-smooth edges
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=actual_supersample/8))
        
        # Additional smoothing pass with SMOOTH_MORE filter
        canvas = canvas.filter(ImageFilter.SMOOTH_MORE)
        
        # Downsample with high-quality LANCZOS resampling
        if actual_supersample > 1:
            new_size = (width // actual_supersample, height // actual_supersample)
            canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)
            
            # Final smoothing pass at target resolution
            canvas = canvas.filter(ImageFilter.SMOOTH)

        # Calculate anchor (front-face top-left)
        anchor_x = margin // actual_supersample
        anchor_y = margin // actual_supersample

        return canvas, (anchor_x, anchor_y)

    def _prepare_letter_sprites(self):
        """Pre-render all letter sprites at maximum scale to avoid pixelation."""
        # Clear existing sprites
        self.letter_sprites = []
        
        # Use MAX scale for initial rendering to ensure no pixelation when scaling up
        base_scale = max(self.start_scale, self.end_scale, self.final_scale, 2.0)
        font_px = int(self.font_size * base_scale)
        font = self._get_font(font_px)
        
        # Calculate layout positions - all letters at y=0, baseline alignment happens during render
        current_x = 0
        for letter in self.text:
            if letter == ' ':
                # Space character - scale it appropriately with letters
                # Use smaller ratio for spaces to avoid excessive gaps
                space_width = max(1, int(font_px * 0.2))  # Reduced from /3 to *0.2 for tighter spacing
                sprite = LetterSprite(
                    char=' ',
                    sprite_3d=None,
                    position=(current_x, 0),  # Relative position in text block
                    width=space_width,
                    height=1,
                    anchor=(0, 0)
                )
                self.letter_sprites.append(sprite)
                current_x += space_width
            else:
                # Render the letter at maximum scale for best quality
                sprite_3d, (ax, ay) = self._render_3d_letter(letter, base_scale, 1.0, 1.0)
                
                sprite = LetterSprite(
                    char=letter,
                    sprite_3d=sprite_3d,
                    position=(current_x, 0),  # All at y=0, baseline alignment in render
                    width=sprite_3d.width if sprite_3d else 0,
                    height=sprite_3d.height if sprite_3d else 0,
                    anchor=(ax, ay)
                )
                self.letter_sprites.append(sprite)
                current_x += sprite_3d.width if sprite_3d else font_px
        
        if self.debug:
            print(f"[Text3DMotion] Prepared {len(self.letter_sprites)} letter sprites")

    def generate_frame(self, frame_number: int, background: np.ndarray) -> np.ndarray:
        """Generate a single frame with letters moving as a unified group."""
        frame = background.copy()
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

        # Calculate animation parameters
        if frame_number < self.shrink_frames:
            # Shrinking phase
            t = frame_number / max(1, self.shrink_frames)
            t_smooth = t * t * (3 - 2 * t)  # smoothstep
            scale = self.start_scale + (self.end_scale - self.start_scale) * t_smooth
            cx = self.start_position[0] + (self.end_position[0] - self.start_position[0]) * t_smooth
            cy = self.start_position[1] + (self.end_position[1] - self.start_position[1]) * t_smooth
            # Start going behind immediately when shrinking starts
            is_behind = t > 0.0  # Changed from 0.5 to 0.0 - goes behind immediately
            # Gradually reduce opacity as text goes behind (from 1.0 to final_alpha)
            if is_behind:
                # Smooth transition from full opacity to final opacity
                alpha = 1.0 + (self.final_alpha - 1.0) * t_smooth
            else:
                alpha = 1.0
        else:
            # Settling phase
            t = (frame_number - self.shrink_frames) / max(1, self.settle_frames)
            t_smooth = t * t * (3 - 2 * t)
            scale = self.end_scale + (self.final_scale - self.end_scale) * t_smooth
            cx = self.end_position[0]
            cy = self.end_position[1]
            is_behind = True
            # Alpha should remain constant at final_alpha during settling
            alpha = self.final_alpha
        
        # Calculate scale ratio from pre-rendered size
        pre_render_scale = max(self.start_scale, self.end_scale, self.final_scale, 2.0)
        scale_ratio = scale / pre_render_scale

        # Calculate total text width at current scale
        total_width = 0
        max_height = 0
        for sprite in self.letter_sprites:
            if sprite.sprite_3d:
                scaled_w = int(sprite.width * scale_ratio)
                scaled_h = int(sprite.height * scale_ratio)
                total_width += scaled_w
                max_height = max(max_height, scaled_h)
            else:
                total_width += int(sprite.width * scale_ratio)

        # Calculate group starting position (centered)
        group_start_x = int(cx - total_width // 2)
        group_start_y = int(cy - max_height // 2)

        # Create a canvas for the entire text group
        canvas = Image.fromarray(frame)
        
        # Render each letter at its position within the group
        current_x = group_start_x
        rendered_sprites = []
        
        for sprite in self.letter_sprites:
            if sprite.char == ' ':
                # Space - just advance position
                current_x += int(sprite.width * scale_ratio)
                rendered_sprites.append(None)
            else:
                # Scale the sprite (DOWN from pre-rendered size for best quality)
                if sprite.sprite_3d:
                    scaled_w = int(sprite.width * scale_ratio)
                    scaled_h = int(sprite.height * scale_ratio)
                    scaled_sprite = sprite.sprite_3d.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
                    
                    # Apply alpha
                    sprite_array = np.array(scaled_sprite)
                    sprite_array[:, :, 3] = (sprite_array[:, :, 3] * alpha).astype(np.uint8)
                    scaled_sprite = Image.fromarray(sprite_array)
                    
                    # Store position for this frame with baseline alignment
                    # Adjust y position to align baselines of all letters
                    # Taller letters (like 'l', 'h') need to be positioned higher
                    # Shorter letters (like 'e', 'o') need to be positioned lower
                    baseline_adjustment = max_height - scaled_h
                    letter_pos = (current_x, group_start_y + baseline_adjustment)
                    
                    # CRITICAL: Store the original unoccluded sprite for handoff
                    original_scaled_sprite = scaled_sprite.copy()
                    
                    # Apply masking if behind subject
                    if is_behind:
                        # CRITICAL FIX: Create a FRESH COPY of the sprite for occlusion
                        # Never modify the original or reuse modified sprites
                        occluded_sprite = scaled_sprite.copy()  # Work with a copy!
                        
                        # ALWAYS extract fresh mask for EVERY frame (no caching!)
                        current_mask = None
                        try:
                            import sys
                            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                            from video.segmentation.segment_extractor import extract_foreground_mask
                            current_rgb = background[:, :, :3] if background.shape[2] == 4 else background
                            
                            # Debug: Log extraction attempt
                            if self.debug and frame_number % 5 == 0:
                                print(f"[MASK_EXTRACT] Frame {frame_number}: Extracting fresh mask...")
                            
                            current_mask = extract_foreground_mask(current_rgb)
                            
                            if current_mask.shape[:2] != (self.resolution[1], self.resolution[0]):
                                current_mask = cv2.resize(current_mask, self.resolution, interpolation=cv2.INTER_LINEAR)
                            
                            current_mask = cv2.GaussianBlur(current_mask, (3, 3), 0)
                            kernel = np.ones((3, 3), np.uint8)
                            current_mask = cv2.dilate(current_mask, kernel, iterations=1)
                            current_mask = (current_mask > 128).astype(np.uint8) * 255
                            
                            if self.debug and frame_number % 5 == 0:
                                print(f"[Text3DMotion] Frame {frame_number}: Dynamic mask extracted")
                        except Exception as e:
                            if self.debug:
                                print(f"[Text3DMotion] Frame {frame_number}: Failed to extract mask: {e}")
                            # CRITICAL FIX: Never use stale mask - better no occlusion than wrong occlusion
                            current_mask = None
                            print(f"[MASK_FIX] Frame {frame_number}: Skipping occlusion (no stale fallback)")
                        
                        # Apply mask if we have one
                        if current_mask is not None:
                            # Work with the copy, not the original!
                            sprite_np = np.array(occluded_sprite)
                            h, w = sprite_np.shape[:2]
                            
                            # Get mask region
                            y1 = max(0, letter_pos[1])
                            y2 = min(frame.shape[0], letter_pos[1] + h)
                            x1 = max(0, letter_pos[0])
                            x2 = min(frame.shape[1], letter_pos[0] + w)
                            
                            if y2 > y1 and x2 > x1:
                                mask_region = current_mask[y1:y2, x1:x2]
                                sprite_region = sprite_np[
                                    max(0, -letter_pos[1]):max(0, -letter_pos[1]) + (y2 - y1),
                                    max(0, -letter_pos[0]):max(0, -letter_pos[0]) + (x2 - x1)
                                ]
                                
                                # Apply mask to alpha channel of the COPY
                                if sprite_region.shape[:2] == mask_region.shape:
                                    sprite_region[:, :, 3] = (
                                        sprite_region[:, :, 3] * (1 - mask_region / 255.0)
                                    ).astype(np.uint8)
                                
                                # Update the occluded copy, not the original
                                occluded_sprite = Image.fromarray(sprite_np)
                        
                        # Use the occluded copy for rendering
                        scaled_sprite = occluded_sprite
                    
                    # Composite the letter onto canvas
                    canvas.paste(scaled_sprite, letter_pos, scaled_sprite)
                    
                    # Store the ORIGINAL sprite for handoff (not the occluded one!)
                    # This ensures dissolve animation starts with clean sprites
                    rendered_sprite = LetterSprite(
                        char=sprite.char,
                        sprite_3d=original_scaled_sprite,  # Use the original, not occluded!
                        position=letter_pos,
                        width=scaled_w,
                        height=scaled_h,
                        anchor=(0, 0)  # Already adjusted during scaling
                    )
                    rendered_sprites.append(rendered_sprite)
                    
                    current_x += scaled_w

        # Store final state on last frame
        if frame_number == self.total_frames - 1:
            # Filter out None entries (spaces)
            final_sprites = [s for s in rendered_sprites if s is not None]
            
            self._final_state = MotionState(
                scale=scale,
                position=(group_start_x, group_start_y),
                text_size=(total_width, max_height),
                center_position=(int(cx), int(cy)),
                is_behind=is_behind,
                alpha=alpha,
                letter_sprites=final_sprites
            )
            
            if self.debug:
                print(f"[Text3DMotion] Final state stored with {len(final_sprites)} letter sprites")

        result = np.array(canvas)
        return result[:, :, :3] if result.shape[2] == 4 else result

    def get_final_state(self) -> Optional[MotionState]:
        """Get the final state for handoff to next animation."""
        return self._final_state