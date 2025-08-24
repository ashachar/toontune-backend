#!/usr/bin/env python3
"""
Combined 3D text animation with CORRECT position calculation.
Text shrinks smoothly behind subject, then dissolves from EXACT position.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import random
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


@dataclass
class Text3DMotionDissolve:
    """
    Combined animation: 3D text motion followed by 3D-aware dissolve.
    FIXED: Letters positioned to maintain same visual center as motion phase.
    """

    # Core animation parameters
    duration: float = 3.0
    fps: int = 30
    resolution: Tuple[int, int] = (1920, 1080)

    # Text properties
    text: str = "HELLO WORLD"
    segment_mask: Optional[Union[str, np.ndarray]] = None
    font_size: int = 120
    text_color: Tuple[int, int, int] = (255, 220, 0)
    depth_color: Tuple[int, int, int] = (200, 170, 0)
    depth_layers: int = 10
    depth_offset: int = 4

    # Motion phase timing
    motion_duration: float = 0.75
    shrink_duration: float = 0.6
    settle_duration: float = 0.15
    
    # Dissolve phase timing
    dissolve_stable_duration: float = 0.17
    dissolve_duration: float = 0.67
    dissolve_stagger: float = 0.33

    # Start/end states for motion
    start_scale: float = 2.0
    end_scale: float = 0.8
    final_scale: float = 0.75
    center_position: Optional[Tuple[int, int]] = None

    # Dissolve parameters
    float_distance: int = 30
    max_dissolve_scale: float = 1.2
    randomize_order: bool = True
    maintain_kerning: bool = True

    # Visual effects
    shadow_offset: int = 8
    outline_width: int = 3
    perspective_angle: float = 25
    supersample_factor: int = 2
    glow_effect: bool = True

    # Control flags
    debug: bool = False

    def __post_init__(self):
        # Font candidates
        self._font_candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]

        # Calculate total duration
        num_letters = len(self.text)
        self.dissolve_total_duration = (
            self.dissolve_stable_duration + 
            max(0, num_letters - 1) * self.dissolve_stagger + 
            self.dissolve_duration
        )
        
        # Update total duration to include both phases
        self.duration = self.motion_duration + self.dissolve_total_duration

        # Validate motion durations
        self.shrink_duration = min(self.shrink_duration, self.motion_duration * 0.9)
        self.settle_duration = self.motion_duration - self.shrink_duration

        # Frame calculations
        self.total_frames = int(self.duration * self.fps)
        self.motion_frames = int(self.motion_duration * self.fps)
        self.shrink_frames = int(self.shrink_duration * self.fps)
        self.settle_frames = self.motion_frames - self.shrink_frames
        self.dissolve_frames = self.total_frames - self.motion_frames

        # Load or create mask
        self.segment_mask = self._load_or_make_mask(self.segment_mask)

        # Default center position
        if self.center_position is None:
            self.center_position = (self.resolution[0] // 2, self.resolution[1] // 2)

        # Initialize mask cache
        self._frame_mask_cache = {}
        
        # Calculate transition scale (scale at which text goes behind)
        self.transition_scale = self.start_scale - 0.6 * (self.start_scale - self.end_scale)
        
        # IMPORTANT: Calculate exact final position from motion phase
        self._calculate_motion_final_state()
        
        # Initialize dissolve state using the final motion position
        self._init_dissolve_state()

        if self.debug:
            self._log(f"Init: Motion + 3D Dissolve animation (CORRECT)")
            self._log(f"Motion frames: 0-{self.motion_frames-1}")
            self._log(f"Dissolve frames: {self.motion_frames}-{self.total_frames-1}")
            self._log(f"Text will remain centered at: {self.center_position}")

    def _calculate_motion_final_state(self):
        """Calculate the exact final position and scale from motion phase."""
        # Calculate final state at last motion frame
        final_frame = self.motion_frames - 1
        t_global = final_frame / max(self.motion_frames - 1, 1)
        smooth_t_global = self._smoothstep(t_global)
        
        # Calculate final scale
        shrink_progress = self.shrink_duration / self.motion_duration
        if smooth_t_global <= shrink_progress:
            local_t = smooth_t_global / shrink_progress
            self.motion_final_scale = self.start_scale - local_t * (self.start_scale - self.end_scale)
        else:
            local_t = (smooth_t_global - shrink_progress) / (1.0 - shrink_progress)
            self.motion_final_scale = self.end_scale - local_t * (self.end_scale - self.final_scale)
        
        # Calculate final Y position
        cx, cy = self.center_position
        start_y = cy - self.resolution[1] * 0.15
        end_y = cy
        self.motion_final_center_y = start_y + smooth_t_global * (end_y - start_y)
        
        # The text should remain centered at cx horizontally
        self.motion_final_center_x = cx
        
        if self.debug:
            self._log(f"Motion final state: scale={self.motion_final_scale:.3f}")
            self._log(f"Motion final center: ({self.motion_final_center_x}, {self.motion_final_center_y:.1f})")

    def _init_dissolve_state(self):
        """Initialize dissolve animation state using exact motion final position."""
        self.letter_3d_sprites = {}
        self.letter_positions = []
        self.letter_kill_masks = {}
        self.dissolve_order = []
        
        # Pre-render 3D letter sprites at MOTION FINAL SCALE
        self._prepare_3d_letter_sprites_centered()
        
        # Determine dissolve order
        if self.randomize_order:
            indices = list(range(len(self.text)))
            random.shuffle(indices)
            self.dissolve_order = indices
        else:
            self.dissolve_order = list(range(len(self.text)))

    def _prepare_3d_letter_sprites_centered(self):
        """Pre-render individual 3D letter sprites centered at motion final position."""
        # Use the scale from the end of motion phase
        scale = self.motion_final_scale
        font_px = int(self.font_size * scale)
        font = self._get_font(font_px)
        
        # Measure full text to get total width
        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        full_bbox = d.textbbox((0, 0), self.text, font=font)
        
        # Calculate total text width (without padding)
        text_width = full_bbox[2] - full_bbox[0]
        text_height = full_bbox[3] - full_bbox[1]
        
        # CRITICAL FIX: Calculate starting X to center text at motion_final_center_x
        # The text should be centered, not left-aligned
        start_x = int(self.motion_final_center_x - text_width // 2)
        start_y = int(self.motion_final_center_y - text_height // 2)
        
        current_x = start_x
        
        if self.debug:
            self._log(f"Letter positioning:")
            self._log(f"  Text width: {text_width}, height: {text_height}")
            self._log(f"  Starting X for centered text: {start_x}")
            self._log(f"  Center will be at: {self.motion_final_center_x}")
        
        # Render each letter with 3D depth
        for i, letter in enumerate(self.text):
            if letter == ' ':
                # Handle space
                space_width = font_px // 3
                self.letter_positions.append((int(current_x), int(start_y)))
                self.letter_3d_sprites[i] = None
                current_x += space_width
                continue
            
            # Measure this letter
            letter_bbox = d.textbbox((0, 0), letter, font=font)
            letter_width = letter_bbox[2] - letter_bbox[0]
            
            # Render 3D letter sprite
            sprite_3d, letter_anchor = self._render_3d_letter(
                letter, scale, 1.0, self.transition_scale
            )
            
            # Store sprite
            self.letter_3d_sprites[i] = sprite_3d
            
            # Position is where the sprite's top-left goes
            # We need to account for the padding in the sprite
            sprite_x = int(current_x - int(letter_anchor[0] - letter_width // 2))
            sprite_y = int(start_y - int(letter_anchor[1] - text_height // 2))
            
            self.letter_positions.append((sprite_x, sprite_y))
            
            # Initialize kill mask for this letter
            if sprite_3d:
                self.letter_kill_masks[i] = np.zeros(
                    (sprite_3d.height, sprite_3d.width), dtype=np.uint8
                )
            
            # Advance position with kerning
            if self.maintain_kerning and i < len(self.text) - 1:
                current_x += int(letter_width * 0.9)
            else:
                current_x += letter_width
        
        if self.debug:
            self._log(f"Letter sprites prepared:")
            for i, (letter, pos) in enumerate(zip(self.text, self.letter_positions)):
                if pos:
                    self._log(f"  '{letter}': position ({pos[0]}, {pos[1]})")
            
            # Calculate actual center of letters
            if self.letter_positions:
                valid_pos = [p for p in self.letter_positions if p is not None]
                if valid_pos:
                    # Approximate bounds
                    min_x = min(p[0] for p in valid_pos)
                    max_x = max(p[0] for p in valid_pos) + int(text_width / len(self.text))  # Approximate
                    actual_center = (min_x + max_x) / 2
                    self._log(f"  Letters actual center X: {actual_center:.1f} (target: {self.motion_final_center_x})")

    def generate_frame(
        self,
        frame_number: int,
        background: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate a single frame of the combined animation."""
        
        if background is None:
            frame = np.zeros((self.resolution[1], self.resolution[0], 4), dtype=np.uint8)
            frame[:, :, 3] = 255
        else:
            if background.shape[2] == 3:
                frame = np.zeros((background.shape[0], background.shape[1], 4), dtype=np.uint8)
                frame[:, :, :3] = background
                frame[:, :, 3] = 255
            else:
                frame = background.copy()

        # Determine animation phase
        if frame_number < self.motion_frames:
            # Motion phase
            return self._generate_motion_frame(frame_number, frame, background)
        else:
            # Dissolve phase
            dissolve_frame = frame_number - self.motion_frames
            return self._generate_3d_dissolve_frame(dissolve_frame, frame, background)

    def _generate_motion_frame(self, frame_number: int, frame: np.ndarray, background: Optional[np.ndarray]) -> np.ndarray:
        """Generate a frame during the motion phase."""
        
        # CONTINUOUS t across entire motion animation
        t_global = frame_number / max(self.motion_frames - 1, 1)
        
        # Apply single smoothstep to entire animation
        smooth_t_global = self._smoothstep(t_global)
        
        # Determine phase for other calculations
        if frame_number < self.shrink_frames:
            phase = "shrink"
            t_local = frame_number / max(self.shrink_frames - 1, 1)
        else:
            phase = "settle"
            t_local = (frame_number - self.shrink_frames) / max(self.settle_frames - 1, 1)

        # CONTINUOUS scale interpolation
        shrink_progress = self.shrink_duration / self.motion_duration
        
        if smooth_t_global <= shrink_progress:
            local_t = smooth_t_global / shrink_progress
            scale = self.start_scale - local_t * (self.start_scale - self.end_scale)
        else:
            local_t = (smooth_t_global - shrink_progress) / (1.0 - shrink_progress)
            scale = self.end_scale - local_t * (self.end_scale - self.final_scale)
        
        # Alpha calculation
        if phase == "shrink":
            if t_local < 0.4:
                alpha = 1.0
                is_behind = False
            elif t_local < 0.6:
                fade_t = (t_local - 0.4) / 0.2
                alpha = 1.0 - fade_t * 0.4
                is_behind = False
            else:
                fade_t = (t_local - 0.6) / 0.4
                k = 3.0
                alpha = 0.6 - 0.4 * (1 - np.exp(-k * fade_t)) / (1 - np.exp(-k))
                is_behind = True
        else:  # settle
            alpha = 0.2
            is_behind = True

        # Use fixed scale for depth when behind
        if is_behind:
            depth_scale = self.transition_scale
        else:
            depth_scale = scale

        # Render 3D text
        text_pil, (anchor_x, anchor_y) = self._render_3d_text(
            self.text, scale, alpha, False, depth_scale
        )

        # Calculate position with CONTINUOUS interpolation
        cx, cy = self.center_position
        start_y = cy - self.resolution[1] * 0.15
        end_y = cy
        
        pos_x = int(cx - anchor_x)
        pos_y = int(start_y + smooth_t_global * (end_y - start_y) - anchor_y)

        # Place text on frame
        text_np = np.array(text_pil)
        tw, th = text_pil.size

        y1 = max(0, pos_y)
        y2 = min(frame.shape[0], pos_y + th)
        x1 = max(0, pos_x)
        x2 = min(frame.shape[1], pos_x + tw)

        ty1 = max(0, -pos_y)
        ty2 = ty1 + (y2 - y1)
        tx1 = max(0, -pos_x)
        tx2 = tx1 + (x2 - x1)

        # Build text layer
        text_layer = np.zeros_like(frame)
        text_layer[y1:y2, x1:x2] = text_np[ty1:ty2, tx1:tx2]

        # Apply masking when behind
        if is_behind:
            if background is not None and background.shape[2] >= 3:
                if frame_number not in self._frame_mask_cache:
                    from utils.segmentation.segment_extractor import extract_foreground_mask
                    
                    if background.shape[2] == 4:
                        current_rgb = background[:, :, :3]
                    else:
                        current_rgb = background
                    
                    current_mask = extract_foreground_mask(current_rgb)
                    
                    if current_mask.shape[:2] != (self.resolution[1], self.resolution[0]):
                        current_mask = cv2.resize(current_mask, self.resolution, interpolation=cv2.INTER_LINEAR)
                    
                    # Minimal processing
                    current_mask = cv2.GaussianBlur(current_mask, (3, 3), 0)
                    kernel = np.ones((3, 3), np.uint8)
                    current_mask = cv2.dilate(current_mask, kernel, iterations=1)
                    current_mask = (current_mask > 128).astype(np.uint8) * 255
                    
                    self._frame_mask_cache[frame_number] = current_mask
                else:
                    current_mask = self._frame_mask_cache[frame_number]
            else:
                current_mask = self.segment_mask
            
            # Apply mask
            mask_region = current_mask[y1:y2, x1:x2]
            text_alpha = text_layer[y1:y2, x1:x2, 3].astype(np.float32)
            mask_factor = mask_region.astype(np.float32) / 255.0
            text_alpha *= (1.0 - mask_factor)
            text_layer[y1:y2, x1:x2, 3] = text_alpha.astype(np.uint8)

        # Composite
        frame_pil = Image.fromarray(frame)
        text_pil = Image.fromarray(text_layer)
        out = Image.alpha_composite(frame_pil, text_pil)
        result = np.array(out)

        if self.debug and frame_number % 10 == 0:
            self._log(f"Motion frame {frame_number}: scale={scale:.3f}, pos=({pos_x}, {pos_y})")

        return result[:, :, :3] if result.shape[2] == 4 else result

    def _generate_3d_dissolve_frame(self, dissolve_frame: int, frame: np.ndarray, background: Optional[np.ndarray]) -> np.ndarray:
        """Generate a frame during the 3D dissolve phase using correct positions."""
        
        # Calculate timing within dissolve phase
        t = dissolve_frame / max(self.dissolve_frames - 1, 1)
        
        # Create canvas for dissolve effect
        canvas = Image.fromarray(frame)
        
        # Process each letter with its 3D sprite
        for idx, letter_idx in enumerate(self.dissolve_order):
            if self.letter_3d_sprites[letter_idx] is None:
                continue  # Skip spaces
            
            # Calculate letter-specific timing
            letter_start = idx * self.dissolve_stagger / self.dissolve_total_duration
            letter_end = letter_start + self.dissolve_duration / self.dissolve_total_duration
            
            if t < letter_start:
                # Letter hasn't started dissolving yet - show at final alpha
                alpha_mult = 0.2  # Match the end alpha from motion phase
                scale = 1.0
                float_y = 0
                add_holes = False
            elif t > letter_end:
                # Letter has fully dissolved
                continue
            else:
                # Letter is dissolving
                letter_t = (t - letter_start) / (letter_end - letter_start)
                
                # Smooth interpolation
                smooth_t = self._smoothstep(letter_t)
                
                # Alpha fade out (starting from 0.2)
                alpha_mult = 0.2 * (1.0 - smooth_t)
                
                # Scale effect
                scale = 1.0 + smooth_t * (self.max_dissolve_scale - 1.0)
                
                # Float upward
                float_y = -smooth_t * self.float_distance
                
                # Add dissolve holes after 30% progress
                add_holes = letter_t > 0.3
                
                if add_holes:
                    self._add_3d_dissolve_holes(letter_idx, letter_t)
            
            # Get the pre-rendered 3D sprite
            sprite_3d = self.letter_3d_sprites[letter_idx]
            pos_x, pos_y = self.letter_positions[letter_idx]
            
            # Create a copy for transformation
            sprite = sprite_3d.copy()
            
            # Apply scale transformation
            if scale != 1.0:
                new_size = (int(sprite.width * scale), int(sprite.height * scale))
                sprite = sprite.resize(new_size, Image.Resampling.LANCZOS)
                # Adjust position to keep centered
                pos_x -= (new_size[0] - sprite_3d.width) // 2
                pos_y -= (new_size[1] - sprite_3d.height) // 2
            
            # Apply float
            pos_y += int(float_y)
            
            # Apply alpha and kill mask to the 3D sprite
            sprite_array = np.array(sprite)
            
            if letter_idx in self.letter_kill_masks and np.any(self.letter_kill_masks[letter_idx]):
                # Resize kill mask if needed
                kill_mask = self.letter_kill_masks[letter_idx]
                if scale != 1.0:
                    kill_mask = cv2.resize(kill_mask, (sprite.width, sprite.height))
                
                # Apply kill mask - this affects ALL depth layers
                alpha_mask = 1.0 - (kill_mask / 255.0)
                sprite_array[:, :, 3] = (sprite_array[:, :, 3] * alpha_mask * alpha_mult).astype(np.uint8)
            else:
                # Just apply alpha multiplier
                sprite_array[:, :, 3] = (sprite_array[:, :, 3] * alpha_mult).astype(np.uint8)
            
            # Apply glow effect during dissolve if enabled
            if self.glow_effect and alpha_mult > 0.01:
                sprite_img = Image.fromarray(sprite_array)
                glow = sprite_img.filter(ImageFilter.GaussianBlur(radius=3))
                # Blend glow with original
                glow_array = np.array(glow)
                glow_array[:, :, 3] = (glow_array[:, :, 3] * 0.5).astype(np.uint8)
                glow_img = Image.fromarray(glow_array)
                sprite_img = Image.alpha_composite(glow_img, sprite_img)
                sprite_array = np.array(sprite_img)
            
            # Composite onto canvas
            sprite_img = Image.fromarray(sprite_array)
            canvas.paste(sprite_img, (int(pos_x), int(pos_y)), sprite_img)
        
        result = np.array(canvas)
        
        if self.debug and dissolve_frame % 10 == 0:
            active_letters = sum(1 for idx in self.dissolve_order 
                               if self.letter_3d_sprites[idx] is not None)
            self._log(f"3D Dissolve frame {dissolve_frame}: t={t:.3f}, active letters: {active_letters}")
        
        return result[:, :, :3] if result.shape[2] == 4 else result

    def _add_3d_dissolve_holes(self, letter_idx: int, letter_t: float):
        """Add dissolve holes to a 3D letter's kill mask."""
        if letter_idx not in self.letter_kill_masks:
            return
        
        kill_mask = self.letter_kill_masks[letter_idx]
        h, w = kill_mask.shape
        
        # More holes as dissolve progresses
        base_holes = 5
        progress_holes = int(letter_t * 15)
        num_holes = base_holes + progress_holes
        
        for _ in range(num_holes):
            # Random hole position
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            
            # Varying hole sizes (larger as dissolve progresses)
            min_radius = 2 + int(letter_t * 2)
            max_radius = 5 + int(letter_t * 5)
            radius = random.randint(min_radius, max_radius)
            
            # Create circular hole with feathered edges
            for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist <= radius:
                        # Gaussian falloff for smooth edges
                        if dist < radius * 0.7:
                            opacity = 255
                        else:
                            falloff = (dist - radius * 0.7) / (radius * 0.3)
                            opacity = int(255 * (1 - falloff))
                        
                        kill_mask[y, x] = min(255, kill_mask[y, x] + opacity)

    def _render_3d_letter(
        self,
        letter: str,
        scale: float,
        alpha: float,
        depth_scale: float,
    ) -> Tuple[Optional[Image.Image], Tuple[float, float]]:
        """Render a single 3D letter with depth layers."""
        
        ss = self.supersample_factor
        
        # Font size
        font_px = max(2, int(round(self.font_size * scale * ss)))
        font = self._get_font(font_px)

        # Measure letter
        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), letter, font=font)
        face_w = max(1, bbox[2] - bbox[0])
        face_h = max(1, bbox[3] - bbox[1])

        # Canvas with padding for 3D depth
        depth_off = int(round(self.depth_offset * depth_scale * ss))
        pad = max(depth_off * self.depth_layers * 2, ss * 8)
        canvas_w = face_w + pad * 2 + depth_off * self.depth_layers
        canvas_h = face_h + pad * 2 + depth_off * self.depth_layers

        img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Center front face
        front_x = (canvas_w - face_w) // 2
        front_y = (canvas_h - face_h) // 2

        # Draw depth layers (80% reduction as before)
        reduced_depth_off = int(round(depth_off * 0.2))
        extra_layers = self.depth_layers * 2
        
        for i in range(extra_layers, 0, -1):
            ox = front_x + i * reduced_depth_off
            oy = front_y - i * reduced_depth_off
            t = (extra_layers - i) / max(extra_layers - 1, 1)
            t = t * t * t * (3.0 - 2.0 * t - 0.5 * t)

            r = int(self.depth_color[0] * (1 - t) + self.text_color[0] * t * 0.75)
            g = int(self.depth_color[1] * (1 - t) + self.text_color[1] * t * 0.75)
            b = int(self.depth_color[2] * (1 - t) + self.text_color[2] * t * 0.75)

            draw.text((ox, oy), letter, font=font, fill=(r, g, b, int(255 * alpha)))

        # Outline
        outline_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
        outline_draw = ImageDraw.Draw(outline_img)
        outline_w = max(1, int(self.outline_width * ss))
        
        for radius in range(outline_w, 0, -1):
            fade = 1.0 - (radius - 1) / max(outline_w, 1) * 0.5
            for ang in range(0, 360, 30):
                ox = int(round(radius * np.cos(np.radians(ang))))
                oy = int(round(radius * np.sin(np.radians(ang))))
                outline_draw.text(
                    (front_x + ox, front_y + oy),
                    letter,
                    font=font,
                    fill=(0, 0, 0, int(110 * alpha * fade)),
                )
        
        outline_img = outline_img.filter(ImageFilter.GaussianBlur(radius=max(1, ss // 2)))
        img = Image.alpha_composite(img, outline_img)

        # Front face
        face_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(face_img).text(
            (front_x, front_y),
            letter,
            font=font,
            fill=(*self.text_color, int(255 * alpha)),
        )
        img = Image.alpha_composite(img, face_img)

        # Anchor point
        anchor_x_ss = front_x + face_w / 2.0
        anchor_y_ss = front_y + face_h / 2.0

        # Shadow
        shadow = np.array(img)
        shadow[:, :, :3] = 0
        shadow[:, :, 3] = (shadow[:, :, 3].astype(np.float32) * 0.4).astype(np.uint8)
        shadow = Image.fromarray(shadow).filter(ImageFilter.GaussianBlur(radius=max(2, ss)))
        final = Image.new("RGBA", img.size, (0, 0, 0, 0))
        shadow_off = int(round(self.shadow_offset * depth_scale * ss * 0.3))
        final.paste(shadow, (shadow_off, shadow_off), shadow)
        final = Image.alpha_composite(final, img)

        # Downsample
        final = final.filter(ImageFilter.GaussianBlur(radius=max(0.0, ss / 6.0)))
        target_w = max(1, final.width // ss)
        target_h = max(1, final.height // ss)
        final = final.resize((target_w, target_h), Image.Resampling.LANCZOS)

        anchor_x = anchor_x_ss / ss
        anchor_y = anchor_y_ss / ss

        return final, (anchor_x, anchor_y)

    def _render_3d_text(
        self,
        text: str,
        scale: float,
        alpha: float,
        apply_perspective: bool,
        depth_scale: float = None,
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """Render full 3D text (for motion phase)."""
        
        if depth_scale is None:
            depth_scale = scale
            
        ss = self.supersample_factor
        
        # Font size uses regular scale
        font_px = max(2, int(round(self.font_size * scale * ss)))
        font = self._get_font(font_px)

        # Measure text
        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), text, font=font)
        face_w = max(1, bbox[2] - bbox[0])
        face_h = max(1, bbox[3] - bbox[1])

        # Canvas
        depth_off = int(round(self.depth_offset * depth_scale * ss))
        pad = max(depth_off * self.depth_layers * 2, ss * 8)
        canvas_w = face_w + pad * 2 + depth_off * self.depth_layers
        canvas_h = face_h + pad * 2 + depth_off * self.depth_layers

        img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Center front face
        front_x = (canvas_w - face_w) // 2
        front_y = (canvas_h - face_h) // 2

        # Depth layers with 80% reduction
        reduced_depth_off = int(round(depth_off * 0.2))
        extra_layers = self.depth_layers * 2
        
        for i in range(extra_layers, 0, -1):
            ox = front_x + i * reduced_depth_off
            oy = front_y - i * reduced_depth_off
            t = (extra_layers - i) / max(extra_layers - 1, 1)
            t = t * t * t * (3.0 - 2.0 * t - 0.5 * t)

            r = int(self.depth_color[0] * (1 - t) + self.text_color[0] * t * 0.75)
            g = int(self.depth_color[1] * (1 - t) + self.text_color[1] * t * 0.75)
            b = int(self.depth_color[2] * (1 - t) + self.text_color[2] * t * 0.75)

            draw.text((ox, oy), text, font=font, fill=(r, g, b, int(255 * alpha)))

        # Outline
        outline_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
        outline_draw = ImageDraw.Draw(outline_img)
        outline_w = max(1, int(self.outline_width * ss))
        
        for radius in range(outline_w, 0, -1):
            fade = 1.0 - (radius - 1) / max(outline_w, 1) * 0.5
            for ang in range(0, 360, 30):
                ox = int(round(radius * np.cos(np.radians(ang))))
                oy = int(round(radius * np.sin(np.radians(ang))))
                outline_draw.text(
                    (front_x + ox, front_y + oy),
                    text,
                    font=font,
                    fill=(0, 0, 0, int(110 * alpha * fade)),
                )
        
        outline_img = outline_img.filter(ImageFilter.GaussianBlur(radius=max(1, ss // 2)))
        img = Image.alpha_composite(img, outline_img)

        # Front face
        face_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(face_img).text(
            (front_x, front_y),
            text,
            font=font,
            fill=(*self.text_color, int(255 * alpha)),
        )
        img = Image.alpha_composite(img, face_img)

        # Anchor
        anchor_x_ss = front_x + face_w / 2.0
        anchor_y_ss = front_y + face_h / 2.0

        # Shadow
        shadow = np.array(img)
        shadow[:, :, :3] = 0
        shadow[:, :, 3] = (shadow[:, :, 3].astype(np.float32) * 0.4).astype(np.uint8)
        shadow = Image.fromarray(shadow).filter(ImageFilter.GaussianBlur(radius=max(2, ss)))
        final = Image.new("RGBA", img.size, (0, 0, 0, 0))
        shadow_off = int(round(self.shadow_offset * depth_scale * ss * 0.3))
        final.paste(shadow, (shadow_off, shadow_off), shadow)
        final = Image.alpha_composite(final, img)

        # Downsample
        final = final.filter(ImageFilter.GaussianBlur(radius=max(0.0, ss / 6.0)))
        target_w = max(1, final.width // ss)
        target_h = max(1, final.height // ss)
        final = final.resize((target_w, target_h), Image.Resampling.LANCZOS)

        anchor_x = anchor_x_ss / ss
        anchor_y = anchor_y_ss / ss

        return final, (anchor_x, anchor_y)

    def _load_or_make_mask(self, segment_mask) -> np.ndarray:
        w, h = self.resolution
        if segment_mask is None:
            mask = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.uint8)
        elif isinstance(segment_mask, str):
            mask_img = Image.open(segment_mask).convert("L")
            mask_img = mask_img.resize(self.resolution, Image.Resampling.LANCZOS)
            mask = np.array(mask_img, dtype=np.uint8)
        else:
            mask = segment_mask
            if mask.shape[:2] != (self.resolution[1], self.resolution[0]):
                mask = cv2.resize(mask, self.resolution, interpolation=cv2.INTER_LINEAR)
        mask = (mask > 128).astype(np.uint8) * 255
        return mask

    def _smoothstep(self, t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[3D_CORRECT] {msg}")

    def _get_font(self, px: int) -> ImageFont.FreeTypeFont:
        for path in self._font_candidates:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, px)
                except:
                    continue
        return ImageFont.load_default()