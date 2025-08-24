#!/usr/bin/env python3
"""
3D text animation that combines motion and dissolve effects.
Each individual letter maintains exact position at transition.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, List
import random
from dataclasses import dataclass

@dataclass
class LetterState:
    """State of an individual letter for precise tracking."""
    char: str
    sprite_3d: Optional[Image.Image]
    position: Tuple[int, int]  # Exact pixel position from motion phase
    width: int
    height: int

class Text3DMotionDissolve:
    """3D text animation with precise per-letter position tracking."""
    
    def __init__(
        self,
        duration: float = 3.0,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        text: str = "HELLO",
        segment_mask: Optional[np.ndarray] = None,
        font_size: int = 120,
        text_color: Tuple[int, int, int] = (255, 220, 0),
        depth_color: Tuple[int, int, int] = (200, 170, 0),
        depth_layers: int = 8,
        depth_offset: int = 3,
        motion_duration: float = 1.0,
        start_scale: float = 2.0,
        end_scale: float = 1.0,
        final_scale: float = 0.9,
        shrink_duration: float = 0.8,
        settle_duration: float = 0.2,
        dissolve_stable_duration: float = 0.2,
        dissolve_duration: float = 0.8,
        dissolve_stagger: float = 0.15,
        float_distance: float = 50,
        max_dissolve_scale: float = 1.5,
        randomize_order: bool = False,
        maintain_kerning: bool = True,
        center_position: Optional[Tuple[int, int]] = None,
        shadow_offset: int = 5,
        outline_width: int = 2,
        perspective_angle: float = 0,
        supersample_factor: int = 2,
        glow_effect: bool = True,
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
        self.motion_duration = motion_duration
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.final_scale = final_scale
        self.shrink_duration = shrink_duration
        self.settle_duration = settle_duration
        self.dissolve_stable_duration = dissolve_stable_duration
        self.dissolve_duration = dissolve_duration
        self.dissolve_stagger = dissolve_stagger
        self.float_distance = float_distance
        self.max_dissolve_scale = max_dissolve_scale
        self.randomize_order = randomize_order
        self.maintain_kerning = maintain_kerning
        self.center_position = center_position or (resolution[0] // 2, resolution[1] // 2)
        self.shadow_offset = shadow_offset
        self.outline_width = outline_width
        self.perspective_angle = perspective_angle
        self.supersample_factor = supersample_factor
        self.glow_effect = glow_effect
        self.debug = debug
        
        # Calculate phase frames
        self.motion_frames = int(motion_duration * fps)
        self.dissolve_total_duration = dissolve_stable_duration + dissolve_duration + dissolve_stagger * (len(text) - 1)
        self.dissolve_frames = int(self.dissolve_total_duration * fps)
        
        # Cache for dynamic masks
        self._frame_mask_cache = {}
        
        # Letter states for precise tracking
        self.letter_states: List[LetterState] = []
        self.dissolve_order = []
        self.letter_kill_masks = {}
        
        # Transition values that will be captured from motion phase
        self.transition_scale = None
        self.transition_text_bbox = None  # Exact bounding box at transition
        
        # Initialize dissolve order
        self._init_dissolve_order()
        
    def _init_dissolve_order(self):
        """Initialize the order in which letters dissolve."""
        if self.randomize_order:
            indices = list(range(len(self.text)))
            random.shuffle(indices)
            self.dissolve_order = indices
        else:
            self.dissolve_order = list(range(len(self.text)))
    
    def _log(self, message: str):
        """Debug logging."""
        if self.debug:
            print(f"[Text3DMotionDissolve] {message}")
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get font at specified size."""
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except:
            return ImageFont.load_default()
    
    def _smoothstep(self, t: float) -> float:
        """Smooth interpolation function."""
        t = max(0, min(1, t))
        return t * t * (3 - 2 * t)
    
    def _render_3d_letter(
        self, 
        letter: str, 
        scale: float, 
        alpha: float,
        depth_scale: float
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """Render a single 3D letter with depth layers."""
        font_px = int(self.font_size * scale * self.supersample_factor)
        font = self._get_font(font_px)
        
        # Measure letter
        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), letter, font=font)
        width = bbox[2] - bbox[0] + self.depth_offset * self.depth_layers * 2
        height = bbox[3] - bbox[1] + self.depth_offset * self.depth_layers * 2
        
        # Create canvas
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        # Render depth layers
        for i in range(self.depth_layers - 1, -1, -1):
            depth_alpha = int(alpha * 255 * (0.3 + 0.7 * (1 - i / self.depth_layers)))
            offset = int(i * self.depth_offset * depth_scale * self.supersample_factor)
            
            if i == 0:
                # Front layer
                color = (*self.text_color, depth_alpha)
            else:
                # Depth layers
                factor = 0.7 - (i / self.depth_layers) * 0.4
                color = tuple(int(c * factor) for c in self.depth_color) + (depth_alpha,)
            
            x = -bbox[0] + offset
            y = -bbox[1] + offset
            draw.text((x, y), letter, font=font, fill=color)
        
        # Downsample if supersampling
        if self.supersample_factor > 1:
            new_size = (width // self.supersample_factor, height // self.supersample_factor)
            canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)
        
        # Calculate anchor
        anchor_x = -bbox[0] // self.supersample_factor
        anchor_y = -bbox[1] // self.supersample_factor
        
        return canvas, (anchor_x, anchor_y)
    
    def _render_3d_text(
        self,
        text: str,
        scale: float,
        alpha: float,
        is_behind: bool,
        depth_scale: float
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """Render full 3D text string."""
        font_px = int(self.font_size * scale * self.supersample_factor)
        font = self._get_font(font_px)
        
        # Measure text
        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0] + self.depth_offset * self.depth_layers * 2
        height = bbox[3] - bbox[1] + self.depth_offset * self.depth_layers * 2
        
        # Create canvas
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        # Render depth layers
        for i in range(self.depth_layers - 1, -1, -1):
            depth_alpha = int(alpha * 255 * (0.3 + 0.7 * (1 - i / self.depth_layers)))
            offset = int(i * self.depth_offset * depth_scale * self.supersample_factor)
            
            if i == 0:
                color = (*self.text_color, depth_alpha)
            else:
                factor = 0.7 - (i / self.depth_layers) * 0.4
                color = tuple(int(c * factor) for c in self.depth_color) + (depth_alpha,)
            
            x = -bbox[0] + offset
            y = -bbox[1] + offset
            draw.text((x, y), text, font=font, fill=color)
        
        # Downsample
        if self.supersample_factor > 1:
            new_size = (width // self.supersample_factor, height // self.supersample_factor)
            canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)
        
        anchor_x = -bbox[0] // self.supersample_factor
        anchor_y = -bbox[1] // self.supersample_factor
        
        return canvas, (anchor_x, anchor_y)
    
    def _capture_motion_final_state(self, frame_number: int, text_pil: Image.Image, pos_x: int, pos_y: int, scale: float):
        """Capture the exact final state from motion phase for letter positioning."""
        self.transition_scale = scale
        self.transition_text_bbox = (pos_x, pos_y, text_pil.width, text_pil.height)
        
        # Now prepare individual letter states with exact positions
        font_px = int(self.font_size * scale)
        font = self._get_font(font_px)
        
        # Clear previous states
        self.letter_states = []
        
        # Track current position within the text
        current_x_offset = 0
        
        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        
        for i, letter in enumerate(self.text):
            if letter == ' ':
                # Handle space
                space_width = font_px // 3
                state = LetterState(
                    char=letter,
                    sprite_3d=None,
                    position=(pos_x + current_x_offset, pos_y),
                    width=space_width,
                    height=0
                )
                self.letter_states.append(state)
                current_x_offset += space_width
            else:
                # Measure this letter
                letter_bbox = d.textbbox((0, 0), letter, font=font)
                letter_width = letter_bbox[2] - letter_bbox[0]
                letter_height = letter_bbox[3] - letter_bbox[1]
                
                # Render 3D sprite for this letter
                sprite_3d, (anchor_x, anchor_y) = self._render_3d_letter(
                    letter, scale, 1.0, 1.0
                )
                
                # Calculate exact position for this letter
                # This matches how text is naturally rendered
                letter_x = pos_x + current_x_offset
                letter_y = pos_y
                
                state = LetterState(
                    char=letter,
                    sprite_3d=sprite_3d,
                    position=(letter_x, letter_y),
                    width=sprite_3d.width,
                    height=sprite_3d.height
                )
                self.letter_states.append(state)
                
                # Move to next letter position
                current_x_offset += letter_width
        
        if self.debug:
            self._log(f"Captured motion final state at frame {frame_number}:")
            self._log(f"  Text position: ({pos_x}, {pos_y})")
            self._log(f"  Text size: {text_pil.width}x{text_pil.height}")
            self._log(f"  Scale: {scale:.3f}")
            self._log(f"  Letter count: {len(self.letter_states)}")
    
    def _generate_motion_frame(self, frame_number: int, frame: np.ndarray, background: Optional[np.ndarray]) -> np.ndarray:
        """Generate a frame during the motion phase."""
        t_global = frame_number / max(self.motion_frames - 1, 1)
        smooth_t_global = self._smoothstep(t_global)
        
        # Calculate scale
        shrink_progress = self.shrink_duration / self.motion_duration
        if smooth_t_global <= shrink_progress:
            local_t = smooth_t_global / shrink_progress
            scale = self.start_scale - local_t * (self.start_scale - self.end_scale)
            depth_scale = 1.0
            is_behind = local_t > 0.5
            base_alpha = 1.0 if local_t <= 0.5 else max(0.2, 1.0 - (local_t - 0.5) * 3.6)
        else:
            local_t = (smooth_t_global - shrink_progress) / (1.0 - shrink_progress)
            scale = self.end_scale - local_t * (self.end_scale - self.final_scale)
            depth_scale = 1.0
            is_behind = True
            base_alpha = 0.2
        
        # Render text
        text_pil, (anchor_x, anchor_y) = self._render_3d_text(
            self.text, scale, base_alpha, is_behind, depth_scale
        )
        
        # Calculate position
        cx, cy = self.center_position
        start_y = cy - self.resolution[1] * 0.15
        end_y = cy
        
        pos_x = int(cx - anchor_x)
        pos_y = int(start_y + smooth_t_global * (end_y - start_y) - anchor_y)
        
        # Capture final state if at last motion frame
        if frame_number == self.motion_frames - 1:
            self._capture_motion_final_state(frame_number, text_pil, pos_x, pos_y, scale)
        
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
        if is_behind and self.segment_mask is not None:
            # Use dynamic mask if available
            if background is not None:
                if frame_number not in self._frame_mask_cache:
                    from utils.segmentation.segment_extractor import extract_foreground_mask
                    current_rgb = background[:, :, :3] if background.shape[2] == 4 else background
                    current_mask = extract_foreground_mask(current_rgb)
                    
                    if current_mask.shape[:2] != (self.resolution[1], self.resolution[0]):
                        current_mask = cv2.resize(current_mask, self.resolution, interpolation=cv2.INTER_LINEAR)
                    
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
        
        return result[:, :, :3] if result.shape[2] == 4 else result
    
    def _add_3d_dissolve_holes(self, letter_idx: int, progress: float):
        """Add dissolve holes to a letter's kill mask."""
        if letter_idx >= len(self.letter_states):
            return
        
        state = self.letter_states[letter_idx]
        if state.sprite_3d is None:
            return
        
        # Initialize kill mask if needed
        if letter_idx not in self.letter_kill_masks:
            self.letter_kill_masks[letter_idx] = np.zeros(
                (state.sprite_3d.height, state.sprite_3d.width), dtype=np.uint8
            )
        
        # Add random holes based on progress
        num_holes = int(progress * 20)
        for _ in range(num_holes):
            x = np.random.randint(0, state.sprite_3d.width)
            y = np.random.randint(0, state.sprite_3d.height)
            radius = np.random.randint(2, 8)
            cv2.circle(self.letter_kill_masks[letter_idx], (x, y), radius, 1, -1)
    
    def _generate_3d_dissolve_frame(self, dissolve_frame: int, frame: np.ndarray, background: Optional[np.ndarray]) -> np.ndarray:
        """Generate a frame during the 3D dissolve phase using exact captured positions."""
        
        # Ensure we have captured the motion final state
        if not self.letter_states:
            self._log("WARNING: No letter states captured from motion phase!")
            return frame
        
        # Calculate timing within dissolve phase
        t = dissolve_frame / max(self.dissolve_frames - 1, 1)
        
        # Create canvas for dissolve effect
        canvas = Image.fromarray(frame)
        
        # Process each letter with its exact position
        for idx in self.dissolve_order:
            if idx >= len(self.letter_states):
                continue
            
            state = self.letter_states[idx]
            if state.sprite_3d is None:
                continue  # Skip spaces
            
            # Calculate letter-specific timing
            letter_start = (idx if not self.randomize_order else self.dissolve_order.index(idx)) * self.dissolve_stagger / self.dissolve_total_duration
            letter_end = letter_start + self.dissolve_duration / self.dissolve_total_duration
            
            if t < letter_start:
                # Letter hasn't started dissolving yet
                alpha_mult = 0.2
                scale = 1.0
                float_y = 0
                add_holes = False
            elif t > letter_end:
                # Letter has fully dissolved
                continue
            else:
                # Letter is dissolving
                letter_t = (t - letter_start) / (letter_end - letter_start)
                smooth_t = self._smoothstep(letter_t)
                
                # Effects
                alpha_mult = 0.2 * (1.0 - smooth_t)
                scale = 1.0 + smooth_t * (self.max_dissolve_scale - 1.0)
                float_y = -smooth_t * self.float_distance
                add_holes = letter_t > 0.3
                
                if add_holes:
                    self._add_3d_dissolve_holes(idx, letter_t)
            
            # Get the sprite
            sprite = state.sprite_3d.copy()
            
            # Use EXACT position from motion phase
            pos_x, pos_y = state.position
            
            # Apply scale transformation
            if scale != 1.0:
                new_size = (int(sprite.width * scale), int(sprite.height * scale))
                sprite = sprite.resize(new_size, Image.Resampling.LANCZOS)
                # Adjust position to keep centered
                pos_x -= (new_size[0] - state.sprite_3d.width) // 2
                pos_y -= (new_size[1] - state.sprite_3d.height) // 2
            
            # Apply float
            pos_y += int(float_y)
            
            # Apply alpha and kill mask
            sprite_array = np.array(sprite)
            
            if idx in self.letter_kill_masks and np.any(self.letter_kill_masks[idx]):
                kill_mask = self.letter_kill_masks[idx]
                if scale != 1.0:
                    kill_mask = cv2.resize(kill_mask, (sprite.width, sprite.height))
                
                # Apply kill mask
                sprite_array[:, :, 3] = sprite_array[:, :, 3] * (1 - kill_mask)
            
            # Apply overall alpha
            sprite_array[:, :, 3] = (sprite_array[:, :, 3] * alpha_mult).astype(np.uint8)
            
            # Convert back to PIL and composite
            sprite = Image.fromarray(sprite_array)
            
            # Paste at exact position
            canvas.paste(sprite, (int(pos_x), int(pos_y)), sprite)
        
        result = np.array(canvas)
        return result[:, :, :3] if result.shape[2] == 4 else result
    
    def generate_frame(self, frame_number: int, background: np.ndarray) -> np.ndarray:
        """Generate a single frame of the animation."""
        frame = background.copy()
        
        # Ensure RGBA
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        
        # Determine which phase we're in
        if frame_number < self.motion_frames:
            # Motion phase
            return self._generate_motion_frame(frame_number, frame, background)
        else:
            # Dissolve phase
            dissolve_frame = frame_number - self.motion_frames
            if dissolve_frame < self.dissolve_frames:
                return self._generate_3d_dissolve_frame(dissolve_frame, frame, background)
            else:
                # Animation complete
                return background[:, :, :3] if background.shape[2] == 4 else background
    
    def generate_video(self, output_path: str, background_frames: List[np.ndarray]):
        """Generate complete video with the animation."""
        height, width = background_frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        total_frames_needed = self.total_frames
        
        for i in range(total_frames_needed):
            bg_idx = i % len(background_frames)
            background = background_frames[bg_idx]
            
            frame = self.generate_frame(i, background)
            
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved to {output_path}")