#!/usr/bin/env python3
"""
3D letter dissolve animation where each letter dissolves individually.
Extracted from Text3DMotionDissolve to be a standalone, reusable animation.

Fixes applied:
- Pre-render letters with symmetric depth margin (matches Text3DMotion).
- Position each sprite at (intended_front_xy - anchor) to avoid jump.
- [POS_HANDOFF] debug logs to prove positions are frozen and consistent.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, List, Dict
import random
from dataclasses import dataclass


@dataclass
class LetterSprite:
    """Individual letter sprite with its 3D rendering and position."""
    char: str
    sprite_3d: Optional[Image.Image]
    position: Tuple[int, int]   # Top-left where we paste the sprite image
    width: int
    height: int
    anchor: Tuple[int, int]     # FRONT-FACE top-left inside sprite coordinates


class Letter3DDissolve:
    """
    3D letter-by-letter dissolve animation.

    This class handles:
    - Individual letter 3D rendering with depth
    - Letter-by-letter dissolve with staggered timing
    - Float and scale effects during dissolve
    - Dissolve holes/particles effect
    - Customizable dissolve order (sequential or random)

    All positions are defined in terms of the FRONT-FACE text layout.
    """

    def __init__(
        self,
        duration: float = 1.5,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        text: str = "HELLO",
        font_size: int = 120,
        text_color: Tuple[int, int, int] = (255, 220, 0),
        depth_color: Tuple[int, int, int] = (200, 170, 0),
        depth_layers: int = 8,
        depth_offset: int = 3,
        initial_scale: float = 0.9,   # Scale at start of dissolve (from motion)
        initial_position: Optional[Tuple[int, int]] = None,  # FRONT-FACE CENTER of text
        stable_duration: float = 0.2,     # How long to show before dissolving
        dissolve_duration: float = 0.8,   # Per-letter dissolve time
        dissolve_stagger: float = 0.1,    # Delay between letters starting
        float_distance: float = 50,       # How far letters float up
        max_dissolve_scale: float = 1.3,  # Max scale during dissolve
        randomize_order: bool = False,    # Random vs sequential dissolve
        shadow_offset: int = 5,
        outline_width: int = 2,
        supersample_factor: int = 2,
        debug: bool = False,
    ):
        self.duration = duration
        self.fps = fps
        self.total_frames = int(duration * fps)
        self.resolution = resolution
        self.text = text
        self.font_size = font_size
        self.text_color = text_color
        self.depth_color = depth_color
        self.depth_layers = depth_layers
        self.depth_offset = depth_offset
        self.initial_scale = initial_scale
        self.initial_position = initial_position or (resolution[0] // 2, resolution[1] // 2)
        self.stable_duration = stable_duration
        self.dissolve_duration = dissolve_duration
        self.dissolve_stagger = dissolve_stagger
        self.float_distance = float_distance
        self.max_dissolve_scale = max_dissolve_scale
        self.randomize_order = randomize_order
        self.shadow_offset = shadow_offset
        self.outline_width = outline_width
        self.supersample_factor = supersample_factor
        self.debug = debug

        # Letter sprites and positions
        self.letter_sprites: List[LetterSprite] = []
        self.dissolve_order: List[int] = []
        self.letter_kill_masks: Dict[int, np.ndarray] = {}

        # Initialize letter sprites and dissolve order
        self._prepare_letter_sprites()
        self._init_dissolve_order()

    def _log(self, message: str):
        """Debug logging (required format)."""
        if self.debug:
            print(f"[POS_HANDOFF] {message}")

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
        """
        Render a single 3D letter with depth layers.

        Returns:
            canvas (PIL.Image)
            anchor (ax, ay): FRONT-FACE top-left inside the canvas (post downsample)
        """
        font_px = int(self.font_size * scale * self.supersample_factor)
        font = self._get_font(font_px)

        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), letter, font=font)
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

        # Symmetric margin to accommodate depth
        margin = int(self.depth_offset * self.depth_layers * self.supersample_factor)
        width = bbox_w + 2 * margin
        height = bbox_h + 2 * margin

        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        for i in range(self.depth_layers - 1, -1, -1):
            depth_alpha = int(alpha * 255 * (0.3 + 0.7 * (1 - i / self.depth_layers)))
            offset = int(i * self.depth_offset * depth_scale * self.supersample_factor)

            if i == 0:
                color = (*self.text_color, depth_alpha)
            else:
                factor = 0.7 - (i / self.depth_layers) * 0.4
                color = tuple(int(c * factor) for c in self.depth_color) + (depth_alpha,)

            x = -bbox[0] + margin + offset
            y = -bbox[1] + margin + offset
            draw.text((x, y), letter, font=font, fill=color)

        if self.supersample_factor > 1:
            new_size = (width // self.supersample_factor, height // self.supersample_factor)
            canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)
            ax = int(round((-bbox[0] + margin) / self.supersample_factor))
            ay = int(round((-bbox[1] + margin) / self.supersample_factor))
        else:
            ax = -bbox[0] + margin
            ay = -bbox[1] + margin

        return canvas, (ax, ay)

    def _prepare_letter_sprites(self):
        """Pre-render individual letter sprites and calculate positions (front-face accurate)."""
        # Use non-supersampled font here just to compute layout metrics (front-face)
        font_px = int(self.font_size * self.initial_scale)
        font = self._get_font(font_px)

        # Measure full text (front-face) to compute centered layout
        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        full_bbox = d.textbbox((0, 0), self.text, font=font)
        text_width = full_bbox[2] - full_bbox[0]
        text_height = full_bbox[3] - full_bbox[1]

        # Center the FRONT-FACE bbox at the initial center
        cx, cy = self.initial_position
        start_x = cx - text_width // 2
        start_y = cy - text_height // 2

        current_x = start_x
        visible_positions: List[Tuple[int, int]] = []

        self.letter_sprites = []
        for i, letter in enumerate(self.text):
            if letter == ' ':
                # Approximate space advance
                space_width = font_px // 3
                sprite = LetterSprite(
                    char=letter,
                    sprite_3d=None,
                    position=(current_x, start_y),
                    width=space_width,
                    height=0,
                    anchor=(0, 0)
                )
                self.letter_sprites.append(sprite)
                visible_positions.append((current_x, start_y))
                current_x += space_width
            else:
                # Get letter advance (front-face)
                letter_bbox = d.textbbox((0, 0), letter, font=font)
                advance = letter_bbox[2] - letter_bbox[0]

                # Render 3D sprite and anchor (sprite includes depth margins)
                sprite_3d, (ax, ay) = self._render_3d_letter(
                    letter, self.initial_scale, 1.0, 1.0
                )

                # We want the FRONT-FACE top-left at (current_x, start_y).
                # Therefore, when pasting the sprite image, we must subtract the anchor.
                paste_x = current_x - ax
                paste_y = start_y - ay

                sprite = LetterSprite(
                    char=letter,
                    sprite_3d=sprite_3d,
                    position=(paste_x, paste_y),
                    width=sprite_3d.width if sprite_3d else 0,
                    height=sprite_3d.height if sprite_3d else 0,
                    anchor=(ax, ay)
                )
                self.letter_sprites.append(sprite)

                visible_positions.append((current_x, start_y))
                current_x += advance

        self._log(
            f"Dissolve layout -> center={self.initial_position}, front_text_bbox=({text_width},{text_height}), "
            f"start_topleft=({start_x},{start_y})"
        )
        self._log(f"Letter positions frozen at: {visible_positions}")

    def _init_dissolve_order(self):
        """Initialize the order in which letters dissolve."""
        if self.randomize_order:
            indices = list(range(len(self.text)))
            random.shuffle(indices)
            self.dissolve_order = indices
        else:
            self.dissolve_order = list(range(len(self.text)))

    def _add_dissolve_holes(self, letter_idx: int, progress: float):
        """Add dissolve holes to a letter's kill mask."""
        if letter_idx >= len(self.letter_sprites):
            return

        sprite = self.letter_sprites[letter_idx]
        if sprite.sprite_3d is None:
            return

        if letter_idx not in self.letter_kill_masks:
            self.letter_kill_masks[letter_idx] = np.zeros(
                (sprite.sprite_3d.height, sprite.sprite_3d.width), dtype=np.uint8
            )

        num_holes = int(progress * 20)
        for _ in range(num_holes):
            x = np.random.randint(0, sprite.sprite_3d.width)
            y = np.random.randint(0, sprite.sprite_3d.height)
            radius = np.random.randint(2, 8)
            cv2.circle(self.letter_kill_masks[letter_idx], (x, y), radius, 1, -1)

    def set_initial_state(self, scale: float, position: Tuple[int, int]):
        """
        Set initial state from previous animation (e.g., Text3DMotion).
        This allows seamless transition between animations.
        """
        self.initial_scale = scale
        self.initial_position = position
        self.letter_sprites = []
        self._log(f"Received handoff -> center={position}, scale={scale:.3f}")
        self._prepare_letter_sprites()

    def generate_frame(self, frame_number: int, background: np.ndarray) -> np.ndarray:
        """Generate a single frame of the dissolve animation."""
        frame = background.copy()

        # Ensure RGBA
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

        # Timeline
        t = frame_number / max(self.total_frames - 1, 1)
        canvas = Image.fromarray(frame)

        # Helpful one-time log
        if self.debug and frame_number == 0 and self.letter_sprites:
            s0 = self.letter_sprites[0]
            self._log(
                f"Frame0 check -> first_letter '{s0.char}' paste_topleft={s0.position}, anchor={s0.anchor}"
            )

        # Process each letter
        for idx in self.dissolve_order:
            if idx >= len(self.letter_sprites):
                continue

            sprite = self.letter_sprites[idx]
            if sprite.sprite_3d is None:
                continue

            # Per-letter timing window
            letter_order_idx = self.dissolve_order.index(idx)
            letter_start = (self.stable_duration + letter_order_idx * self.dissolve_stagger) / self.duration
            letter_end = letter_start + self.dissolve_duration / self.duration

            if t < letter_start:
                # Show stable (match motion's final alpha ~0.2)
                alpha_mult = 0.2
                scale = 1.0
                float_y = 0
                add_holes = False
            elif t > letter_end:
                # Already dissolved
                continue
            else:
                # Dissolving now
                letter_t = (t - letter_start) / (letter_end - letter_start)
                smooth_t = self._smoothstep(letter_t)

                alpha_mult = 0.2 * (1.0 - smooth_t)
                scale = 1.0 + smooth_t * (self.max_dissolve_scale - 1.0)
                float_y = -smooth_t * self.float_distance
                add_holes = letter_t > 0.3

                if add_holes:
                    self._add_dissolve_holes(idx, letter_t)

            sprite_img = sprite.sprite_3d.copy()

            # Scale around sprite center (visually pleasant). Since we placed the
            # front-face correctly at frame 0, small re-centering during dissolve is OK.
            pos_x, pos_y = sprite.position
            if scale != 1.0:
                new_w = int(round(sprite_img.width * scale))
                new_h = int(round(sprite_img.height * scale))
                sprite_img = sprite_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # Keep visual center of sprite stable during scale
                pos_x -= (new_w - sprite.sprite_3d.width) // 2
                pos_y -= (new_h - sprite.sprite_3d.height) // 2

            # Float upwards
            pos_y += int(round(float_y))

            # Convert to array for alpha edits
            sprite_array = np.array(sprite_img)

            # Apply kill mask if any
            if idx in self.letter_kill_masks and np.any(self.letter_kill_masks[idx]):
                kill_mask = self.letter_kill_masks[idx]
                if scale != 1.0:
                    kill_mask = cv2.resize(kill_mask, (sprite_img.width, sprite_img.height))
                sprite_array[:, :, 3] = (sprite_array[:, :, 3] * (1 - kill_mask)).astype(np.uint8)

            # Overall alpha
            sprite_array[:, :, 3] = (sprite_array[:, :, 3] * alpha_mult).astype(np.uint8)
            sprite_img = Image.fromarray(sprite_array)

            # Paste
            canvas.paste(sprite_img, (int(pos_x), int(pos_y)), sprite_img)

        result = np.array(canvas)
        return result[:, :, :3] if result.shape[2] == 4 else result