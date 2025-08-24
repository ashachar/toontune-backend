#!/usr/bin/env python3
"""
3D letter dissolve animation where each letter dissolves individually.

FRAME-ACCURATE FIXES:
- Per-letter schedule computed in integer frames (start/end/fade_end).
- Guaranteed minimum fade frames (prevents skipped fades at low FPS).
- 1-frame "safety hold" at dissolve start so alpha begins EXACTLY at stable_alpha.
- Optional overlap guard: previous letter's fade extends at least until next letter's start.

DEBUG:
- [JUMP_CUT] logs print the full schedule and per-frame transitions.
- [POS_HANDOFF] logs from motion remain supported.
- [TEXT_QUALITY] Robust TTF/OTF font discovery + optional --font path.
- [TEXT_QUALITY] Logs to verify font and supersampling.

Original authorship retained; this refactor targets the jump-cut described in the issue.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, List, Dict
import random
from dataclasses import dataclass

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class LetterSprite:
    """Individual letter sprite with its 3D rendering and position."""
    char: str
    sprite_3d: Optional[Image.Image]
    position: Tuple[int, int]   # paste top-left
    width: int
    height: int
    anchor: Tuple[int, int]     # FRONT-FACE top-left inside sprite


@dataclass
class _LetterTiming:
    """Frame-accurate per-letter schedule."""
    start: int           # first frame letter is DEFINITELY drawn at stable_alpha
    hold_end: int        # last frame of the "safety hold" at stable_alpha (>= start)
    end: int             # last frame of the dissolve window (after hold)
    fade_end: int        # last frame of the fade-out tail
    order_index: int     # 0..N-1 sequential in dissolve_order


class Letter3DDissolve:
    """
    3D letter-by-letter dissolve animation with frame-accurate timing.

    Key parameters:
    - duration: total dissolve clip length (seconds)
    - dissolve_duration: time a letter spends in the dissolve window (seconds)
    - dissolve_stagger: delay between starts (seconds)
    - post_fade_seconds: additional fade after dissolve (seconds) -> converted to frames (min 2)
    - pre_dissolve_hold_frames: safety frames at stable_alpha before letter starts to change
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
        initial_scale: float = 0.9,                   # handoff scale
        initial_position: Optional[Tuple[int, int]] = None,  # FRONT-FACE CENTER
        stable_duration: float = 0.2,                 # pre-letter dissolve lead-in (sec)
        stable_alpha: float = 0.3,                    # opacity when not dissolving (0..1)
        dissolve_duration: float = 0.8,               # per-letter dissolve (sec)
        dissolve_stagger: float = 0.1,                # delay between letter starts (sec)
        float_distance: float = 50,
        max_dissolve_scale: float = 1.3,
        randomize_order: bool = False,
        segment_mask: Optional[np.ndarray] = None,
        is_behind: bool = False,
        shadow_offset: int = 5,
        outline_width: int = 2,
        supersample_factor: int = 2,
        # --- New robustness knobs ---
        post_fade_seconds: float = 0.10,              # tail after dissolve (sec); min 2 frames
        pre_dissolve_hold_frames: int = 1,            # hold N frames at stable_alpha at start
        ensure_no_gap: bool = True,                   # extend previous fade to next start
        font_path: Optional[str] = None,              # NEW: explicit font path
        debug: bool = False,
    ):
        self.duration = duration
        self.fps = fps
        self.total_frames = int(round(duration * fps))
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
        self.stable_alpha = max(0.0, min(1.0, stable_alpha))
        self.dissolve_duration = dissolve_duration
        self.dissolve_stagger = dissolve_stagger
        self.float_distance = float_distance
        self.max_dissolve_scale = max(1.0, max_dissolve_scale)
        self.randomize_order = randomize_order
        self.segment_mask = segment_mask
        self.is_behind = is_behind
        self.shadow_offset = shadow_offset
        self.outline_width = outline_width
        self.supersample_factor = supersample_factor
        self.post_fade_seconds = max(0.0, post_fade_seconds)
        self.pre_dissolve_hold_frames = max(0, int(pre_dissolve_hold_frames))
        self.ensure_no_gap = ensure_no_gap
        self.font_path = font_path
        self.debug = debug

        # Runtime
        self.letter_sprites: List[LetterSprite] = []
        self.dissolve_order: List[int] = []
        self.letter_kill_masks: Dict[int, np.ndarray] = {}
        self._frame_mask_cache: Dict[int, np.ndarray] = {}
        self._timeline: Dict[int, _LetterTiming] = {}
        self._entered_dissolve_logged: Dict[int, bool] = {}
        self._hold_logged: Dict[int, bool] = {}

        # Build sprites and schedule
        self._prepare_letter_sprites()
        self._init_dissolve_order()
        self._build_frame_timeline()
        
        if self.debug:
            print(f"[TEXT_QUALITY] Supersample factor: {self.supersample_factor}")

    # -----------------------------
    # Utilities / logging
    # -----------------------------
    def _log_pos(self, message: str):
        if self.debug:
            print(f"[POS_HANDOFF] {message}")

    def _log_jump(self, message: str):
        if self.debug:
            print(f"[JUMP_CUT] {message}")

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get font at specified size - prioritize vector fonts."""
        candidates = []
        if self.font_path:
            candidates.append(self.font_path)
        
        # Environment overrides
        for key in ("T3D_FONT", "TEXT_FONT", "FONT_PATH"):
            p = os.environ.get(key)
            if p:
                candidates.append(p)
        
        # Common cross-OS paths
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        
        for p in candidates:
            try:
                if p and os.path.isfile(p):
                    if self.debug:
                        print(f"[TEXT_QUALITY] Using TTF font: {p}")
                    return ImageFont.truetype(p, size)
            except Exception:
                continue
        
        if self.debug:
            print("[TEXT_QUALITY] WARNING: Falling back to PIL bitmap font (edges may look jagged). "
                  "Install a TTF/OTF and pass font_path parameter.")
        return ImageFont.load_default()

    @staticmethod
    def _smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3 - 2 * t)

    # -----------------------------
    # Rendering helpers
    # -----------------------------
    def _render_3d_letter(
        self, letter: str, scale: float, alpha: float, depth_scale: float
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        from PIL import ImageFilter
        
        font_px = int(self.font_size * scale * self.supersample_factor)
        font = self._get_font(font_px)

        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), letter, font=font)
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

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
            
            # Add stroke for front layer to improve antialiasing
            if i == 0 and self.supersample_factor >= 4:
                stroke_width = max(1, self.supersample_factor // 8)
                draw.text((x, y), letter, font=font, fill=color, stroke_width=stroke_width, stroke_fill=color)
            else:
                draw.text((x, y), letter, font=font, fill=color)

        # Apply Gaussian blur for antialiasing before downsampling
        if self.supersample_factor >= 4:
            blur_radius = self.supersample_factor / 5.0
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Progressive downsampling for better quality
        if self.supersample_factor >= 8:
            # Two-step downsampling for very high supersample factors
            intermediate_size = (width // (self.supersample_factor // 2), height // (self.supersample_factor // 2))
            canvas = canvas.resize(intermediate_size, Image.Resampling.LANCZOS)
            
            final_size = (intermediate_size[0] // 2, intermediate_size[1] // 2)
            canvas = canvas.resize(final_size, Image.Resampling.LANCZOS)
            
            ax = int(round((-bbox[0] + margin) / self.supersample_factor))
            ay = int(round((-bbox[1] + margin) / self.supersample_factor))
        elif self.supersample_factor > 1:
            new_size = (width // self.supersample_factor, height // self.supersample_factor)
            canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)
            ax = int(round((-bbox[0] + margin) / self.supersample_factor))
            ay = int(round((-bbox[1] + margin) / self.supersample_factor))
        else:
            ax = -bbox[0] + margin
            ay = -bbox[1] + margin

        return canvas, (ax, ay)

    # -----------------------------
    # Layout & order
    # -----------------------------
    def _prepare_letter_sprites(self):
        """Pre-render letter sprites and compute front-face layout."""
        font_px = int(self.font_size * self.initial_scale)
        font = self._get_font(font_px)

        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        full_bbox = d.textbbox((0, 0), self.text, font=font)
        text_width = full_bbox[2] - full_bbox[0]
        text_height = full_bbox[3] - full_bbox[1]

        cx, cy = self.initial_position
        start_x = cx - text_width // 2
        start_y = cy - text_height // 2

        current_x = start_x
        visible_positions: List[Tuple[int, int]] = []

        self.letter_sprites = []
        for letter in self.text:
            if letter == ' ':
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
                letter_bbox = d.textbbox((0, 0), letter, font=font)
                advance = letter_bbox[2] - letter_bbox[0]

                sprite_3d, (ax, ay) = self._render_3d_letter(letter, self.initial_scale, 1.0, 1.0)
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

        self._log_pos(
            f"Dissolve layout -> center={self.initial_position}, front_text_bbox=({text_width},{text_height}), "
            f"start_topleft=({start_x},{start_y})"
        )
        self._log_pos(f"Letter positions frozen at: {visible_positions}")

    def _init_dissolve_order(self):
        if self.randomize_order:
            indices = [i for i, ch in enumerate(self.text) if ch != ' ']
            random.shuffle(indices)
            self.dissolve_order = indices
        else:
            self.dissolve_order = [i for i, ch in enumerate(self.text) if ch != ' ']
        self._log_pos(f"Dissolve order (excluding spaces): {self.dissolve_order}")

    # -----------------------------
    # Frame-accurate schedule
    # -----------------------------
    def _build_frame_timeline(self):
        """Compute per-letter schedule in integer frames and log it."""
        min_fade_frames = max(2, int(round(self.post_fade_seconds * self.fps)))
        dissolve_frames = max(1, int(round(self.dissolve_duration * self.fps)))
        self._timeline.clear()
        self._entered_dissolve_logged.clear()
        self._hold_logged.clear()

        # 1) initial pass
        for order_idx, letter_idx in enumerate(self.dissolve_order):
            start_seconds = self.stable_duration + order_idx * self.dissolve_stagger
            start_frame = int(round(start_seconds * self.fps))
            hold_end = start_frame + max(0, self.pre_dissolve_hold_frames - 1)
            end_frame = hold_end + dissolve_frames  # dissolve begins AFTER hold
            fade_end = end_frame + min_fade_frames

            # clamp into clip range
            start_frame = max(0, min(self.total_frames - 1, start_frame))
            hold_end = max(start_frame, min(self.total_frames - 1, hold_end))
            end_frame = max(hold_end, min(self.total_frames - 1, end_frame))
            fade_end = max(end_frame, min(self.total_frames - 1, fade_end))

            self._timeline[letter_idx] = _LetterTiming(
                start=start_frame,
                hold_end=hold_end,
                end=end_frame,
                fade_end=fade_end,
                order_index=order_idx
            )

        # 2) Optional pass to prevent gaps: ensure prev.fade_end >= next.start
        if self.ensure_no_gap and len(self.dissolve_order) > 1:
            for i in range(len(self.dissolve_order) - 1):
                a = self._timeline[self.dissolve_order[i]]
                b = self._timeline[self.dissolve_order[i + 1]]
                if a.fade_end < b.start:
                    # extend a.fade_end up to b.start
                    new_fade_end = b.start
                    self._timeline[self.dissolve_order[i]] = _LetterTiming(
                        start=a.start, hold_end=a.hold_end, end=a.end,
                        fade_end=new_fade_end, order_index=a.order_index
                    )
                    self._log_jump(
                        f"Extended fade: letter#{i} fade_end {a.fade_end} -> {new_fade_end} to meet next.start {b.start}"
                    )

        # 3) Log schedule
        lines = []
        for i, idx in enumerate(self.dissolve_order):
            t = self._timeline[idx]
            ch = self.letter_sprites[idx].char if 0 <= idx < len(self.letter_sprites) else '?'
            lines.append(
                f"[JUMP_CUT] schedule[{i}] '{ch}' idx={idx}: start={t.start}, hold_end={t.hold_end}, "
                f"end={t.end}, fade_end={t.fade_end}"
            )
        if self.debug:
            print("\n".join(lines))

    def debug_print_schedule(self):
        """Public helper for tests."""
        self._build_frame_timeline()

    # -----------------------------
    # External handoff
    # -----------------------------
    def set_initial_state(self, scale: float, position: Tuple[int, int], alpha: float = None,
                          is_behind: bool = None, segment_mask: np.ndarray = None):
        self.initial_scale = scale
        self.initial_position = position
        if alpha is not None:
            self.stable_alpha = max(0.0, min(1.0, alpha))
        if is_behind is not None:
            self.is_behind = is_behind
        if segment_mask is not None:
            self.segment_mask = segment_mask
        self.letter_sprites = []
        self._log_pos(f"Received handoff -> center={position}, scale={scale:.3f}, "
                      f"alpha={self.stable_alpha:.3f}, is_behind={self.is_behind}")
        self._prepare_letter_sprites()
        self._build_frame_timeline()

    # -----------------------------
    # Kill mask helper
    # -----------------------------
    def _add_dissolve_holes(self, letter_idx: int, progress_0_1: float):
        sprite = self.letter_sprites[letter_idx]
        if sprite.sprite_3d is None:
            return
        if letter_idx not in self.letter_kill_masks:
            self.letter_kill_masks[letter_idx] = np.zeros(
                (sprite.sprite_3d.height, sprite.sprite_3d.width), dtype=np.uint8
            )
        num_holes = int(progress_0_1 * 20)
        for _ in range(num_holes):
            x = np.random.randint(0, sprite.sprite_3d.width)
            y = np.random.randint(0, sprite.sprite_3d.height)
            radius = np.random.randint(2, 8)
            cv2.circle(self.letter_kill_masks[letter_idx], (x, y), radius, 1, -1)

    # -----------------------------
    # Frame generation
    # -----------------------------
    def generate_frame(self, frame_number: int, background: np.ndarray) -> np.ndarray:
        frame = background.copy()
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

        # Optional dynamic mask when behind subject
        current_mask = None
        if self.is_behind and self.segment_mask is not None:
            if frame_number not in self._frame_mask_cache:
                try:
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
                    if self.debug and frame_number % 10 == 0:
                        self._log_pos(f"Dynamic mask extracted for frame {frame_number}")
                except Exception as e:
                    current_mask = self.segment_mask
                    if self.debug:
                        self._log_pos(f"Using static mask for frame {frame_number}: {e}")
            else:
                current_mask = self._frame_mask_cache[frame_number]

        canvas = Image.fromarray(frame)

        # Helpful first-frame log
        if self.debug and frame_number == 0 and self.letter_sprites:
            s0 = self.letter_sprites[self.dissolve_order[0]]
            self._log_pos(
                f"Frame0 check -> first letter '{s0.char}' paste_topleft={s0.position}, anchor={s0.anchor}"
            )

        # Process letters
        for idx in self.dissolve_order:
            sprite = self.letter_sprites[idx]
            if sprite.sprite_3d is None:
                continue

            timing = self._timeline[idx]
            f = frame_number

            # Determine phase by frame number (frame-accurate)
            if f < timing.start:
                phase = "stable"
                alpha_mult = self.stable_alpha
                scale = 1.0
                float_y = 0
                add_holes = False
            elif timing.start <= f <= timing.hold_end:
                # 1-frame (or more) safety hold at EXACT stable_alpha
                if not self._hold_logged.get(idx):
                    self._log_jump(
                        f"'{sprite.char}' enters HOLD at frame {f} (alpha={self.stable_alpha:.3f})"
                    )
                    self._hold_logged[idx] = True
                phase = "hold"
                alpha_mult = self.stable_alpha
                scale = 1.0
                float_y = 0
                add_holes = False
            elif timing.hold_end < f <= timing.end:
                # Dissolve begins AFTER hold; progress starts at ~0 on first dissolve frame
                denom = max(1, (timing.end - timing.hold_end))
                letter_t = (f - timing.hold_end) / denom
                smooth_t = self._smoothstep(letter_t)
                if not self._entered_dissolve_logged.get(idx):
                    self._log_jump(
                        f"'{sprite.char}' begins DISSOLVE at frame {f} "
                        f"(t={letter_t:.3f}, alpha≈{self.stable_alpha * (1.0 - 0.98 * 0):.3f})"
                    )
                    self._entered_dissolve_logged[idx] = True
                phase = "dissolve"
                alpha_mult = self.stable_alpha * (1.0 - smooth_t * 0.98)  # approaches ~0.02*stable_alpha
                scale = 1.0 + smooth_t * (self.max_dissolve_scale - 1.0)
                float_y = -smooth_t * self.float_distance
                add_holes = letter_t > 0.3
                if add_holes:
                    self._add_dissolve_holes(idx, letter_t)
            elif timing.end < f <= timing.fade_end:
                # Fade tail (guaranteed >= 2 frames)
                fade_denom = max(1, (timing.fade_end - timing.end))
                fade_t = (f - timing.end) / fade_denom
                phase = "fade"
                alpha_mult = self.stable_alpha * 0.02 * (1.0 - fade_t)
                scale = self.max_dissolve_scale
                float_y = -self.float_distance
                add_holes = True
                if idx not in self.letter_kill_masks:
                    # ensure it's "holey" in fade
                    self.letter_kill_masks[idx] = np.ones(
                        (sprite.sprite_3d.height, sprite.sprite_3d.width), dtype=np.uint8
                    )
            else:
                # Completely gone
                continue

            # Copy and transform sprite
            sprite_img = sprite.sprite_3d.copy()
            pos_x, pos_y = sprite.position

            if scale != 1.0:
                new_w = int(round(sprite_img.width * scale))
                new_h = int(round(sprite_img.height * scale))
                sprite_img = sprite_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                pos_x -= (new_w - sprite.sprite_3d.width) // 2
                pos_y -= (new_h - sprite.sprite_3d.height) // 2

            pos_y += int(round(float_y))
            sprite_array = np.array(sprite_img)

            # Apply kill mask if any (scaled to current sprite size)
            if idx in self.letter_kill_masks and np.any(self.letter_kill_masks[idx]):
                kill_mask = self.letter_kill_masks[idx]
                if (sprite_img.width, sprite_img.height) != (kill_mask.shape[1], kill_mask.shape[0]):
                    kill_mask = cv2.resize(kill_mask, (sprite_img.width, sprite_img.height),
                                           interpolation=cv2.INTER_NEAREST)
                sprite_array[:, :, 3] = (sprite_array[:, :, 3] * (1 - kill_mask)).astype(np.uint8)

            # Overall alpha
            sprite_array[:, :, 3] = (sprite_array[:, :, 3] * alpha_mult).astype(np.uint8)
            sprite_img = Image.fromarray(sprite_array)

            # Occlusion if behind subject
            if self.is_behind and current_mask is not None:
                sprite_np = np.array(sprite_img)
                sp_h, sp_w = sprite_np.shape[:2]
                y1 = max(0, int(pos_y)); y2 = min(self.resolution[1], int(pos_y) + sp_h)
                x1 = max(0, int(pos_x)); x2 = min(self.resolution[0], int(pos_x) + sp_w)
                sy1 = max(0, -int(pos_y)); sy2 = sy1 + (y2 - y1)
                sx1 = max(0, -int(pos_x)); sx2 = sx1 + (x2 - x1)

                if y2 > y1 and x2 > x1:
                    mask_region = current_mask[y1:y2, x1:x2]
                    sprite_alpha = sprite_np[sy1:sy2, sx1:sx2, 3].astype(np.float32)
                    mask_factor = mask_region.astype(np.float32) / 255.0
                    sprite_alpha *= (1.0 - mask_factor)
                    sprite_np[sy1:sy2, sx1:sx2, 3] = sprite_alpha.astype(np.uint8)
                    sprite_img = Image.fromarray(sprite_np)

            canvas.paste(sprite_img, (int(pos_x), int(pos_y)), sprite_img)

        result = np.array(canvas)
        return result[:, :, :3] if result.shape[2] == 4 else result