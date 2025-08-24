#!/usr/bin/env python3
"""
3D text motion animation with shrinking and moving behind subject.
Extracted from Text3DMotionDissolve to be a standalone, reusable animation.

Fixes applied:
- Correct centering by front-face center (not canvas top-left).
- Consistent anchor computation with depth margins.
- [POS_HANDOFF] debug logs to verify handoff to dissolve.
- [TEXT_QUALITY] Robust TTF/OTF font discovery + optional --font path.
- [TEXT_QUALITY] Logs to verify font and supersampling.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class MotionState:
    """State captured at the end of motion animation for handoff to next animation."""
    scale: float
    position: Tuple[int, int]           # Top-left position used during final composite (debug)
    text_size: Tuple[int, int]          # Rendered sprite size (debug)
    center_position: Tuple[int, int]    # Intended front-face center (what dissolve should use)
    is_behind: bool                      # Whether text is behind subject at end of motion


class Text3DMotion:
    """
    3D text animation that shrinks and moves behind a subject.

    This class handles:
    - 3D text rendering with depth layers
    - Smooth shrinking from large to small
    - Movement from start position to end position (interpreted as FRONT-FACE CENTER)
    - Occlusion when passing behind subject
    - Dynamic mask recalculation for moving subjects
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
        start_position: Optional[Tuple[int, int]] = None,  # FRONT-FACE CENTER
        end_position: Optional[Tuple[int, int]] = None,    # FRONT-FACE CENTER
        shrink_duration: float = 0.8,
        settle_duration: float = 0.2,
        final_alpha: float = 0.3,  # Final opacity when behind subject (0.0-1.0)
        shadow_offset: int = 5,
        outline_width: int = 2,
        perspective_angle: float = 0,
        supersample_factor: int = 2,
        glow_effect: bool = True,
        font_path: Optional[str] = None,   # NEW: explicit font path
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

        # Positions are interpreted as FRONT-FACE CENTER of the text (not top-left).
        default_center = (resolution[0] // 2, resolution[1] // 2)
        self.start_position = start_position or default_center
        self.end_position = end_position or default_center

        self.shrink_duration = shrink_duration
        self.settle_duration = settle_duration
        self.final_alpha = final_alpha
        self.shadow_offset = shadow_offset
        self.outline_width = outline_width
        self.perspective_angle = perspective_angle
        self.supersample_factor = supersample_factor
        self.glow_effect = glow_effect
        self.font_path = font_path
        self.debug = debug

        # Cache for dynamic masks
        self._frame_mask_cache: Dict[int, np.ndarray] = {}

        # Final state for handoff
        self._final_state: Optional[MotionState] = None
        
        if self.debug:
            print(f"[TEXT_QUALITY] Supersample factor: {self.supersample_factor}")

    def _log(self, message: str):
        """Debug logging (required format)."""
        if self.debug:
            print(f"[POS_HANDOFF] {message}")

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

    def _smoothstep(self, t: float) -> float:
        """Smooth interpolation function."""
        t = max(0, min(1, t))
        return t * t * (3 - 2 * t)

    def _render_3d_text(
        self,
        text: str,
        scale: float,
        alpha: float,
        depth_scale: float
    ) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int]]:
        """
        Render 3D text with depth layers.

        Returns:
            canvas (PIL.Image)
            anchor (ax, ay): FRONT-FACE top-left *inside* canvas coordinates (post downsample)
            front_size (fw, fh): FRONT-FACE bbox size (post downsample)
        """
        from PIL import ImageFilter
        
        # Work at supersampled resolution for quality
        font_px = int(self.font_size * scale * self.supersample_factor)
        font = self._get_font(font_px)

        # FRONT-FACE bbox (no depth)
        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), text, font=font)  # (l, t, r, b)
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

        # Margin to accommodate depth layers on both sides (symmetric canvas)
        margin = int(self.depth_offset * self.depth_layers * self.supersample_factor)

        width = bbox_w + 2 * margin
        height = bbox_h + 2 * margin

        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # Render depth layers back-to-front
        for i in range(self.depth_layers - 1, -1, -1):
            depth_alpha = int(alpha * 255 * (0.3 + 0.7 * (1 - i / self.depth_layers)))
            offset = int(i * self.depth_offset * depth_scale * self.supersample_factor)
            if i == 0:
                color = (*self.text_color, depth_alpha)
            else:
                factor = 0.7 - (i / self.depth_layers) * 0.4
                color = tuple(int(c * factor) for c in self.depth_color) + (depth_alpha,)

            # Draw with symmetric margin + per-layer offset (only +x,+y)
            x = -bbox[0] + margin + offset
            y = -bbox[1] + margin + offset
            
            # Add stroke for front layer to improve antialiasing
            if i == 0 and self.supersample_factor >= 4:
                stroke_width = max(1, self.supersample_factor // 8)
                draw.text((x, y), text, font=font, fill=color, stroke_width=stroke_width, stroke_fill=color)
            else:
                draw.text((x, y), text, font=font, fill=color)

        # Apply Gaussian blur for antialiasing before downsampling
        if self.supersample_factor >= 4:
            blur_radius = self.supersample_factor / 5.0
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Progressive downsampling for better quality
        if self.supersample_factor >= 8:
            # Two-step downsampling for very high supersample factors
            intermediate_size = (width // (self.supersample_factor // 2), height // (self.supersample_factor // 2))
            canvas = canvas.resize(intermediate_size, Image.Resampling.LANCZOS)
            
            new_size = (intermediate_size[0] // 2, intermediate_size[1] // 2)
            canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)
            # Convert supersampled coordinates to final coordinates
            ax = int(round((-bbox[0] + margin) / self.supersample_factor))
            ay = int(round((-bbox[1] + margin) / self.supersample_factor))
            fw = int(round(bbox_w / self.supersample_factor))
            fh = int(round(bbox_h / self.supersample_factor))
        elif self.supersample_factor > 1:
            new_size = (width // self.supersample_factor, height // self.supersample_factor)
            canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)
            # Convert supersampled coordinates to final coordinates
            ax = int(round((-bbox[0] + margin) / self.supersample_factor))
            ay = int(round((-bbox[1] + margin) / self.supersample_factor))
            fw = int(round(bbox_w / self.supersample_factor))
            fh = int(round(bbox_h / self.supersample_factor))
        else:
            ax = -bbox[0] + margin
            ay = -bbox[1] + margin
            fw = bbox_w
            fh = bbox_h

        return canvas, (ax, ay), (fw, fh)

    def generate_frame(self, frame_number: int, background: np.ndarray) -> np.ndarray:
        """Generate a single frame of the motion animation."""
        frame = background.copy()

        # Ensure RGBA
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

        # Progress
        t_global = frame_number / max(self.total_frames - 1, 1)
        smooth_t_global = self._smoothstep(t_global)

        # Scale phases
        shrink_progress = self.shrink_duration / self.duration
        if smooth_t_global <= shrink_progress:
            local_t = smooth_t_global / shrink_progress
            scale = self.start_scale - local_t * (self.start_scale - self.end_scale)
            depth_scale = 1.0
            is_behind = local_t > 0.5
            # Fade from full opacity to final_alpha during shrink
            base_alpha = 1.0 if local_t <= 0.5 else max(self.final_alpha, 1.0 - (local_t - 0.5) * (2.0 * (1.0 - self.final_alpha)))
        else:
            # Settle phase - maintain final_alpha
            local_t = (smooth_t_global - shrink_progress) / (1.0 - shrink_progress)
            scale = self.end_scale - local_t * (self.end_scale - self.final_scale)
            depth_scale = 1.0
            is_behind = True
            base_alpha = self.final_alpha

        # Render text (get front-face anchor + size)
        text_pil, (anchor_x, anchor_y), (front_w, front_h) = self._render_3d_text(
            self.text, scale, base_alpha, depth_scale
        )

        # Interpolate FRONT-FACE CENTER from start to end
        cx = self.start_position[0] + smooth_t_global * (self.end_position[0] - self.start_position[0])
        cy = self.start_position[1] + smooth_t_global * (self.end_position[1] - self.start_position[1])

        # Place such that FRONT-FACE CENTER == (cx, cy)
        pos_x = int(round(cx - (anchor_x + front_w / 2.0)))
        pos_y = int(round(cy - (anchor_y + front_h / 2.0)))

        # Store final state for handoff (last frame)
        if frame_number == self.total_frames - 1:
            self._final_state = MotionState(
                scale=scale,
                position=(pos_x, pos_y),
                text_size=(text_pil.width, text_pil.height),
                center_position=(int(round(cx)), int(round(cy))),
                is_behind=is_behind,
            )
            self._log(
                f"Motion final snapshot -> center=({cx:.1f},{cy:.1f}), "
                f"front_size=({front_w},{front_h}), anchor=({anchor_x},{anchor_y}), "
                f"paste_topleft=({pos_x},{pos_y}), scale={scale:.3f}, is_behind={is_behind}"
            )

        # Composite onto frame
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

            # Apply mask
            mask_region = current_mask[y1:y2, x1:x2]
            text_alpha = text_layer[y1:y2, x1:x2, 3].astype(np.float32)
            mask_factor = mask_region.astype(np.float32) / 255.0
            text_alpha *= (1.0 - mask_factor)
            text_layer[y1:y2, x1:x2, 3] = text_alpha.astype(np.uint8)

        # Composite
        frame_pil = Image.fromarray(frame)
        text_pil_img = Image.fromarray(text_layer)
        out = Image.alpha_composite(frame_pil, text_pil_img)
        result = np.array(out)

        return result[:, :, :3] if result.shape[2] == 4 else result

    def get_final_state(self) -> Optional[MotionState]:
        """Get the final state for handoff to next animation."""
        return self._final_state