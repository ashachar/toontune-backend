"""
Text 3D Behind Segment (Improved)
---------------------------------

Fixes:
  • Pixelated 3D / jagged edges  → SSAA + LANCZOS + cubic perspective warp.
  • Focal point drift while shrinking → anchor the *front face* center and keep it aligned to `center_position`
    even after perspective transform.

Adds:
  • Debug logs with the tag: [3D_PIXELATED] ...
  • Backward-compatible API (render_3d_text(...) still returns PIL.Image)
  • Tunable supersample_factor

Usage hint:
  Set `center_position=(x, y)` to the *target center of the sentence* you want to lock onto.
"""

import os
from typing import List, Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2


class Text3DBehindSegment:
    """
    Animation where 3D text with depth moves from foreground to background behind a segmented object.

    Key params (unchanged names for compatibility):
      - text, segment_mask, font_size, font_path, text_color, depth_color, depth_layers, depth_offset,
        start_scale, end_scale, phase1_duration, phase2_duration, phase3_duration,
        center_position, shadow_offset, outline_width, perspective_angle

    New params:
      - supersample_factor: int = 3     # anti-aliasing quality (3–4 recommended)
      - debug: bool = True              # print [3D_PIXELATED] logs
      - perspective_during_shrink: bool = False  # keep shrink phase clean by default
    """

    def __init__(
        self,
        duration: float = 3.0,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        text: str = "START",
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
        perspective_angle: float = 25.0,
        # New:
        supersample_factor: int = 4,  # Increased for smoother depth on all letters
        debug: bool = True,
        perspective_during_shrink: bool = False,
        **kwargs,
    ):
        # Basic
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
        self.perspective_angle = float(perspective_angle)

        # New toggles
        self.supersample_factor = max(1, int(supersample_factor))
        self.debug = bool(debug)
        self.perspective_during_shrink = bool(perspective_during_shrink)

        # Segment mask
        self.segment_mask = self._load_or_make_mask(segment_mask)

        # Font (will be re-created per scale; we keep a path list to try)
        self._font_candidates = []
        if self.font_path and os.path.exists(self.font_path):
            self._font_candidates.append(self.font_path)
        # Reasonable fallbacks:
        self._font_candidates += [
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]

        # Phase frame counts
        self.phase1_frames = int(self.phase1_duration * self.fps)
        self.phase2_frames = int(self.phase2_duration * self.fps)
        self.phase3_frames = int(self.phase3_duration * self.fps)

        # Fit inside total_frames if needed
        total_phase_frames = self.phase1_frames + self.phase2_frames + self.phase3_frames
        if total_phase_frames > self.total_frames:
            scale_factor = self.total_frames / max(total_phase_frames, 1)
            self.phase1_frames = max(1, int(round(self.phase1_frames * scale_factor)))
            self.phase2_frames = max(1, int(round(self.phase2_frames * scale_factor)))
            self.phase3_frames = max(1, self.total_frames - self.phase1_frames - self.phase2_frames)

        self._log(f"Init: res={self.resolution}, fps={self.fps}, total_frames={self.total_frames}")
        self._log(f"Supersample={self.supersample_factor}, perspective_angle={self.perspective_angle}, perspective_during_shrink={self.perspective_during_shrink}")
        self._log(f"Center target (lock point)={self.center_position}")
        
        # Cache for frame masks to avoid redundant calculations
        self._frame_mask_cache = {}

    # ----------------------
    # Public API (compatible)
    # ----------------------

    def render_3d_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        scale: float = 1.0,
        alpha: float = 1.0,
        apply_perspective: bool = True,
    ) -> Image.Image:
        """
        Backward-compatible: returns only the RGBA image (anchor discarded).
        """
        img, _anchor = self._render_3d_text_with_anchor(text, scale, alpha, apply_perspective)
        return img

    def generate_frame(self, frame_number: int, background: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a single frame of the 3D text animation with:
          • SSAA anti-aliasing
          • *Front face* anchor kept locked to `center_position`
          • Optional perspective (disabled during shrink by default)
          • DYNAMIC mask recalculation every frame when behind
        """
        # Base frame RGBA
        if background is not None:
            frame = background.copy()
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
            elif frame.shape[2] == 3:
                frame = np.concatenate([frame, np.full((*frame.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
        else:
            frame = np.zeros((*self.resolution[::-1], 4), dtype=np.uint8)
            frame[:, :, 3] = 255

        # Phase selection + scalars (FIXED: fade timing and no slant)
        if frame_number < self.phase1_frames:
            phase = "shrink"
            t = self._smoothstep(frame_number / max(self.phase1_frames - 1, 1))
            scale = self.start_scale + (self.end_scale - self.start_scale) * t
            
            # FIX 1: Start fading at 50% through shrink phase
            if t < 0.5:
                # First half of shrink - fully opaque (in foreground)
                alpha = 1.0
                is_behind = False
            else:
                # Second half of shrink - start exponential fade
                fade_t = (t - 0.5) * 2.0  # Normalize 0.5-1.0 to 0.0-1.0
                # Exponential fade: rapidly fade to 50% opacity
                k = 4.0  # Exponential factor for sharp fade
                alpha = 1.0 - 0.5 * (1 - np.exp(-k * fade_t)) / (1 - np.exp(-k))
                is_behind = True  # Start masking as it passes behind
            
            # FIX 2: NO perspective during shrink (prevents early slant)
            apply_persp = False
            
        elif frame_number < self.phase1_frames + self.phase2_frames:
            phase = "transition"
            t = (frame_number - self.phase1_frames) / max(self.phase2_frames - 1, 1)
            scale = self.end_scale
            # Continue fading to final 50% opacity
            alpha = 0.75 - 0.25 * (1 - np.exp(-3.0 * t)) / (1 - np.exp(-3.0))
            is_behind = True
            # FIX 2: NO perspective in transition (prevents slant)
            apply_persp = False
            
        else:
            phase = "stable"
            scale = self.end_scale
            alpha = 0.5
            is_behind = True
            # FIX 2: NO perspective when stable (prevents final slant)
            apply_persp = False

        # Render text + get anchor (front-face center after perspective)
        text_img, face_anchor = self._render_3d_text_with_anchor(self.text, scale, alpha, apply_persp)
        text_np = np.array(text_img)
        th, tw = text_np.shape[:2]

        # Place so that face_anchor lands at self.center_position
        cx, cy = self.center_position
        ax, ay = face_anchor
        pos_x = int(round(cx - ax))
        pos_y = int(round(cy - ay))

        # Clamp in frame
        pos_x = max(0, min(pos_x, self.resolution[0] - tw))
        pos_y = max(0, min(pos_y, self.resolution[1] - th))

        # Compose
        y1, y2 = pos_y, min(pos_y + th, self.resolution[1])
        x1, x2 = pos_x, min(pos_x + tw, self.resolution[0])

        ty1, ty2 = 0, y2 - y1
        tx1, tx2 = 0, x2 - x1

        # Build text layer
        text_layer = np.zeros_like(frame)
        text_layer[y1:y2, x1:x2] = text_np[ty1:ty2, tx1:tx2]

        # If behind, cut by mask - RECALCULATE mask from current frame!
        if is_behind:
            # CRITICAL FIX: Recalculate mask from CURRENT frame, not stored mask
            if background is not None and background.shape[2] >= 3:
                # Check cache first
                if frame_number not in self._frame_mask_cache:
                    # Extract foreground mask from current frame
                    from utils.segmentation.segment_extractor import extract_foreground_mask
                    
                    # Convert current background to RGB if needed
                    if background.shape[2] == 4:
                        current_rgb = background[:, :, :3]
                    else:
                        current_rgb = background
                    
                    # Get fresh mask for THIS frame
                    if self.debug:
                        self._log(f"[MASK_FIX] Calculating mask for frame {frame_number}")
                    
                    current_mask = extract_foreground_mask(current_rgb)
                    
                    # Ensure mask is right size
                    if current_mask.shape[:2] != (self.resolution[1], self.resolution[0]):
                        current_mask = cv2.resize(current_mask, self.resolution, interpolation=cv2.INTER_LINEAR)
                    
                    # Binarize with threshold
                    current_mask = (current_mask > 128).astype(np.uint8) * 255
                    
                    # Cache it
                    self._frame_mask_cache[frame_number] = current_mask
                else:
                    current_mask = self._frame_mask_cache[frame_number]
                    if self.debug and frame_number % 30 == 0:
                        self._log(f"[MASK_FIX] Using cached mask for frame {frame_number}")
            else:
                # Fallback to stored mask if no background
                current_mask = self.segment_mask
            
            # Apply the CURRENT mask to occlude text
            mask_region = current_mask[y1:y2, x1:x2]
            text_alpha = text_layer[y1:y2, x1:x2, 3].astype(np.float32)
            text_alpha *= (1.0 - (mask_region.astype(np.float32) / 255.0))
            text_layer[y1:y2, x1:x2, 3] = text_alpha.astype(np.uint8)

            # Debug: measure how much was occluded
            if self.debug and (frame_number in (0, self.phase1_frames - 1, self.phase1_frames,
                                                self.phase1_frames + self.phase2_frames - 1, self.total_frames - 1)):
                nonzero_before = np.count_nonzero(text_np[:, :, 3] > 0)
                nonzero_after = np.count_nonzero(text_layer[:, :, 3] > 0)
                self._log(f"Phase={phase}, occlusion={(1 - (nonzero_after / max(nonzero_before, 1))):.3f}, scale={scale:.3f}, alpha={alpha:.3f}")

        # Composite
        frame_pil = Image.fromarray(frame)
        text_pil = Image.fromarray(text_layer)
        out = Image.alpha_composite(frame_pil, text_pil)
        result = np.array(out)

        # Log focal alignment occasionally
        if self.debug and (frame_number in (0, self.phase1_frames - 1, self.phase1_frames,
                                            self.phase1_frames + self.phase2_frames - 1, self.total_frames - 1)):
            self._log(f"Frame {frame_number}: phase={phase}, scale={scale:.3f}, apply_perspective={apply_persp}, placed_at=({pos_x},{pos_y}), anchor={face_anchor}, target_center={self.center_position}, text_size=({tw}x{th})")

        # Return RGB if caller expects video without alpha
        return result[:, :, :3] if result.shape[2] == 4 else result

    # ----------------------
    # Internals / helpers
    # ----------------------

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
        # Binarize
        mask = (mask > 128).astype(np.uint8) * 255
        return mask

    def _smoothstep(self, t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[3D_PIXELATED] {msg}")

    def _get_font(self, px: int) -> ImageFont.FreeTypeFont:
        for path in self._font_candidates:
            try:
                if os.path.exists(path):
                    return ImageFont.truetype(path, px)
            except Exception:
                continue
        # Last resort: default bitmap font (less ideal, but safe)
        return ImageFont.load_default()

    def _render_3d_text_with_anchor(
        self,
        text: str,
        scale: float,
        alpha: float,
        apply_perspective: bool,
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        Draw the 3D text with SSAA and return:
          (RGBA PIL.Image, (anchor_x, anchor_y))
        where anchor is the *front-face center* **after all transforms**.
        """
        ss = self.supersample_factor
        # 1) Scaled font at supersampled resolution
        font_px = max(2, int(round(self.font_size * scale * ss)))
        font = self._get_font(font_px)

        # 2) Measure text face size
        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), text, font=font)
        face_w = max(1, bbox[2] - bbox[0])
        face_h = max(1, bbox[3] - bbox[1])

        # 3) Build a canvas big enough for extrusion (biased +x, -y)
        depth_off = int(round(self.depth_offset * scale * ss))
        pad = max(depth_off * self.depth_layers * 2, ss * 8)
        canvas_w = face_w + pad * 2 + depth_off * self.depth_layers
        canvas_h = face_h + pad * 2 + depth_off * self.depth_layers

        img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Center the FRONT FACE within the canvas:
        front_x = (canvas_w - face_w) // 2
        front_y = (canvas_h - face_h) // 2

        # 4) Depth/extrusion layers (back → front) with smooth gradient
        # REDUCED DEPTH BY 80% - now only 20% of original
        reduced_depth_off = int(round(depth_off * 0.2))
        
        # Use more layers for smoother gradients
        extra_layers = self.depth_layers * 2
        
        for i in range(extra_layers, 0, -1):
            # Positive x, negative y for a top-right extrusion look
            ox = front_x + i * reduced_depth_off
            oy = front_y - i * reduced_depth_off
            t = (extra_layers - i) / max(extra_layers - 1, 1)
            # Smoother curve for better gradient
            t = t * t * t * (3.0 - 2.0 * t - 0.5 * t)

            r = int(self.depth_color[0] * (1 - t) + self.text_color[0] * t * 0.75)
            g = int(self.depth_color[1] * (1 - t) + self.text_color[1] * t * 0.75)
            b = int(self.depth_color[2] * (1 - t) + self.text_color[2] * t * 0.75)

            draw.text((ox, oy), text, font=font, fill=(r, g, b, int(255 * alpha)))

        # 5) Soft outline (anti-aliased), then face
        outline_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
        outline_draw = ImageDraw.Draw(outline_img)
        outline_w = max(1, int(self.outline_width * ss))
        for radius in range(outline_w, 0, -1):
            fade = 1.0 - (radius - 1) / max(outline_w, 1) * 0.5
            # sample around circle (30-deg steps)
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

        face_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(face_img).text(
            (front_x, front_y),
            text,
            font=font,
            fill=(*self.text_color, int(255 * alpha)),
        )
        img = Image.alpha_composite(img, face_img)

        # Anchor BEFORE perspective (front-face center, in SS coords)
        anchor_x_ss = front_x + face_w / 2.0
        anchor_y_ss = front_y + face_h / 2.0

        # 6) Optional perspective warp (apply to image + transform anchor)
        if apply_perspective and self.perspective_angle > 0:
            arr = np.array(img)
            H, W = arr.shape[:2]

            angle = np.radians(self.perspective_angle)
            offset = int(H * np.tan(angle) * 0.2)

            src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            dst = np.float32([[offset, 0], [W - offset, 0], [W, H], [0, H]])

            M = cv2.getPerspectiveTransform(src, dst)
            arr = cv2.warpPerspective(
                arr,
                M,
                (W, H),
                flags=cv2.INTER_CUBIC,  # high-quality interpolation
                borderMode=cv2.BORDER_TRANSPARENT,
            )

            # Transform the anchor with the same homography
            pt = np.array([anchor_x_ss, anchor_y_ss, 1.0], dtype=np.float64)
            M33 = M @ pt
            denom = max(M33[2], 1e-6)
            anchor_x_ss = float(M33[0] / denom)
            anchor_y_ss = float(M33[1] / denom)

            img = Image.fromarray(arr)

        # 7) Shadow (soft, then offset) - reduced to match smaller depth
        shadow = np.array(img)
        shadow[:, :, :3] = 0
        shadow[:, :, 3] = (shadow[:, :, 3].astype(np.float32) * 0.4).astype(np.uint8)
        shadow = Image.fromarray(shadow).filter(ImageFilter.GaussianBlur(radius=max(2, ss)))
        final = Image.new("RGBA", img.size, (0, 0, 0, 0))
        # Reduce shadow offset to match the 80% smaller depth
        shadow_off = int(round(self.shadow_offset * scale * ss * 0.3))
        final.paste(shadow, (shadow_off, shadow_off), shadow)
        final = Image.alpha_composite(final, img)

        # 8) Pre-downsample blur (very slight) for nicer AA, then LANCZOS downsample
        final = final.filter(ImageFilter.GaussianBlur(radius=max(0.0, ss / 6.0)))

        target_w = max(1, final.width // ss)
        target_h = max(1, final.height // ss)
        final = final.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Downscaled anchor to target coords
        anchor_x = anchor_x_ss / ss
        anchor_y = anchor_y_ss / ss

        # Debug once per call site (caller throttles)
        self._log(f"Render: scale={scale:.3f}, alpha={alpha:.3f}, ss={ss}, canvas={canvas_w}x{canvas_h}→{target_w}x{target_h}, anchor={(int(round(anchor_x)), int(round(anchor_y)))}")

        return final, (anchor_x, anchor_y)