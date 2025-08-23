#!/usr/bin/env python3
"""
3D text animation with FIXED O occlusion.
More aggressive masking for seated figures that rembg misses.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


@dataclass
class Text3DBehindSegment:
    """
    3D text animation with fixed occlusion for seated figures.
    """

    # Core animation parameters
    duration: float = 2.0
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

    # Animation timing (backwards only)
    shrink_duration: float = 1.5
    wiggle_duration: float = 0.5

    # Start/end states
    start_scale: float = 2.0
    end_scale: float = 0.8
    center_position: Optional[Tuple[int, int]] = None

    # Visual effects
    shadow_offset: int = 8
    outline_width: int = 3
    perspective_angle: float = 25
    supersample_factor: int = 2

    # Control flags
    debug: bool = False

    def __post_init__(self):
        # Font candidates
        self._font_candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]

        # Validate durations
        self.shrink_duration = min(self.shrink_duration, self.duration * 0.9)
        self.wiggle_duration = self.duration - self.shrink_duration

        # Frame calculations
        self.total_frames = int(self.duration * self.fps)
        self.shrink_frames = int(self.shrink_duration * self.fps)
        self.wiggle_frames = self.total_frames - self.shrink_frames

        # Load or create mask
        self.segment_mask = self._load_or_make_mask(self.segment_mask)

        # Default center position
        if self.center_position is None:
            self.center_position = (self.resolution[0] // 2, self.resolution[1] // 2)

        # Initialize mask cache
        self._frame_mask_cache = {}

        if self.debug:
            self._log(f"Init: FIXED O occlusion version")
            self._log(f"res={self.resolution}, fps={self.fps}, total_frames={self.total_frames}")

    def _improve_mask_for_seated_figures(self, mask: np.ndarray) -> np.ndarray:
        """
        Special processing to better detect seated figures that rembg misses.
        """
        # 1. First pass - fill small gaps with morphological closing
        kernel_close = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 2. Connect nearby components (helps with partial detections)
        # Dilate then erode to connect close regions
        kernel_connect = np.ones((9, 9), np.uint8)
        mask = cv2.dilate(mask, kernel_connect, iterations=1)
        mask = cv2.erode(mask, kernel_connect, iterations=1)
        
        # 3. Fill interior holes
        # Find contours and fill them
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, -1)
        
        # 4. Expand to cover missed edges (seated figures often have soft edges)
        kernel_expand = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel_expand, iterations=2)
        
        # 5. Smooth the result
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # 6. Apply a lower threshold to be more inclusive
        mask = (mask > 50).astype(np.uint8) * 255
        
        if self.debug:
            coverage = np.sum(mask > 128) / mask.size
            self._log(f"[MASK_FIX] Improved mask coverage: {coverage:.1%}")
        
        return mask

    def generate_frame(
        self,
        frame_number: int,
        background: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate a single frame with fixed O occlusion."""
        
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

        # Animation phases
        if frame_number < self.shrink_frames:
            phase = "shrink"
            t = frame_number / max(self.shrink_frames - 1, 1)
        else:
            phase = "wiggle"
            t = (frame_number - self.shrink_frames) / max(self.wiggle_frames - 1, 1)

        # Apply easing
        smooth_t = self._smoothstep(t)

        # Calculate parameters
        if phase == "shrink":
            scale = self.start_scale - smooth_t * (self.start_scale - self.end_scale)
            
            # Fade timing
            if t < 0.4:
                alpha = 1.0
                is_behind = False
            elif t < 0.6:
                fade_t = (t - 0.4) / 0.2
                alpha = 1.0 - fade_t * 0.4
                is_behind = False
            else:
                fade_t = (t - 0.6) / 0.4
                k = 3.0
                alpha = 0.6 - 0.4 * (1 - np.exp(-k * fade_t)) / (1 - np.exp(-k))
                is_behind = True
        else:
            wiggle_amount = np.sin(t * np.pi * 4) * 0.02
            scale = self.end_scale * (1.0 + wiggle_amount)
            alpha = 0.2
            is_behind = True

        # Render 3D text
        text_pil, (anchor_x, anchor_y) = self._render_3d_text(
            self.text, scale, alpha, False
        )

        # Calculate position
        cx, cy = self.center_position
        
        if phase == "shrink":
            start_y = cy - self.resolution[1] * 0.15
            end_y = cy
            pos_x = int(cx - anchor_x)
            pos_y = int(start_y + smooth_t * (end_y - start_y) - anchor_y)
        else:
            pos_x = int(cx - anchor_x)
            pos_y = int(cy - anchor_y)

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

        # Apply improved occlusion when behind
        if is_behind:
            if background is not None and background.shape[2] >= 3:
                if frame_number not in self._frame_mask_cache:
                    from utils.segmentation.segment_extractor import extract_foreground_mask
                    
                    if background.shape[2] == 4:
                        current_rgb = background[:, :, :3]
                    else:
                        current_rgb = background
                    
                    if self.debug:
                        self._log(f"[MASK] Extracting mask for frame {frame_number}")
                    
                    current_mask = extract_foreground_mask(current_rgb)
                    
                    if current_mask.shape[:2] != (self.resolution[1], self.resolution[0]):
                        current_mask = cv2.resize(current_mask, self.resolution, interpolation=cv2.INTER_LINEAR)
                    
                    # Apply improved processing for seated figures
                    current_mask = self._improve_mask_for_seated_figures(current_mask)
                    
                    self._frame_mask_cache[frame_number] = current_mask
                    
                    if self.debug and frame_number == 30:
                        # Check O region specifically
                        o_region = current_mask[250:350, 550:650]
                        o_coverage = np.sum(o_region > 128) / o_region.size
                        self._log(f"[O_CHECK] Frame 30 O region coverage: {o_coverage:.1%}")
                else:
                    current_mask = self._frame_mask_cache[frame_number]
            else:
                current_mask = self.segment_mask
            
            # Apply mask with full strength
            mask_region = current_mask[y1:y2, x1:x2]
            text_alpha = text_layer[y1:y2, x1:x2, 3].astype(np.float32)
            mask_factor = mask_region.astype(np.float32) / 255.0
            text_alpha *= (1.0 - mask_factor)
            text_layer[y1:y2, x1:x2, 3] = text_alpha.astype(np.uint8)
            
            if self.debug and frame_number % 10 == 0:
                occluded = np.sum(mask_region > 128) / max(mask_region.size, 1)
                self._log(f"[OCCLUSION] Frame {frame_number}: {occluded:.1%} of text occluded")

        # Composite
        frame_pil = Image.fromarray(frame)
        text_pil = Image.fromarray(text_layer)
        out = Image.alpha_composite(frame_pil, text_pil)
        result = np.array(out)

        return result[:, :, :3] if result.shape[2] == 4 else result

    def _render_3d_text(
        self,
        text: str,
        scale: float,
        alpha: float,
        apply_perspective: bool,
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """Render 3D text."""
        ss = self.supersample_factor
        
        # Scaled font
        font_px = max(2, int(round(self.font_size * scale * ss)))
        font = self._get_font(font_px)

        # Measure text
        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), text, font=font)
        face_w = max(1, bbox[2] - bbox[0])
        face_h = max(1, bbox[3] - bbox[1])

        # Canvas
        depth_off = int(round(self.depth_offset * scale * ss))
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
        shadow_off = int(round(self.shadow_offset * scale * ss * 0.3))
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
            print(f"[3D_FIXED_O] {msg}")

    def _get_font(self, px: int) -> ImageFont.FreeTypeFont:
        for path in self._font_candidates:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, px)
                except:
                    continue
        return ImageFont.load_default()