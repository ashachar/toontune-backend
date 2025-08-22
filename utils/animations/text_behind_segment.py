"""
Text Behind Segment animation.
Text moves from foreground to background behind a segmented object.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

from .animate import Animation


class TextBehindSegment(Animation):
    """
    Animation where text moves from foreground to background behind a segmented object.

    This creates a dolly zoom / parallax-like effect where text appears to move
    through 3D space, going from in front of the subject to behind them.

    Additional Parameters:
    ---------------------
    text : str
        Text to animate (default 'START')
    segment_mask : Union[np.ndarray, str]
        Either a numpy array of the segment mask or path to mask image
        (Alpha 255 -> subject (foreground), 0 -> background)
    font_size : int
        Base font size (default 150)
    font_path : str
        Path to font file (optional, uses system font if None)
    text_color : Tuple[int, int, int]
        RGB color for text (default (255, 220, 0) - golden yellow)
    start_scale : float
        Initial scale factor for text (default 2.0)
    end_scale : float
        Final scale factor for text (default 1.0)
    phase1_duration : float
        Duration of shrinking phase in seconds (default 1.0)
    phase2_duration : float
        Duration of moving-behind phase in seconds (default 0.67)
    phase3_duration : float
        Duration of stable-behind phase in seconds (default 1.33)
    center_position : Optional[Tuple[int, int]]
        Center position for text (default: frame center)
    shadow_offset : int
        Shadow offset in pixels (default 3)
    outline_width : int
        Outline width in pixels (default 2)
    """

    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        text: str = "START",
        segment_mask: Union[np.ndarray, str, Image.Image, None] = None,
        font_size: int = 150,
        font_path: Optional[str] = None,
        text_color: Tuple[int, int, int] = (255, 220, 0),
        start_scale: float = 2.0,
        end_scale: float = 1.0,
        phase1_duration: float = 1.0,
        phase2_duration: float = 0.67,
        phase3_duration: float = 1.33,
        center_position: Optional[Tuple[int, int]] = None,
        shadow_offset: int = 3,
        outline_width: int = 2,
        direction: float = 0,
        start_frame: int = 0,
        animation_start_frame: int = 0,
        path: Optional[List[Tuple[int, int, int]]] = None,
        fps: int = 30,
        duration: float = 3.0,
        temp_dir: Optional[str] = None
    ):
        total_duration = phase1_duration + phase2_duration + phase3_duration

        super().__init__(
            element_path=element_path,
            background_path=background_path,
            position=position,
            direction=direction,
            start_frame=start_frame,
            animation_start_frame=animation_start_frame,
            path=path,
            fps=fps,
            duration=total_duration,
            temp_dir=temp_dir
        )

        self.text = text
        self.segment_mask = segment_mask
        self.font_size = font_size
        self.font_path = font_path
        self.text_color = text_color
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.phase1_duration = phase1_duration
        self.phase2_duration = phase2_duration
        self.phase3_duration = phase3_duration
        self.center_position = center_position
        self.shadow_offset = shadow_offset
        self.outline_width = outline_width

        # Phase frame boundaries
        self.phase1_end = int(phase1_duration * fps)
        self.phase2_end = self.phase1_end + int(phase2_duration * fps)
        self.phase3_end = self.phase2_end + int(phase3_duration * fps)

        # Handoff storage
        self.letter_positions_history: List[Dict[str, Any]] = []
        self.final_letter_positions: Optional[List[Tuple[int, int, str]]] = None
        self.final_font_size: Optional[int] = None
        self.final_center_position: Optional[Tuple[int, int]] = None
        self.final_text_origin: Optional[Tuple[int, int]] = None
        self.final_scale: Optional[float] = None

        # Frozen final frame
        self.final_text_rgba: Optional[np.ndarray] = None
        self.final_occlusion: bool = False

        # Exact geometry
        self.final_word_bbox: Optional[Tuple[int, int, int, int]] = None
        self.final_letter_bboxes: Optional[List[Tuple[int, int, int, int]]] = None
        self.final_letter_centers: Optional[List[Tuple[float, float]]] = None

        # Load segment mask if path or PIL image provided
        if isinstance(segment_mask, str) and os.path.exists(segment_mask):
            mask_img = Image.open(segment_mask)
            self.segment_mask = np.array(mask_img)
        elif isinstance(segment_mask, Image.Image):
            self.segment_mask = np.array(segment_mask)

        # Normalize to single-channel uint8 (defer final sizing until per-frame)
        if isinstance(self.segment_mask, np.ndarray):
            if self.segment_mask.ndim == 3:
                # Prefer alpha, else luminance
                if self.segment_mask.shape[2] == 4:
                    self.segment_mask = self.segment_mask[:, :, 3]
                else:
                    self.segment_mask = (0.2989 * self.segment_mask[:, :, 0] +
                                         0.5870 * self.segment_mask[:, :, 1] +
                                         0.1140 * self.segment_mask[:, :, 2]).astype(np.uint8)

    # -------------------------------
    # Utilities
    # -------------------------------
    def load_font(self, size: int):
        """Load font with specified size."""
        if self.font_path and os.path.exists(self.font_path):
            print(f"[FONT_DEBUG] Loading TrueType font: {self.font_path} at size {size}")
            font = ImageFont.truetype(self.font_path, size)
            # Test the font size
            from PIL import Image, ImageDraw
            test_img = Image.new('RGB', (200, 200))
            test_draw = ImageDraw.Draw(test_img)
            test_bbox = test_draw.textbbox((0, 0), "H", font=font)
            print(f"[FONT_DEBUG] Font loaded, test char 'H' bbox: {test_bbox}, height: {test_bbox[3] - test_bbox[1]}")
            return font
        else:
            system_fonts = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "arial.ttf"
            ]
            for font_path in system_fonts:
                if os.path.exists(font_path):
                    print(f"[FONT_DEBUG] Found fallback font: {font_path} at size {size}")
                    return ImageFont.truetype(font_path, size)
            print(f"[FONT_DEBUG] WARNING: Using default font (tiny!)")
            return ImageFont.load_default()

    def _ensure_mask_float01(self, mask: Optional[Union[np.ndarray, Image.Image]], w: int, h: int) -> Optional[np.ndarray]:
        """
        Returns a (h, w) float32 numpy array in [0,1] where:
          1.0 -> subject/foreground
          0.0 -> background
        """
        if mask is None:
            return None

        if isinstance(mask, Image.Image):
            mask_np = np.array(mask)
        else:
            mask_np = mask

        # Channel selection to single channel
        if mask_np.ndim == 3:
            if mask_np.shape[2] == 4:
                mask_np = mask_np[:, :, 3]
            else:
                # convert RGB to luminance
                mask_np = (0.2989 * mask_np[:, :, 0] +
                           0.5870 * mask_np[:, :, 1] +
                           0.1140 * mask_np[:, :, 2]).astype(np.float32)

        # Resize if needed
        if mask_np.shape[0] != h or mask_np.shape[1] != w:
            pil_m = Image.fromarray(mask_np.astype(np.uint8))
            pil_m = pil_m.resize((w, h), Image.BILINEAR)
            mask_np = np.array(pil_m)

        mask_f = mask_np.astype(np.float32)
        # Normalize to [0,1] if values likely in [0,255]
        if mask_f.max() > 1.0:
            mask_f /= 255.0

        # Optional: slight feather to avoid jaggy edges
        # (comment out if you want hard edges)
        try:
            pil = Image.fromarray((mask_f * 255).astype(np.uint8))
            pil = pil.filter(ImageFilter.GaussianBlur(radius=0.75))
            mask_f = (np.array(pil).astype(np.float32)) / 255.0
        except Exception:
            # If PIL filter missing for any reason, keep as-is.
            pass

        return np.clip(mask_f, 0.0, 1.0)

    def _basic_letter_positions(
        self, center_x: int, center_y: int, current_scale: float
    ) -> Tuple[List[Tuple[int, int, str]], Tuple[int, int], int]:
        """Approximate per-letter top-lefts; fast path for intermediate frames."""
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)

        base_font = self.load_font(self.font_size)
        current_font_size = int(self.font_size * current_scale)
        scaled_font = self.load_font(current_font_size)

        base_bbox = draw.textbbox((0, 0), self.text, font=base_font)
        base_width = base_bbox[2] - base_bbox[0]
        base_height = base_bbox[3] - base_bbox[1]
        base_x = center_x - base_width // 2
        base_y = center_y - base_height // 2

        scaled_bbox = draw.textbbox((0, 0), self.text, font=scaled_font)
        scaled_width = scaled_bbox[2] - scaled_bbox[0]
        scaled_height = scaled_bbox[3] - scaled_bbox[1]

        text_x = base_x - (scaled_width - base_width) // 2
        text_y = base_y - (scaled_height - base_height) // 2

        letter_positions: List[Tuple[int, int, str]] = []
        for i, letter in enumerate(self.text):
            prefix = self.text[:i]
            if prefix:
                prefix_bbox = draw.textbbox((0, 0), prefix, font=scaled_font)
                prefix_width = prefix_bbox[2] - prefix_bbox[0]
            else:
                prefix_width = 0
            letter_x = text_x + prefix_width
            letter_positions.append((letter_x, text_y, letter))

        return letter_positions, (text_x, text_y), current_font_size

    def _precise_letter_geometry(
        self, center_x: int, center_y: int, current_scale: float
    ) -> Tuple[
        List[Tuple[int, int, str]],
        Tuple[int, int],
        int,
        List[Tuple[int, int, int, int]],
        Tuple[int, int, int, int]
    ]:
        """
        Kerning-accurate per-letter positions AND absolute bboxes for the CURRENT scale
        using difference-images.
        """
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)

        base_font = self.load_font(self.font_size)
        current_font_size = int(self.font_size * current_scale)
        scaled_font = self.load_font(current_font_size)

        base_bbox = draw.textbbox((0, 0), self.text, font=base_font)
        base_width = base_bbox[2] - base_bbox[0]
        base_height = base_bbox[3] - base_bbox[1]
        base_x = center_x - base_width // 2
        base_y = center_y - base_height // 2

        scaled_bbox = draw.textbbox((0, 0), self.text, font=scaled_font)
        scaled_width = scaled_bbox[2] - scaled_bbox[0]
        scaled_height = scaled_bbox[3] - scaled_bbox[1]
        text_x = base_x - (scaled_width - base_width) // 2
        text_y = base_y - (scaled_height - base_height) // 2

        pad = max(8, self.outline_width * 2 + 4)
        W = scaled_width + pad * 2
        H = scaled_height + pad * 2
        origin = (pad, pad)

        letter_positions: List[Tuple[int, int, str]] = []
        letter_bboxes_abs: List[Tuple[int, int, int, int]] = []

        for i, letter in enumerate(self.text):
            prefix = self.text[:i]
            prefix_plus = self.text[:i + 1]

            imgA = Image.new('L', (W, H), 0)
            imgB = Image.new('L', (W, H), 0)
            dA = ImageDraw.Draw(imgA)
            dB = ImageDraw.Draw(imgB)

            if prefix:
                dA.text(origin, prefix, font=scaled_font, fill=255)
            dB.text(origin, prefix_plus, font=scaled_font, fill=255)

            diff = ImageChops.difference(imgB, imgA)
            bbox = diff.getbbox()

            if not bbox:
                prefix_bbox = draw.textbbox((0, 0), prefix, font=scaled_font) if prefix else (0, 0, 0, 0)
                prefix_width = prefix_bbox[2] - prefix_bbox[0]
                letter_x = text_x + prefix_width
                letter_y = text_y
                letter_positions.append((letter_x, letter_y, letter))
                letter_bboxes_abs.append((letter_x, letter_y, letter_x + 1, letter_y + 1))
                continue

            dx0, dy0, dx1, dy1 = bbox
            letter_x = text_x + (dx0 - origin[0])
            letter_y = text_y + (dy0 - origin[1])
            letter_w = dx1 - dx0
            letter_h = dy1 - dy0

            letter_positions.append((letter_x, letter_y, letter))
            letter_bboxes_abs.append((letter_x, letter_y, letter_x + letter_w, letter_y + letter_h))

        word_bbox = (text_x, text_y, scaled_width, scaled_height)
        return letter_positions, (text_x, text_y), current_font_size, letter_bboxes_abs, word_bbox

    def calculate_letter_positions(self, center_x: int, center_y: int, current_scale: float) -> List[Tuple[int, int, str]]:
        """Fast approximate positions for intermediate frames."""
        approx_positions, _, _ = self._basic_letter_positions(center_x, center_y, current_scale)
        return approx_positions

    def render_text_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Render text onto a single frame with animation.
        NOTE: If `mask` is None, we will use `self.segment_mask` automatically.
        """
        h, w = frame.shape[:2]

        # Determine center position
        if self.center_position:
            center_x, center_y = self.center_position
        else:
            center_x = w // 2
            center_y = int(h * 0.45)

        # Determine phase & occlusion
        if frame_idx <= self.phase1_end:
            phase = "foreground"
            phase_progress = frame_idx / self.phase1_end if self.phase1_end > 0 else 1
            current_scale = self.start_scale - (self.start_scale - 1.3) * phase_progress
            occlusion = False
        elif frame_idx <= self.phase2_end:
            phase = "transition"
            phase_progress = (frame_idx - self.phase1_end) / (self.phase2_end - self.phase1_end)
            current_scale = 1.3 - (1.3 - self.end_scale) * phase_progress
            occlusion = phase_progress > 0.3
        else:
            phase = "background"
            current_scale = self.end_scale
            occlusion = True

        # Create text layer
        text_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        base_font = self.load_font(self.font_size)
        base_bbox = draw.textbbox((0, 0), self.text, font=base_font)
        base_width = base_bbox[2] - base_bbox[0]
        base_height = base_bbox[3] - base_bbox[1]
        base_x = center_x - base_width // 2
        base_y = center_y - base_height // 2

        current_font_size = int(self.font_size * current_scale)
        scaled_font = self.load_font(current_font_size)

        scaled_bbox = draw.textbbox((0, 0), self.text, font=scaled_font)
        scaled_width = scaled_bbox[2] - scaled_bbox[0]
        scaled_height = scaled_bbox[3] - scaled_bbox[1]

        text_x = base_x - (scaled_width - base_width) // 2
        text_y = base_y - (scaled_height - base_height) // 2

        # History (approx positions)
        letter_positions = self.calculate_letter_positions(center_x, center_y, current_scale)
        self.letter_positions_history.append({
            'frame_idx': frame_idx,
            'positions': letter_positions,
            'font_size': current_font_size,
            'center_position': (center_x, center_y),
            'scale': current_scale,
            'text_origin': (text_x, text_y),
        })

        # Draw shadow
        shadow_px = max(2, int(self.shadow_offset * current_scale))
        draw.text((text_x + shadow_px, text_y + shadow_px), self.text, font=scaled_font, fill=(0, 0, 0, 100))

        # Outline (perimeter)
        for dx in range(-self.outline_width, self.outline_width + 1):
            for dy in range(-self.outline_width, self.outline_width + 1):
                if abs(dx) == self.outline_width or abs(dy) == self.outline_width:
                    draw.text((text_x + dx, text_y + dy), self.text, font=scaled_font, fill=(255, 255, 255, 150))

        # Main text
        draw.text((text_x, text_y), self.text, font=scaled_font, fill=(*self.text_color, 255))

        text_layer_np = np.array(text_layer)  # (h, w, 4)
        text_alpha = (text_layer_np[:, :, 3].astype(np.float32) / 255.0)[..., None]  # (h, w, 1)
        text_rgb = text_layer_np[:, :, :3].astype(np.float32)  # (h, w, 3)
        
        # Apply transparency ONLY AFTER text passes behind subject
        if phase == "foreground":
            # During shrink (phase 1), text stays FULLY OPAQUE
            # No fading at all - text is still in front
            pass  # Keep text_alpha as is (fully opaque)
        elif phase == "transition":
            # During transition (phase 2), text passes behind - START FADING HERE
            # Use exponential curve for dramatic fade as it goes behind
            phase_progress = (frame_idx - self.phase1_end) / (self.phase2_end - self.phase1_end)
            
            # Exponential function for smooth fade from 1.0 to 0.5
            import math
            k = 3.0  # Curve factor - higher = more exponential
            exp_progress = (math.exp(k * phase_progress) - 1) / (math.exp(k) - 1)
            
            # Fade from fully opaque (1.0) to semi-transparent (0.5)
            target_alpha = 1.0 - (0.5 * exp_progress)
            text_alpha = text_alpha * target_alpha
        elif phase == "background":
            # During stable behind (phase 3), maintain 0.5 alpha
            # The final fade out will happen during dissolve
            text_alpha = text_alpha * 0.5

        # Use provided mask or fallback to self.segment_mask
        if mask is None:
            mask = self.segment_mask

        subject_mask = self._ensure_mask_float01(mask, w=w, h=h)  # (h, w) in [0,1] or None
        subject_cov = float(subject_mask.mean()) if subject_mask is not None else 0.0

        # Compute visible alpha with proper occlusion:
        # - When occlusion=True: text is ONLY visible where background is present: (1 - subject_mask).
        # - When occlusion=False: text is fully visible with its own alpha.
        if occlusion and subject_mask is not None:
            visible_alpha = text_alpha * (1.0 - subject_mask[..., None])  # (h, w, 1)
            occlusion_mode = "masked"
        elif occlusion and subject_mask is None:
            # Fallback: no mask -> cannot hide behind, draw on top (warn)
            visible_alpha = text_alpha
            occlusion_mode = "no-mask-fallback"
        else:
            visible_alpha = text_alpha
            occlusion_mode = "foreground"

        # Debug diagnostics (required prefix)
        if frame_idx % 5 == 0:
            text_pixels = int((text_layer_np[:, :, 3] > 0).sum())
            print(
                f"[OCCLUSION] frame={frame_idx} phase={phase} occlusion={occlusion} "
                f"mode={occlusion_mode} text_pixels={text_pixels} "
                f"mask_present={subject_mask is not None} mask_cov={subject_cov:.3f}"
            )

        # Final composite
        base = frame.astype(np.float32)
        out = base * (1.0 - visible_alpha) + text_rgb * visible_alpha
        result = out.astype(np.uint8)

        # Freeze "final" geometry at end of phase 3 (for handoff)
        if frame_idx == self.phase3_end - 1 or phase == "background":
            precise_positions, _, _, letter_bboxes_abs, word_bbox = self._precise_letter_geometry(
                center_x, center_y, current_scale
            )
            centers = [((x0 + x1) / 2.0, (y0 + y1) / 2.0) for (x0, y0, x1, y1) in letter_bboxes_abs]

            print(f"[OCCLUSION] freeze centers={[(round(cx,1), round(cy,1)) for (cx,cy) in centers]}")
            print(f"[OCCLUSION] freeze word_bbox={word_bbox}")

            self.final_letter_positions = precise_positions
            self.final_font_size = current_font_size
            self.final_center_position = (center_x, center_y)
            self.final_text_origin = (text_x, text_y)
            self.final_scale = current_scale
            self.final_word_bbox = word_bbox
            self.final_letter_bboxes = letter_bboxes_abs
            self.final_letter_centers = centers

            # Apply 0.5 alpha to the frozen RGBA since text is now behind
            frozen_rgba = text_layer_np.copy()
            frozen_rgba[:, :, 3] = (frozen_rgba[:, :, 3] * 0.5).astype(np.uint8)
            self.final_text_rgba = frozen_rgba
            self.final_occlusion = bool(occlusion)

            shape_str = self.final_text_rgba.shape if self.final_text_rgba is not None else None
            print(f"[OCCLUSION] frozen RGBA shape={shape_str} occlusion={self.final_occlusion}")

        return result

    def get_handoff_data(self) -> dict:
        """Return all data required for pixel-perfect WD handoff."""
        if self.final_text_rgba is None:
            print("[OCCLUSION] WARNING: final_text_rgba is None; WD will redraw (risk of drift).")
        if self.final_letter_bboxes is None or self.final_word_bbox is None:
            print("[OCCLUSION] WARNING: final_letter_bboxes/word_bbox missing; WD may re-measure.")
        if self.final_letter_centers is None:
            print("[OCCLUSION] WARNING: final_letter_centers missing; WD will fallback to bbox centers.")

        return {
            'final_letter_positions': self.final_letter_positions,
            'final_font_size': self.final_font_size or int(self.font_size * self.end_scale),
            'final_center_position': self.final_center_position,
            'final_text_origin': self.final_text_origin,
            'scale': self.final_scale if self.final_scale is not None else self.end_scale,
            'text': self.text,
            'text_color': self.text_color,
            'font_path': self.font_path,
            'outline_width': self.outline_width,
            'shadow_offset': self.shadow_offset,
            'positions_history': self.letter_positions_history,
            'final_text_rgba': self.final_text_rgba,
            'final_occlusion': self.final_occlusion,
            'final_word_bbox': self.final_word_bbox,
            'final_letter_bboxes': self.final_letter_bboxes,
            'final_letter_centers': self.final_letter_centers,
        }

    def process_frames(self) -> list:
        return []

    def animate(self, output_path: str) -> bool:
        print(f"[OCCLUSION] TBS animation; Text='{self.text}' Durations: {self.phase1_duration}/{self.phase2_duration}/{self.phase3_duration}")
        print(f"[OCCLUSION] ✓ TBS complete")
        return True