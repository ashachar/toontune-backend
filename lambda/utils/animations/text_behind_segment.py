"""
Text Behind Segment animation.
Text moves from foreground to background behind a segmented object.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image, ImageDraw, ImageFont, ImageChops
from .animate import Animation


class TextBehindSegment(Animation):
    """
    Animation where text moves from foreground to background behind a segmented object.
    
    This creates a famous dolly zoom/parallax effect where text appears to move
    through 3D space, going from in front of the subject to behind them.
    
    Additional Parameters:
    ---------------------
    text : str
        Text to animate (default 'START')
    segment_mask : Union[np.ndarray, str]
        Either a numpy array of the segment mask or path to mask image
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
        Duration of moving behind phase in seconds (default 0.67)
    phase3_duration : float
        Duration of stable behind phase in seconds (default 1.33)
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
        segment_mask: Union[np.ndarray, str] = None,
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
        self.final_letter_centers: Optional[List[Tuple[float, float]]] = None  # NEW: for exclusive partition
        
        # Load segment mask if path provided
        if isinstance(segment_mask, str) and os.path.exists(segment_mask):
            mask_img = Image.open(segment_mask)
            self.segment_mask = np.array(mask_img)
            if len(self.segment_mask.shape) == 3:
                self.segment_mask = self.segment_mask[:, :, 0]
            self.segment_mask = self.segment_mask > 128
    
    def load_font(self, size: int):
        """Load font with specified size."""
        if self.font_path and os.path.exists(self.font_path):
            return ImageFont.truetype(self.font_path, size)
        else:
            system_fonts = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "arial.ttf"
            ]
            for font in system_fonts:
                if os.path.exists(font):
                    return ImageFont.truetype(font, size)
            return ImageFont.load_default()
    
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
    ) -> Tuple[List[Tuple[int, int, str]], Tuple[int, int], int, List[Tuple[int, int, int, int]], Tuple[int, int, int, int]]:
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
            prefix_plus = self.text[:i+1]

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
        """
        h, w = frame.shape[:2]
        
        # Determine center position
        if self.center_position:
            center_x, center_y = self.center_position
        else:
            center_x = w // 2
            center_y = int(h * 0.45)
        
        # Phase and parameters
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
        
        text_layer_np = np.array(text_layer)

        # Freeze at final frame of phase 3
        if frame_idx == self.phase3_end - 1 or phase == "background":
            precise_positions, _, _, letter_bboxes_abs, word_bbox = self._precise_letter_geometry(
                center_x, center_y, current_scale
            )

            # Centers for exclusive partition
            centers = [((x0 + x1) / 2.0, (y0 + y1) / 2.0) for (x0, y0, x1, y1) in letter_bboxes_abs]

            print(f"[SPRITE_OVERLAP] freeze centers={[(round(cx,1), round(cy,1)) for (cx,cy) in centers]}")
            print(f"[SPRITE_OVERLAP] freeze word_bbox={word_bbox}")

            self.final_letter_positions = precise_positions
            self.final_font_size = current_font_size
            self.final_center_position = (center_x, center_y)
            self.final_text_origin = (text_x, text_y)
            self.final_scale = current_scale
            self.final_word_bbox = word_bbox
            self.final_letter_bboxes = letter_bboxes_abs
            self.final_letter_centers = centers

            self.final_text_rgba = text_layer_np.copy()
            self.final_occlusion = bool(occlusion)

            shape_str = self.final_text_rgba.shape if self.final_text_rgba is not None else None
            print(f"[SPRITE_OVERLAP] frozen RGBA shape={shape_str} occlusion={self.final_occlusion}")
        
        # Composite with frame
        result = frame.copy()
        if occlusion and mask is not None:
            bg_mask = ~mask
            for c in range(3):
                text_visible = (text_layer_np[:, :, 3] > 0) & bg_mask
                if np.any(text_visible):
                    alpha_blend = text_layer_np[text_visible, 3] / 255.0
                    result[text_visible, c] = (
                        result[text_visible, c] * (1 - alpha_blend) +
                        text_layer_np[text_visible, c] * alpha_blend
                    ).astype(np.uint8)
        else:
            for c in range(3):
                text_visible = text_layer_np[:, :, 3] > 0
                if np.any(text_visible):
                    alpha_blend = text_layer_np[text_visible, 3] / 255.0
                    result[text_visible, c] = (
                        result[text_visible, c] * (1 - alpha_blend) +
                        text_layer_np[text_visible, c] * alpha_blend
                    ).astype(np.uint8)
        return result
    
    def get_handoff_data(self) -> dict:
        """Return all data required for pixel-perfect WD handoff."""
        if self.final_text_rgba is None:
            print("[SPRITE_OVERLAP] WARNING: final_text_rgba is None; WD will redraw (risk of drift).")
        if self.final_letter_bboxes is None or self.final_word_bbox is None:
            print("[SPRITE_OVERLAP] WARNING: final_letter_bboxes/word_bbox missing; WD may re-measure.")
        if self.final_letter_centers is None:
            print("[SPRITE_OVERLAP] WARNING: final_letter_centers missing; WD will fallback to bbox centers.")

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
            'final_letter_centers': self.final_letter_centers,   # NEW
        }
    
    def process_frames(self) -> list:
        return []
    
    def animate(self, output_path: str) -> bool:
        print(f"[SPRITE_OVERLAP] TBS animation; Text='{self.text}' Durations: {self.phase1_duration}/{self.phase2_duration}/{self.phase3_duration}")
        print(f"[SPRITE_OVERLAP] ✓ TBS complete")
        return True