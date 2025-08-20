"""
Word Dissolve animation - EXCLUSIVE SPRITE PARTITION
Prevents cross-letter influence by partitioning the final RGBA into
non-overlapping per-letter sprites via nearest-center labeling.
"""

import os
import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from .animate import Animation


class WordDissolve(Animation):
    """
    Animation where a word dissolves letter by letter.
    With exclusive sprites, each pixel belongs to exactly one letter.
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        word: str = "START",
        font_size: int = 150,
        font_path: Optional[str] = None,
        text_color: Tuple[int, int, int] = (255, 220, 0),
        stable_duration: float = 0.17,
        dissolve_duration: float = 0.67,
        dissolve_stagger: float = 0.33,
        float_distance: int = 30,
        max_scale: float = 1.2,
        randomize_order: bool = True,
        glow_effect: bool = True,
        maintain_kerning: bool = True,
        center_position: Optional[Tuple[int, int]] = None,
        handoff_data: Optional[dict] = None,
        outline_width: int = 2,
        shadow_offset: int = 3,
        direction: float = 0,
        start_frame: int = 0,
        animation_start_frame: int = 0,
        path: Optional[List[Tuple[int, int, int]]] = None,
        fps: int = 30,
        duration: Optional[float] = None,
        temp_dir: Optional[str] = None
    ):
        num_letters = len(word)
        if duration is None:
            duration = stable_duration + (num_letters - 1) * dissolve_stagger + dissolve_duration
        
        super().__init__(
            element_path=element_path,
            background_path=background_path,
            position=position,
            direction=direction,
            start_frame=start_frame,
            animation_start_frame=animation_start_frame,
            path=path,
            fps=fps,
            duration=duration,
            temp_dir=temp_dir
        )
        
        self.word = word
        self.font_size = font_size
        self.font_path = font_path
        self.text_color = text_color
        self.stable_duration = stable_duration
        self.dissolve_duration = dissolve_duration
        self.dissolve_stagger = dissolve_stagger
        self.float_distance = float_distance
        self.max_scale = max_scale
        self.randomize_order = randomize_order
        self.glow_effect = glow_effect
        self.maintain_kerning = maintain_kerning
        self.center_position = center_position
        
        self.outline_width = outline_width
        self.shadow_offset = shadow_offset
        
        # Handoff
        self.handoff_data = handoff_data
        self.frozen_letter_positions: Optional[List[Tuple[int, int, str]]] = None
        self.frozen_font_size: Optional[int] = None
        self.frozen_center_position: Optional[Tuple[int, int]] = None
        self.frozen_text_origin: Optional[Tuple[int, int]] = None
        self.final_scale: float = 1.0

        self.frozen_text_rgba: Optional[np.ndarray] = None
        self.frozen_occlusion: bool = False

        self.word_bbox: Optional[Tuple[int, int, int, int]] = None
        self.letter_bboxes_abs: Optional[List[Tuple[int, int, int, int]]] = None
        self.letter_centers: Optional[List[Tuple[float, float]]] = None  # NEW

        # Prepared exclusive sprites
        # Each entry: { "sprite": np.ndarray(H,W,4), "center": (cx,cy) }
        self.letter_sprites: List[Optional[Dict[str, Any]]] = []

        if handoff_data:
            self.frozen_letter_positions = handoff_data.get('final_letter_positions', None)
            self.frozen_font_size = handoff_data.get('final_font_size', font_size)
            self.frozen_center_position = handoff_data.get('final_center_position', center_position)
            self.frozen_text_origin = handoff_data.get('final_text_origin', None)
            self.final_scale = handoff_data.get('scale', 1.0)

            if handoff_data.get('text'):
                self.word = handoff_data['text']
            if handoff_data.get('font_path'):
                self.font_path = handoff_data['font_path']
            if handoff_data.get('text_color'):
                self.text_color = handoff_data['text_color']
            if handoff_data.get('outline_width') is not None:
                self.outline_width = handoff_data['outline_width']
            if handoff_data.get('shadow_offset') is not None:
                self.shadow_offset = handoff_data['shadow_offset']

            self.frozen_text_rgba = handoff_data.get('final_text_rgba', None)
            self.frozen_occlusion = bool(handoff_data.get('final_occlusion', False))

            self.word_bbox = handoff_data.get('final_word_bbox', None)
            self.letter_bboxes_abs = handoff_data.get('final_letter_bboxes', None)
            self.letter_centers = handoff_data.get('final_letter_centers', None)

        self.stable_frames = int(stable_duration * fps)
        self.dissolve_frames = int(dissolve_duration * fps)
        self.stagger_frames = int(dissolve_stagger * fps)
        
        self.letter_indices = list(range(len(self.word)))
        if randomize_order:
            random.shuffle(self.letter_indices)
        
        self.letter_dissolvers: List[Dict[str, Any]] = []
        self.prepare_letter_dissolvers()

        self._prepare_letter_sprites()

        print(f"[SPRITE_OVERLAP] WD init: origin={self.frozen_text_origin} font_size={self.frozen_font_size} "
              f"stable_frames={self.stable_frames} letters={len(self.word)}")
        if self.frozen_letter_positions is not None:
            preview = [(x, y, ch) for (x, y, ch) in self.frozen_letter_positions]
            print(f"[SPRITE_OVERLAP] WD frozen per-letter positions sample={preview[:min(3, len(preview))]}")
        if self.frozen_text_rgba is not None:
            print(f"[SPRITE_OVERLAP] WD frozen RGBA shape={self.frozen_text_rgba.shape} occlusion={self.frozen_occlusion}")
        else:
            print(f"[SPRITE_OVERLAP] WD WARNING: no frozen RGBA; stable phase may redraw (risk of drift)")
        if self.letter_bboxes_abs is not None:
            print(f"[SPRITE_OVERLAP] WD letter_bboxes count={len(self.letter_bboxes_abs)} word_bbox={self.word_bbox}")
        if self.letter_centers is not None:
            print(f"[SPRITE_OVERLAP] WD centers={[(round(cx,1), round(cy,1)) for (cx,cy) in self.letter_centers]}")

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

    def _prepare_letter_sprites(self) -> None:
        """
        Build exclusive per-letter sprites by partitioning the final text RGBA.
        """
        self.letter_sprites = [None] * len(self.word)
        if self.frozen_text_rgba is None:
            print("[SPRITE_OVERLAP] WD WARNING: no frozen RGBA; cannot build sprites.")
            return

        H, W = self.frozen_text_rgba.shape[:2]
        alpha = self.frozen_text_rgba[:, :, 3] > 0

        # Centers: prefer those from TBS; otherwise fallback to bbox centers
        centers: List[Tuple[float, float]] = []
        if self.letter_centers and len(self.letter_centers) == len(self.word):
            centers = self.letter_centers
        elif self.letter_bboxes_abs:
            centers = [((x0 + x1) / 2.0, (y0 + y1) / 2.0) for (x0, y0, x1, y1) in self.letter_bboxes_abs]
            print("[SPRITE_OVERLAP] WD WARNING: using bbox centers (no centers from TBS).")
        else:
            print("[SPRITE_OVERLAP] WD ERROR: no centers/bboxes; cannot partition sprites.")
            return

        # Build nearest-center label map over all text pixels
        yy, xx = np.indices((H, W))
        centers_np = np.array(centers, dtype=np.float32)  # (N,2)
        if centers_np.shape[0] == 0:
            print("[SPRITE_OVERLAP] WD ERROR: zero centers; cannot partition.")
            return

        # Distances for all pixels to all centers
        # dists shape: (H, W, N)
        dists = []
        for (cx, cy) in centers_np:
            d = (xx - cx) ** 2 + (yy - cy) ** 2
            dists.append(d)
        dists = np.stack(dists, axis=-1)
        labels = np.argmin(dists, axis=-1)  # (H, W)
        labels[~alpha] = -1

        assigned_px = int(np.sum(labels != -1))
        total_px = int(np.sum(alpha))
        print(f"[SPRITE_OVERLAP] label coverage: assigned={assigned_px} / alpha={total_px} "
              f"({(assigned_px / max(1,total_px))*100:.1f}%)")

        # Build exclusive sprites: crop to bbox of each label, blank out other pixels
        for idx in range(len(self.word)):
            mask_i = labels == idx
            if not np.any(mask_i):
                print(f"[SPRITE_OVERLAP] sprite {idx} ('{self.word[idx]}'): EMPTY")
                continue

            ys, xs = np.where(mask_i)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1

            sprite = self.frozen_text_rgba[y0:y1, x0:x1].copy()
            region_mask = mask_i[y0:y1, x0:x1]
            # Zero alpha where not belonging to this letter
            sprite[~region_mask, 3] = 0

            # Center for placement
            cx, cy = centers[idx]
            self.letter_sprites[idx] = {
                "sprite": sprite,
                "center": (cx, cy),
                "bbox": (x0, y0, x1, y1),
            }
            h_i, w_i = sprite.shape[:2]
            print(f"[SPRITE_OVERLAP] sprite {idx} ('{self.word[idx]}'): bbox=({x0},{y0},{x1},{y1}) size={w_i}x{h_i}")

        built = sum(1 for s in self.letter_sprites if s is not None)
        print(f"[SPRITE_OVERLAP] WD prepared EXCLUSIVE sprites: {built}/{len(self.word)}")

    def _alpha_blit(
        self,
        dst: np.ndarray,
        src_rgba: np.ndarray,
        top_left: Tuple[int, int],
        visible_mask: Optional[np.ndarray] = None
    ) -> None:
        """Alpha-composite src_rgba onto dst at top_left (x, y)."""
        y, x = top_left[1], top_left[0]
        H, W = dst.shape[:2]
        h, w = src_rgba.shape[:2]

        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        if x0 >= x1 or y0 >= y1:
            return

        sx0 = x0 - x; sy0 = y0 - y
        sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)

        src_view = src_rgba[sy0:sy1, sx0:sx1]
        if visible_mask is not None:
            vm = visible_mask[y0:y1, x0:x1]
            src_alpha = src_view[:, :, 3] > 0
            vis = src_alpha & vm
        else:
            vis = src_view[:, :, 3] > 0

        if not np.any(vis):
            return

        alpha = src_view[:, :, 3].astype(np.float32) / 255.0
        alpha = alpha * vis.astype(np.float32)

        for c in range(3):
            dst_region = dst[y0:y1, x0:x1, c].astype(np.float32)
            src_region = src_view[:, :, c].astype(np.float32)
            out = dst_region * (1.0 - alpha) + src_region * alpha
            dst[y0:y1, x0:x1, c] = out.astype(np.uint8)

    def _scaled_sprite(self, sprite_rgba: np.ndarray, scale: float, opacity: int) -> np.ndarray:
        """Scale sprite by 'scale' and apply opacity to alpha channel."""
        if scale <= 0:
            scale = 0.001
        h, w = sprite_rgba.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        img = Image.fromarray(sprite_rgba, mode='RGBA')
        img = img.resize((new_w, new_h), Image.LANCZOS)
        out = np.array(img)

        if opacity < 255:
            a = out[:, :, 3].astype(np.float32)
            a = a * (opacity / 255.0)
            out[:, :, 3] = np.clip(a, 0, 255).astype(np.uint8)
        return out

    def _compute_default_positions(self) -> List[Tuple[int, int]]:
        """Best-effort default positions when no handoff is provided."""
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        font = self.load_font(self.font_size)

        if self.center_position:
            center_x, center_y = self.center_position
        else:
            center_x, center_y = 640, 360

        full_bbox = draw.textbbox((0, 0), self.word, font=font)
        full_w = full_bbox[2] - full_bbox[0]
        full_h = full_bbox[3] - full_bbox[1]
        base_x = center_x - full_w // 2
        base_y = center_y - full_h // 2

        positions: List[Tuple[int, int]] = []
        if self.maintain_kerning:
            for i in range(len(self.word)):
                prefix = self.word[:i]
                if prefix:
                    pb = draw.textbbox((0, 0), prefix, font=font)
                    pw = pb[2] - pb[0]
                else:
                    pw = 0
                positions.append((base_x + pw, base_y))
        else:
            cx = base_x
            for _letter in self.word:
                lb = draw.textbbox((0, 0), _letter, font=font)
                lw = lb[2] - lb[0]
                positions.append((cx, base_y))
                cx += lw
        return positions
    
    def calculate_letter_positions(self) -> List[Tuple[int, int]]:
        """Return frozen handoff (x,y) if provided; otherwise compute defaults."""
        if self.frozen_letter_positions:
            print("[SPRITE_OVERLAP] WD using frozen kerning-accurate positions from TBS.")
            return [(x, y) for (x, y, _ch) in self.frozen_letter_positions]
        print("[SPRITE_OVERLAP] WD no handoff; using default positions.")
        return self._compute_default_positions()
    
    def prepare_letter_dissolvers(self):
        """Set timing/order per letter."""
        for i, letter in enumerate(self.word):
            dissolve_idx = self.letter_indices.index(i)
            start_frame = self.stable_frames + (dissolve_idx * self.stagger_frames)
            self.letter_dissolvers.append({
                'letter': letter,
                'index': i,
                'dissolve_order': dissolve_idx,
                'start_frame': start_frame
            })
        if self.letter_dissolvers:
            first_idx = self.letter_dissolvers[0]['index']
            print(f"[SPRITE_OVERLAP] WD dissolve schedule: first index={first_idx} char='{self.word[first_idx]}' at frame={self.stable_frames}")

    def render_word_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Render word with dissolving letters onto a frame, using exclusive sprites.
        """
        h, w = frame.shape[:2]
        
        if self.center_position:
            center_x, center_y = self.center_position
        else:
            center_x = w // 2
            center_y = int(h * 0.45)
            self.center_position = (center_x, center_y)
        
        # Phase A: exact stability before any dissolve starts
        if frame_idx < self.stable_frames:
            if self.frozen_text_rgba is not None:
                text_layer_np = self.frozen_text_rgba
                result = frame.copy()
                if self.frozen_occlusion and mask is not None:
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
                if frame_idx == 0:
                    print("[SPRITE_OVERLAP] WD stable-phase uses frozen RGBA (pixel-perfect).")
                return result

            # Fallback redraw (rare)
            print("[SPRITE_OVERLAP] WD WARNING: no frozen RGBA; redrawing stable frame.")
            text_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_layer)
            effective_font_size = self.frozen_font_size if self.frozen_font_size else self.font_size
            base_font = self.load_font(effective_font_size)
            if self.frozen_text_origin is not None:
                text_x, text_y = self.frozen_text_origin
            else:
                full_bbox = draw.textbbox((0, 0), self.word, font=base_font)
                full_w = full_bbox[2] - full_bbox[0]
                full_h = full_bbox[3] - full_bbox[1]
                text_x = center_x - full_w // 2
                text_y = center_y - full_h // 2
            shadow_px = max(2, int(self.shadow_offset * self.final_scale))
            draw.text((text_x + shadow_px, text_y + shadow_px), self.word, font=base_font, fill=(0, 0, 0, 100))
            for dx in range(-self.outline_width, self.outline_width + 1):
                for dy in range(-self.outline_width, self.outline_width + 1):
                    if abs(dx) == self.outline_width or abs(dy) == self.outline_width:
                        draw.text((text_x + dx, text_y + dy), self.word, font=base_font, fill=(255, 255, 255, 150))
            draw.text((text_x, text_y), self.word, font=base_font, fill=(*self.text_color, 255))
            text_layer_np = np.array(text_layer)
            result = frame.copy()
            if mask is not None:
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
        
        # Phase B: dissolve begins
        if frame_idx == self.stable_frames:
            first_idx = self.letter_indices[0] if self.letter_indices else None
            first_ch = self.word[first_idx] if first_idx is not None else None
            print(f"[SPRITE_OVERLAP] WD dissolve begins at frame {frame_idx}. first index={first_idx} char='{first_ch}'")
        
        use_sprites = (self.frozen_text_rgba is not None and self.letter_sprites and any(self.letter_sprites))
        bg_mask = ~mask if (mask is not None and self.frozen_occlusion) else None
        result = frame.copy()

        if use_sprites:
            # Pass 1: draw all non-started letters (fully opaque)
            non_started_indices: List[int] = []
            dissolving_indices: List[int] = []
            for i in range(len(self.word)):
                info = next(d for d in self.letter_dissolvers if d['index'] == i)
                start_frame = info['start_frame']
                if frame_idx < start_frame:
                    non_started_indices.append(i)
                elif frame_idx < start_frame + self.dissolve_frames:
                    dissolving_indices.append(i)
                # else: finished

            for i in non_started_indices:
                S = self.letter_sprites[i]
                if S is None: 
                    continue
                spr = S["sprite"]
                cx, cy = S["center"]
                th, tw = spr.shape[:2]
                x = int(round(cx - tw / 2.0))
                y = int(round(cy - th / 2.0))
                self._alpha_blit(result, spr, (x, y), visible_mask=bg_mask)

            # Pass 2: draw dissolving letters (scale/fade/float)
            for i in dissolving_indices:
                S = self.letter_sprites[i]
                if S is None:
                    continue
                spr = S["sprite"]
                cx, cy = S["center"]

                info = next(d for d in self.letter_dissolvers if d['index'] == i)
                start_frame = info['start_frame']
                progress = (frame_idx - start_frame) / max(1, self.dissolve_frames)
                opacity = int(255 * (1 - progress))
                if opacity <= 0:
                    continue
                current_scale = 1.0 + (self.max_scale - 1.0) * progress
                float_offset = int(progress * self.float_distance)

                spr_scaled = self._scaled_sprite(spr, current_scale, opacity)
                nx_center = cx
                ny_center = cy - float_offset

                sh, sw = spr_scaled.shape[:2]
                nx = int(round(nx_center - sw / 2.0))
                ny = int(round(ny_center - sh / 2.0))

                # Optional faint glow
                if self.glow_effect and progress > 0.2:
                    glow_alpha = int(opacity * 0.35 * min(1.0, progress * 2))
                    if glow_alpha > 0:
                        for radius in [5, 3, 1]:
                            for angle in range(0, 360, 90):
                                gx = int(radius * np.cos(np.radians(angle)))
                                gy = int(radius * np.sin(np.radians(angle)))
                                glow = spr_scaled.copy()
                                a = glow[:, :, 3].astype(np.float32)
                                a = a * (glow_alpha / 255.0)
                                glow[:, :, 3] = np.clip(a, 0, 255).astype(np.uint8)
                                self._alpha_blit(result, glow, (nx + gx, ny + gy), visible_mask=bg_mask)

                self._alpha_blit(result, spr_scaled, (nx, ny), visible_mask=bg_mask)

            return result

        # ===== Fallback (no sprites) =====
        stable_positions = self.calculate_letter_positions()
        text_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)
        effective_font_size = self.frozen_font_size if self.frozen_font_size else self.font_size
        base_font = self.load_font(effective_font_size)
        
        non_started_indices: List[int] = []
        dissolving_indices: List[int] = []
        for i in range(len(self.word)):
            info = next(d for d in self.letter_dissolvers if d['index'] == i)
            start_frame = info['start_frame']
            if frame_idx < start_frame:
                non_started_indices.append(i)
            elif frame_idx < start_frame + self.dissolve_frames:
                dissolving_indices.append(i)

        # Non-dissolving letters
        for idx in non_started_indices:
            letter = self.word[idx]
            lx, ly = stable_positions[idx]
            shadow_px = max(2, int(self.shadow_offset * self.final_scale))
            draw.text((lx + shadow_px, ly + shadow_px), letter, font=base_font, fill=(0, 0, 0, 100))
            for ddx in range(-self.outline_width, self.outline_width + 1):
                for ddy in range(-self.outline_width, self.outline_width + 1):
                    if abs(ddx) == self.outline_width or abs(ddy) == self.outline_width:
                        draw.text((lx + ddx, ly + ddy), letter, font=base_font, fill=(255, 255, 255, 150))
            draw.text((lx, ly), letter, font=base_font, fill=(*self.text_color, 255))
        
        # Dissolving letters
        for idx in dissolving_indices:
            letter = self.word[idx]
            lx, ly = stable_positions[idx]
            info = next(d for d in self.letter_dissolvers if d['index'] == i)
            start_frame = info['start_frame']
            progress = (frame_idx - start_frame) / max(1, self.dissolve_frames)
            float_offset = int(progress * self.float_distance)
            opacity = int(255 * (1 - progress))
            if opacity <= 0:
                continue
            current_scale = 1.0 + (self.max_scale - 1.0) * progress
            base_size = self.frozen_font_size if self.frozen_font_size else self.font_size
            scaled_font_size = max(1, int(base_size * current_scale))
            scaled_font = self.load_font(scaled_font_size)
            temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            tmp_draw = ImageDraw.Draw(temp_img)
            orig_bbox = tmp_draw.textbbox((0, 0), letter, font=base_font)
            scaled_bbox = tmp_draw.textbbox((0, 0), letter, font=scaled_font)
            orig_w = orig_bbox[2] - orig_bbox[0]
            orig_h = orig_bbox[3] - orig_bbox[1]
            scaled_w = scaled_bbox[2] - scaled_bbox[0]
            scaled_h = scaled_bbox[3] - scaled_bbox[1]
            dx = (scaled_w - orig_w) // 2
            dy = (scaled_h - orig_h) // 2
            ax = lx - dx
            ay = ly - float_offset - dy
            shadow_alpha = max(10, int(100 * (1 - progress)))
            if shadow_alpha > 0:
                draw.text((ax + 2, ay + 2), letter, font=scaled_font, fill=(0, 0, 0, shadow_alpha))
            if self.glow_effect and progress > 0.2:
                glow_alpha = int(opacity * 0.4 * min(1.0, progress * 2))
                for radius in [5, 3, 1]:
                    for angle in range(0, 360, 60):
                        gx = int(radius * np.cos(np.radians(angle)))
                        gy = int(radius * np.sin(np.radians(angle)))
                        draw.text((ax + gx, ay + gy), letter, font=scaled_font, fill=(255, 240, 180, glow_alpha))
            if progress < 0.4:
                outline_alpha = int(150 * (1 - progress * 2.5))
                for ddx in range(-self.outline_width, self.outline_width + 1):
                    for ddy in range(-self.outline_width, self.outline_width + 1):
                        if abs(ddx) == self.outline_width or abs(ddy) == self.outline_width:
                            draw.text((ax + ddx, ay + ddy), letter, font=scaled_font, fill=(255, 255, 255, outline_alpha))
            draw.text((ax, ay), letter, font=scaled_font, fill=(*self.text_color, opacity))
        
        text_layer_np = np.array(text_layer)
        if bg_mask is not None:
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
    
    def process_frames(self) -> list:
        return []
    
    def animate(self, output_path: str) -> bool:
        order = ''.join([self.word[i] for i in self.letter_indices])
        print(f"[SPRITE_OVERLAP] WD animation; Word='{self.word}' Letters={len(self.word)} "
              f"Dissolve order='{order}' Total duration={self.duration}s")
        print(f"[SPRITE_OVERLAP] ✓ WD complete")
        return True