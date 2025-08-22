"""
Word Dissolve animation - EXCLUSIVE SPRITE PARTITION with SOFT OCCLUSION
- Exact per-letter sprites via prefix–difference (no Voronoi cracks).
- Matches TextBehindSegment's soft occlusion to avoid "broken" edges.
- Adds [SPRITE_BREAK] debug logs for diagnosis.
"""

import os
import random
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import math

from .animate import Animation


class WordDissolve(Animation):
    """
    Animation where a word dissolves letter by letter.
    With exclusive sprites, each pixel belongs to exactly one letter.
    Soft occlusion matches TBS to prevent edge "cracks".
    """

    # ----- Unified debug -----
    def _dbg(self, msg: str) -> None:
        """Unified debug printing with required prefix."""
        if getattr(self, "debug", False):
            print(f"[FRAME_DISSOLVE] {msg}")

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
        temp_dir: Optional[str] = None,
        sprite_pad_ratio: float = 0.25,
        debug: Optional[bool] = None
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

        # New diagnostics & rendering knobs
        # Debug is on when env FRAME_DISSOLVE_DEBUG!=0 unless explicitly set
        env_debug = os.getenv('FRAME_DISSOLVE_DEBUG', '0') != '0'
        self.debug = (debug if debug is not None else env_debug)
        self.sprite_pad_ratio = float(sprite_pad_ratio)

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
        self.letter_centers: Optional[List[Tuple[float, float]]] = None  # absolute frame centers

        # Exclusive sprites per letter
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
                self.text_color = tuple(handoff_data['text_color'])
            if handoff_data.get('outline_width') is not None:
                self.outline_width = int(handoff_data['outline_width'])
            if handoff_data.get('shadow_offset') is not None:
                self.shadow_offset = int(handoff_data['shadow_offset'])

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

        print(f"[SPRITE_BREAK] WD init: word='{self.word}' letters={len(self.word)} "
              f"stable={self.stable_frames}f dissolve={self.dissolve_frames}f "
              f"stagger={self.stagger_frames}f fps={self.fps}")
        if self.frozen_text_rgba is None:
            print("[SPRITE_BREAK] WD WARNING: no frozen RGBA; stable phase will redraw (risk of drift)")
        if self.letter_bboxes_abs is not None:
            print(f"[SPRITE_BREAK] WD letter_bboxes count={len(self.letter_bboxes_abs)} word_bbox={self.word_bbox}")
        if self.letter_centers is not None:
            centers_str = [(round(cx, 1), round(cy, 1)) for (cx, cy) in self.letter_centers]
            print(f"[SPRITE_BREAK] WD centers={centers_str}")

    # -------------------------------
    # Utilities
    # -------------------------------
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

    def _ensure_mask_float01(self, mask: Optional[Union[np.ndarray, Image.Image]], w: int, h: int) -> Optional[np.ndarray]:
        """
        Returns (h, w) float32 in [0,1]; 1.0 -> subject/foreground; 0.0 -> background.
        Includes a small Gaussian blur to feather edges like TBS.
        """
        if mask is None:
            return None

        if isinstance(mask, Image.Image):
            mask_np = np.array(mask)
        else:
            mask_np = mask

        if mask_np.ndim == 3:
            if mask_np.shape[2] == 4:
                mask_np = mask_np[:, :, 3]
            else:
                mask_np = (0.2989 * mask_np[:, :, 0] +
                           0.5870 * mask_np[:, :, 1] +
                           0.1140 * mask_np[:, :, 2]).astype(np.float32)

        if mask_np.shape[0] != h or mask_np.shape[1] != w:
            pil_m = Image.fromarray(mask_np.astype(np.uint8))
            pil_m = pil_m.resize((w, h), Image.BILINEAR)
            mask_np = np.array(pil_m)

        mask_f = mask_np.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0

        try:
            pil = Image.fromarray((mask_f * 255).astype(np.uint8))
            pil = pil.filter(ImageFilter.GaussianBlur(radius=0.75))
            mask_f = (np.array(pil).astype(np.float32)) / 255.0
        except Exception:
            pass

        return np.clip(mask_f, 0.0, 1.0)

    # -------------------------------
    # Exact per-letter masks (prefix–difference)
    # -------------------------------
    def _get_final_text_origin(self, W: int, H: int, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
        """Choose text origin consistent with TBS handoff (fallback to centering)."""
        if self.frozen_text_origin is not None:
            return self.frozen_text_origin

        tmp = Image.new('L', (1, 1), 0)
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), self.word, font=font)
        w_text = bbox[2] - bbox[0]
        h_text = bbox[3] - bbox[1]
        if self.center_position:
            cx, cy = self.center_position
        else:
            cx, cy = W // 2, int(H * 0.45)
        return (cx - w_text // 2, cy - h_text // 2)

    def _build_per_letter_masks_exact(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Returns:
          - masks: list of boolean (H,W) masks, one per letter (exclusive after fill)
          - labels: int32 (H,W) label map in [-1..N-1] over text alpha
        """
        if self.frozen_text_rgba is None:
            print("[SPRITE_BREAK] WD ERROR: cannot build masks; frozen RGBA missing.")
            return [], np.zeros((1, 1), dtype=np.int32)

        H, W = self.frozen_text_rgba.shape[:2]
        alpha = self.frozen_text_rgba[:, :, 3] > 0

        # Font and origin must match TBS geometry
        effective_font_size = self.frozen_font_size if self.frozen_font_size else self.font_size
        font = self.load_font(effective_font_size)
        text_x, text_y = self._get_final_text_origin(W, H, font)

        # Build seed masks with prefix–difference (no outline/shadow needed for seeding).
        masks_seed: List[np.ndarray] = []
        prefix = ""
        for i, ch in enumerate(self.word):
            imgA = Image.new('L', (W, H), 0)
            imgB = Image.new('L', (W, H), 0)
            dA = ImageDraw.Draw(imgA)
            dB = ImageDraw.Draw(imgB)

            if prefix:
                dA.text((text_x, text_y), prefix, font=font, fill=255)
            dB.text((text_x, text_y), prefix + ch, font=font, fill=255)

            diff = ImageChops.difference(imgB, imgA)
            seed = np.array(diff) > 0
            masks_seed.append(seed)
            prefix += ch

        # Initialize labels with seeds.
        labels = np.full((H, W), -1, dtype=np.int32)
        for i, m in enumerate(masks_seed):
            labels[m] = i

        seed_assigned = int(np.sum(labels != -1))
        total_alpha = int(np.sum(alpha))
        print(f"[SPRITE_BREAK] seeds coverage: assigned={seed_assigned} / alpha={total_alpha} "
              f"({(seed_assigned / max(1, total_alpha)) * 100:.1f}%)")

        # Assign leftover text pixels (outline/halo overlaps) by nearest seeded letter center.
        missing = alpha & (labels == -1)
        n_missing = int(missing.sum())
        if n_missing > 0:
            # centers: prefer TBS centers; else centroid of seed; else bbox centers
            if self.letter_centers and len(self.letter_centers) == len(self.word):
                centers = self.letter_centers
            else:
                centers = []
                for i, m in enumerate(masks_seed):
                    ys, xs = np.where(m)
                    if len(xs) == 0:
                        # fallback to bbox center if available
                        if self.letter_bboxes_abs and i < len(self.letter_bboxes_abs):
                            x0, y0, x1, y1 = self.letter_bboxes_abs[i]
                            centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
                        else:
                            centers.append((W / 2.0, H / 2.0))
                    else:
                        centers.append((float(xs.mean()), float(ys.mean())))

            yy, xx = np.indices((H, W))
            centers_np = np.array(centers, dtype=np.float32)  # (N,2)
            dists = []
            for (cx, cy) in centers_np:
                d = (xx - cx) ** 2 + (yy - cy) ** 2
                dists.append(d)
            dists = np.stack(dists, axis=-1)  # (H,W,N)
            nearest = np.argmin(dists, axis=-1)
            labels[missing] = nearest[missing]

            print(f"[SPRITE_BREAK] filled leftover pixels: {n_missing} assigned by nearest seeded center")

        # Convert to boolean masks per label (exclusive and covering alpha)
        masks: List[np.ndarray] = []
        for i in range(len(self.word)):
            masks.append((labels == i) & alpha)

        covered = int(np.sum((labels != -1) & alpha))
        print(f"[SPRITE_BREAK] final label coverage: assigned={covered} / alpha={total_alpha} "
              f"({(covered / max(1, total_alpha)) * 100:.1f}%)")
        return masks, labels

    # -------------------------------
    # Sprite preparation
    # -------------------------------
    def _prepare_letter_sprites(self) -> None:
        """
        Build exclusive per-letter sprites by **exact** partition of the final text RGBA.
        Anchors to bbox center. Adds padding around each sprite so glow/blur is never clipped.
        """
        self.letter_sprites = [None] * len(self.word)
        if self.frozen_text_rgba is None:
            self._dbg("WARNING: no frozen RGBA; cannot build sprites.")
            return

        H, W = self.frozen_text_rgba.shape[:2]
        masks, labels = self._build_per_letter_masks_exact()
        if len(masks) != len(self.word):
            self._dbg("ERROR: mask build failed; sprites not prepared.")
            return

        for idx in range(len(self.word)):
            mask_i = masks[idx]
            if not np.any(mask_i):
                self._dbg(f"sprite {idx} ('{self.word[idx]}'): EMPTY")
                continue

            ys, xs = np.where(mask_i)
            y0, y1 = int(ys.min()), int(ys.max() + 1)
            x0, x1 = int(xs.min()), int(xs.max() + 1)

            sprite = self.frozen_text_rgba[y0:y1, x0:x1].copy()
            region_mask = mask_i[y0:y1, x0:x1]
            sprite[~region_mask, 3] = 0

            # --- NEW: pad sprite to prevent rectangular-box glow artifacts ---
            h_i, w_i = sprite.shape[:2]
            max_dim = max(w_i, h_i)

            # Ensure pad covers:
            #  - max blur (~20% of size at progress=1, increased from 16%)
            #  - some safety for up-scaling (self.max_scale)
            #  - Extra margin to prevent any boundary artifacts
            blur_max = int(round(max_dim * self.max_scale * 0.20))
            pad_ratio = max(0.15, min(0.50, float(getattr(self, "sprite_pad_ratio", 0.30))))
            pad_by_ratio = int(round(max_dim * pad_ratio))

            # Increase minimum padding to ensure no boundaries are visible
            pad_px = max(blur_max + 5, pad_by_ratio, 10)

            padded = np.zeros((h_i + 2*pad_px, w_i + 2*pad_px, 4), dtype=np.uint8)
            padded[pad_px:pad_px + h_i, pad_px:pad_px + w_i] = sprite

            # Use bbox center as pivot (handoff alignment)
            bbox_cx = (x0 + x1) / 2.0
            bbox_cy = (y0 + y1) / 2.0

            self.letter_sprites[idx] = {
                "sprite": padded,
                "pivot": (bbox_cx, bbox_cy),
                # Store **padded** bbox (for hole placement)
                "bbox": (x0 - pad_px, y0 - pad_px, x1 + pad_px, y1 + pad_px),
                "pad_px": pad_px,
                "orig_bbox": (x0, y0, x1, y1),
            }

            self._dbg(
                f"sprite {idx} '{self.word[idx]}': "
                f"orig_bbox=({x0},{y0},{x1},{y1}) size={w_i}x{h_i} "
                f"pad={pad_px}px -> padded={padded.shape[1]}x{padded.shape[0]}"
            )

        built = sum(1 for s in self.letter_sprites if s is not None)
        self._dbg(f"prepared EXCLUSIVE sprites: {built}/{len(self.word)}")

        # Track active indices and compute true last frame
        self.active_indices = [i for i, s in enumerate(self.letter_sprites) if s is not None]
        active_letters = len(self.active_indices)
        self.last_start_frame = self.stable_frames + max(0, active_letters - 1) * self.stagger_frames
        self.last_end_frame = self.last_start_frame + self.dissolve_frames
        self._dbg(
            f"timing: active_letters={active_letters} "
            f"stable={self.stable_frames} stagger={self.stagger_frames} "
            f"dissolve={self.dissolve_frames} last_start={self.last_start_frame} last_end={self.last_end_frame}"
        )

    # -------------------------------
    # Compositing helpers
    # -------------------------------
    def _top_left_from_pivot(self, pivot_x: float, pivot_y: float, sprite_w: int, sprite_h: int) -> Tuple[int, int]:
        """
        Given a pivot point (cx, cy) and sprite dimensions, compute the top-left corner.
        This ensures consistent positioning regardless of sprite scaling.
        """
        x = int(round(pivot_x - sprite_w / 2.0))
        y = int(round(pivot_y - sprite_h / 2.0))
        return (x, y)
    
    def _alpha_blit(
        self,
        dst: np.ndarray,             # RGB uint8
        src_rgba: np.ndarray,        # RGBA uint8
        top_left: Tuple[int, int],   # (x, y)
        visible_mask: Optional[np.ndarray] = None  # None | bool[h,w] | float[h,w] (0..1)
    ) -> None:
        """Alpha-composite src_rgba onto dst at top_left (x, y), honoring an optional visibility mask."""
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
        src_alpha = src_view[:, :, 3].astype(np.float32) / 255.0  # (h,w)

        if visible_mask is not None:
            vm = visible_mask[y0:y1, x0:x1]
            if vm.dtype == bool:
                vmf = vm.astype(np.float32)
            else:
                vmf = vm.astype(np.float32)
            alpha = src_alpha * vmf
        else:
            alpha = src_alpha

        if not np.any(alpha > 0.0):
            return

        for c in range(3):
            dst_region = dst[y0:y1, x0:x1, c].astype(np.float32)
            src_region = src_view[:, :, c].astype(np.float32)
            out = dst_region * (1.0 - alpha) + src_region * alpha
            dst[y0:y1, x0:x1, c] = np.clip(out, 0, 255).astype(np.uint8)

    def _scaled_sprite(self, sprite_rgba: np.ndarray, scale: float, opacity: int) -> np.ndarray:
        """Scale sprite by `scale` and apply opacity to alpha channel."""
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

    def _soft_glow_sprite(self, spr_scaled: np.ndarray, opacity: int, progress: float) -> np.ndarray:
        """
        Build a single blurred/tinted glow from spr_scaled's alpha.
        Returns an RGBA sprite (same size as spr_scaled).
        NOTE: glow radius depends on scaled size. Padding is guaranteed upstream.
        """
        h_sc, w_sc = spr_scaled.shape[:2]
        a = spr_scaled[:, :, 3].astype(np.uint8)

        # Max blur at progress=1 is ~16% of the max dimension (empirical sweet spot).
        blur_px = max(1, int(round(max(h_sc, w_sc) * (0.06 + 0.10 * progress))))
        try:
            a_blur = np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=blur_px)))
        except Exception:
            # Extremely rare: fall back to identity
            a_blur = a

        # Glow strength ~ 40% of current letter alpha
        strength = int(opacity * 0.4)
        glow_a = (a_blur.astype(np.float32) * (strength / 255.0)).clip(0, 255).astype(np.uint8)

        glow_rgb = np.zeros_like(spr_scaled[:, :, :3], dtype=np.uint8)
        glow_rgb[:, :, 0] = self.text_color[0]
        glow_rgb[:, :, 1] = self.text_color[1]
        glow_rgb[:, :, 2] = self.text_color[2]
        out = np.dstack([glow_rgb, glow_a])

        # Debug (rate-limited)
        if self.debug and strength > 0 and blur_px % 7 == 0:
            self._dbg(f"glow: size={w_sc}x{h_sc} blur={blur_px}px strength={strength} progress={progress:.2f}")
        return out
    
    def _union_hole_mask(self, H: int, W: int, indices: List[int], dilate_px: int = 0) -> Optional[np.ndarray]:
        """
        Build a union mask (float32 [0..1], shape (H,W)) of the *current dissolving* letters,
        by splatting each letter's sprite alpha into frame space. Optional dilation widens the hole
        so outline/halo pixels assigned to neighbors are also removed.
        """
        if not indices:
            return None

        hole = np.zeros((H, W), dtype=np.float32)
        for i in indices:
            S = self.letter_sprites[i]
            if not S:
                continue
            x0, y0, x1, y1 = S["bbox"]
            a = S["sprite"][:, :, 3].astype(np.float32) / 255.0
            sub = hole[y0:y1, x0:x1]
            np.maximum(sub, a, out=sub)  # union

        if dilate_px > 0:
            try:
                pil = Image.fromarray((hole * 255).astype(np.uint8))
                # Prefer MaxFilter (morphological dilation); fallback to Gaussian if unavailable.
                pil = pil.filter(ImageFilter.MaxFilter(size=dilate_px * 2 + 1))
                hole = np.array(pil).astype(np.float32) / 255.0
            except Exception:
                pil = Image.fromarray((hole * 255).astype(np.uint8))
                pil = pil.filter(ImageFilter.GaussianBlur(radius=dilate_px))
                hole = np.array(pil).astype(np.float32) / 255.0

        return np.clip(hole, 0.0, 1.0)

    # -------------------------------
    # Layout (fallback positions)
    # -------------------------------
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
            print("[SPRITE_BREAK] WD using frozen kerning-accurate positions from TBS.")
            return [(x, y) for (x, y, _ch) in self.frozen_letter_positions]
        print("[SPRITE_BREAK] WD no handoff; using default positions.")
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
            print(f"[SPRITE_BREAK] WD dissolve schedule: first index={first_idx} "
                  f"char='{self.word[first_idx]}' at frame={self.stable_frames}")

    # -------------------------------
    # Main render
    # -------------------------------
    def render_word_frame(
        self,
        frame: np.ndarray,                     # RGB uint8
        frame_idx: int,
        mask: Optional[Union[np.ndarray, Image.Image, bool]] = None
    ) -> np.ndarray:
        """
        Render word with dissolving letters onto a frame, using exclusive sprites.
        Applies **soft occlusion** to match TBS.
        """
        h, w = frame.shape[:2]

        if self.center_position:
            center_x, center_y = self.center_position
        else:
            center_x = w // 2
            center_y = int(h * 0.45)
            self.center_position = (center_x, center_y)

        # Prepare soft background visibility if occluded handoff + mask provided
        bg_vis: Optional[np.ndarray] = None
        if self.frozen_occlusion and (mask is not None):
            subject_mask = self._ensure_mask_float01(mask, w=w, h=h)  # (h,w) float [0..1]
            if subject_mask is not None:
                bg_vis = 1.0 - subject_mask  # 1=background visible, 0=behind subject
                cov = float(subject_mask.mean())
                if frame_idx % 5 == 0:
                    print(f"[SPRITE_BREAK] soft-occlusion: frame={frame_idx} mask_present=True cov={cov:.3f}")

        # ----------------- Phase A: Stable (pre-dissolve) -----------------
        if frame_idx < self.stable_frames:
            if self.frozen_text_rgba is not None:
                text_layer_np = self.frozen_text_rgba
                text_alpha = (text_layer_np[:, :, 3].astype(np.float32) / 255.0)  # (h,w)
                if bg_vis is not None:
                    visible_alpha = text_alpha * bg_vis
                else:
                    visible_alpha = text_alpha

                if frame_idx == 0:
                    print("[SPRITE_BREAK] stable-phase uses FROZEN RGBA "
                          f"+ {'soft occlusion' if bg_vis is not None else 'no occlusion'}.")

                base = frame.astype(np.float32)
                text_rgb = text_layer_np[:, :, :3].astype(np.float32)
                out = base * (1.0 - visible_alpha[..., None]) + text_rgb * visible_alpha[..., None]
                return np.clip(out, 0, 255).astype(np.uint8)

            # Fallback redraw (rare)
            print("[SPRITE_BREAK] WD WARNING: no frozen RGBA; redrawing stable frame.")
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

            text_alpha = (text_layer_np[:, :, 3].astype(np.float32) / 255.0)
            if bg_vis is not None:
                visible_alpha = text_alpha * bg_vis
            else:
                visible_alpha = text_alpha
            base = frame.astype(np.float32)
            out = base * (1.0 - visible_alpha[..., None]) + text_layer_np[:, :, :3].astype(np.float32) * visible_alpha[..., None]
            return np.clip(out, 0, 255).astype(np.uint8)

        # ----------------- Phase B: Dissolve begins -----------------
        if frame_idx == self.stable_frames:
            first_idx = self.letter_indices[0] if self.letter_indices else None
            first_ch = self.word[first_idx] if first_idx is not None else None
            self._dbg(f"dissolve starts at t={frame_idx}; first='{first_ch}' (index={first_idx})")
        
        # Periodic progress ping
        if frame_idx % 30 == 0:
            le = getattr(self, "last_end_frame", None)
            if le is not None:
                print(f"[DISSOLVE_TIMING] t={frame_idx} / last_end={le}")

        use_sprites = (self.frozen_text_rgba is not None and self.letter_sprites and any(self.letter_sprites))
        result = frame.copy()

        # ---------- Sprite-based path (preferred & pixel-accurate) ----------
        if use_sprites:
            # Categorize letters by their state
            not_started_indices: List[int] = []
            dissolving_indices: List[int] = []
            completed_indices: List[int] = []
            
            for i in range(len(self.word)):
                info = next(d for d in self.letter_dissolvers if d['index'] == i)
                start_frame = info['start_frame']
                end_frame = start_frame + self.dissolve_frames
                
                if frame_idx < start_frame:
                    not_started_indices.append(i)
                elif frame_idx < end_frame:
                    dissolving_indices.append(i)
                else:
                    completed_indices.append(i)
            
            # Debug: Log letter states every 10 frames
            if frame_idx % 10 == 0 or frame_idx == self.stable_frames:
                self._dbg(f"[LIFECYCLE] frame={frame_idx}: not_started={not_started_indices}, "
                         f"dissolving={dissolving_indices}, completed={completed_indices}")

            # 2.a Build the base "frozen word" layer and punch holes for dissolving AND completed letters
            base_rgba = self.frozen_text_rgba  # (H,W,4)
            base_rgb = base_rgba[:, :, :3].astype(np.float32)
            base_alpha = (base_rgba[:, :, 3].astype(np.float32) / 255.0)  # (H,W)

            # Create holes for BOTH dissolving AND completed letters
            # This prevents completed letters from reappearing
            all_removed_indices = dissolving_indices + completed_indices
            
            # Dynamic hole: outline + glow extent (avoid edge remnants that look like a 'frame')
            glow_for_hole = 0
            if all_removed_indices:
                dims = []
                for i in all_removed_indices:
                    S = self.letter_sprites[i]
                    if S is not None:
                        sh, sw = S['sprite'].shape[:2]
                        dims.append(max(sh, sw))
                if dims:
                    # Increase glow calculation to 20% to better catch artifacts
                    glow_for_hole = int(round(max(dims) * 0.20))
            
            # Increase base hole radius for better artifact removal
            hole_radius = max(2, int(self.outline_width + 2), glow_for_hole // 2 + 2)
            hole = self._union_hole_mask(h, w, all_removed_indices, dilate_px=hole_radius)
            
            if all_removed_indices:
                self._dbg(f"[HOLE] frame={frame_idx} hole_radius={hole_radius}px "
                         f"for dissolving={dissolving_indices} completed={completed_indices}")
            if hole is not None:
                base_alpha = base_alpha * (1.0 - hole)

            # Respect soft occlusion (same rule as TBS)
            if bg_vis is not None:
                base_alpha = base_alpha * bg_vis

            # Composite the base (all non-dissolving letters in their exact frozen look)
            result_f = result.astype(np.float32)
            out = result_f * (1.0 - base_alpha[..., None]) + base_rgb * base_alpha[..., None]
            result = np.clip(out, 0, 255).astype(np.uint8)

            # Debug: when a letter *starts* dissolving, log hole size & bounds
            for i in dissolving_indices:
                info = next(d for d in self.letter_dissolvers if d['index'] == i)
                if frame_idx == info['start_frame']:
                    S = self.letter_sprites[i]
                    if S:
                        x0, y0, x1, y1 = S["bbox"]
                        hole_px = int((hole[y0:y1, x0:x1] > 0.01).sum()) if hole is not None else 0
                        self._dbg(
                            f"start letter idx={i} '{self.word[i]}' bbox=({x0},{y0},{x1},{y1}) "
                            f"pad={S.get('pad_px')}px hole_radius={hole_radius}px hole_pixels={hole_px}"
                        )

            # 2.b Render the dissolving letters on top (scaled/faded/floated)
            for i in dissolving_indices:
                S = self.letter_sprites[i]
                if S is None:
                    continue

                spr = S["sprite"]
                pivot_x, pivot_y = S["pivot"]

                info = next(d for d in self.letter_dissolvers if d['index'] == i)
                start_frame = info['start_frame']
                progress = (frame_idx - start_frame) / max(1, self.dissolve_frames)
                opacity = int(255 * (1 - progress))
                
                # Debug progress for each dissolving letter
                if frame_idx % 15 == 0 or opacity <= 10:
                    self._dbg(f"[DISSOLVE] frame={frame_idx} letter='{self.word[i]}' idx={i} "
                             f"progress={progress:.2f} opacity={opacity} "
                             f"start={start_frame} end={start_frame+self.dissolve_frames}")
                
                if opacity <= 0:
                    self._dbg(f"[SKIP] frame={frame_idx} letter='{self.word[i]}' idx={i} opacity=0")
                    continue

                current_scale = 1.0 + (self.max_scale - 1.0) * progress
                float_offset = int(progress * self.float_distance)

                spr_scaled = self._scaled_sprite(spr, current_scale, opacity)
                sh, sw = spr_scaled.shape[:2]
                nx, ny = self._top_left_from_pivot(pivot_x, (pivot_y - float_offset), sw, sh)

                # Log sprite bounds for debugging artifacts
                if frame_idx == start_frame or frame_idx % 30 == 0:
                    self._dbg(f"[BOUNDS] frame={frame_idx} letter='{self.word[i]}' "
                             f"sprite_size=({sw},{sh}) pos=({nx},{ny}) scale={current_scale:.2f}")

                # Optional soft glow (arc-safe)
                if self.glow_effect and progress > 0.2:
                    glow = self._soft_glow_sprite(spr_scaled, opacity, progress)
                    self._alpha_blit(result, glow, (nx, ny), visible_mask=bg_vis)

                self._alpha_blit(result, spr_scaled, (nx, ny), visible_mask=bg_vis)

            return result

        # ---------- Fallback path (no sprites; redraw letters) ----------
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

        for idx in dissolving_indices:
            letter = self.word[idx]
            lx, ly = stable_positions[idx]
            info = next(d for d in self.letter_dissolvers if d['index'] == idx)  # fixed
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
        text_alpha = (text_layer_np[:, :, 3].astype(np.float32) / 255.0)
        if bg_vis is not None:
            visible_alpha = text_alpha * bg_vis
        else:
            visible_alpha = text_alpha
        base = frame.astype(np.float32)
        out = base * (1.0 - visible_alpha[..., None]) + text_layer_np[:, :, :3].astype(np.float32) * visible_alpha[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    # -------------------------------
    # API
    # -------------------------------
    def process_frames(self) -> list:
        return []

    def animate(self, output_path: str) -> bool:
        order = ''.join([self.word[i] for i in self.letter_indices])
        print(f"[SPRITE_BREAK] WD animation; Word='{self.word}' Letters={len(self.word)} "
              f"Dissolve order='{order}' Total duration={self.duration}s")
        print(f"[SPRITE_BREAK] ✓ WD complete")
        return True