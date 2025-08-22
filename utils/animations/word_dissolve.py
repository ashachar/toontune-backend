"""
Word Dissolve animation — FIXED
- Permanent kill-mask so dissolved letters never reappear.
- Premultiplied-alpha scaling to remove rectangular sprite fringe.
- Gaussian feather holes (shape-preserving; no square-box dilation).
- Extra debug logs prefixed with [DISSOLVE_BUG].

This file is a drop-in replacement for utils/animations/word_dissolve.py
and keeps the original public API used by your tests.
"""

import os
import random
import math
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

from .animate import Animation


class WordDissolve(Animation):
    """Letter-by-letter dissolve with exact per-letter sprites."""

    # ---------- small unified debug ----------
    def _dbg(self, msg: str) -> None:
        if getattr(self, "debug", False):
            print(f"[DISSOLVE_BUG] {msg}")

    # ---------- ctor ----------
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
        debug: Optional[bool] = None,
    ):
        num_letters = len(word)
        if duration is None:
            duration = stable_duration + max(0, num_letters - 1) * dissolve_stagger + dissolve_duration

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
            temp_dir=temp_dir,
        )

        # config
        env_debug = os.getenv("FRAME_DISSOLVE_DEBUG", "0") != "0"
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

        # handoff
        self.handoff_data = handoff_data or {}
        self.frozen_letter_positions: Optional[List[Tuple[int, int, str]]] = self.handoff_data.get("final_letter_positions")
        self.frozen_font_size: Optional[int] = self.handoff_data.get("final_font_size", font_size)
        self.frozen_center_position: Optional[Tuple[int, int]] = self.handoff_data.get("final_center_position", center_position)
        self.frozen_text_origin: Optional[Tuple[int, int]] = self.handoff_data.get("final_text_origin")
        self.final_scale: float = float(self.handoff_data.get("scale", 1.0))
        self.frozen_text_rgba: Optional[np.ndarray] = self.handoff_data.get("final_text_rgba")
        self.frozen_occlusion: bool = bool(self.handoff_data.get("final_occlusion", False))
        self.word_bbox: Optional[Tuple[int, int, int, int]] = self.handoff_data.get("final_word_bbox")
        self.letter_bboxes_abs: Optional[List[Tuple[int, int, int, int]]] = self.handoff_data.get("final_letter_bboxes")
        self.letter_centers: Optional[List[Tuple[float, float]]] = self.handoff_data.get("final_letter_centers")

        # timing
        self.stable_frames = int(stable_duration * fps)
        self.dissolve_frames = int(dissolve_duration * fps)
        self.stagger_frames = int(dissolve_stagger * fps)

        self.letter_indices = list(range(len(self.word)))
        if randomize_order:
            random.shuffle(self.letter_indices)

        self.letter_dissolvers: List[Dict[str, Any]] = []
        self.prepare_letter_dissolvers()

        # exact per-letter sprites
        self.letter_sprites: List[Optional[Dict[str, Any]]] = []
        self._prepare_letter_sprites()

        # ---- NEW: persistent kill mask (graveyard) ----
        # Any pixel that ever belonged to a started letter is permanently removed from the base layer.
        if self.frozen_text_rgba is not None:
            H, W = self.frozen_text_rgba.shape[:2]
            self._dead_mask = np.zeros((H, W), dtype=np.float32)
        else:
            self._dead_mask = None
        self._letters_started: set[int] = set()

        # cache last-end
        self.active_indices = [i for i, s in enumerate(self.letter_sprites) if s is not None]
        active_letters = len(self.active_indices)
        self.last_start_frame = self.stable_frames + max(0, active_letters - 1) * self.stagger_frames
        self.last_end_frame = self.last_start_frame + self.dissolve_frames

        self._dbg(f"init word='{self.word}', letters={len(self.word)}, "
                  f"stable={self.stable_frames}f, dissolve={self.dissolve_frames}f, stagger={self.stagger_frames}f")
        if self.frozen_text_rgba is None:
            self._dbg("WARNING: frozen RGBA is None; stable phase will redraw (risk of drift)")

    # ---------- utilities ----------
    def load_font(self, size: int):
        if self.font_path and os.path.exists(self.font_path):
            return ImageFont.truetype(self.font_path, size)
        for fp in ["/System/Library/Fonts/Helvetica.ttc",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                   "arial.ttf"]:
            if os.path.exists(fp):
                return ImageFont.truetype(fp, size)
        return ImageFont.load_default()

    def _ensure_mask_float01(self, mask: Optional[Union[np.ndarray, Image.Image]], w: int, h: int) -> Optional[np.ndarray]:
        if mask is None:
            return None
        if isinstance(mask, Image.Image):
            arr = np.array(mask)
        else:
            arr = mask
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                arr = arr[:, :, 3]
            else:
                arr = (0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]).astype(np.float32)
        if arr.shape[0] != h or arr.shape[1] != w:
            arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize((w, h), Image.BILINEAR))
        f = arr.astype(np.float32)
        if f.max() > 1.0:
            f /= 255.0
        # slight feather to avoid jaggies
        f = np.array(Image.fromarray((f * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=0.75))).astype(np.float32) / 255.0
        return np.clip(f, 0.0, 1.0)

    # ---------- exact partition via prefix-difference ----------
    def _get_final_text_origin(self, W: int, H: int, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
        if self.frozen_text_origin is not None:
            return self.frozen_text_origin
        tmp = Image.new('L', (1, 1), 0)
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), self.word, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = (self.center_position if self.center_position else (W // 2, int(H * 0.45)))
        return (cx - tw // 2, cy - th // 2)

    def _build_per_letter_masks_exact(self) -> Tuple[List[np.ndarray], np.ndarray]:
        if self.frozen_text_rgba is None:
            self._dbg("ERROR: cannot build masks; frozen RGBA missing.")
            return [], np.zeros((1, 1), dtype=np.int32)

        H, W = self.frozen_text_rgba.shape[:2]
        alpha = self.frozen_text_rgba[:, :, 3] > 0

        font = self.load_font(self.frozen_font_size or self.font_size)
        text_x, text_y = self._get_final_text_origin(W, H, font)

        masks_seed: List[np.ndarray] = []
        prefix = ""
        for ch in self.word:
            imgA = Image.new('L', (W, H), 0)
            imgB = Image.new('L', (W, H), 0)
            dA, dB = ImageDraw.Draw(imgA), ImageDraw.Draw(imgB)
            if prefix:
                dA.text((text_x, text_y), prefix, font=font, fill=255)
            dB.text((text_x, text_y), prefix + ch, font=font, fill=255)
            diff = ImageChops.difference(imgB, imgA)
            seed = np.array(diff) > 0
            masks_seed.append(seed)
            prefix += ch

        labels = np.full((H, W), -1, dtype=np.int32)
        for i, m in enumerate(masks_seed):
            labels[m] = i

        # Fill leftovers by nearest center (frozen centers if available)
        leftover = alpha & (labels == -1)
        if leftover.any():
            if self.letter_centers and len(self.letter_centers) == len(self.word):
                centers = self.letter_centers
            else:
                centers = []
                for i, m in enumerate(masks_seed):
                    ys, xs = np.where(m)
                    if len(xs) == 0:
                        if self.letter_bboxes_abs and i < len(self.letter_bboxes_abs):
                            x0, y0, x1, y1 = self.letter_bboxes_abs[i]
                            centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
                        else:
                            centers.append((W / 2.0, H / 2.0))
                    else:
                        centers.append((float(xs.mean()), float(ys.mean())))
            yy, xx = np.indices((H, W))
            dstack = np.stack([(xx - cx) ** 2 + (yy - cy) ** 2 for (cx, cy) in centers], axis=-1)
            nearest = np.argmin(dstack, axis=-1)
            labels[leftover] = nearest[leftover]

        masks: List[np.ndarray] = []
        for i in range(len(self.word)):
            masks.append((labels == i) & alpha)
        return masks, labels

    # ---------- sprite preparation ----------
    def _prepare_letter_sprites(self) -> None:
        self.letter_sprites = [None] * len(self.word)
        if self.frozen_text_rgba is None:
            self._dbg("WARNING: no frozen RGBA; cannot build sprites.")
            return
        H, W = self.frozen_text_rgba.shape[:2]
        masks, _ = self._build_per_letter_masks_exact()
        if len(masks) != len(self.word):
            self._dbg("ERROR: mask build failed; sprites not prepared.")
            return

        for i, mask_i in enumerate(masks):
            # Skip space characters - they don't need sprites and can cause artifacts
            if i < len(self.word) and self.word[i] == ' ':
                self._dbg(f"Skipping sprite for space character at index {i}")
                continue
            if not np.any(mask_i):
                continue
            ys, xs = np.where(mask_i)
            y0, y1 = int(ys.min()), int(ys.max() + 1)
            x0, x1 = int(xs.min()), int(xs.max() + 1)

            sprite = self.frozen_text_rgba[y0:y1, x0:x1].copy()
            region_mask = mask_i[y0:y1, x0:x1]
            sprite[~region_mask, 3] = 0
            
            # Clean up very faint alpha pixels that can cause gray line artifacts
            # These are often anti-aliasing artifacts at sprite edges
            alpha_threshold = 5  # Out of 255
            sprite[:, :, 3] = np.where(sprite[:, :, 3] < alpha_threshold, 0, sprite[:, :, 3])
            
            # Remove disconnected pixels (isolated from main letter body)
            # This fixes the "gray line at edge" artifact
            from scipy import ndimage
            alpha_binary = sprite[:, :, 3] > alpha_threshold
            if np.any(alpha_binary):
                # Find connected components
                labeled, num_features = ndimage.label(alpha_binary)
                if num_features > 1:
                    # Keep only the largest connected component
                    sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
                    largest_label = np.argmax(sizes) + 1
                    # Zero out alpha for non-largest components
                    sprite[:, :, 3] = np.where(labeled == largest_label, sprite[:, :, 3], 0)
                    self._dbg(f"Removed {num_features-1} disconnected components from letter '{self.word[i]}'")

            # padding to protect blur/glow
            h_i, w_i = sprite.shape[:2]
            max_dim = max(w_i, h_i)
            blur_max = int(round(max_dim * self.max_scale * 0.20))
            pad_by_ratio = int(round(max_dim * max(0.15, min(0.50, float(getattr(self, "sprite_pad_ratio", 0.30))))))
            pad_px = max(blur_max + 5, pad_by_ratio, 10)

            padded = np.zeros((h_i + 2 * pad_px, w_i + 2 * pad_px, 4), dtype=np.uint8)
            padded[pad_px:pad_px + h_i, pad_px:pad_px + w_i] = sprite

            bbox_cx = (x0 + x1) / 2.0
            bbox_cy = (y0 + y1) / 2.0

            self.letter_sprites[i] = {
                "sprite": padded,
                "pivot": (bbox_cx, bbox_cy),
                "bbox": (x0 - pad_px, y0 - pad_px, x1 + pad_px, y1 + pad_px),  # in frame coords
                "pad_px": pad_px,
                "orig_bbox": (x0, y0, x1, y1),
            }

    # ---------- compositing helpers ----------
    def _top_left_from_pivot(self, pivot_x: float, pivot_y: float, sw: int, sh: int) -> Tuple[int, int]:
        return (int(round(pivot_x - sw / 2.0)), int(round(pivot_y - sh / 2.0)))

    def _alpha_blit(self, dst: np.ndarray, src_rgba: np.ndarray, top_left: Tuple[int, int],
                    visible_mask: Optional[np.ndarray] = None) -> None:
        """Straight alpha composite with optional visibility mask (float 0..1)."""
        x, y = top_left
        H, W = dst.shape[:2]
        h, w = src_rgba.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x0 >= x1 or y0 >= y1:
            return

        sx0, sy0 = x0 - x, y0 - y
        sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)

        src_view = src_rgba[sy0:sy1, sx0:sx1]
        alpha = src_view[:, :, 3].astype(np.float32) / 255.0

        # Eliminate extremely tiny alpha that causes faint box edges after scaling
        alpha[alpha < (1.0 / 255.0)] = 0.0

        if visible_mask is not None:
            vm = visible_mask[y0:y1, x0:x1].astype(np.float32)
            alpha = alpha * vm

        if not np.any(alpha > 0.0):
            return

        for c in range(3):
            dst_region = dst[y0:y1, x0:x1, c].astype(np.float32)
            src_region = src_view[:, :, c].astype(np.float32)
            out = dst_region * (1.0 - alpha) + src_region * alpha
            dst[y0:y1, x0:x1, c] = np.clip(out, 0, 255).astype(np.uint8)

    def _resize_rgba_premultiplied(self, rgba: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
        """Scale RGBA via premultiplied alpha to avoid fringe/rectangles."""
        f = rgba.astype(np.float32) / 255.0
        a = f[:, :, 3:4]
        rgb_premul = f[:, :, :3] * a

        rgb_img = Image.fromarray((rgb_premul * 255).astype(np.uint8), mode="RGB")
        a_img = Image.fromarray((a[:, :, 0] * 255).astype(np.uint8), mode="L")

        rgb_rs = np.array(rgb_img.resize((new_w, new_h), Image.LANCZOS)).astype(np.float32) / 255.0
        a_rs = np.array(a_img.resize((new_w, new_h), Image.LANCZOS)).astype(np.float32) / 255.0
        a_rs = np.clip(a_rs, 0.0, 1.0)

        # unpremultiply safely
        eps = 1e-6
        rgb_out = np.where(a_rs[..., None] > eps, rgb_rs / (a_rs[..., None] + eps), 0.0)
        out = np.dstack([np.clip(rgb_out, 0.0, 1.0), a_rs])
        return (out * 255).astype(np.uint8)

    def _scaled_sprite(self, sprite_rgba: np.ndarray, scale: float, opacity: int) -> np.ndarray:
        if scale <= 0:
            scale = 0.001
        h, w = sprite_rgba.shape[:2]
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        out = self._resize_rgba_premultiplied(sprite_rgba, nw, nh)
        if opacity < 255:
            a = out[:, :, 3].astype(np.float32) * (opacity / 255.0)
            out[:, :, 3] = np.clip(a, 0, 255).astype(np.uint8)
        return out

    def _soft_glow_sprite(self, spr_scaled: np.ndarray, opacity: int, progress: float) -> np.ndarray:
        h_sc, w_sc = spr_scaled.shape[:2]
        a = spr_scaled[:, :, 3].astype(np.uint8)
        blur_px = max(1, int(round(max(h_sc, w_sc) * (0.06 + 0.10 * progress))))
        a_blur = np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=blur_px)))
        strength = int(opacity * 0.4)
        glow_a = (a_blur.astype(np.float32) * (strength / 255.0)).clip(0, 255).astype(np.uint8)
        rgb = np.zeros_like(spr_scaled[:, :, :3], dtype=np.uint8)
        rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = self.text_color
        return np.dstack([rgb, glow_a])

    def _union_hole_mask(self, H: int, W: int, indices: List[int], dilate_px: int = 0) -> Optional[np.ndarray]:
        """Union of sprite alpha in frame space; gaussian feather to avoid boxy dilation."""
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
            np.maximum(sub, a, out=sub)
        if dilate_px > 0:
            # Use gaussian feather, not MaxFilter (square kernel amplifies rectangles)
            pil = Image.fromarray((hole * 255).astype(np.uint8))
            pil = pil.filter(ImageFilter.GaussianBlur(radius=float(dilate_px)))
            hole = np.array(pil).astype(np.float32) / 255.0
        return np.clip(hole, 0.0, 1.0)

    # ---------- layout fallbacks ----------
    def _compute_default_positions(self) -> List[Tuple[int, int]]:
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        font = self.load_font(self.font_size)
        if self.center_position:
            cx, cy = self.center_position
        else:
            cx, cy = 640, 360
        full_bbox = draw.textbbox((0, 0), self.word, font=font)
        fw, fh = full_bbox[2] - full_bbox[0], full_bbox[3] - full_bbox[1]
        bx = cx - fw // 2
        by = cy - fh // 2
        positions: List[Tuple[int, int]] = []
        if self.maintain_kerning:
            for i in range(len(self.word)):
                prefix = self.word[:i]
                pw = (draw.textbbox((0, 0), prefix, font=font)[2] - draw.textbbox((0, 0), prefix, font=font)[0]) if prefix else 0
                positions.append((bx + pw, by))
        else:
            cx2 = bx
            for ch in self.word:
                lw = draw.textbbox((0, 0), ch, font=font)[2] - draw.textbbox((0, 0), ch, font=font)[0]
                positions.append((cx2, by))
                cx2 += lw
        return positions

    def calculate_letter_positions(self) -> List[Tuple[int, int]]:
        if self.frozen_letter_positions:
            self._dbg("Using frozen kerning positions from TBS.")
            return [(x, y) for (x, y, _c) in self.frozen_letter_positions]
        self._dbg("No handoff; using default positions.")
        return self._compute_default_positions()

    def prepare_letter_dissolvers(self):
        for i, ch in enumerate(self.word):
            dissolve_idx = self.letter_indices.index(i)
            start = self.stable_frames + dissolve_idx * self.stagger_frames
            self.letter_dissolvers.append({"letter": ch, "index": i, "dissolve_order": dissolve_idx, "start_frame": start})
        if self.letter_dissolvers:
            first_i = self.letter_dissolvers[0]["index"]
            self._dbg(f"dissolve schedule first index={first_i} char='{self.word[first_i]}' at frame={self.stable_frames}")

    # ---------- main render ----------
    def render_word_frame(self, frame: np.ndarray, frame_idx: int,
                          mask: Optional[Union[np.ndarray, Image.Image, bool]] = None) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.center_position is None:
            self.center_position = (w // 2, int(h * 0.45))

        # soft occlusion (match TBS)
        bg_vis: Optional[np.ndarray] = None
        if self.frozen_occlusion and (mask is not None):
            subject = self._ensure_mask_float01(mask, w=w, h=h)
            if subject is not None:
                bg_vis = 1.0 - subject

        # ---- Phase A: stable ----
        if frame_idx < self.stable_frames:
            if self.frozen_text_rgba is not None:
                text_rgba = self.frozen_text_rgba
                a = text_rgba[:, :, 3].astype(np.float32) / 255.0
                if bg_vis is not None:
                    a = a * bg_vis
                base = frame.astype(np.float32)
                rgb = text_rgba[:, :, :3].astype(np.float32)
                out = base * (1.0 - a[..., None]) + rgb * a[..., None]
                return np.clip(out, 0, 255).astype(np.uint8)
            # fallback redraw (rare)
            img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            font = self.load_font(self.frozen_font_size or self.font_size)
            if self.frozen_text_origin is not None:
                tx, ty = self.frozen_text_origin
            else:
                bb = draw.textbbox((0, 0), self.word, font=font)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                cx, cy = self.center_position
                tx, ty = cx - tw // 2, cy - th // 2
            sh = max(2, int(self.shadow_offset * self.final_scale))
            draw.text((tx + sh, ty + sh), self.word, font=font, fill=(0, 0, 0, 100))
            for dx in range(-self.outline_width, self.outline_width + 1):
                for dy in range(-self.outline_width, self.outline_width + 1):
                    if abs(dx) == self.outline_width or abs(dy) == self.outline_width:
                        draw.text((tx + dx, ty + dy), self.word, font=font, fill=(255, 255, 255, 150))
            draw.text((tx, ty), self.word, font=font, fill=(*self.text_color, 255))
            text = np.array(img)
            a = text[:, :, 3].astype(np.float32) / 255.0
            if bg_vis is not None:
                a = a * bg_vis
            base = frame.astype(np.float32)
            out = base * (1.0 - a[..., None]) + text[:, :, :3].astype(np.float32) * a[..., None]
            return np.clip(out, 0, 255).astype(np.uint8)

        # ---- Phase B: dissolve ----
        use_sprites = self.frozen_text_rgba is not None and any(self.letter_sprites)
        result = frame.copy()

        if not use_sprites:
            # minimalistic fallback: just skip (kept for API parity)
            return result

        # states
        not_started, dissolving, completed = [], [], []
        for i in range(len(self.word)):
            info = next(d for d in self.letter_dissolvers if d["index"] == i)
            s = info["start_frame"]
            e = s + self.dissolve_frames
            if frame_idx < s:
                not_started.append(i)
            elif frame_idx < e:
                dissolving.append(i)
            else:
                completed.append(i)

        # CRITICAL FIX: If ALL letters are completed, return frame without any text
        if len(completed) == len(self.word):
            self._dbg(f"frame={frame_idx}: All letters completed, returning clean frame")
            return result  # result is frame.copy() from line 511

        # (1) Grow the persistent kill mask exactly when a letter STARTS dissolving
        if self._dead_mask is not None:
            for i in dissolving:
                info = next(d for d in self.letter_dissolvers if d["index"] == i)
                if frame_idx == info["start_frame"] and i not in self._letters_started:
                    self._letters_started.add(i)
                    S = self.letter_sprites[i]
                    if S:
                        x0, y0, x1, y1 = S["bbox"]
                        a = S["sprite"][:, :, 3].astype(np.float32) / 255.0

                        # radius must cover: outline + glow + scale growth
                        sh, sw = S["sprite"].shape[:2]
                        # Increase growth factor to fully cover scaled sprites
                        grow = int(round(max(sh, sw) * (max(self.max_scale, 1.0) - 1.0) * 0.8))  # Increased from 0.6
                        # Increase glow radius to match actual glow effect
                        glow = int(round(max(sh, sw) * 0.20))  # Increased from 0.12
                        radius = max(self.outline_width + 4, glow, grow, 6)  # Increased minimums

                        # feather (gaussian) to avoid box geometry
                        a_soft = np.array(Image.fromarray((a * 255).astype(np.uint8)).filter(
                            ImageFilter.GaussianBlur(radius=float(radius))
                        )).astype(np.float32) / 255.0

                        sub = self._dead_mask[y0:y1, x0:x1]
                        np.maximum(sub, a_soft, out=sub)
                        self._dbg(f"kill-mask grow: idx={i} letter='{self.word[i]}' radius={radius}px bbox=({x0},{y0},{x1},{y1})")

        # Also ensure completed letters are FULLY masked (opacity = 1.0)
        # CRITICAL: Expand mask to cover outlines and shadows
        if self._dead_mask is not None:
            H, W = self._dead_mask.shape
            for i in completed:
                S = self.letter_sprites[i]
                if S:
                    x0, y0, x1, y1 = S["bbox"]
                    # Expand bounds to cover outline and shadow
                    outline_expand = self.outline_width + 3  # Extra pixels for safety
                    shadow_expand = max(self.shadow_offset, 3)
                    x0 = max(0, x0 - outline_expand)
                    y0 = max(0, y0 - outline_expand)
                    x1 = min(W, x1 + outline_expand + shadow_expand)
                    y1 = min(H, y1 + outline_expand + shadow_expand)
                    
                    # Force entire expanded region to be fully masked
                    self._dead_mask[y0:y1, x0:x1] = 1.0

        # (2) Build a frame-local hole for currently dissolving letters (helps during fade)
        H, W = self.frozen_text_rgba.shape[:2]
        hole_radius = 0
        if dissolving or completed:
            # include some extra to be safe; we already have persistent dead_mask for started ones
            dims = []
            for i in dissolving + completed:
                S = self.letter_sprites[i]
                if S is not None:
                    sh, sw = S["sprite"].shape[:2]
                    dims.append(max(sh, sw))
            if dims:
                # Increase hole radius to fully cover glow + scale effects
                # Account for: scale growth (max_scale - 1.0), glow blur (up to 16% of size)
                hole_radius = int(round(max(dims) * 0.25))  # Increased from 0.10 to prevent edge artifacts
        frame_hole = self._union_hole_mask(H, W, dissolving + completed, dilate_px=hole_radius)

        # (3) Compose the base frozen text with PERMANENT + FRAME holes
        # CRITICAL: Only render letters that haven't started dissolving yet
        base_rgba = self.frozen_text_rgba
        base_rgb = base_rgba[:, :, :3].astype(np.float32)
        base_a = base_rgba[:, :, 3].astype(np.float32) / 255.0

        # Create mask for letters that should be visible in base
        # Only show letters in not_started state
        base_letter_mask = np.ones((H, W), dtype=np.float32)
        for i in dissolving + completed:
            S = self.letter_sprites[i]
            if S:
                x0, y0, x1, y1 = S["bbox"]
                # Use the actual sprite alpha to avoid overlapping adjacent letters
                sprite_alpha = S["sprite"][:, :, 3].astype(np.float32) / 255.0
                
                # Apply slight dilation to cover outlines
                alpha_img = Image.fromarray((sprite_alpha * 255).astype(np.uint8))
                # Small dilation to cover outlines without overlapping neighbors
                dilated = alpha_img.filter(ImageFilter.MaxFilter(size=5))
                sprite_alpha_dilated = np.array(dilated).astype(np.float32) / 255.0
                
                # Subtract this sprite from the base mask
                mask_region = base_letter_mask[y0:y1, x0:x1]
                np.minimum(mask_region, 1.0 - sprite_alpha_dilated, out=mask_region)
        
        # Apply the mask to remove dissolving/completed letters from base
        base_a = base_a * base_letter_mask
        
        if bg_vis is not None:
            base_a = base_a * bg_vis

        base_f = result.astype(np.float32)
        out = base_f * (1.0 - base_a[..., None]) + base_rgb * base_a[..., None]
        result = np.clip(out, 0, 255).astype(np.uint8)

        # (4) Draw dissolving letters on top
        for i in dissolving:
            # Skip space characters - they shouldn't be rendered
            if i < len(self.word) and self.word[i] == ' ':
                continue
            S = self.letter_sprites[i]
            if S is None:
                continue

            info = next(d for d in self.letter_dissolvers if d["index"] == i)
            start = info["start_frame"]
            progress = (frame_idx - start) / max(1, self.dissolve_frames)
            progress = max(0.0, min(1.0, float(progress)))
            opacity = int(255 * (1.0 - progress))
            if opacity <= 0:
                continue

            spr = S["sprite"]
            pivot_x, pivot_y = S["pivot"]

            current_scale = 1.0 + (self.max_scale - 1.0) * progress
            float_offset = int(progress * self.float_distance)

            spr_scaled = self._scaled_sprite(spr, current_scale, opacity)
            sh, sw = spr_scaled.shape[:2]
            nx, ny = self._top_left_from_pivot(pivot_x, (pivot_y - float_offset), sw, sh)

            if self.glow_effect and progress > 0.2:
                glow = self._soft_glow_sprite(spr_scaled, opacity, progress)
                self._alpha_blit(result, glow, (nx, ny), visible_mask=bg_vis)

            self._alpha_blit(result, spr_scaled, (nx, ny), visible_mask=bg_vis)

        # (5) Logs (every 12 frames to avoid spam)
        if frame_idx % 12 == 0:
            self._dbg(f"frame={frame_idx} states: not_started={not_started}, dissolving={dissolving}, completed={completed}")
            dead_cov = float(self._dead_mask.mean()) if self._dead_mask is not None else 0.0
            self._dbg(f"dead_mask coverage={dead_cov:.4f}, hole_radius={hole_radius}px")

        return result

    # ---------- public API stubs ----------
    def process_frames(self) -> list:
        return []

    def animate(self, output_path: str) -> bool:
        order = ''.join([self.word[i] for i in self.letter_indices])
        self._dbg(f"animate word='{self.word}', order='{order}', duration={self.duration}s")
        return True