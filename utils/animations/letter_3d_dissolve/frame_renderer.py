"""
Frame rendering logic for dissolve animation.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Any


class FrameRenderer:
    """Handles frame-by-frame rendering of the dissolve animation."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
    @staticmethod
    def smoothstep(t: float) -> float:
        """Smooth interpolation function."""
        t = max(0.0, min(1.0, t))
        return t * t * (3 - 2 * t)
    
    def compute_letter_state(
        self,
        frame_number: int,
        timing: Any,  # LetterTiming
        stable_alpha: float,
        max_dissolve_scale: float,
        float_distance: float
    ) -> Tuple[str, float, float, float, bool]:
        """Compute the state of a letter at a given frame."""
        f = frame_number
        
        # Determine phase by frame number (frame-accurate)
        if f < timing.start:
            phase = "stable"
            target_alpha = stable_alpha
            scale = 1.0
            float_y = 0
            add_holes = False
        elif timing.start <= f <= timing.hold_end:
            # 1-frame (or more) safety hold at EXACT stable_alpha
            phase = "hold"
            target_alpha = stable_alpha
            scale = 1.0
            float_y = 0
            add_holes = False
        elif timing.hold_end < f <= timing.end:
            # Dissolve begins AFTER hold; progress starts at ~0 on first dissolve frame
            denom = max(1, (timing.end - timing.hold_end))
            letter_t = (f - timing.hold_end) / denom
            smooth_t = self.smoothstep(letter_t)
            phase = "dissolve"
            target_alpha = stable_alpha * (1.0 - smooth_t * 0.98)
            scale = 1.0 + smooth_t * (max_dissolve_scale - 1.0)
            float_y = -smooth_t * float_distance
            add_holes = letter_t > 0.3
        elif timing.end < f <= timing.fade_end:
            # Fade tail (guaranteed >= 2 frames)
            fade_denom = max(1, (timing.fade_end - timing.end))
            fade_t = (f - timing.end) / fade_denom
            phase = "fade"
            target_alpha = stable_alpha * 0.02 * (1.0 - fade_t)
            scale = max_dissolve_scale
            float_y = -float_distance
            add_holes = True
        else:
            # Completely gone
            phase = "gone"
            target_alpha = 0.0
            scale = 1.0
            float_y = 0
            add_holes = False
        
        return phase, target_alpha, scale, float_y, add_holes
    
    def transform_sprite(
        self,
        sprite: Any,  # LetterSprite
        scale: float,
        float_y: float,
        frame_number: int
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """Transform sprite with scale and float effects."""
        sprite_img = sprite.sprite_3d.copy()
        pos_x, pos_y = sprite.position
        
        # Track position changes
        original_pos = (pos_x, pos_y)
        
        # Calculate visual center BEFORE scaling
        visual_center_x = pos_x + sprite.sprite_3d.width // 2
        visual_center_y = pos_y + sprite.sprite_3d.height // 2
        
        # Debug logging for hold phase
        if not hasattr(sprite, f"_logged_hold_pos_{frame_number}"):
            if self.debug:
                print(f"[LETTER_SHIFT] Frame {frame_number}: '{sprite.char}' in HOLD phase")
                print(f"  Position: {sprite.position}, Visual center: ({visual_center_x}, {visual_center_y})")
            setattr(sprite, f"_logged_hold_pos_{frame_number}", True)
        
        # Apply scaling
        if scale != 1.0:
            new_w = int(round(sprite_img.width * scale))
            new_h = int(round(sprite_img.height * scale))
            sprite_img = sprite_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            # Maintain visual center after scaling
            pos_x = visual_center_x - new_w // 2
            pos_y = visual_center_y - new_h // 2
            
            # Debug logging for scale start
            if abs(scale - 1.0) < 0.01 and not hasattr(sprite, "_logged_scale_start"):
                if self.debug:
                    print(f"[LETTER_SHIFT] Frame {frame_number}: '{sprite.char}' START SCALING")
                    print(f"  Scale: {scale:.4f}, Original pos: {original_pos}")
                    print(f"  New dimensions: {new_w}x{new_h}, New pos: ({pos_x}, {pos_y})")
                    print(f"  Position shift: dx={pos_x-original_pos[0]}, dy={pos_y-original_pos[1]}")
                setattr(sprite, "_logged_scale_start", True)
        
        # Apply vertical float after scaling
        float_shift = int(round(float_y))
        pos_y += float_shift
        
        # Debug logging for position changes
        if not hasattr(sprite, f"_logged_dissolve_{frame_number}"):
            total_dx = pos_x - original_pos[0]
            total_dy = pos_y - original_pos[1]
            if abs(total_dx) > 0 or abs(total_dy) > 0:
                if self.debug:
                    print(f"[LETTER_SHIFT] Frame {frame_number}: '{sprite.char}' POSITION CHANGED")
                    print(f"  Scale: {scale:.3f}, Float: {float_y:.1f}")
                    print(f"  Original: {original_pos} → Final: ({pos_x}, {pos_y})")
                    print(f"  Total shift: dx={total_dx}, dy={total_dy} (float_shift={float_shift})")
                setattr(sprite, f"_logged_dissolve_{frame_number}", True)
        
        return sprite_img, (pos_x, pos_y)
    
    def apply_alpha_and_kill_mask(
        self,
        sprite_array: np.ndarray,
        target_alpha: float,
        kill_mask: Optional[np.ndarray],
        sprite_size: Tuple[int, int],
        handoff_sprite_alpha: Optional[float],
        frame_number: int,
        sprite_char: str,
        phase: str,
        timing_start: int
    ) -> np.ndarray:
        """Apply alpha and kill mask to sprite."""
        # Apply kill mask if any
        if kill_mask is not None and np.any(kill_mask):
            if (sprite_size[0], sprite_size[1]) != (kill_mask.shape[1], kill_mask.shape[0]):
                kill_mask = cv2.resize(kill_mask, sprite_size, interpolation=cv2.INTER_NEAREST)
            sprite_array[:, :, 3] = (sprite_array[:, :, 3] * (1 - kill_mask)).astype(np.uint8)
        
        # Apply alpha using relative multiplier if sprites were premultiplied
        if handoff_sprite_alpha is not None:
            # Sprites are premultiplied by motion's alpha, so we need relative multiplier
            denom = max(handoff_sprite_alpha, 1e-6)
            relative_mult = target_alpha / denom
            # Clamp to reasonable range
            relative_mult = float(np.clip(relative_mult, 0.0, 4.0))
            
            # Debug logging at important moments
            if self.debug and (phase in ("stable", "hold")) and (frame_number == 0 or frame_number == timing_start):
                print(f"[OPACITY_BLINK] frame={frame_number} letter='{sprite_char}' phase={phase} "
                      f"target_alpha={target_alpha:.3f} incoming={handoff_sprite_alpha:.3f} "
                      f"relative_mult={relative_mult:.3f}")
            
            # Apply relative multiplier
            sprite_array[:, :, 3] = np.clip(
                sprite_array[:, :, 3].astype(np.float32) * relative_mult, 0, 255
            ).astype(np.uint8)
            
            # Debug check for first frame
            if self.debug and frame_number == 0 and phase == "stable":
                non_zero_after = sprite_array[:, :, 3][sprite_array[:, :, 3] > 0]
                if len(non_zero_after) > 0:
                    print(f"[OPACITY_BLINK] After applying relative_mult={relative_mult:.3f}:")
                    print(f"  Min alpha: {non_zero_after.min()}, Max alpha: {non_zero_after.max()}, "
                          f"Mean: {non_zero_after.mean():.1f}")
        else:
            # Standalone mode: use absolute multiplier
            sprite_array[:, :, 3] = (sprite_array[:, :, 3] * target_alpha).astype(np.uint8)
        
        return sprite_array