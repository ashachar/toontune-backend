#!/usr/bin/env python3
"""
Fix for two issues:
1. Text should start fading at midpoint of shrink (exponential fade)
2. Text should NOT slant at final position (no perspective)
"""

import os
import numpy as np
from utils.animations import text_3d_behind_segment

# Store original method
original_generate_frame = text_3d_behind_segment.Text3DBehindSegment.generate_frame

def fixed_generate_frame(self, frame_number: int, background: Optional[np.ndarray] = None) -> np.ndarray:
    """Fixed version with proper fade timing and no slanting"""
    from typing import Optional
    import cv2
    from PIL import Image
    
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

    # FIXED PHASE LOGIC
    if frame_number < self.phase1_frames:
        # Phase 1: Shrinking
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
            # Exponential fade: alpha = 1 - 0.5 * (1 - e^(-kt)) / (1 - e^(-k))
            k = 4.0  # Exponential factor
            alpha = 1.0 - 0.5 * (1 - np.exp(-k * fade_t)) / (1 - np.exp(-k))
            is_behind = True  # Start masking as it passes behind
        
        # FIX 2: NEVER apply perspective during shrink
        apply_persp = False
        
    elif frame_number < self.phase1_frames + self.phase2_frames:
        # Phase 2: Transition (already behind, continue fading if needed)
        phase = "transition"
        t = (frame_number - self.phase1_frames) / max(self.phase2_frames - 1, 1)
        scale = self.end_scale
        
        # Continue exponential fade to 50% opacity
        # We're already partially faded from phase 1
        # Continue from wherever we left off to 0.5
        start_alpha = 0.75  # Approximate where phase 1 ended
        alpha = start_alpha - (start_alpha - 0.5) * (1 - np.exp(-3.0 * t)) / (1 - np.exp(-3.0))
        is_behind = True
        
        # FIX 2: NO perspective in transition either
        apply_persp = False
        
    else:
        # Phase 3: Stable behind
        phase = "stable"
        scale = self.end_scale
        alpha = 0.5  # Final opacity
        is_behind = True
        
        # FIX 2: NO perspective when stable (prevents slanting)
        apply_persp = False

    # Debug log the key changes
    if self.debug and frame_number % 10 == 0:
        self._log(f"[FADE_FIX] Frame {frame_number}: phase={phase}, scale={scale:.3f}, alpha={alpha:.3f}, is_behind={is_behind}, apply_persp={apply_persp}")

    # Render text with fixes
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

    # If behind, cut by mask
    if is_behind:
        mask_region = self.segment_mask[y1:y2, x1:x2]
        text_alpha = text_layer[y1:y2, x1:x2, 3].astype(np.float32)
        text_alpha *= (1.0 - (mask_region.astype(np.float32) / 255.0))
        text_layer[y1:y2, x1:x2, 3] = text_alpha.astype(np.uint8)

    # Composite
    frame_pil = Image.fromarray(frame)
    text_pil = Image.fromarray(text_layer)
    out = Image.alpha_composite(frame_pil, text_pil)
    result = np.array(out)

    # Return RGB if caller expects video without alpha
    return result[:, :, :3] if result.shape[2] == 4 else result

# Apply the fix
text_3d_behind_segment.Text3DBehindSegment.generate_frame = fixed_generate_frame

print("✅ Fixes applied to Text3DBehindSegment:")
print("  1. Text starts fading at 50% of shrink phase (exponential)")
print("  2. No perspective transform (prevents slanting)")
print("\nNow any new animations will have these fixes!")