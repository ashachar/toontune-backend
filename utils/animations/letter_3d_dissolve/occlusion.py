"""
Occlusion handling for letters behind foreground objects.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class OcclusionHandler:
    """Handles occlusion of letters behind foreground objects."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
    def extract_fresh_mask(
        self,
        background: np.ndarray,
        frame_number: int,
        resolution: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Extract fresh foreground mask for current frame - NO CACHING!"""
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from video.segmentation.segment_extractor import extract_foreground_mask
            
            current_rgb = background[:, :, :3] if background.shape[2] == 4 else background
            
            # CRITICAL FIX: Extract mask for EVERY frame
            print(f"[EXTRACT_DEBUG] Frame {frame_number}: Calling extract_foreground_mask")
            print(f"[EXTRACT_DEBUG]   Input shape: {current_rgb.shape}")
            current_mask = extract_foreground_mask(current_rgb)
            print(f"[EXTRACT_DEBUG]   Mask shape: {current_mask.shape}")
            print(f"[EXTRACT_DEBUG]   Mask unique values: {np.unique(current_mask)[:5]}")
            
            # DEBUG: Prove mask is changing every frame
            if frame_number % 5 == 0:
                mask_pixels = np.sum(current_mask > 128)
                y_coords, x_coords = np.where(current_mask > 128)
                if len(x_coords) > 0:
                    mask_left = x_coords.min()
                    mask_right = x_coords.max()
                    mask_center_x = (mask_left + mask_right) // 2
                    print(f"[MASK_DEBUG] Frame {frame_number}: Fresh mask extracted - {mask_pixels:,} pixels")
                    print(f"[MASK_DEBUG]   Mask bounds: x=[{mask_left}-{mask_right}], center_x={mask_center_x}")
                    
                    # Save mask visualization for key frames
                    if frame_number in [25, 30, 35]:
                        cv2.imwrite(f'outputs/mask_frame_{frame_number}.png', current_mask)
                        print(f"[MASK_DEBUG]   Saved mask to outputs/mask_frame_{frame_number}.png")
            
            if current_mask.shape[:2] != (resolution[1], resolution[0]):
                current_mask = cv2.resize(current_mask, resolution, interpolation=cv2.INTER_LINEAR)
            
            # Apply morphological operations for cleaner mask
            current_mask = cv2.GaussianBlur(current_mask, (3, 3), 0)
            kernel = np.ones((3, 3), np.uint8)
            current_mask = cv2.dilate(current_mask, kernel, iterations=1)
            current_mask = (current_mask > 128).astype(np.uint8) * 255
            
            # Log EVERY successful extraction to verify it's working
            if self.debug:
                mask_pixels = np.sum(current_mask > 0)
                print(f"[MASK_UPDATE] Frame {frame_number}: Fresh mask extracted, {mask_pixels} foreground pixels")
                
            return current_mask
            
        except Exception as e:
            # CRITICAL: Do NOT fallback to stale mask - this causes the bug!
            # Better to have no occlusion than wrong occlusion
            print(f"[MASK_ERROR] Frame {frame_number}: Extraction failed: {e}")
            print(f"[MASK_ERROR] Exception type: {type(e).__name__}")
            import traceback
            print(f"[MASK_ERROR] Traceback:\n{traceback.format_exc()}")
            return None
    
    def apply_occlusion(
        self,
        sprite_img: np.ndarray,
        sprite_position: Tuple[int, int],
        current_mask: np.ndarray,
        resolution: Tuple[int, int],
        frame_number: int,
        sprite_char: str = '?'
    ) -> np.ndarray:
        """Apply occlusion mask to sprite."""
        pos_x, pos_y = sprite_position
        sp_h, sp_w = sprite_img.shape[:2]
        
        # Calculate bounds
        y1 = max(0, int(pos_y))
        y2 = min(resolution[1], int(pos_y) + sp_h)
        x1 = max(0, int(pos_x))
        x2 = min(resolution[0], int(pos_x) + sp_w)
        sy1 = max(0, -int(pos_y))
        sy2 = sy1 + (y2 - y1)
        sx1 = max(0, -int(pos_x))
        sx2 = sx1 + (x2 - x1)
        
        if y2 > y1 and x2 > x1:
            mask_region = current_mask[y1:y2, x1:x2]
            sprite_alpha = sprite_img[sy1:sy2, sx1:sx2, 3].astype(np.float32)
            
            # DEBUG: Track mask application
            if sprite_char == 'H':
                occluding_pixels = np.sum(mask_region > 128)
                print(f"[APPLY_DEBUG] Frame {frame_number}: Applying mask to 'H'")
                print(f"[APPLY_DEBUG]   H position: ({int(pos_x)}, {int(pos_y)})")
                print(f"[APPLY_DEBUG]   H bounds: x=[{x1}-{x2}], y=[{y1}-{y2}]")
                print(f"[APPLY_DEBUG]   Mask pixels in region: {occluding_pixels}")
                
                # Find mask edge position
                if occluding_pixels > 0:
                    mask_x_in_region = np.where(mask_region > 128)[1]
                    if len(mask_x_in_region) > 0:
                        edge_x = x1 + mask_x_in_region.min()
                        print(f"[APPLY_DEBUG]   Mask edge at x={edge_x}")
            
            # DEBUG: Show exactly what's being occluded
            if frame_number % 5 == 0 and sprite_char == 'H':
                occluding_pixels = np.sum(mask_region > 128)
                print(f"[OCCLUSION_DEBUG]   Letter 'H' region: x=[{x1}-{x2}], y=[{y1}-{y2}]")
                print(f"[OCCLUSION_DEBUG]   Mask pixels in H's region: {occluding_pixels}")
                
                # Find where mask edge is within the letter's region
                if occluding_pixels > 0:
                    mask_x_coords = np.where(mask_region > 128)[1]
                    if len(mask_x_coords) > 0:
                        local_mask_left = mask_x_coords.min()
                        local_mask_right = mask_x_coords.max()
                        global_mask_left = x1 + local_mask_left
                        global_mask_right = x1 + local_mask_right
                        print(f"[OCCLUSION_DEBUG]   Mask edge in H region: local x=[{local_mask_left}-{local_mask_right}]")
                        print(f"[OCCLUSION_DEBUG]   Mask edge global: x=[{global_mask_left}-{global_mask_right}]")
            
            # Apply occlusion
            visible_before = np.sum(sprite_alpha > 0)
            occluding_pixels = np.sum(mask_region > 128)
            
            mask_factor = mask_region.astype(np.float32) / 255.0
            sprite_alpha *= (1.0 - mask_factor)
            sprite_img[sy1:sy2, sx1:sx2, 3] = sprite_alpha.astype(np.uint8)
            
            visible_after = np.sum(sprite_img[sy1:sy2, sx1:sx2, 3] > 0)
            hidden_count = visible_before - visible_after
            
            if self.debug and (hidden_count > 0 or frame_number % 10 == 0):
                print(f"[OCCLUSION_PROOF] Frame {frame_number}, '{sprite_char}' at ({x1},{y1}): "
                      f"{occluding_pixels} mask pixels → {hidden_count} pixels hidden "
                      f"(was {visible_before}, now {visible_after})")
        
        return sprite_img