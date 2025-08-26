"""
Post-processing system for scene objects.

Post-processors apply effects based on object state, independent of animations.
This ensures effects like occlusion are consistently applied whenever needed.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from abc import ABC, abstractmethod


class PostProcessor(ABC):
    """Base class for post-processing effects."""
    
    @abstractmethod
    def process(
        self,
        object_sprite: np.ndarray,
        object_state: Any,
        frame_number: int,
        background: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Apply post-processing to object sprite."""
        pass
    
    @abstractmethod
    def should_process(self, object_state: Any) -> bool:
        """Check if this processor should be applied to the object."""
        pass


class OcclusionProcessor(PostProcessor):
    """
    Handles occlusion for objects behind foreground elements.
    
    CRITICAL: This processor ALWAYS recalculates masks for objects
    with is_behind=True, regardless of animation state.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._mask_cache: Dict[int, np.ndarray] = {}
        self._last_processed_frame = -1
        
    def should_process(self, object_state: Any) -> bool:
        """Process if object is behind foreground."""
        return hasattr(object_state, 'is_behind') and object_state.is_behind
    
    def extract_foreground_mask(
        self,
        background: np.ndarray,
        frame_number: int
    ) -> Optional[np.ndarray]:
        """
        Extract foreground mask from background.
        ALWAYS extracts fresh mask - no caching for is_behind objects.
        """
        try:
            # Import segmentation module
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from video.segmentation.segment_extractor import extract_foreground_mask
            
            # Extract RGB if RGBA
            if background.shape[2] == 4:
                background_rgb = background[:, :, :3]
            else:
                background_rgb = background
            
            # CRITICAL: Always extract fresh mask
            if self.debug and frame_number % 5 == 0:
                print(f"[OCCLUSION_PROC] Frame {frame_number}: Extracting FRESH mask")
            
            mask = extract_foreground_mask(background_rgb)
            
            # Clean up mask
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = (mask > 128).astype(np.uint8) * 255
            
            if self.debug and frame_number % 5 == 0:
                mask_pixels = np.sum(mask > 128)
                print(f"[OCCLUSION_PROC] Frame {frame_number}: Mask has {mask_pixels:,} pixels")
            
            return mask
            
        except Exception as e:
            if self.debug:
                print(f"[OCCLUSION_PROC] Frame {frame_number}: Mask extraction failed: {e}")
            return None
    
    def process(
        self,
        object_sprite: np.ndarray,
        object_state: Any,
        frame_number: int,
        background: np.ndarray,
        object_bounds: Optional[Tuple[int, int, int, int]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Apply occlusion to object sprite.
        
        CRITICAL: This is called EVERY FRAME for objects with is_behind=True,
        ensuring the mask is always current.
        """
        if not self.should_process(object_state):
            return object_sprite
        
        # Extract fresh mask EVERY TIME for is_behind objects
        mask = self.extract_foreground_mask(background, frame_number)
        if mask is None:
            return object_sprite
        
        # Get object bounds
        if object_bounds is None:
            x1, y1 = int(object_state.position[0]), int(object_state.position[1])
            x2 = x1 + object_sprite.shape[1]
            y2 = y1 + object_sprite.shape[0]
        else:
            x1, y1, x2, y2 = object_bounds
        
        # Ensure bounds are within frame
        frame_h, frame_w = background.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return object_sprite
        
        # Get mask region corresponding to object
        mask_region = mask[y1:y2, x1:x2]
        
        # Calculate sprite region (accounting for clipping)
        sprite_x1 = max(0, -int(object_state.position[0]))
        sprite_y1 = max(0, -int(object_state.position[1]))
        sprite_x2 = sprite_x1 + (x2 - x1)
        sprite_y2 = sprite_y1 + (y2 - y1)
        
        # Apply occlusion to alpha channel
        if object_sprite.shape[2] == 4:
            sprite_region = object_sprite[sprite_y1:sprite_y2, sprite_x1:sprite_x2]
            
            # Only process if regions match
            if sprite_region.shape[:2] == mask_region.shape:
                # Calculate occlusion
                mask_factor = mask_region.astype(np.float32) / 255.0
                
                # Apply to alpha channel
                original_alpha = sprite_region[:, :, 3].astype(np.float32)
                occluded_alpha = original_alpha * (1.0 - mask_factor)
                sprite_region[:, :, 3] = occluded_alpha.astype(np.uint8)
                
                if self.debug and frame_number % 10 == 0:
                    visible_before = np.sum(original_alpha > 0)
                    visible_after = np.sum(occluded_alpha > 0)
                    hidden = visible_before - visible_after
                    if hidden > 0:
                        obj_id = getattr(object_state, 'object_id', 'unknown')
                        print(f"[OCCLUSION_PROC] Frame {frame_number}, {obj_id}: "
                              f"{hidden} pixels hidden (was {visible_before}, now {visible_after})")
        
        return object_sprite


class EffectsProcessor(PostProcessor):
    """Apply visual effects like shadows, glow, etc."""
    
    def should_process(self, object_state: Any) -> bool:
        """Check if effects should be applied."""
        # Could check for specific effect flags in object_state
        return False
    
    def process(
        self,
        object_sprite: np.ndarray,
        object_state: Any,
        frame_number: int,
        background: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Apply visual effects."""
        # Placeholder for effects like shadow, glow, etc.
        return object_sprite