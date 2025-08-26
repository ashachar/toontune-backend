"""
Render pipeline that orchestrates object rendering with post-processing.

This pipeline ensures that:
1. Objects are rendered based on their current state
2. Post-processing (like occlusion) is ALWAYS applied when needed
3. The final composite is correct regardless of animation state
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from .scene_object import SceneObject
from .post_processor import PostProcessor, OcclusionProcessor


class RenderPipeline:
    """
    Manages the rendering of all scene objects with proper post-processing.
    
    CRITICAL: This pipeline ensures occlusion is recalculated EVERY FRAME
    for objects with is_behind=True, fixing the stale mask bug.
    """
    
    def __init__(self, width: int, height: int, debug: bool = False):
        self.width = width
        self.height = height
        self.debug = debug
        
        # Scene objects organized by z-order
        self.objects: List[SceneObject] = []
        
        # Post-processors to apply
        self.post_processors: List[PostProcessor] = [
            OcclusionProcessor(debug=debug)
        ]
        
        # Frame counter for debugging
        self.frame_counter = 0
        
    def add_object(self, obj: SceneObject):
        """Add an object to the scene."""
        self.objects.append(obj)
        # Sort by z_order
        self.objects.sort(key=lambda x: x.state.z_order)
        if self.debug:
            print(f"[RENDER_PIPELINE] Added {obj.object_id}, total objects: {len(self.objects)}")
    
    def remove_object(self, obj: SceneObject):
        """Remove an object from the scene."""
        if obj in self.objects:
            self.objects.remove(obj)
            if self.debug:
                print(f"[RENDER_PIPELINE] Removed {obj.object_id}")
    
    def clear_objects(self):
        """Remove all objects from the scene."""
        self.objects.clear()
        if self.debug:
            print(f"[RENDER_PIPELINE] Cleared all objects")
    
    def render_frame(
        self,
        background: np.ndarray,
        frame_number: int,
        animations: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Render a complete frame with all objects and post-processing.
        
        Args:
            background: Background frame (RGB or RGBA)
            frame_number: Current frame number
            animations: Optional animation states to apply to objects
            
        Returns:
            Composite frame with all objects rendered
        """
        self.frame_counter = frame_number
        
        # Start with background
        if background.shape[2] == 3:
            composite = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
        else:
            composite = background.copy()
        
        if self.debug and frame_number % 30 == 0:
            print(f"\n[RENDER_PIPELINE] Frame {frame_number}: Rendering {len(self.objects)} objects")
        
        # Process each object
        for obj in self.objects:
            if not obj.state.is_visible:
                continue
            
            # Apply animations if provided
            if animations and obj.object_id in animations:
                anim_data = animations[obj.object_id]
                if 'transform' in anim_data:
                    obj.apply_animation_transform(
                        anim_data['type'],
                        anim_data['transform']
                    )
            
            # Render the object
            obj_sprite = obj.render(frame_number)
            if obj_sprite is None:
                continue
            
            # CRITICAL: Apply post-processing based on object STATE, not animation
            for processor in self.post_processors:
                if processor.should_process(obj.state):
                    if self.debug and frame_number % 30 == 0:
                        print(f"[RENDER_PIPELINE] Frame {frame_number}: "
                              f"Applying {processor.__class__.__name__} to {obj.object_id}")
                    
                    # Get object bounds for occlusion processing
                    bounds = obj.get_bounds()
                    
                    # Process the sprite
                    obj_sprite = processor.process(
                        obj_sprite,
                        obj.state,
                        frame_number,
                        background,  # Original background, not composite
                        object_bounds=bounds
                    )
            
            # Composite the object onto the frame
            composite = self._composite_object(composite, obj_sprite, obj.state.position)
        
        # Convert back to RGB if needed
        if composite.shape[2] == 4:
            composite = cv2.cvtColor(composite, cv2.COLOR_RGBA2RGB)
        
        return composite
    
    def _composite_object(
        self,
        background: np.ndarray,
        sprite: np.ndarray,
        position: Tuple[float, float]
    ) -> np.ndarray:
        """
        Composite an object sprite onto the background.
        
        Args:
            background: Background image (RGBA)
            sprite: Object sprite (RGBA)
            position: Position to place the sprite
            
        Returns:
            Composited image
        """
        if sprite is None or sprite.size == 0:
            return background
        
        x, y = int(position[0]), int(position[1])
        h, w = sprite.shape[:2]
        
        # Calculate valid regions
        bg_h, bg_w = background.shape[:2]
        
        # Source region in sprite
        src_x1 = max(0, -x)
        src_y1 = max(0, -y)
        src_x2 = min(w, bg_w - x)
        src_y2 = min(h, bg_h - y)
        
        # Destination region in background
        dst_x1 = max(0, x)
        dst_y1 = max(0, y)
        dst_x2 = min(bg_w, x + w)
        dst_y2 = min(bg_h, y + h)
        
        # Check if any part is visible
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return background
        
        # Get regions
        sprite_region = sprite[src_y1:src_y2, src_x1:src_x2]
        bg_region = background[dst_y1:dst_y2, dst_x1:dst_x2]
        
        # Alpha blend
        if sprite_region.shape[2] == 4:
            alpha = sprite_region[:, :, 3:4].astype(np.float32) / 255.0
            
            # Blend RGB channels
            blended = bg_region[:, :, :3] * (1 - alpha) + sprite_region[:, :, :3] * alpha
            background[dst_y1:dst_y2, dst_x1:dst_x2, :3] = blended.astype(np.uint8)
            
            # Update alpha if background has alpha channel
            if background.shape[2] == 4:
                bg_alpha = bg_region[:, :, 3:4].astype(np.float32) / 255.0
                new_alpha = bg_alpha + alpha * (1 - bg_alpha)
                background[dst_y1:dst_y2, dst_x1:dst_x2, 3:4] = (new_alpha * 255).astype(np.uint8)
        else:
            # No alpha, just copy
            background[dst_y1:dst_y2, dst_x1:dst_x2, :3] = sprite_region
        
        return background
    
    def update_object_state(self, object_id: str, **state_updates):
        """Update the state of a specific object."""
        for obj in self.objects:
            if obj.object_id == object_id:
                for key, value in state_updates.items():
                    if hasattr(obj.state, key):
                        setattr(obj.state, key, value)
                        if self.debug:
                            print(f"[RENDER_PIPELINE] Updated {object_id}.{key} = {value}")
                break
    
    def get_object(self, object_id: str) -> Optional[SceneObject]:
        """Get an object by ID."""
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None
    
    def set_all_objects_behind(self, is_behind: bool):
        """Set all objects to be behind/in-front of foreground."""
        for obj in self.objects:
            obj.set_behind(is_behind)
        if self.debug:
            print(f"[RENDER_PIPELINE] Set all {len(self.objects)} objects is_behind={is_behind}")