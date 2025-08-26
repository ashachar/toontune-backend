"""
Scene object base classes with state management.

Objects maintain their state (position, visibility, is_behind, etc.)
independent of animations. This ensures properties like occlusion
are always correctly applied based on object state, not animation state.
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from PIL import Image
from dataclasses import dataclass, field


@dataclass
class ObjectState:
    """State of a scene object, persists across animations."""
    position: Tuple[float, float] = (0, 0)
    scale: float = 1.0
    rotation: float = 0.0
    opacity: float = 1.0
    is_visible: bool = True
    is_behind: bool = False
    z_order: int = 0
    
    # Cached values for optimization
    needs_mask_update: bool = True
    last_mask_frame: int = -1
    cached_mask: Optional[np.ndarray] = None
    
    # Additional attributes
    metadata: Dict[str, Any] = field(default_factory=dict)


class SceneObject:
    """
    Base class for all scene objects.
    
    Key principles:
    1. Objects maintain their own state
    2. State persists across animations
    3. Post-processing is determined by state, not animation
    """
    
    def __init__(self, object_id: str, initial_state: Optional[ObjectState] = None):
        self.object_id = object_id
        self.state = initial_state or ObjectState()
        self.animations: List[Any] = []  # Current animations affecting this object
        self.sprite: Optional[Image.Image] = None
        self.base_sprite: Optional[Image.Image] = None  # Original, unmodified sprite
        
        # Track state changes
        self._state_changed = True
        self._last_rendered_frame = -1
        
    def set_behind(self, is_behind: bool):
        """
        Set whether object is behind foreground.
        CRITICAL: When is_behind=True, mask MUST be recalculated every frame.
        """
        if self.state.is_behind != is_behind:
            self.state.is_behind = is_behind
            self.state.needs_mask_update = True
            self._state_changed = True
            print(f"[OBJECT_STATE] {self.object_id}: is_behind set to {is_behind}")
    
    def update_position(self, x: float, y: float):
        """Update object position."""
        self.state.position = (x, y)
        self._state_changed = True
        if self.state.is_behind:
            self.state.needs_mask_update = True
    
    def update_scale(self, scale: float):
        """Update object scale."""
        self.state.scale = scale
        self._state_changed = True
        if self.state.is_behind:
            self.state.needs_mask_update = True
    
    def update_opacity(self, opacity: float):
        """Update object opacity."""
        self.state.opacity = max(0.0, min(1.0, opacity))
        self._state_changed = True
    
    def requires_mask_update(self, current_frame: int) -> bool:
        """
        Check if mask needs updating.
        CRITICAL: Always returns True if is_behind=True to ensure fresh mask every frame.
        """
        if not self.state.is_behind:
            return False
        
        # ALWAYS recalculate mask when behind, regardless of animation state
        return True  # Force recalculation every frame when behind
    
    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Get object bounding box (x1, y1, x2, y2)."""
        if self.sprite is None:
            return (0, 0, 0, 0)
        
        x, y = self.state.position
        w, h = self.sprite.width, self.sprite.height
        
        # Account for scale
        if self.state.scale != 1.0:
            w = int(w * self.state.scale)
            h = int(h * self.state.scale)
        
        return (int(x), int(y), int(x + w), int(y + h))
    
    def render(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Render object for current frame.
        Returns numpy array of rendered object or None if not visible.
        """
        if not self.state.is_visible or self.sprite is None:
            return None
        
        # This will be overridden by subclasses
        rendered = np.array(self.sprite)
        
        # Apply opacity
        if rendered.shape[2] == 4 and self.state.opacity < 1.0:
            rendered[:, :, 3] = (rendered[:, :, 3] * self.state.opacity).astype(np.uint8)
        
        self._last_rendered_frame = frame_number
        return rendered


class LetterObject(SceneObject):
    """
    Letter object with text-specific properties.
    """
    
    def __init__(
        self, 
        char: str,
        position: Tuple[float, float],
        sprite_3d: Optional[Image.Image] = None,
        width: int = 0,
        height: int = 0,
        anchor: Tuple[int, int] = (0, 0)
    ):
        super().__init__(object_id=f"letter_{char}_{id(self)}")
        
        self.char = char
        self.state.position = position
        self.sprite = sprite_3d
        self.base_sprite = sprite_3d.copy() if sprite_3d else None
        self.width = width
        self.height = height
        self.anchor = anchor
        
        # Letter-specific state
        self.base_position = position  # Original position before animations
        self.dissolve_progress = 0.0
        self.motion_progress = 0.0
        
        # Track which animation phases this letter has been through
        self.animation_history = {
            'motion_completed': False,
            'dissolve_started': False,
            'dissolve_completed': False
        }
    
    def apply_animation_transform(self, animation_type: str, transform_data: Dict[str, Any]):
        """
        Apply animation transformation to the letter.
        This updates the object's state based on animation, but doesn't handle occlusion.
        """
        if animation_type == 'motion':
            if 'position' in transform_data:
                self.update_position(*transform_data['position'])
            if 'scale' in transform_data:
                self.update_scale(transform_data['scale'])
            if 'opacity' in transform_data:
                self.update_opacity(transform_data['opacity'])
            self.motion_progress = transform_data.get('progress', 0.0)
            
        elif animation_type == 'dissolve':
            if 'scale' in transform_data:
                self.update_scale(transform_data['scale'])
            if 'opacity' in transform_data:
                self.update_opacity(transform_data['opacity'])
            if 'position_offset' in transform_data:
                # Add offset to current position
                current_x, current_y = self.state.position
                offset_x, offset_y = transform_data['position_offset']
                self.update_position(current_x + offset_x, current_y + offset_y)
            self.dissolve_progress = transform_data.get('progress', 0.0)
            
            if self.dissolve_progress > 0 and not self.animation_history['dissolve_started']:
                self.animation_history['dissolve_started'] = True
    
    def __repr__(self):
        return (f"LetterObject('{self.char}', pos={self.state.position}, "
                f"is_behind={self.state.is_behind}, opacity={self.state.opacity:.2f})")