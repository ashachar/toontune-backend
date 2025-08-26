"""
Animation adapters for the object system.

These adapters translate existing animations (motion, dissolve) to work
with the object-based architecture, ensuring animations only update object
state, while post-processing handles effects like occlusion.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from .scene_object import LetterObject
import math


class AnimationAdapter:
    """Base class for animation adapters."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def apply(
        self,
        objects: List[LetterObject],
        frame_number: int,
        total_frames: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply animation to objects.
        
        Returns:
            Dict mapping object_id to animation transform data
        """
        raise NotImplementedError


class MotionAnimationAdapter(AnimationAdapter):
    """
    Adapts motion animation to work with object system.
    Updates object positions during motion phase.
    """
    
    def __init__(
        self,
        motion_duration_frames: int = 20,
        debug: bool = False
    ):
        super().__init__(debug)
        self.motion_duration_frames = motion_duration_frames
        
    def apply(
        self,
        objects: List[LetterObject],
        frame_number: int,
        total_frames: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Apply motion animation to letters."""
        animations = {}
        
        if frame_number >= self.motion_duration_frames:
            return animations
        
        # Calculate motion progress
        progress = frame_number / self.motion_duration_frames
        
        # Easing function
        eased_progress = self._ease_out_cubic(progress)
        
        for obj in objects:
            if not isinstance(obj, LetterObject):
                continue
            
            # Calculate position offset from base
            start_x = obj.base_position[0] - 50  # Start 50px left
            start_y = obj.base_position[1] - 30  # Start 30px up
            target_x, target_y = obj.base_position
            
            # Interpolate position
            current_x = start_x + (target_x - start_x) * eased_progress
            current_y = start_y + (target_y - start_y) * eased_progress
            
            # Scale effect (start small, grow to normal)
            scale = 0.3 + 0.7 * eased_progress
            
            # Opacity fade in
            opacity = min(1.0, eased_progress * 1.2)
            
            animations[obj.object_id] = {
                'type': 'motion',
                'transform': {
                    'position': (current_x, current_y),
                    'scale': scale,
                    'opacity': opacity,
                    'progress': progress
                }
            }
            
            if self.debug and frame_number % 5 == 0:
                print(f"[MOTION_ADAPTER] Frame {frame_number}: {obj.char} "
                      f"pos=({current_x:.1f}, {current_y:.1f}), "
                      f"scale={scale:.2f}, opacity={opacity:.2f}")
        
        return animations
    
    def _ease_out_cubic(self, t: float) -> float:
        """Cubic ease-out function."""
        return 1 - pow(1 - t, 3)


class DissolveAnimationAdapter(AnimationAdapter):
    """
    Adapts dissolve animation to work with object system.
    Updates object opacity and position during dissolve phase.
    """
    
    def __init__(
        self,
        motion_duration_frames: int = 20,
        safety_hold_frames: int = 20,
        dissolve_duration_frames: int = 60,
        debug: bool = False
    ):
        super().__init__(debug)
        self.motion_duration_frames = motion_duration_frames
        self.safety_hold_frames = safety_hold_frames
        self.dissolve_duration_frames = dissolve_duration_frames
        self.dissolve_start_frame = motion_duration_frames + safety_hold_frames
        
    def apply(
        self,
        objects: List[LetterObject],
        frame_number: int,
        total_frames: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Apply dissolve animation to letters."""
        animations = {}
        
        # Only apply during dissolve phase
        if frame_number < self.dissolve_start_frame:
            return animations
        
        if frame_number >= self.dissolve_start_frame + self.dissolve_duration_frames:
            # Animation complete - ensure all objects are invisible
            for obj in objects:
                if isinstance(obj, LetterObject):
                    animations[obj.object_id] = {
                        'type': 'dissolve',
                        'transform': {
                            'opacity': 0.0,
                            'scale': 0.1,
                            'progress': 1.0
                        }
                    }
            return animations
        
        # Calculate dissolve timing for each letter
        num_letters = len([o for o in objects if isinstance(o, LetterObject)])
        if num_letters == 0:
            return animations
        
        # Stagger the dissolve across letters
        stagger_frames = self.dissolve_duration_frames * 0.6  # 60% for stagger
        frames_per_letter = stagger_frames / max(1, num_letters - 1) if num_letters > 1 else 0
        
        letter_idx = 0
        for obj in objects:
            if not isinstance(obj, LetterObject):
                continue
            
            # Calculate when this letter starts dissolving
            letter_start_offset = letter_idx * frames_per_letter
            letter_dissolve_start = self.dissolve_start_frame + letter_start_offset
            
            # Duration for this letter's dissolve
            letter_dissolve_duration = self.dissolve_duration_frames * 0.5
            
            if frame_number < letter_dissolve_start:
                # Not dissolving yet
                letter_idx += 1
                continue
            
            # Calculate progress for this letter
            frames_since_start = frame_number - letter_dissolve_start
            progress = min(1.0, frames_since_start / letter_dissolve_duration)
            
            # Easing
            eased_progress = self._ease_in_cubic(progress)
            
            # Calculate effects
            opacity = 1.0 - eased_progress
            scale = 1.0 - (eased_progress * 0.9)  # Shrink to 10% size
            
            # Add slight upward drift
            drift_y = -eased_progress * 20  # Move up 20 pixels
            drift_x = math.sin(eased_progress * math.pi) * 5  # Slight horizontal wobble
            
            animations[obj.object_id] = {
                'type': 'dissolve',
                'transform': {
                    'opacity': opacity,
                    'scale': scale,
                    'position_offset': (drift_x, drift_y),
                    'progress': progress
                }
            }
            
            if self.debug and frame_number % 10 == 0 and progress > 0:
                print(f"[DISSOLVE_ADAPTER] Frame {frame_number}: {obj.char} "
                      f"progress={progress:.2f}, opacity={opacity:.2f}, "
                      f"scale={scale:.2f}")
            
            letter_idx += 1
        
        return animations
    
    def _ease_in_cubic(self, t: float) -> float:
        """Cubic ease-in function."""
        return t * t * t