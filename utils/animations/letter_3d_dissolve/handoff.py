"""Handoff handling from motion animation."""

from typing import Optional, Tuple, List, Any
import numpy as np
from PIL import Image

try:
    from ..text_3d_motion import LetterSprite
except ImportError:
    from text_3d_motion import LetterSprite


class HandoffHandler:
    """Manages handoff from motion animation to dissolve animation."""
    
    def __init__(self, sprite_manager, timing_calc, debug: bool = False):
        self.sprite_manager = sprite_manager
        self.timing_calc = timing_calc
        self.debug = debug
        self._handoff_sprite_alpha: Optional[float] = None
    
    def handle_initial_state(
        self,
        scale: float,
        position: Tuple[int, int],
        alpha: Optional[float],
        is_behind: Optional[bool],
        segment_mask: Optional[np.ndarray],
        letter_sprites: Optional[List[Any]],
        text: str,
        randomize_order: bool,
        stable_duration: float,
        dissolve_stagger: float,
        dissolve_duration: float,
        post_fade_seconds: float,
        pre_dissolve_hold_frames: int,
        ensure_no_gap: bool
    ) -> dict:
        """Process handoff from motion animation and return updated state."""
        has_existing_sprites = bool(self.sprite_manager.letter_sprites)
        
        if has_existing_sprites:
            if self.debug:
                print(f"[LETTER_SHIFT] Keeping {len(self.sprite_manager.letter_sprites)} existing sprites")
            # Store base positions if not already stored
            for sprite in self.sprite_manager.letter_sprites:
                if sprite.base_position is None:
                    sprite.base_position = sprite.position
        
        # Store incoming alpha for opacity fix
        if alpha is not None:
            stable_alpha = max(0.0, min(1.0, alpha))
            self._handoff_sprite_alpha = float(stable_alpha)
            if self.debug:
                print(f"[OPACITY_BLINK] Handoff received. motion_final_alpha={alpha:.3f}")
        else:
            stable_alpha = None
            self._handoff_sprite_alpha = None
        
        if self.debug:
            print(f"[POS_HANDOFF] Received handoff -> center={position}, scale={scale:.3f}")
        
        # Handle letter sprites
        if letter_sprites is not None and not has_existing_sprites:
            self._import_motion_sprites(letter_sprites, text)
        elif not has_existing_sprites:
            self.sprite_manager.prepare_letter_sprites(text, position, scale)
            self.sprite_manager.init_dissolve_order(text, randomize_order)
        
        # Build timeline
        timeline = self.timing_calc.build_frame_timeline(
            self.sprite_manager.dissolve_order,
            stable_duration,
            dissolve_stagger,
            dissolve_duration,
            post_fade_seconds,
            pre_dissolve_hold_frames,
            ensure_no_gap
        )
        self.timing_calc.log_schedule(
            self.sprite_manager.dissolve_order,
            self.sprite_manager.letter_sprites
        )
        
        return {
            'stable_alpha': stable_alpha,
            'is_behind': is_behind,
            'segment_mask': segment_mask,
            'handoff_sprite_alpha': self._handoff_sprite_alpha
        }
    
    def _import_motion_sprites(self, letter_sprites: List[Any], text: str):
        """Import sprites from motion animation."""
        self.sprite_manager.letter_sprites = []
        
        for motion_sprite in letter_sprites:
            # Add spaces between words if needed
            while len(self.sprite_manager.letter_sprites) < len(text):
                expected_char = text[len(self.sprite_manager.letter_sprites)]
                
                if expected_char == ' ':
                    space_sprite = LetterSprite(
                        char=' ', sprite_3d=None,
                        position=(0, 0), width=self.sprite_manager.renderer.font_size // 3,
                        height=1, anchor=(0, 0)
                    )
                    self.sprite_manager.letter_sprites.append(space_sprite)
                else:
                    break
            
            # Add the actual letter sprite
            if motion_sprite.char == text[len(self.sprite_manager.letter_sprites)]:
                dissolve_sprite = LetterSprite(
                    char=motion_sprite.char,
                    sprite_3d=motion_sprite.sprite_3d,
                    position=motion_sprite.position,
                    width=motion_sprite.width,
                    height=motion_sprite.height,
                    anchor=motion_sprite.anchor
                )
                self.sprite_manager.letter_sprites.append(dissolve_sprite)
        
        self.sprite_manager.init_dissolve_order(text, False)