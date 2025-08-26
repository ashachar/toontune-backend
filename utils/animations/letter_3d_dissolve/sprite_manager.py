"""Letter sprite management and layout."""
import random
from typing import List, Tuple, Optional, Any
from PIL import Image, ImageDraw
import numpy as np

try:
    from ..text_3d_motion import LetterSprite
except ImportError:
    from text_3d_motion import LetterSprite


class SpriteManager:
    """Manages letter sprites, their layout and ordering."""
    
    def __init__(self, renderer, debug: bool = False):
        self.renderer = renderer
        self.debug = debug
        self.letter_sprites: List[LetterSprite] = []
        self.dissolve_order: List[int] = []
        self.letter_kill_masks: dict = {}
        
    def prepare_letter_sprites_at_position(
        self, 
        text: str,
        text_topleft: Tuple[int, int],
        initial_scale: float
    ) -> None:
        """Prepare letter sprites at exact position from motion animation."""
        start_x, start_y = text_topleft
        
        font_px = int(self.renderer.font_size * initial_scale * self.renderer.supersample_factor)
        font = self.renderer.get_font(font_px)
        
        current_x = start_x
        visible_positions = []
        
        for letter in text:
            if letter == ' ':
                # Space
                sprite = LetterSprite(
                    char=' ',
                    sprite_3d=None,
                    position=(current_x, start_y),
                    width=max(1, font_px // 3),
                    height=1,
                    anchor=(0, 0)
                )
                self.letter_sprites.append(sprite)
                current_x += max(1, font_px // 3)
            else:
                # Render letter
                sprite_3d, (ax, ay) = self.renderer.render_3d_letter(letter, initial_scale, 1.0, 1.0)
                advance = int(sprite_3d.width * 1.1) if sprite_3d else font_px
                
                # Position letter at current x
                paste_x = current_x
                paste_y = start_y
                
                sprite = LetterSprite(
                    char=letter,
                    sprite_3d=sprite_3d,
                    position=(paste_x, paste_y),
                    width=sprite_3d.width if sprite_3d else 0,
                    height=sprite_3d.height if sprite_3d else 0,
                    anchor=(ax, ay)
                )
                self.letter_sprites.append(sprite)
                visible_positions.append((current_x, start_y))
                current_x += advance
        
        if self.debug:
            print(f"[POS_HANDOFF] Letter sprites created at exact motion position: start=({start_x},{start_y})")
            print(f"[POS_HANDOFF] Letter positions: {visible_positions}")
    
    def prepare_letter_sprites(
        self,
        text: str,
        initial_position: Tuple[int, int],
        initial_scale: float
    ) -> None:
        """Pre-render letter sprites and compute front-face layout."""
        font_px = int(self.renderer.font_size * initial_scale)
        font = self.renderer.get_font(font_px)

        tmp = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        full_bbox = d.textbbox((0, 0), text, font=font)
        text_width = full_bbox[2] - full_bbox[0]
        text_height = full_bbox[3] - full_bbox[1]

        cx, cy = initial_position
        start_x = cx - text_width // 2
        start_y = cy - text_height // 2

        current_x = start_x
        visible_positions: List[Tuple[int, int]] = []

        self.letter_sprites = []
        for letter in text:
            if letter == ' ':
                space_width = font_px // 3
                sprite = LetterSprite(
                    char=letter,
                    sprite_3d=None,
                    position=(current_x, start_y),
                    width=space_width,
                    height=0,
                    anchor=(0, 0)
                )
                self.letter_sprites.append(sprite)
                visible_positions.append((current_x, start_y))
                current_x += space_width
            else:
                letter_bbox = d.textbbox((0, 0), letter, font=font)
                advance = letter_bbox[2] - letter_bbox[0]

                sprite_3d, (ax, ay) = self.renderer.render_3d_letter(letter, initial_scale, 1.0, 1.0)
                paste_x = current_x - ax
                paste_y = start_y - ay

                sprite = LetterSprite(
                    char=letter,
                    sprite_3d=sprite_3d,
                    position=(paste_x, paste_y),
                    width=sprite_3d.width if sprite_3d else 0,
                    height=sprite_3d.height if sprite_3d else 0,
                    anchor=(ax, ay)
                )
                self.letter_sprites.append(sprite)
                visible_positions.append((current_x, start_y))
                current_x += advance

        if self.debug:
            print(
                f"[POS_HANDOFF] Dissolve layout -> center={initial_position}, "
                f"front_text_bbox=({text_width},{text_height}), start_topleft=({start_x},{start_y})"
            )
            print(f"[POS_HANDOFF] Letter positions frozen at: {visible_positions}")
    
    def update_sprites_for_handoff(self, initial_scale: float) -> None:
        """Update existing sprites for new scale while preserving positions."""
        if not self.letter_sprites:
            return
            
        # Re-render sprites at new scale but keep positions stable
        for sprite in self.letter_sprites:
            if sprite.char != ' ' and sprite.sprite_3d is not None:
                # Store current visual center before re-rendering
                old_center_x = sprite.position[0] + sprite.width // 2
                old_center_y = sprite.position[1] + sprite.height // 2
                
                # Re-render at new scale
                sprite_3d, (ax, ay) = self.renderer.render_3d_letter(
                    sprite.char, initial_scale, 1.0, 1.0
                )
                sprite.sprite_3d = sprite_3d
                sprite.width = sprite_3d.width if sprite_3d else 0
                sprite.height = sprite_3d.height if sprite_3d else 0
                sprite.anchor = (ax, ay)
                
                # Adjust position to maintain the same visual center
                new_pos_x = old_center_x - sprite.width // 2
                new_pos_y = old_center_y - sprite.height // 2
                sprite.position = (new_pos_x, new_pos_y)
                
                if self.debug:
                    print(f"[LETTER_SHIFT] '{sprite.char}': center preserved at ({old_center_x}, {old_center_y})")
                    print(f"  New position: {sprite.position}, size: {sprite.width}x{sprite.height}")
            elif sprite.char == ' ':
                sprite.width = max(1, int(self.renderer.font_size * initial_scale) // 3)
        
        if self.debug:
            print(f"[LETTER_SHIFT] Updated {len(self.letter_sprites)} sprites")
    
    def init_dissolve_order(self, text: str, randomize: bool = False) -> None:
        """Initialize the dissolve order for letters."""
        if randomize:
            indices = [i for i, ch in enumerate(text) if ch != ' ']
            random.shuffle(indices)
            self.dissolve_order = indices
        else:
            self.dissolve_order = [i for i, ch in enumerate(text) if ch != ' ']
        
        if self.debug:
            print(f"[POS_HANDOFF] Dissolve order (excluding spaces): {self.dissolve_order}")
    
    def add_dissolve_holes(self, letter_idx: int, progress_0_1: float) -> None:
        """Add dissolve holes to a letter sprite."""
        import cv2
        sprite = self.letter_sprites[letter_idx]
        if sprite.sprite_3d is None:
            return
        if letter_idx not in self.letter_kill_masks:
            self.letter_kill_masks[letter_idx] = np.zeros(
                (sprite.sprite_3d.height, sprite.sprite_3d.width), dtype=np.uint8)
        num_holes = int(progress_0_1 * 20)
        for _ in range(num_holes):
            x = np.random.randint(0, sprite.sprite_3d.width)
            y = np.random.randint(0, sprite.sprite_3d.height)
            radius = np.random.randint(2, 8)
            cv2.circle(self.letter_kill_masks[letter_idx], (x, y), radius, 1, -1)