"""
3D Text Morphing animations - text transforms from one phrase to another
"""

import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base_3d_text_animation import Base3DTextAnimation, Animation3DConfig, Letter3D
from PIL import Image, ImageDraw, ImageFont


class TextMorph3D(Base3DTextAnimation):
    """
    3D text morphing animation - smoothly transforms from one text to another
    This replicates the effect seen in real_estate.mov at 9-11 seconds
    """
    
    def __init__(self, config: Animation3DConfig,
                 target_text: str = None,
                 morph_start: float = 0.3,  # When morphing starts (0-1)
                 morph_end: float = 0.7,    # When morphing ends (0-1)
                 blur_peak: float = 15.0,   # Max blur during transition
                 dissolve_overlap: float = 0.2):  # How much dissolve overlaps
        super().__init__(config)
        
        # Store original and target text
        self.source_text = config.text
        self.target_text = target_text or "MORPHED TEXT"
        self.morph_start = morph_start
        self.morph_end = morph_end
        self.blur_peak = blur_peak
        self.dissolve_overlap = dissolve_overlap
        
        # Create target letters (second set of text)
        self.target_letters = []
        self._initialize_target_letters()
        
        # Store whether we've switched to target text
        self.has_switched = False
    
    def _initialize_target_letters(self):
        """Initialize the target text letters"""
        # Temporarily change config text to create target letters
        original_text = self.config.text
        self.config.text = self.target_text
        
        # Create temporary animation to get target letter positions
        temp_letters = []
        temp_positions = []
        
        # Create a temporary image to measure text
        temp_img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        
        # Calculate total text width for target text
        total_width = 0
        letter_widths = []
        
        for char in self.target_text:
            bbox = draw.textbbox((0, 0), char, font=self.font)
            char_width = bbox[2] - bbox[0]
            letter_widths.append(char_width)
            total_width += char_width * self.config.letter_spacing
        
        # Starting position (centered)
        start_x = self.config.position[0] - total_width / 2
        current_x = start_x
        
        # Create target letter objects
        for i, char in enumerate(self.target_text):
            if char == ' ':
                current_x += letter_widths[i] * self.config.letter_spacing
                continue
            
            position = np.array([
                current_x + letter_widths[i] / 2,
                self.config.position[1],
                self.config.position[2]
            ], dtype=np.float32)
            
            letter = Letter3D(
                character=char,
                index=i,
                position=position.copy(),
                rotation=np.zeros(3, dtype=np.float32),
                scale=np.ones(3, dtype=np.float32),
                opacity=0.0,  # Start invisible
                color=self.config.font_color,
                depth_color=self.config.depth_color
            )
            
            self.target_letters.append(letter)
            temp_positions.append(position.copy())
            
            current_x += letter_widths[i] * self.config.letter_spacing
        
        # Store target positions
        self.target_positions = temp_positions
        
        # Restore original text
        self.config.text = original_text
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with morphing effect"""
        
        # Calculate morph progress
        if progress < self.morph_start:
            # Before morph - show source text normally
            morph_progress = 0.0
            source_state = "visible"
            target_state = "hidden"
        elif progress < self.morph_end:
            # During morph - transition
            morph_progress = (progress - self.morph_start) / (self.morph_end - self.morph_start)
            source_state = "fading"
            target_state = "appearing"
        else:
            # After morph - show target text
            morph_progress = 1.0
            source_state = "hidden"
            target_state = "visible"
        
        # Calculate blur amount (peaks in middle of transition)
        if morph_progress > 0 and morph_progress < 1:
            # Blur peaks at midpoint
            blur_factor = 1.0 - abs(morph_progress - 0.5) * 2.0
            current_blur = self.blur_peak * blur_factor
        else:
            current_blur = 0.0
        
        # Update source letters
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            if source_state == "visible":
                # Normal appearance
                letter.opacity = letter_progress
                letter.blur_amount = 0.0
            elif source_state == "fading":
                # Fading out with blur
                fade_out_progress = morph_progress
                
                # Add stagger to fade
                letter_fade = fade_out_progress + (i * 0.02)
                letter_fade = min(1.0, letter_fade)
                
                letter.opacity = letter_progress * (1.0 - letter_fade)
                letter.blur_amount = current_blur
                
                # Slight scale down as it fades
                scale_factor = 1.0 - (letter_fade * 0.1)
                letter.scale = np.array([scale_factor, scale_factor, scale_factor])
            else:
                # Hidden
                letter.opacity = 0.0
                letter.blur_amount = 0.0
        
        # Update target letters
        for i, letter in enumerate(self.target_letters):
            if target_state == "hidden":
                # Not visible yet
                letter.opacity = 0.0
                letter.blur_amount = 0.0
            elif target_state == "appearing":
                # Fading in with blur
                fade_in_progress = morph_progress
                
                # Add stagger to appearance
                letter_fade = fade_in_progress - (i * 0.02)
                letter_fade = max(0.0, min(1.0, letter_fade))
                
                letter.opacity = letter_fade
                letter.blur_amount = current_blur * (1.0 - letter_fade)
                
                # Slight scale up as it appears
                scale_factor = 0.9 + (letter_fade * 0.1)
                letter.scale = np.array([scale_factor, scale_factor, scale_factor])
            else:
                # Fully visible
                letter.opacity = 1.0
                letter.blur_amount = 0.0
                letter.scale = np.ones(3, dtype=np.float32)
    
    def apply_frame(self, background_frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Override to render both source and target letters"""
        if background_frame is None:
            return None
        
        # Calculate progress
        progress = min(1.0, (frame_number / fps) * 1000 / self.config.duration_ms)
        
        # Update letter states
        self.update_letters(progress, frame_number, fps)
        
        # Start with background
        result = background_frame.copy()
        
        # Determine which letters to render based on morph progress
        morph_progress = 0.0
        if progress >= self.morph_start and progress <= self.morph_end:
            morph_progress = (progress - self.morph_start) / (self.morph_end - self.morph_start)
        elif progress > self.morph_end:
            morph_progress = 1.0
        
        # Render source letters if not fully morphed
        if morph_progress < 1.0:
            for letter in self.letters:
                if letter.opacity > 0.01:
                    sprite, bounds = self.render_letter_3d(letter)
                    if sprite is not None and bounds is not None:
                        result = self._composite_sprite(result, sprite, bounds)
        
        # Render target letters if morphing has started
        if morph_progress > 0.0:
            for letter in self.target_letters:
                if letter.opacity > 0.01:
                    sprite, bounds = self.render_letter_3d(letter)
                    if sprite is not None and bounds is not None:
                        result = self._composite_sprite(result, sprite, bounds)
        
        return result


class CrossDissolve3D(Base3DTextAnimation):
    """
    Simpler cross-dissolve between two texts
    One text fades out while another fades in
    """
    
    def __init__(self, config: Animation3DConfig,
                 target_text: str = None,
                 transition_start: float = 0.4,
                 transition_duration: float = 0.3):
        super().__init__(config)
        
        self.source_text = config.text
        self.target_text = target_text or "NEW TEXT"
        self.transition_start = transition_start
        self.transition_duration = transition_duration
        
        # Create target letters
        self.target_letters = []
        self._initialize_target_letters()
    
    def _initialize_target_letters(self):
        """Initialize the target text letters (same as TextMorph3D)"""
        original_text = self.config.text
        self.config.text = self.target_text
        
        temp_img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        
        total_width = 0
        letter_widths = []
        
        for char in self.target_text:
            bbox = draw.textbbox((0, 0), char, font=self.font)
            char_width = bbox[2] - bbox[0]
            letter_widths.append(char_width)
            total_width += char_width * self.config.letter_spacing
        
        start_x = self.config.position[0] - total_width / 2
        current_x = start_x
        
        for i, char in enumerate(self.target_text):
            if char == ' ':
                current_x += letter_widths[i] * self.config.letter_spacing
                continue
            
            position = np.array([
                current_x + letter_widths[i] / 2,
                self.config.position[1],
                self.config.position[2]
            ], dtype=np.float32)
            
            letter = Letter3D(
                character=char,
                index=i,
                position=position.copy(),
                rotation=np.zeros(3, dtype=np.float32),
                scale=np.ones(3, dtype=np.float32),
                opacity=0.0,
                color=self.config.font_color,
                depth_color=self.config.depth_color
            )
            
            self.target_letters.append(letter)
            current_x += letter_widths[i] * self.config.letter_spacing
        
        self.config.text = original_text
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with cross-dissolve"""
        
        transition_end = self.transition_start + self.transition_duration
        
        if progress < self.transition_start:
            # Show source only
            source_opacity = 1.0
            target_opacity = 0.0
        elif progress < transition_end:
            # Transition
            t = (progress - self.transition_start) / self.transition_duration
            source_opacity = 1.0 - t
            target_opacity = t
        else:
            # Show target only
            source_opacity = 0.0
            target_opacity = 1.0
        
        # Update source letters
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            letter.opacity = letter_progress * source_opacity
        
        # Update target letters  
        for i, letter in enumerate(self.target_letters):
            letter.opacity = target_opacity
    
    def apply_frame(self, background_frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Render both sets of letters"""
        if background_frame is None:
            return None
        
        progress = min(1.0, (frame_number / fps) * 1000 / self.config.duration_ms)
        self.update_letters(progress, frame_number, fps)
        
        result = background_frame.copy()
        
        # Render both source and target letters
        for letter in self.letters:
            if letter.opacity > 0.01:
                sprite, bounds = self.render_letter_3d(letter)
                if sprite is not None and bounds is not None:
                    result = self._composite_sprite(result, sprite, bounds)
        
        for letter in self.target_letters:
            if letter.opacity > 0.01:
                sprite, bounds = self.render_letter_3d(letter)
                if sprite is not None and bounds is not None:
                    result = self._composite_sprite(result, sprite, bounds)
        
        return result