"""
3D Word-based animations where words appear sequentially
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base_3d_text_animation import Base3DTextAnimation, Animation3DConfig, Letter3D
from PIL import Image, ImageDraw, ImageFont


class WordRiseSequence3D(Base3DTextAnimation):
    """
    Words rise from below one after another to the center position.
    Each word appears separately, sliding up from the bottom.
    """
    
    def __init__(self, config: Animation3DConfig,
                 word_spacing_ms: int = 500,  # Time between each word
                 rise_distance: float = 300,   # How far below words start
                 rise_duration_ms: int = 400,  # Time for each word to rise
                 overshoot: float = 0.1,       # Bounce overshoot amount
                 fade_in: bool = True,         # Fade as they rise
                 stack_mode: bool = False):    # Stack words on top of each other
        super().__init__(config)
        
        self.word_spacing_ms = word_spacing_ms
        self.rise_distance = rise_distance
        self.rise_duration_ms = rise_duration_ms
        self.overshoot = overshoot
        self.fade_in = fade_in
        self.stack_mode = stack_mode
        
        # Parse text into words and organize letters by word
        self.words = []
        self.word_letters = []
        self._organize_words()
        
        # Calculate center positions for each word
        self.word_center_positions = []
        self.word_start_positions = []
        self._calculate_word_positions()
    
    def _organize_words(self):
        """Organize letters into words"""
        words = self.config.text.split()
        self.words = words
        
        # Group letters by word - map to actual letter array indices
        # Note: self.letters only contains non-space characters
        letter_idx = 0
        for word in words:
            word_letter_group = []
            for char in word:
                if char != ' ' and letter_idx < len(self.letters):
                    # Store both the letter and its actual index in the letters array
                    self.letters[letter_idx].array_index = letter_idx
                    word_letter_group.append(self.letters[letter_idx])
                    letter_idx += 1
            if word_letter_group:  # Only add non-empty groups
                self.word_letters.append(word_letter_group)
    
    def _calculate_word_positions(self):
        """Calculate center and start positions for each word"""
        for word_idx, word_letters in enumerate(self.word_letters):
            if not word_letters:
                continue
            
            # Get word bounds
            min_x = min(letter.position[0] for letter in word_letters)
            max_x = max(letter.position[0] for letter in word_letters)
            word_center_x = (min_x + max_x) / 2
            
            # Center position (final position)
            if self.stack_mode and word_idx > 0:
                # Stack words vertically
                center_y = self.config.position[1] - (word_idx * 80)
            else:
                # All words go to same center position
                center_y = self.config.position[1]
            
            center_pos = np.array([word_center_x, center_y, 0])
            
            # Start position (below screen)
            start_pos = np.array([word_center_x, center_y + self.rise_distance, 0])
            
            self.word_center_positions.append(center_pos)
            self.word_start_positions.append(start_pos)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with word rise animation"""
        current_time_ms = progress * self.config.duration_ms
        
        for word_idx, word_letters in enumerate(self.word_letters):
            # Calculate when this word should start animating
            word_start_time = word_idx * self.word_spacing_ms
            word_end_time = word_start_time + self.rise_duration_ms
            
            if current_time_ms < word_start_time:
                # Word hasn't started yet - keep below
                word_progress = 0.0
                state = "waiting"
            elif current_time_ms < word_end_time:
                # Word is rising
                word_progress = (current_time_ms - word_start_time) / self.rise_duration_ms
                state = "rising"
            else:
                # Word has arrived
                word_progress = 1.0
                state = "arrived"
            
            # Apply easing for smooth motion
            if state == "rising":
                # Use gentler ease-in-out-sine for smooth, gentle motion
                # This creates a very smooth S-curve without harsh acceleration
                eased_progress = (1 - np.cos(word_progress * np.pi)) / 2
                
                # Only apply minimal overshoot if enabled, and make it very smooth
                if self.overshoot > 0 and word_progress > 0.8:
                    # Very gentle overshoot only at the very end
                    overshoot_progress = (word_progress - 0.8) / 0.2
                    # Use a gentler curve for overshoot
                    gentle_bounce = (1 - np.cos(overshoot_progress * np.pi)) / 2 * self.overshoot * 0.3
                    eased_progress = min(1.0, eased_progress + gentle_bounce)
            else:
                eased_progress = word_progress
            
            # Calculate word offset from start to center
            start_pos = self.word_start_positions[word_idx]
            center_pos = self.word_center_positions[word_idx]
            word_offset = (center_pos - start_pos) * eased_progress
            
            # Apply to each letter in the word
            for letter_idx, letter in enumerate(word_letters):
                # Calculate new position
                # Use array_index if available, otherwise fall back to finding index
                idx = getattr(letter, 'array_index', None)
                if idx is None:
                    # Find the letter in the main array
                    for i, l in enumerate(self.letters):
                        if l == letter:
                            idx = i
                            break
                if idx is None:
                    idx = letter_idx  # Fallback
                
                original_y = self.original_positions[idx][1]
                new_y = start_pos[1] + word_offset[1]
                
                # Maintain relative X position within word
                letter.position[0] = self.original_positions[idx][0]
                letter.position[1] = new_y
                
                # Keep Z position stable to avoid any flickering
                letter.position[2] = 0
                
                # Handle opacity with smooth transitions
                if self.fade_in:
                    if state == "waiting":
                        letter.opacity = 0.0
                    elif state == "rising":
                        # Smooth fade in as it rises using the same gentle easing
                        # Start fading in immediately but gradually
                        fade_progress = eased_progress  # Use the same smooth easing for fade
                        letter.opacity = fade_progress * fade_progress  # Square for even softer fade-in
                    else:
                        letter.opacity = 1.0
                else:
                    # Instant appearance when rising starts
                    letter.opacity = 1.0 if state != "waiting" else 0.0
                
                # No rotation to ensure smooth, flicker-free animation
                letter.rotation[0] = 0
                letter.rotation[1] = 0
                letter.rotation[2] = 0


class WordDropIn3D(Base3DTextAnimation):
    """
    Words drop from above one at a time with bounce effect.
    Similar to WordRiseSequence3D but from top.
    """
    
    def __init__(self, config: Animation3DConfig,
                 word_spacing_ms: int = 400,
                 drop_height: float = 400,
                 drop_duration_ms: int = 500,
                 bounce_count: int = 2,
                 bounce_damping: float = 0.5):
        super().__init__(config)
        
        self.word_spacing_ms = word_spacing_ms
        self.drop_height = drop_height
        self.drop_duration_ms = drop_duration_ms
        self.bounce_count = bounce_count
        self.bounce_damping = bounce_damping
        
        # Parse and organize words
        self.words = []
        self.word_letters = []
        self._organize_words()
        
        self.word_center_positions = []
        self._calculate_word_positions()
    
    def _organize_words(self):
        """Organize letters into words"""
        words = self.config.text.split()
        self.words = words
        
        letter_idx = 0
        for word in words:
            word_letter_group = []
            for char in word:
                if char != ' ' and letter_idx < len(self.letters):
                    self.letters[letter_idx].array_index = letter_idx
                    word_letter_group.append(self.letters[letter_idx])
                    letter_idx += 1
            if word_letter_group:
                self.word_letters.append(word_letter_group)
    
    def _calculate_word_positions(self):
        """Calculate center positions for each word"""
        for word_letters in self.word_letters:
            if not word_letters:
                continue
            
            # Get word center
            min_x = min(letter.position[0] for letter in word_letters)
            max_x = max(letter.position[0] for letter in word_letters)
            word_center_x = (min_x + max_x) / 2
            
            center_pos = np.array([word_center_x, self.config.position[1], 0])
            self.word_center_positions.append(center_pos)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with word drop animation"""
        current_time_ms = progress * self.config.duration_ms
        
        for word_idx, word_letters in enumerate(self.word_letters):
            word_start_time = word_idx * self.word_spacing_ms
            word_end_time = word_start_time + self.drop_duration_ms
            
            if current_time_ms < word_start_time:
                # Word hasn't started
                word_progress = 0.0
            elif current_time_ms < word_end_time:
                # Word is dropping
                word_progress = (current_time_ms - word_start_time) / self.drop_duration_ms
            else:
                # Word has landed
                word_progress = 1.0
            
            # Calculate drop position with bounce
            if word_progress < 1.0:
                # Falling with acceleration
                fall_progress = word_progress * word_progress  # Quadratic for acceleration
                y_offset = -self.drop_height * (1 - fall_progress)
            else:
                # Bounce after landing
                time_after_land = current_time_ms - word_end_time
                bounce_period = 200  # ms per bounce
                
                y_offset = 0
                for bounce_num in range(self.bounce_count):
                    bounce_start = bounce_num * bounce_period
                    bounce_end = bounce_start + bounce_period
                    
                    if time_after_land >= bounce_start and time_after_land < bounce_end:
                        bounce_progress = (time_after_land - bounce_start) / bounce_period
                        bounce_height = 50 * pow(self.bounce_damping, bounce_num)
                        y_offset = -bounce_height * abs(np.sin(bounce_progress * np.pi))
                        break
            
            # Apply to letters
            for letter in word_letters:
                idx = getattr(letter, 'array_index', self.letters.index(letter) if letter in self.letters else 0)
                original_y = self.original_positions[idx][1]
                letter.position[1] = original_y + y_offset
                
                # Opacity
                letter.opacity = 1.0 if word_progress > 0 else 0.0
                
                # Add slight rotation during fall
                if word_progress < 1.0:
                    letter.rotation[2] = (1 - word_progress) * 0.2


class WordWave3D(Base3DTextAnimation):
    """
    Words appear in a wave pattern from center outward.
    Middle word appears first, then adjacent words ripple outward.
    """
    
    def __init__(self, config: Animation3DConfig,
                 wave_speed_ms: int = 200,
                 rise_distance: float = 200,
                 scale_effect: bool = True):
        super().__init__(config)
        
        self.wave_speed_ms = wave_speed_ms
        self.rise_distance = rise_distance
        self.scale_effect = scale_effect
        
        # Organize words
        self.words = []
        self.word_letters = []
        self._organize_words()
        
        # Calculate wave order (center outward)
        self.word_order = self._calculate_wave_order()
    
    def _organize_words(self):
        """Organize letters into words"""
        words = self.config.text.split()
        self.words = words
        
        letter_idx = 0
        for word in words:
            word_letter_group = []
            for char in word:
                if char != ' ' and letter_idx < len(self.letters):
                    self.letters[letter_idx].array_index = letter_idx
                    word_letter_group.append(self.letters[letter_idx])
                    letter_idx += 1
            if word_letter_group:
                self.word_letters.append(word_letter_group)
    
    def _calculate_wave_order(self):
        """Calculate order for center-outward wave"""
        num_words = len(self.words)
        if num_words == 0:
            return []
        
        center_idx = num_words // 2
        order = [center_idx]
        
        # Add words alternating left and right from center
        for distance in range(1, num_words):
            if center_idx - distance >= 0:
                order.append(center_idx - distance)
            if center_idx + distance < num_words:
                order.append(center_idx + distance)
        
        return order
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with wave animation"""
        current_time_ms = progress * self.config.duration_ms
        
        for wave_position, word_idx in enumerate(self.word_order):
            if word_idx >= len(self.word_letters):
                continue
                
            word_letters = self.word_letters[word_idx]
            
            # Calculate timing based on wave position
            word_start_time = wave_position * self.wave_speed_ms
            word_duration = 400  # ms for each word to appear
            word_end_time = word_start_time + word_duration
            
            if current_time_ms < word_start_time:
                word_progress = 0.0
            elif current_time_ms < word_end_time:
                word_progress = (current_time_ms - word_start_time) / word_duration
            else:
                word_progress = 1.0
            
            # Smooth easing
            eased_progress = 1 - pow(1 - word_progress, 3)
            
            # Apply animation
            for letter in word_letters:
                # Rise from below
                idx = getattr(letter, 'array_index', self.letters.index(letter) if letter in self.letters else 0)
                original_y = self.original_positions[idx][1]
                y_offset = self.rise_distance * (1 - eased_progress)
                letter.position[1] = original_y + y_offset
                
                # Scale effect
                if self.scale_effect:
                    scale = 0.5 + 0.5 * eased_progress
                    letter.scale = np.array([scale, scale, scale])
                
                # Opacity
                letter.opacity = eased_progress
                
                # Slight rotation for dynamic feel
                letter.rotation[1] = (1 - eased_progress) * 0.3