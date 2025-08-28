"""
Progressive reveal text animations
"""

import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_text_animation import BaseTextAnimation, AnimationConfig


class TypewriterAnimation(BaseTextAnimation):
    """Character-by-character reveal"""
    
    def __init__(self, config: AnimationConfig,
                 chars_per_second: float = 10,
                 cursor_visible: bool = False,
                 cursor_blink: bool = True):
        super().__init__(config)
        self.chars_per_second = chars_per_second
        self.cursor_visible = cursor_visible
        self.cursor_blink = cursor_blink
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply typewriter animation"""
        progress = self.get_progress(frame_number, fps)
        
        # Calculate how many characters to show
        total_chars = len(self.config.text)
        chars_to_show = int(total_chars * progress)
        
        # Get text to display
        visible_text = self.config.text[:chars_to_show]
        
        if not visible_text:
            return frame
        
        # Add cursor if enabled
        if self.cursor_visible and chars_to_show < total_chars:
            if self.cursor_blink:
                # Blink cursor
                blink_rate = 2  # blinks per second
                time_seconds = frame_number / fps
                if int(time_seconds * blink_rate * 2) % 2 == 0:
                    visible_text += "|"
            else:
                visible_text += "|"
        
        return self.draw_text_with_shadow(
            frame, visible_text, self.config.position,
            self.config.font_color, 1.0
        )


class WordRevealAnimation(BaseTextAnimation):
    """Word-by-word reveal"""
    
    def __init__(self, config: AnimationConfig,
                 words_per_second: float = 3,
                 fade_words: bool = True):
        super().__init__(config)
        self.words_per_second = words_per_second
        self.fade_words = fade_words
        self._words = None
    
    def get_words(self):
        """Split text into words"""
        if self._words is None:
            self._words = self.config.text.split()
        return self._words
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply word reveal animation"""
        progress = self.get_progress(frame_number, fps)
        
        words = self.get_words()
        total_words = len(words)
        
        if total_words == 0:
            return frame
        
        # Calculate how many words to show
        words_to_show = int(total_words * progress)
        
        # Handle the currently appearing word
        if words_to_show < total_words:
            current_word_progress = (progress * total_words) - words_to_show
            
            if self.fade_words and current_word_progress > 0:
                # Show partial current word with fade
                visible_text = " ".join(words[:words_to_show])
                if visible_text:
                    visible_text += " "
                
                # Draw completed words
                frame = self.draw_text_with_shadow(
                    frame, visible_text, self.config.position,
                    self.config.font_color, 1.0
                )
                
                # Draw fading current word
                if words_to_show < total_words:
                    # Calculate position of current word
                    text_size, _ = cv2.getTextSize(
                        visible_text, self.font, self.config.font_size / 30,
                        self.config.font_thickness
                    )
                    
                    current_word_pos = (
                        self.config.position[0] + text_size[0],
                        self.config.position[1]
                    )
                    
                    frame = self.draw_text_with_shadow(
                        frame, words[words_to_show], current_word_pos,
                        self.config.font_color, current_word_progress
                    )
            else:
                # Show completed words only
                visible_text = " ".join(words[:words_to_show])
                frame = self.draw_text_with_shadow(
                    frame, visible_text, self.config.position,
                    self.config.font_color, 1.0
                )
        else:
            # All words visible
            frame = self.draw_text_with_shadow(
                frame, self.config.text, self.config.position,
                self.config.font_color, 1.0
            )
        
        return frame


class LineStaggerAnimation(BaseTextAnimation):
    """Multi-line text with staggered reveal"""
    
    def __init__(self, config: AnimationConfig,
                 line_delay_ms: int = 300,
                 line_fade_duration_ms: int = 500):
        super().__init__(config)
        self.line_delay_ms = line_delay_ms
        self.line_fade_duration_ms = line_fade_duration_ms
        self._lines = None
    
    def get_lines(self):
        """Split text into lines"""
        if self._lines is None:
            self._lines = self.config.text.split('\n')
        return self._lines
    
    def apply_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> np.ndarray:
        """Apply line stagger animation"""
        current_time_ms = (frame_number / fps) * 1000
        lines = self.get_lines()
        
        # Calculate line height
        _, baseline = cv2.getTextSize(
            "Tg", self.font, self.config.font_size / 30,
            self.config.font_thickness
        )
        line_height = int(baseline * 3.5)
        
        # Draw each line with staggered timing
        for i, line in enumerate(lines):
            # Calculate when this line starts appearing
            line_start_time = i * self.line_delay_ms
            line_progress_time = current_time_ms - line_start_time
            
            if line_progress_time <= 0:
                continue  # Line hasn't started yet
            
            # Calculate opacity for this line
            line_progress = min(1.0, line_progress_time / self.line_fade_duration_ms)
            opacity = self.ease_value(line_progress, self.config.easing)
            
            # Calculate position for this line
            line_position = (
                self.config.position[0],
                self.config.position[1] + i * line_height
            )
            
            # Draw the line
            frame = self.draw_text_with_shadow(
                frame, line, line_position,
                self.config.font_color, opacity
            )
        
        return frame