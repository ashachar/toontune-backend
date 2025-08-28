"""
3D Digital/Glitch text animations with tech effects
"""

import numpy as np
import cv2
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base_3d_text_animation import Base3DTextAnimation, Animation3DConfig, Letter3D


class Glitch3D(Base3DTextAnimation):
    """3D glitch animation with digital interference effects"""
    
    def __init__(self, config: Animation3DConfig,
                 glitch_intensity: float = 0.7,
                 glitch_frequency: float = 0.3,
                 rgb_shift_amount: int = 10,
                 scan_lines: bool = True,
                 digital_noise: bool = True,
                 displacement: bool = True):
        super().__init__(config)
        self.glitch_intensity = glitch_intensity
        self.glitch_frequency = glitch_frequency
        self.rgb_shift_amount = rgb_shift_amount
        self.scan_lines = scan_lines
        self.digital_noise = digital_noise
        self.displacement = displacement
        
        # Pre-calculate glitch timeline
        self.glitch_timeline = self._generate_glitch_timeline()
    
    def _generate_glitch_timeline(self):
        """Generate when glitches occur"""
        timeline = []
        total_duration_sec = self.config.duration_ms / 1000.0
        
        # Create glitch bursts
        current_time = 0
        while current_time < total_duration_sec:
            if random.random() < self.glitch_frequency:
                timeline.append({
                    'start': current_time,
                    'duration': random.uniform(0.05, 0.2),  # 50-200ms glitches
                    'intensity': random.uniform(0.3, self.glitch_intensity)
                })
            current_time += random.uniform(0.1, 0.5)
        
        return timeline
    
    def _is_glitching(self, progress: float):
        """Check if currently glitching and return intensity"""
        current_time = progress * self.config.duration_ms / 1000.0
        
        for glitch in self.glitch_timeline:
            if glitch['start'] <= current_time < glitch['start'] + glitch['duration']:
                return True, glitch['intensity']
        return False, 0
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with glitch effects"""
        is_glitch, intensity = self._is_glitching(progress)
        
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Base appearance animation
            letter.opacity = letter_progress
            
            if is_glitch and letter_progress > 0.1:
                # Apply glitch effects
                
                # Random displacement
                if self.displacement:
                    letter.position[0] = self.original_positions[i][0] + random.randint(-20, 20) * intensity
                    letter.position[1] = self.original_positions[i][1] + random.randint(-10, 10) * intensity
                
                # Random opacity flicker
                if random.random() < intensity:
                    letter.opacity *= random.uniform(0.3, 1.0)
                
                # Color corruption
                if random.random() < intensity * 0.5:
                    # Temporarily change color to cyan/magenta/yellow
                    glitch_colors = [
                        (0, 255, 255),  # Cyan
                        (255, 0, 255),  # Magenta
                        (255, 255, 0),  # Yellow
                    ]
                    letter.color = random.choice(glitch_colors)
                else:
                    letter.color = self.config.font_color
                
                # Z-axis glitch
                letter.position[2] = self.original_positions[i][2] + random.randint(-50, 50) * intensity
            else:
                # Reset to normal when not glitching
                letter.position = self.original_positions[i].copy()
                letter.color = self.config.font_color
    
    def apply_frame_post_effects(self, frame: np.ndarray, progress: float) -> np.ndarray:
        """Apply post-processing glitch effects to the entire frame"""
        is_glitch, intensity = self._is_glitching(progress)
        
        if not is_glitch:
            return frame
        
        result = frame.copy()
        
        # RGB channel shift
        if self.rgb_shift_amount > 0 and random.random() < intensity:
            shift = int(self.rgb_shift_amount * intensity)
            h, w = result.shape[:2]
            
            # Shift red channel right
            if shift < w:
                result[:, shift:, 2] = frame[:, :-shift, 2]
            
            # Shift blue channel left
            if shift < w:
                result[:, :-shift, 0] = frame[:, shift:, 0]
        
        # Scan lines
        if self.scan_lines and random.random() < intensity:
            line_height = max(2, int(10 * (1 - intensity)))
            for y in range(0, result.shape[0], line_height * 2):
                result[y:y+line_height, :] = result[y:y+line_height, :] * 0.3
        
        # Digital noise
        if self.digital_noise and random.random() < intensity:
            noise = np.random.randint(0, int(100 * intensity), result.shape, dtype=np.uint8)
            result = cv2.add(result, noise)
        
        # Random horizontal bars
        if random.random() < intensity * 0.5:
            bar_height = random.randint(5, 30)
            bar_y = random.randint(0, max(1, result.shape[0] - bar_height))
            result[bar_y:bar_y+bar_height, :] = 255 - result[bar_y:bar_y+bar_height, :]
        
        return result


class Digital3D(Base3DTextAnimation):
    """3D digital matrix-style reveal animation"""
    
    def __init__(self, config: Animation3DConfig,
                 matrix_rain: bool = True,
                 binary_fade: bool = True,
                 terminal_cursor: bool = True):
        super().__init__(config)
        self.matrix_rain = matrix_rain
        self.binary_fade = binary_fade
        self.terminal_cursor = terminal_cursor
        
        # Generate random binary/hex strings for each letter
        self.letter_codes = []
        for letter in self.letters:
            if self.binary_fade:
                code = ''.join([random.choice('01') for _ in range(8)])
            else:
                code = ''.join([random.choice('0123456789ABCDEF') for _ in range(4)])
            self.letter_codes.append(code)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with digital reveal effect"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Letters materialize from digital code
            if letter_progress < 0.3:
                # Show as code
                letter.opacity = 0.5
                letter.color = (0, 255, 0)  # Matrix green
                # Could modify character here but requires sprite regeneration
            elif letter_progress < 0.6:
                # Transitioning
                letter.opacity = 0.5 + 0.5 * ((letter_progress - 0.3) / 0.3)
                # Fade from green to white
                green_amount = int(255 * (1 - (letter_progress - 0.3) / 0.3))
                letter.color = (green_amount, 255, green_amount)
            else:
                # Fully materialized
                letter.opacity = 1.0
                letter.color = self.config.font_color
            
            # Matrix rain effect - letters fall from above
            if self.matrix_rain:
                fall_offset = (1 - letter_progress) * 100
                letter.position[1] = self.original_positions[i][1] - fall_offset
            
            # Add digital flicker
            if random.random() < 0.1 * (1 - letter_progress):
                letter.opacity *= 0.7


class Hologram3D(Base3DTextAnimation):
    """3D holographic projection animation"""
    
    def __init__(self, config: Animation3DConfig,
                 scan_speed: float = 2.0,
                 flicker_amount: float = 0.3,
                 chromatic_aberration: bool = True):
        super().__init__(config)
        self.scan_speed = scan_speed
        self.flicker_amount = flicker_amount
        self.chromatic_aberration = chromatic_aberration
        
        # Override colors for holographic look
        config.font_color = (100, 200, 255)  # Cyan-blue
        config.depth_color = (50, 100, 150)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with holographic effect"""
        # Calculate scan line position
        scan_position = (progress * self.scan_speed) % 1.0
        
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Base opacity
            letter.opacity = letter_progress * 0.8  # Holograms are semi-transparent
            
            # Scan line effect
            letter_y_normalized = (i / len(self.letters))
            distance_from_scan = abs(letter_y_normalized - scan_position)
            
            if distance_from_scan < 0.1:
                # Brighten at scan line
                letter.opacity = min(1.0, letter.opacity + 0.3)
                letter.glow_intensity = 2.0
            else:
                letter.glow_intensity = 0.5
            
            # Holographic flicker
            if random.random() < self.flicker_amount * (1 - letter_progress):
                letter.opacity *= random.uniform(0.5, 1.0)
            
            # Floating effect
            float_offset = np.sin(progress * np.pi * 4 + i * 0.5) * 5
            letter.position[1] = self.original_positions[i][1] + float_offset
            
            # Slight rotation for 3D effect
            letter.rotation[1] = np.sin(progress * np.pi * 2) * 0.2


class Static3D(Base3DTextAnimation):
    """3D TV static/noise reveal animation"""
    
    def __init__(self, config: Animation3DConfig,
                 static_intensity: float = 0.8,
                 tune_in_effect: bool = True,
                 horizontal_hold: bool = True):
        super().__init__(config)
        self.static_intensity = static_intensity
        self.tune_in_effect = tune_in_effect
        self.horizontal_hold = horizontal_hold
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with static/interference effect"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Static intensity decreases as text becomes clear
            current_static = self.static_intensity * (1 - letter_progress)
            
            if current_static > 0.5:
                # Heavy static - barely visible
                letter.opacity = random.uniform(0, 0.3)
                # Random position jitter
                letter.position[0] = self.original_positions[i][0] + random.randint(-10, 10)
                letter.position[1] = self.original_positions[i][1] + random.randint(-5, 5)
            elif current_static > 0.2:
                # Medium static - flickering
                letter.opacity = 0.3 + letter_progress * 0.5 + random.uniform(-0.2, 0.2)
                # Smaller jitter
                letter.position[0] = self.original_positions[i][0] + random.randint(-3, 3)
                letter.position[1] = self.original_positions[i][1] + random.randint(-2, 2)
            else:
                # Clear signal
                letter.opacity = letter_progress
                letter.position = self.original_positions[i].copy()
            
            # Horizontal hold problem (rolling image)
            if self.horizontal_hold and current_static > 0.3:
                roll_amount = np.sin(frame_number * 0.1) * 20 * current_static
                letter.position[1] = self.original_positions[i][1] + roll_amount
            
            # Color distortion in static
            if current_static > 0.3:
                # Random color channel emphasis
                channel = random.randint(0, 2)
                color_list = list(self.config.font_color)
                color_list[channel] = min(255, color_list[channel] + 100)
                letter.color = tuple(color_list)