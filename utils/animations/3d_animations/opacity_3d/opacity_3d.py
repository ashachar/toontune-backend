"""
3D Opacity-based text animations with individual letter control
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base_3d_text_animation import Base3DTextAnimation, Animation3DConfig, Letter3D


class Fade3D(Base3DTextAnimation):
    """3D fade animation with per-letter control"""
    
    def __init__(self, config: Animation3DConfig,
                 start_opacity: float = 0.0,
                 end_opacity: float = 1.0,
                 fade_mode: str = "uniform",  # uniform, wave, spiral
                 depth_fade: bool = True):
        super().__init__(config)
        self.start_opacity = start_opacity
        self.end_opacity = end_opacity
        self.fade_mode = fade_mode
        self.depth_fade = depth_fade
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letter opacities with 3D effects"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Calculate base opacity - simple linear fade
            opacity = self.start_opacity + (self.end_opacity - self.start_opacity) * letter_progress
            
            # Apply fade mode variations
            if self.fade_mode == "wave":
                # NO wave modulation - just use staggered timing for wave effect
                # The stagger already creates a wave-like appearance
                pass  # Simply use base opacity
            elif self.fade_mode == "spiral":
                # Spiral fade from center - smooth transition without oscillation
                center_distance = abs(i - len(self.letters) / 2) / len(self.letters)
                # Use a smooth offset that doesn't oscillate
                spiral_offset = center_distance * 0.1 * (1 - letter_progress)
                opacity = max(0, min(1, opacity - spiral_offset))
            
            # Apply depth fade (letters further back fade differently)
            if self.depth_fade:
                # Smooth depth variation without sine oscillation
                z_offset = (i / len(self.letters)) * 30  # Linear depth, no oscillation
                letter.position[2] = self.original_positions[i][2] + z_offset * (1 - letter_progress)
            
            letter.opacity = max(0, min(1, opacity))


class BlurFade3D(Base3DTextAnimation):
    """3D fade with blur effect on individual letters"""
    
    def __init__(self, config: Animation3DConfig,
                 start_blur: float = 20.0,
                 end_blur: float = 0.0,
                 start_opacity: float = 0.0,
                 end_opacity: float = 1.0,
                 blur_varies_with_depth: bool = True):
        super().__init__(config)
        self.start_blur = start_blur
        self.end_blur = end_blur
        self.start_opacity = start_opacity
        self.end_opacity = end_opacity
        self.blur_varies_with_depth = blur_varies_with_depth
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letter blur and opacity"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Calculate opacity
            letter.opacity = self.start_opacity + (self.end_opacity - self.start_opacity) * letter_progress
            
            # Calculate blur
            base_blur = self.start_blur + (self.end_blur - self.start_blur) * letter_progress
            
            # Vary blur with depth if enabled
            if self.blur_varies_with_depth:
                # Letters start at different depths
                z_offset = np.sin(i * 0.5 + progress * np.pi) * 100
                letter.position[2] = self.original_positions[i][2] + z_offset * (1 - letter_progress)
                
                # Further letters are more blurred
                depth_blur_factor = max(0, letter.position[2] / 100.0)
                letter.blur_amount = base_blur + depth_blur_factor * 5
            else:
                letter.blur_amount = base_blur
            
            # Ensure values are in valid range
            letter.blur_amount = max(0, letter.blur_amount)


class GlowPulse3D(Base3DTextAnimation):
    """3D glow pulse animation with per-letter effects"""
    
    def __init__(self, config: Animation3DConfig,
                 glow_radius: int = 8,
                 pulse_count: int = 3,
                 max_glow_intensity: float = 2.0,
                 glow_cascade: bool = True,
                 start_opacity: float = 0.0,
                 end_opacity: float = 1.0):
        super().__init__(config)
        config.enable_glow = True  # Ensure glow is enabled
        config.glow_radius = glow_radius
        self.pulse_count = pulse_count
        self.max_glow_intensity = max_glow_intensity
        self.glow_cascade = glow_cascade
        self.start_opacity = start_opacity
        self.end_opacity = end_opacity
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letter glow and opacity with pulsing effect"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Calculate base opacity
            letter.opacity = self.start_opacity + (self.end_opacity - self.start_opacity) * letter_progress
            
            # Calculate pulsing glow
            if self.glow_cascade:
                # Cascade effect - each letter pulses with delay
                pulse_phase = (letter_progress * self.pulse_count + i * 0.2) * 2 * np.pi
            else:
                # All letters pulse together
                pulse_phase = letter_progress * self.pulse_count * 2 * np.pi
            
            # Create smooth pulse
            glow_intensity = (np.sin(pulse_phase) * 0.5 + 0.5) * self.max_glow_intensity
            letter.glow_intensity = glow_intensity * letter_progress
            
            # Add subtle Z movement with glow
            z_pulse = np.sin(pulse_phase) * 20
            letter.position[2] = self.original_positions[i][2] + z_pulse


class Dissolve3D(Base3DTextAnimation):
    """3D dissolve effect where letters fade and float away"""
    
    def __init__(self, config: Animation3DConfig,
                 float_distance: float = 100,
                 dissolve_direction: str = "up",  # up, down, random, spiral
                 max_rotation: float = np.pi,
                 scale_during_dissolve: float = 1.5):
        super().__init__(config)
        self.float_distance = float_distance
        self.dissolve_direction = dissolve_direction
        self.max_rotation = max_rotation
        self.scale_during_dissolve = scale_during_dissolve
        
        # Generate random directions for each letter
        self.random_directions = []
        for i in range(len(self.letters)):
            if dissolve_direction == "random":
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
            elif dissolve_direction == "spiral":
                angle = i * np.pi / 4
                direction = np.array([np.cos(angle), np.sin(angle), 0.5])
            elif dissolve_direction == "up":
                direction = np.array([0, -1, 0.2])
            elif dissolve_direction == "down":
                direction = np.array([0, 1, 0.2])
            else:
                direction = np.array([0, -1, 0])
            
            self.random_directions.append(direction)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters for dissolve effect"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Opacity decreases as dissolve progresses
            letter.opacity = 1.0 - letter_progress
            
            # Float away
            float_offset = self.random_directions[i] * self.float_distance * letter_progress
            letter.position = self.original_positions[i] + float_offset
            
            # Add rotation during dissolve
            letter.rotation[0] = letter_progress * self.max_rotation * np.sin(i)
            letter.rotation[1] = letter_progress * self.max_rotation * np.cos(i)
            letter.rotation[2] = letter_progress * self.max_rotation * 0.5
            
            # Scale effect
            scale_factor = 1.0 + (self.scale_during_dissolve - 1.0) * letter_progress
            letter.scale = np.array([scale_factor, scale_factor, scale_factor])
            
            # Add blur as letter dissolves
            letter.blur_amount = letter_progress * 10


class Materialize3D(Base3DTextAnimation):
    """3D materialization effect - letters form from particles"""
    
    def __init__(self, config: Animation3DConfig,
                 particle_spread: float = 200,
                 rotation_speed: float = 2.0,
                 materialize_from: str = "center"):  # center, edges, random
        super().__init__(config)
        self.particle_spread = particle_spread
        self.rotation_speed = rotation_speed
        self.materialize_from = materialize_from
        
        # Generate starting positions for materialization
        self.start_positions = []
        for i, letter in enumerate(self.letters):
            if materialize_from == "center":
                # All letters start from center
                start_pos = np.array([config.position[0], config.position[1], config.position[2] + 100])
            elif materialize_from == "edges":
                # Letters come from screen edges
                angle = i * 2 * np.pi / len(self.letters)
                start_pos = self.original_positions[i] + np.array([
                    np.cos(angle) * particle_spread,
                    np.sin(angle) * particle_spread,
                    100
                ])
            else:  # random
                start_pos = self.original_positions[i] + np.random.randn(3) * particle_spread
            
            self.start_positions.append(start_pos)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters for materialization effect"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Opacity increases as materialization progresses
            letter.opacity = letter_progress
            
            # Move from start position to final position
            letter.position = self.start_positions[i] + \
                            (self.original_positions[i] - self.start_positions[i]) * letter_progress
            
            # Rotation decreases as letter materializes
            rotation_amount = (1 - letter_progress) * self.rotation_speed
            letter.rotation = np.array([
                rotation_amount * np.sin(i * 0.5),
                rotation_amount * np.cos(i * 0.5),
                rotation_amount * 0.5
            ])
            
            # Scale grows from small to normal
            scale_factor = 0.1 + 0.9 * letter_progress
            letter.scale = np.array([scale_factor, scale_factor, scale_factor])
            
            # Blur decreases as letter forms
            letter.blur_amount = (1 - letter_progress) * 15
            
            # Glow effect during materialization
            letter.glow_intensity = np.sin(letter_progress * np.pi) * 1.5