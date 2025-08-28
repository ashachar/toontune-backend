"""
3D Progressive text animations with individual letter control
These animations reveal text progressively with 3D effects
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base_3d_text_animation import Base3DTextAnimation, Animation3DConfig, Letter3D


class Typewriter3D(Base3DTextAnimation):
    """3D typewriter effect with mechanical key press simulation"""
    
    def __init__(self, config: Animation3DConfig,
                 key_press_depth: float = 20,
                 key_bounce: bool = True,
                 mechanical_sound: bool = False,  # For future sound integration
                 cursor_3d: bool = True):
        super().__init__(config)
        self.key_press_depth = key_press_depth
        self.key_bounce = key_bounce
        self.cursor_3d = cursor_3d
        
        # Override stagger for typewriter effect
        self.config.stagger_type = "sequential"
        self.config.stagger_ms = int(config.duration_ms / len(self.letters))
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with typewriter effect"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            if letter_progress <= 0:
                # Letter not typed yet - invisible
                letter.opacity = 0
                letter.position[2] = self.original_positions[i][2] - self.key_press_depth
            elif letter_progress < 0.2:
                # Key press phase
                press_t = letter_progress / 0.2
                letter.opacity = press_t
                
                # Simulate key being pressed
                letter.position[2] = self.original_positions[i][2] - self.key_press_depth * (1 - press_t)
                
                # Scale effect during press
                letter.scale = np.array([
                    1.0 - 0.2 * (1 - press_t),
                    1.0 - 0.2 * (1 - press_t),
                    1.0
                ])
            elif self.key_bounce and letter_progress < 0.4:
                # Bounce phase
                bounce_t = (letter_progress - 0.2) / 0.2
                letter.opacity = 1.0
                
                # Bounce back slightly past original position
                overshoot = np.sin(bounce_t * np.pi) * 5
                letter.position[2] = self.original_positions[i][2] + overshoot
                
                # Return to normal scale
                letter.scale = np.ones(3)
            else:
                # Letter fully typed
                letter.opacity = 1.0
                letter.position[2] = self.original_positions[i][2]
                letter.scale = np.ones(3)
            
            # Add slight vibration to recently typed letters
            if 0.2 < letter_progress < 0.6:
                vibration = np.sin(frame_number * 0.5) * 0.5
                letter.position[0] = self.original_positions[i][0] + vibration
                letter.position[1] = self.original_positions[i][1] + vibration * 0.5


class WordReveal3D(Base3DTextAnimation):
    """3D word-by-word reveal with depth effects"""
    
    def __init__(self, config: Animation3DConfig,
                 reveal_style: str = "flip",  # flip, rise, unfold, materialize
                 word_spacing_factor: float = 1.5):
        super().__init__(config)
        self.reveal_style = reveal_style
        self.word_spacing_factor = word_spacing_factor
        
        # Parse words
        self.words = []
        current_word = []
        for i, letter in enumerate(self.letters):
            if letter.character == ' ':
                if current_word:
                    self.words.append(current_word)
                    current_word = []
            else:
                current_word.append(i)
        if current_word:
            self.words.append(current_word)
        
        # Calculate timing for each word
        self.word_timings = []
        stagger_per_word = 1.0 / len(self.words)
        for w, word in enumerate(self.words):
            self.word_timings.append(w * stagger_per_word)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with word reveal effect"""
        for w, word_indices in enumerate(self.words):
            word_start_time = self.word_timings[w]
            word_duration = 1.0 / len(self.words)
            word_progress = max(0, min(1, (progress - word_start_time) / word_duration))
            
            for letter_index in word_indices:
                letter = self.letters[letter_index]
                
                if self.reveal_style == "flip":
                    # 3D flip reveal
                    if word_progress < 0.5:
                        # First half - flip in
                        flip_angle = np.pi * (1 - word_progress * 2)
                        letter.rotation[1] = flip_angle
                        letter.scale[0] = abs(np.cos(flip_angle))
                        letter.opacity = word_progress * 2
                    else:
                        # Second half - stabilize
                        letter.rotation[1] = 0
                        letter.scale = np.ones(3)
                        letter.opacity = 1.0
                
                elif self.reveal_style == "rise":
                    # Rise from below with rotation
                    rise_height = 100
                    letter.position[1] = self.original_positions[letter_index][1] + \
                                       rise_height * (1 - word_progress)
                    letter.rotation[0] = (1 - word_progress) * np.pi / 4
                    letter.opacity = word_progress
                    
                elif self.reveal_style == "unfold":
                    # Unfold like paper
                    if word_progress < 0.3:
                        # Unfolding phase
                        unfold_t = word_progress / 0.3
                        letter.rotation[0] = (1 - unfold_t) * np.pi
                        letter.scale[1] = unfold_t
                        letter.opacity = unfold_t
                    else:
                        # Settled phase
                        letter.rotation[0] = 0
                        letter.scale = np.ones(3)
                        letter.opacity = 1.0
                
                elif self.reveal_style == "materialize":
                    # Materialize from particles
                    if word_progress < 0.7:
                        # Gathering phase
                        gather_t = word_progress / 0.7
                        
                        # Particles converge
                        scatter_radius = 50
                        scatter_x = np.cos(letter_index * 2) * scatter_radius * (1 - gather_t)
                        scatter_y = np.sin(letter_index * 2) * scatter_radius * (1 - gather_t)
                        scatter_z = np.cos(letter_index * 3) * scatter_radius * (1 - gather_t)
                        
                        letter.position = self.original_positions[letter_index] + \
                                        np.array([scatter_x, scatter_y, scatter_z])
                        
                        # Spinning while gathering
                        letter.rotation = np.array([1, 1, 1]) * (1 - gather_t) * np.pi * 2
                        
                        # Scale and opacity
                        letter.scale = np.ones(3) * (0.5 + 0.5 * gather_t)
                        letter.opacity = gather_t * 0.8
                    else:
                        # Solidify phase
                        solidify_t = (word_progress - 0.7) / 0.3
                        letter.position = self.original_positions[letter_index]
                        letter.rotation = np.zeros(3)
                        letter.scale = np.ones(3)
                        letter.opacity = 0.8 + 0.2 * solidify_t


class WaveReveal3D(Base3DTextAnimation):
    """3D wave reveal where letters appear in a wave pattern"""
    
    def __init__(self, config: Animation3DConfig,
                 wave_amplitude: float = 100,
                 wave_frequency: float = 2.0,
                 wave_direction: str = "horizontal",  # horizontal, vertical, radial, spiral
                 depth_wave: bool = True):
        super().__init__(config)
        self.wave_amplitude = wave_amplitude
        self.wave_frequency = wave_frequency
        self.wave_direction = wave_direction
        self.depth_wave = depth_wave
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with wave reveal"""
        total_letters = len(self.letters)
        
        for i, letter in enumerate(self.letters):
            # Calculate wave position based on direction
            if self.wave_direction == "horizontal":
                wave_position = i / total_letters
            elif self.wave_direction == "vertical":
                # Simulate vertical based on letter groups
                wave_position = (i % 5) / 5.0
            elif self.wave_direction == "radial":
                # Distance from center
                center = total_letters / 2
                wave_position = abs(i - center) / center
            elif self.wave_direction == "spiral":
                # Spiral pattern
                angle = i * np.pi / 4
                wave_position = (np.sin(angle) + 1) / 2
            else:
                wave_position = i / total_letters
            
            # Calculate wave progress for this letter
            wave_offset = wave_position * 0.5  # Spread the wave
            wave_progress = max(0, min(1, (progress - wave_offset) * 2))
            
            # Wave motion
            wave_height = np.sin(wave_progress * np.pi) * self.wave_amplitude
            
            # Update position
            letter.position[1] = self.original_positions[i][1] - wave_height
            
            if self.depth_wave:
                # Add Z-axis wave
                letter.position[2] = self.original_positions[i][2] + \
                                   np.sin(wave_progress * np.pi + i * 0.5) * 30
            
            # Rotation during wave - smoother without rapid oscillation
            letter.rotation[2] = np.sin(wave_progress * np.pi) * 0.3
            
            # Scale pulse
            scale_factor = 1.0 + np.sin(wave_progress * np.pi) * 0.3
            letter.scale = np.array([scale_factor, scale_factor, 1.0])
            
            # Opacity follows wave
            letter.opacity = wave_progress


class Cascade3D(Base3DTextAnimation):
    """3D cascading reveal like dominos or waterfall"""
    
    def __init__(self, config: Animation3DConfig,
                 cascade_style: str = "domino",  # domino, waterfall, cards, fountain
                 cascade_speed: float = 2.0,
                 cascade_overlap: float = 0.7):  # How much letters overlap in timing
        super().__init__(config)
        self.cascade_style = cascade_style
        self.cascade_speed = cascade_speed
        self.cascade_overlap = cascade_overlap
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with cascade effect"""
        total_letters = len(self.letters)
        
        for i, letter in enumerate(self.letters):
            # Calculate cascade timing
            cascade_delay = (i / total_letters) * self.cascade_overlap
            cascade_progress = max(0, min(1, (progress - cascade_delay) / (1 - self.cascade_overlap)))
            
            if self.cascade_style == "domino":
                # Domino falling effect
                if cascade_progress < 0.5:
                    # Falling phase
                    fall_angle = cascade_progress * np.pi
                    letter.rotation[0] = fall_angle
                    letter.position[1] = self.original_positions[i][1] + \
                                       np.sin(fall_angle) * 30
                    letter.opacity = 1.0
                else:
                    # Settled phase
                    letter.rotation[0] = np.pi
                    letter.position[1] = self.original_positions[i][1] + 30
                    letter.opacity = 1.0 - (cascade_progress - 0.5) * 0.5
                
            elif self.cascade_style == "waterfall":
                # Waterfall drop effect
                drop_height = 200
                letter.position[1] = self.original_positions[i][1] - drop_height * (1 - cascade_progress)
                
                # Splash effect at bottom
                if cascade_progress > 0.8:
                    splash_t = (cascade_progress - 0.8) / 0.2
                    letter.scale = np.array([
                        1.0 + splash_t * 0.5,
                        1.0 - splash_t * 0.3,
                        1.0
                    ])
                
                # Fade in during fall
                letter.opacity = min(1.0, cascade_progress * 2)
                
            elif self.cascade_style == "cards":
                # Card dealing effect
                if cascade_progress < 0.3:
                    # Dealing phase
                    deal_t = cascade_progress / 0.3
                    
                    # Arc trajectory
                    arc_height = -100 * np.sin(deal_t * np.pi)
                    letter.position[1] = self.original_positions[i][1] + arc_height
                    letter.position[0] = self.original_positions[i][0] - 200 * (1 - deal_t)
                    
                    # Spin while dealing
                    letter.rotation[2] = deal_t * np.pi * 2
                    letter.opacity = deal_t
                else:
                    # Settled
                    letter.position = self.original_positions[i]
                    letter.rotation[2] = 0
                    letter.opacity = 1.0
                
            elif self.cascade_style == "fountain":
                # Fountain eruption effect
                if cascade_progress < 0.6:
                    # Rising phase
                    rise_t = cascade_progress / 0.6
                    
                    # Parabolic trajectory
                    height = -rise_t * (rise_t - 1) * 400  # Parabola
                    spread = rise_t * 100 * np.sin(i * 0.5)
                    
                    letter.position[1] = self.original_positions[i][1] - height
                    letter.position[0] = self.original_positions[i][0] + spread
                    
                    # Rotation during rise
                    letter.rotation = np.array([rise_t, rise_t * 2, rise_t * 0.5]) * np.pi
                    
                    letter.opacity = rise_t
                else:
                    # Falling/settling phase
                    fall_t = (cascade_progress - 0.6) / 0.4
                    
                    letter.position[1] = self.original_positions[i][1] - 100 * (1 - fall_t)
                    letter.position[0] = self.original_positions[i][0] + 100 * np.sin(i * 0.5) * (1 - fall_t)
                    
                    letter.rotation *= (1 - fall_t)
                    letter.opacity = 1.0


class Build3D(Base3DTextAnimation):
    """3D building/construction animation"""
    
    def __init__(self, config: Animation3DConfig,
                 build_style: str = "blocks",  # blocks, scaffold, print, assemble
                 build_from_direction: str = "bottom"):
        super().__init__(config)
        self.build_style = build_style
        self.build_from_direction = build_from_direction
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with building effect"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            if self.build_style == "blocks":
                # Building blocks stacking
                if letter_progress < 0.3:
                    # Block falling into place
                    fall_t = letter_progress / 0.3
                    fall_height = 200 * (1 - fall_t)
                    
                    letter.position[1] = self.original_positions[i][1] - fall_height
                    letter.scale = np.array([fall_t, fall_t, fall_t])
                    letter.opacity = fall_t
                elif letter_progress < 0.5:
                    # Impact wobble
                    wobble_t = (letter_progress - 0.3) / 0.2
                    wobble = np.sin(wobble_t * np.pi * 4) * (1 - wobble_t)
                    
                    letter.position[1] = self.original_positions[i][1] + wobble * 10
                    letter.scale = np.array([
                        1.0 + wobble * 0.1,
                        1.0 - wobble * 0.1,
                        1.0
                    ])
                    letter.opacity = 1.0
                else:
                    # Settled
                    letter.position = self.original_positions[i]
                    letter.scale = np.ones(3)
                    letter.opacity = 1.0
                    
            elif self.build_style == "scaffold":
                # Scaffolding construction
                if letter_progress < 0.4:
                    # Frame appears
                    frame_t = letter_progress / 0.4
                    letter.scale = np.array([frame_t, frame_t, 0.1])
                    letter.opacity = frame_t * 0.5
                    letter.blur_amount = 10 * (1 - frame_t)
                else:
                    # Fill in
                    fill_t = (letter_progress - 0.4) / 0.6
                    letter.scale = np.array([1.0, 1.0, fill_t])
                    letter.opacity = 0.5 + 0.5 * fill_t
                    letter.blur_amount = 0
                    
            elif self.build_style == "print":
                # 3D printing effect
                print_layers = 10
                current_layer = int(letter_progress * print_layers)
                layer_progress = (letter_progress * print_layers) % 1
                
                # Build layer by layer
                letter.scale[1] = current_layer / print_layers + layer_progress / print_layers
                letter.position[1] = self.original_positions[i][1] + \
                                   (1 - letter.scale[1]) * 30
                
                # Glow on current printing layer
                if layer_progress < 0.5:
                    letter.glow_intensity = layer_progress * 2
                else:
                    letter.glow_intensity = (1 - layer_progress) * 2
                
                letter.opacity = min(1.0, current_layer / 3)
                
            elif self.build_style == "assemble":
                # Parts assembling
                if letter_progress < 0.6:
                    # Parts flying in
                    assemble_t = letter_progress / 0.6
                    
                    # Different parts from different directions
                    angle = i * np.pi / 3
                    start_pos = np.array([
                        np.cos(angle) * 200,
                        np.sin(angle) * 200,
                        (np.random.random() - 0.5) * 100
                    ])
                    
                    letter.position = self.original_positions[i] + start_pos * (1 - assemble_t)
                    letter.rotation = np.array([1, 1, 1]) * (1 - assemble_t) * np.pi
                    letter.scale = np.ones(3) * (0.5 + 0.5 * assemble_t)
                    letter.opacity = assemble_t
                else:
                    # Locked in place
                    lock_t = (letter_progress - 0.6) / 0.4
                    
                    letter.position = self.original_positions[i]
                    letter.rotation = np.zeros(3)
                    letter.scale = np.ones(3)
                    letter.opacity = 1.0
                    
                    # Flash when locked
                    if lock_t < 0.2:
                        letter.glow_intensity = (0.2 - lock_t) * 5