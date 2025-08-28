"""
3D Compound text animations that combine multiple effects
These animations layer multiple 3D effects for complex motion
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base_3d_text_animation import Base3DTextAnimation, Animation3DConfig, Letter3D


class FadeSlide3D(Base3DTextAnimation):
    """Combined 3D fade and slide with depth"""
    
    def __init__(self, config: Animation3DConfig,
                 slide_direction: np.ndarray = None,
                 slide_distance: float = 200,
                 spiral_slide: bool = False,
                 depth_fade: bool = True):
        super().__init__(config)
        self.slide_direction = slide_direction or np.array([-1, 0, 0.5])
        self.slide_distance = slide_distance
        self.spiral_slide = spiral_slide
        self.depth_fade = depth_fade
        
        # Normalize direction
        self.slide_direction = self.slide_direction / np.linalg.norm(self.slide_direction)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update with combined fade and slide"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Slide component
            if self.spiral_slide:
                # Spiral trajectory
                angle = letter_progress * np.pi * 2
                radius = self.slide_distance * (1 - letter_progress)
                slide_offset = np.array([
                    np.cos(angle + i * 0.5) * radius,
                    np.sin(angle + i * 0.5) * radius * 0.5,
                    radius * 0.3
                ])
            else:
                # Linear slide
                slide_offset = self.slide_direction * self.slide_distance * (1 - letter_progress)
            
            letter.position = self.original_positions[i] + slide_offset
            
            # Fade component
            base_opacity = letter_progress
            
            if self.depth_fade:
                # Opacity varies with Z depth
                depth_factor = 1.0 - (letter.position[2] - self.original_positions[i][2]) / 100
                depth_factor = max(0.2, min(1.0, depth_factor))
                letter.opacity = base_opacity * depth_factor
            else:
                letter.opacity = base_opacity
            
            # Add rotation during slide
            letter.rotation[1] = (1 - letter_progress) * np.pi


class ZoomBlur3D(Base3DTextAnimation):
    """Combined 3D zoom with motion blur effect"""
    
    def __init__(self, config: Animation3DConfig,
                 zoom_origin: str = "center",  # center, random, spiral
                 max_zoom: float = 5.0,
                 blur_intensity: float = 20):
        super().__init__(config)
        self.zoom_origin = zoom_origin
        self.max_zoom = max_zoom
        self.blur_intensity = blur_intensity
        
        # Calculate zoom centers
        self.zoom_centers = []
        for i, letter in enumerate(self.letters):
            if zoom_origin == "random":
                center = self.original_positions[i] + np.random.randn(3) * 100
            elif zoom_origin == "spiral":
                angle = i * np.pi / 3
                center = self.original_positions[i] + np.array([
                    np.cos(angle) * 100,
                    np.sin(angle) * 100,
                    0
                ])
            else:  # center
                center = np.array([config.position[0], config.position[1], config.position[2]])
            
            self.zoom_centers.append(center)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update with zoom and blur"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Zoom component
            zoom_factor = self.max_zoom * (1 - letter_progress) + letter_progress
            letter.scale = np.ones(3) * zoom_factor
            
            # Move from zoom center
            zoom_center = self.zoom_centers[i]
            direction = self.original_positions[i] - zoom_center
            letter.position = zoom_center + direction * zoom_factor
            
            # Blur based on zoom speed
            zoom_speed = abs(self.max_zoom - 1.0) * (1 - letter_progress)
            letter.blur_amount = self.blur_intensity * zoom_speed
            
            # Opacity
            letter.opacity = letter_progress
            
            # Rotation adds dynamism
            letter.rotation[2] = (1 - letter_progress) * np.pi


class RotateExplode3D(Base3DTextAnimation):
    """Letters rotate while exploding outward"""
    
    def __init__(self, config: Animation3DConfig,
                 explosion_force: float = 300,
                 rotation_speed: float = 4.0,
                 implosion_first: bool = True):
        super().__init__(config)
        self.explosion_force = explosion_force
        self.rotation_speed = rotation_speed
        self.implosion_first = implosion_first
        
        # Generate explosion vectors
        self.explosion_vectors = []
        center = np.array([config.position[0], config.position[1], config.position[2]])
        
        for i, letter in enumerate(self.letters):
            direction = self.original_positions[i] - center
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
            
            self.explosion_vectors.append(direction * explosion_force)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update with rotation and explosion"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            if self.implosion_first and letter_progress < 0.3:
                # Implosion phase
                implode_t = letter_progress / 0.3
                letter.position = self.original_positions[i] - \
                                self.explosion_vectors[i] * 0.3 * implode_t
                letter.scale = np.ones(3) * (1.0 - 0.3 * implode_t)
                letter.rotation = np.array([implode_t, implode_t * 2, implode_t * 0.5]) * np.pi
                letter.opacity = 1.0
            else:
                # Explosion phase
                if self.implosion_first:
                    explode_t = (letter_progress - 0.3) / 0.7
                else:
                    explode_t = letter_progress
                
                # Position
                letter.position = self.original_positions[i] + \
                                self.explosion_vectors[i] * explode_t
                
                # Rotation accelerates
                rotation_amount = explode_t * explode_t * self.rotation_speed
                letter.rotation = np.array([
                    rotation_amount * np.sin(i),
                    rotation_amount * np.cos(i),
                    rotation_amount * 0.5
                ]) * np.pi
                
                # Scale pulses then shrinks
                if explode_t < 0.5:
                    letter.scale = np.ones(3) * (1.0 + explode_t)
                else:
                    letter.scale = np.ones(3) * (2.0 - explode_t)
                
                # Fade out
                letter.opacity = 1.0 - explode_t * 0.7


class WaveFloat3D(Base3DTextAnimation):
    """Combined wave motion with floating effect"""
    
    def __init__(self, config: Animation3DConfig,
                 wave_amplitude: float = 50,
                 float_height: float = 100,
                 wave_speed: float = 2.0):
        super().__init__(config)
        self.wave_amplitude = wave_amplitude
        self.float_height = float_height
        self.wave_speed = wave_speed
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update with wave and float"""
        time = frame_number / fps
        
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Float upward
            float_y = -self.float_height * letter_progress
            
            # Wave motion
            wave_x = np.sin(time * self.wave_speed + i * 0.5) * self.wave_amplitude * letter_progress
            wave_z = np.cos(time * self.wave_speed * 0.5 + i * 0.3) * self.wave_amplitude * 0.5 * letter_progress
            
            # Combined position
            letter.position = self.original_positions[i] + np.array([wave_x, float_y, wave_z])
            
            # Gentle rotation
            letter.rotation = np.array([
                np.sin(time + i) * 0.2,
                np.cos(time + i) * 0.3,
                np.sin(time * 0.5 + i) * 0.1
            ]) * letter_progress
            
            # Scale pulse
            scale_pulse = 1.0 + np.sin(time * 3 + i) * 0.1 * letter_progress
            letter.scale = np.ones(3) * scale_pulse
            
            # Fade in
            letter.opacity = letter_progress


class TypewriterBounce3D(Base3DTextAnimation):
    """Typewriter effect with bounce physics"""
    
    def __init__(self, config: Animation3DConfig,
                 bounce_height: float = 50,
                 typing_speed: float = 10):
        super().__init__(config)
        self.bounce_height = bounce_height
        self.typing_speed = typing_speed
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update with typewriter and bounce"""
        for i, letter in enumerate(self.letters):
            # Typewriter timing
            type_delay = i / len(self.letters) * 0.7
            type_progress = max(0, min(1, (progress - type_delay) / 0.3))
            
            if type_progress <= 0:
                letter.opacity = 0
                letter.position[1] = self.original_positions[i][1] - self.bounce_height
            elif type_progress < 0.5:
                # Bounce in
                bounce_t = type_progress * 2
                
                # Physics-based bounce
                gravity = 9.8 * 100  # pixels/s²
                initial_velocity = np.sqrt(2 * gravity * self.bounce_height)
                
                t = bounce_t * 0.5  # time in seconds
                height = initial_velocity * t - 0.5 * gravity * t * t
                
                letter.position[1] = self.original_positions[i][1] - self.bounce_height + height
                letter.opacity = bounce_t
                
                # Spin while bouncing
                letter.rotation[2] = bounce_t * np.pi
            else:
                # Settle with damped oscillation
                settle_t = (type_progress - 0.5) * 2
                oscillation = np.sin(settle_t * np.pi * 4) * np.exp(-settle_t * 3) * 10
                
                letter.position[1] = self.original_positions[i][1] + oscillation
                letter.opacity = 1.0
                letter.rotation[2] = 0


class SpiralMaterialize3D(Base3DTextAnimation):
    """Spiral motion with materialization effect"""
    
    def __init__(self, config: Animation3DConfig,
                 spiral_radius: float = 150,
                 spiral_rotations: float = 3,
                 particle_effect: bool = True):
        super().__init__(config)
        self.spiral_radius = spiral_radius
        self.spiral_rotations = spiral_rotations
        self.particle_effect = particle_effect
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update with spiral and materialization"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Spiral motion
            angle = letter_progress * np.pi * 2 * self.spiral_rotations
            radius = self.spiral_radius * (1 - letter_progress)
            
            spiral_x = np.cos(angle) * radius
            spiral_y = np.sin(angle) * radius * 0.5
            spiral_z = -radius
            
            letter.position = self.original_positions[i] + np.array([spiral_x, spiral_y, spiral_z])
            
            # Materialization effect
            if self.particle_effect:
                # Particle-like behavior
                if letter_progress < 0.3:
                    # Gathering particles
                    gather_t = letter_progress / 0.3
                    letter.scale = np.ones(3) * gather_t * 0.5
                    letter.blur_amount = (1 - gather_t) * 20
                    letter.opacity = gather_t * 0.5
                elif letter_progress < 0.7:
                    # Forming shape
                    form_t = (letter_progress - 0.3) / 0.4
                    letter.scale = np.ones(3) * (0.5 + 0.5 * form_t)
                    letter.blur_amount = (1 - form_t) * 5
                    letter.opacity = 0.5 + 0.3 * form_t
                else:
                    # Solidified
                    solid_t = (letter_progress - 0.7) / 0.3
                    letter.scale = np.ones(3)
                    letter.blur_amount = 0
                    letter.opacity = 0.8 + 0.2 * solid_t
                    
                    # Glow when solidified
                    letter.glow_intensity = solid_t * 1.5
            else:
                # Simple fade
                letter.scale = np.ones(3) * letter_progress
                letter.opacity = letter_progress
            
            # Rotation during spiral
            letter.rotation = np.array([angle * 0.5, angle, angle * 0.3])


class ElasticPop3D(Base3DTextAnimation):
    """Elastic pop effect with overshoot"""
    
    def __init__(self, config: Animation3DConfig,
                 overshoot_scale: float = 1.5,
                 elasticity: float = 0.3,
                 pop_rotation: bool = True):
        super().__init__(config)
        self.overshoot_scale = overshoot_scale
        self.elasticity = elasticity
        self.pop_rotation = pop_rotation
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update with elastic pop"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Elastic easing function
            if letter_progress < 0.6:
                # Growing phase with overshoot
                t = letter_progress / 0.6
                # Elastic formula
                scale = self.overshoot_scale * (1 - np.exp(-6 * t) * np.cos(np.pi * 2 * t / self.elasticity))
            else:
                # Settling phase
                t = (letter_progress - 0.6) / 0.4
                scale = self.overshoot_scale - (self.overshoot_scale - 1.0) * t
            
            letter.scale = np.ones(3) * scale
            
            # Pop rotation if enabled
            if self.pop_rotation:
                if letter_progress < 0.3:
                    # Initial spin
                    spin_t = letter_progress / 0.3
                    letter.rotation[2] = spin_t * np.pi * 2
                else:
                    # Stabilize
                    letter.rotation[2] = 0
            
            # Z-axis pop
            z_pop = np.sin(letter_progress * np.pi) * 50
            letter.position[2] = self.original_positions[i][2] - z_pop
            
            # Opacity
            letter.opacity = min(1.0, letter_progress * 3)
            
            # Glow during pop
            if letter_progress < 0.6:
                letter.glow_intensity = np.sin(letter_progress / 0.6 * np.pi) * 2