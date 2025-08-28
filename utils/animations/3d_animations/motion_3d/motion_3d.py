"""
3D Motion-based text animations with individual letter control
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base_3d_text_animation import Base3DTextAnimation, Animation3DConfig, Letter3D


class Slide3D(Base3DTextAnimation):
    """3D slide animation where letters slide in from various directions"""
    
    def __init__(self, config: Animation3DConfig,
                 slide_direction: str = "left",  # left, right, top, bottom, front, back
                 slide_distance: float = 300,
                 rotation_during_slide: bool = True,
                 fade_during_slide: bool = True,
                 curved_path: bool = False):
        super().__init__(config)
        self.slide_direction = slide_direction
        self.slide_distance = slide_distance
        self.rotation_during_slide = rotation_during_slide
        self.fade_during_slide = fade_during_slide
        self.curved_path = curved_path
        
        # Calculate slide vectors for each letter
        self.slide_vectors = []
        for i in range(len(self.letters)):
            if slide_direction == "left":
                vector = np.array([-slide_distance, 0, 0])
            elif slide_direction == "right":
                vector = np.array([slide_distance, 0, 0])
            elif slide_direction == "top":
                vector = np.array([0, -slide_distance, 0])
            elif slide_direction == "bottom":
                vector = np.array([0, slide_distance, 0])
            elif slide_direction == "front":
                vector = np.array([0, 0, -slide_distance])
            elif slide_direction == "back":
                vector = np.array([0, 0, slide_distance])
            else:
                vector = np.array([-slide_distance, 0, 0])
            
            # Add variation for each letter
            variation = np.array([
                np.sin(i * 0.5) * 20,
                np.cos(i * 0.5) * 20,
                np.sin(i * 0.3) * 30
            ])
            self.slide_vectors.append(vector + variation)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters for 3D slide animation"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Calculate position along slide path
            if self.curved_path:
                # Curved trajectory (arc)
                t = 1 - letter_progress
                curve_height = np.sin(t * np.pi) * 50
                position_offset = self.slide_vectors[i] * t
                position_offset[1] -= curve_height  # Add arc to trajectory
                letter.position = self.original_positions[i] + position_offset
            else:
                # Linear slide
                letter.position = self.original_positions[i] + self.slide_vectors[i] * (1 - letter_progress)
            
            # Rotation during slide
            if self.rotation_during_slide:
                rotation_amount = (1 - letter_progress) * np.pi * 2
                letter.rotation = np.array([
                    rotation_amount * 0.2,
                    rotation_amount,
                    rotation_amount * 0.1
                ])
            
            # Fade during slide
            if self.fade_during_slide:
                letter.opacity = letter_progress
            else:
                letter.opacity = 1.0


class Float3D(Base3DTextAnimation):
    """3D floating animation where letters float and bob"""
    
    def __init__(self, config: Animation3DConfig,
                 float_height: float = 100,
                 float_pattern: str = "wave",  # wave, random, circular
                 bob_amplitude: float = 20,
                 bob_frequency: float = 2.0):
        super().__init__(config)
        self.float_height = float_height
        self.float_pattern = float_pattern
        self.bob_amplitude = bob_amplitude
        self.bob_frequency = bob_frequency
        
        # Generate phase offsets for each letter
        self.phase_offsets = [i * 0.3 for i in range(len(self.letters))]
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters for floating animation"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Base float up
            base_y = self.original_positions[i][1] - self.float_height * letter_progress
            
            # Add bobbing motion
            time = frame_number / fps
            
            if self.float_pattern == "wave":
                # Wave pattern
                bob_y = np.sin(time * self.bob_frequency + self.phase_offsets[i]) * self.bob_amplitude
                bob_x = np.cos(time * self.bob_frequency * 0.5 + self.phase_offsets[i]) * self.bob_amplitude * 0.5
                bob_z = np.sin(time * self.bob_frequency * 0.3 + self.phase_offsets[i]) * self.bob_amplitude * 0.3
            elif self.float_pattern == "circular":
                # Circular motion
                angle = time * self.bob_frequency + self.phase_offsets[i]
                bob_x = np.cos(angle) * self.bob_amplitude
                bob_y = np.sin(angle) * self.bob_amplitude
                bob_z = np.sin(angle * 0.5) * self.bob_amplitude * 0.5
            else:  # random
                # Perlin-noise-like random motion
                bob_x = np.sin(time * 1.3 + i) * self.bob_amplitude * 0.7
                bob_y = np.cos(time * 1.7 + i * 2) * self.bob_amplitude
                bob_z = np.sin(time * 0.9 + i * 3) * self.bob_amplitude * 0.5
            
            # Apply position
            letter.position = np.array([
                self.original_positions[i][0] + bob_x * letter_progress,
                base_y + bob_y * letter_progress,
                self.original_positions[i][2] + bob_z * letter_progress
            ])
            
            # Gentle rotation while floating
            letter.rotation = np.array([
                np.sin(time * 0.5 + self.phase_offsets[i]) * 0.1 * letter_progress,
                np.cos(time * 0.3 + self.phase_offsets[i]) * 0.1 * letter_progress,
                np.sin(time * 0.7 + self.phase_offsets[i]) * 0.2 * letter_progress
            ])
            
            # Fade in
            letter.opacity = letter_progress


class Bounce3D(Base3DTextAnimation):
    """3D bounce animation with physics simulation"""
    
    def __init__(self, config: Animation3DConfig,
                 bounce_height: float = 300,
                 bounce_count: int = 3,
                 gravity: float = 980,  # pixels/s^2
                 damping: float = 0.7,
                 spin_on_bounce: bool = True):
        super().__init__(config)
        self.bounce_height = bounce_height
        self.bounce_count = bounce_count
        self.gravity = gravity
        self.damping = damping
        self.spin_on_bounce = spin_on_bounce
        
        # Calculate initial velocities for each letter
        self.initial_velocities = []
        for i in range(len(self.letters)):
            # Vary initial velocity slightly for each letter
            velocity_variation = 1.0 + (i * 0.05)
            initial_vy = -np.sqrt(2 * gravity * bounce_height) * velocity_variation
            self.initial_velocities.append(initial_vy)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with physics-based bounce"""
        time = progress * self.config.duration_ms / 1000.0  # Convert to seconds
        
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            letter_time = letter_progress * self.config.duration_ms / 1000.0
            
            # Physics simulation
            y = self.original_positions[i][1]
            velocity = self.initial_velocities[i]
            current_y = y
            
            # Simple bounce simulation
            t = letter_time
            bounce_num = 0
            
            while bounce_num < self.bounce_count and t > 0:
                # Time to hit ground
                time_to_ground = (-velocity + np.sqrt(velocity**2 + 2*self.gravity*current_y)) / self.gravity
                
                if t < time_to_ground:
                    # Still falling/rising in this bounce
                    current_y = y + velocity * t + 0.5 * self.gravity * t**2
                    break
                else:
                    # Hit ground and bounce
                    t -= time_to_ground
                    velocity = -velocity * self.damping
                    bounce_num += 1
                    current_y = y
            
            letter.position[1] = current_y
            
            # Add horizontal movement for variety
            letter.position[0] = self.original_positions[i][0] + np.sin(i * 0.5) * 30 * letter_progress
            
            # Z-axis variation
            letter.position[2] = self.original_positions[i][2] + np.cos(i * 0.3) * 50 * letter_progress
            
            # Spin on bounce if enabled
            if self.spin_on_bounce:
                spin_amount = bounce_num * np.pi * 2
                letter.rotation = np.array([
                    spin_amount * 0.3,
                    spin_amount,
                    spin_amount * 0.1
                ])
            
            # Fade in
            letter.opacity = min(1.0, letter_progress * 2)


class Orbit3D(Base3DTextAnimation):
    """3D orbital motion animation"""
    
    def __init__(self, config: Animation3DConfig,
                 orbit_radius: float = 150,
                 orbit_speed: float = 2.0,
                 orbit_axis: str = "y",  # x, y, z, mixed
                 elliptical: bool = False):
        super().__init__(config)
        self.orbit_radius = orbit_radius
        self.orbit_speed = orbit_speed
        self.orbit_axis = orbit_axis
        self.elliptical = elliptical
        
        # Calculate orbit center
        self.orbit_center = np.array([
            config.position[0],
            config.position[1],
            config.position[2]
        ])
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters in orbital motion"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Calculate orbital angle
            angle = letter_progress * np.pi * 2 * self.orbit_speed + i * (2 * np.pi / len(self.letters))
            
            # Calculate radius (elliptical if enabled)
            if self.elliptical:
                radius_x = self.orbit_radius
                radius_y = self.orbit_radius * 0.6
                radius_z = self.orbit_radius * 0.8
            else:
                radius_x = radius_y = radius_z = self.orbit_radius
            
            # Calculate position based on orbit axis
            if self.orbit_axis == "y":
                # Orbit around Y axis
                x = self.orbit_center[0] + np.cos(angle) * radius_x * (1 - letter_progress) + \
                    self.original_positions[i][0] * letter_progress
                y = self.original_positions[i][1]
                z = self.orbit_center[2] + np.sin(angle) * radius_z * (1 - letter_progress) + \
                    self.original_positions[i][2] * letter_progress
            elif self.orbit_axis == "x":
                # Orbit around X axis
                x = self.original_positions[i][0]
                y = self.orbit_center[1] + np.cos(angle) * radius_y * (1 - letter_progress) + \
                    self.original_positions[i][1] * letter_progress
                z = self.orbit_center[2] + np.sin(angle) * radius_z * (1 - letter_progress) + \
                    self.original_positions[i][2] * letter_progress
            elif self.orbit_axis == "z":
                # Orbit around Z axis
                x = self.orbit_center[0] + np.cos(angle) * radius_x * (1 - letter_progress) + \
                    self.original_positions[i][0] * letter_progress
                y = self.orbit_center[1] + np.sin(angle) * radius_y * (1 - letter_progress) + \
                    self.original_positions[i][1] * letter_progress
                z = self.original_positions[i][2]
            else:  # mixed
                # Complex 3D orbit
                x = self.orbit_center[0] + np.cos(angle) * radius_x * np.sin(angle * 0.5) * \
                    (1 - letter_progress) + self.original_positions[i][0] * letter_progress
                y = self.orbit_center[1] + np.sin(angle * 0.7) * radius_y * \
                    (1 - letter_progress) + self.original_positions[i][1] * letter_progress
                z = self.orbit_center[2] + np.cos(angle * 0.3) * radius_z * \
                    (1 - letter_progress) + self.original_positions[i][2] * letter_progress
            
            letter.position = np.array([x, y, z])
            
            # Letters face the direction of motion
            letter.rotation[1] = angle
            
            # Fade in during orbit
            letter.opacity = letter_progress


class Swarm3D(Base3DTextAnimation):
    """3D swarm animation where letters move like a swarm/flock"""
    
    def __init__(self, config: Animation3DConfig,
                 swarm_radius: float = 200,
                 swarm_speed: float = 3.0,
                 coherence: float = 0.7,  # How much letters stick together
                 formation_time: float = 0.5):  # Time to form final text (0-1)
        super().__init__(config)
        self.swarm_radius = swarm_radius
        self.swarm_speed = swarm_speed
        self.coherence = coherence
        self.formation_time = formation_time
        
        # Generate random swarm starting positions
        self.swarm_positions = []
        for i in range(len(self.letters)):
            angle = np.random.random() * 2 * np.pi
            radius = np.random.random() * swarm_radius
            pos = np.array([
                np.cos(angle) * radius,
                np.sin(angle) * radius,
                (np.random.random() - 0.5) * 100
            ], dtype=np.float32)
            self.swarm_positions.append(pos)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with swarm behavior"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            if letter_progress < self.formation_time:
                # Swarming phase
                swarm_t = letter_progress / self.formation_time
                time = frame_number / fps
                
                # Flocking behavior simulation
                center_pull = np.array([0.0, 0.0, 0.0])
                for j, other_letter in enumerate(self.letters):
                    if i != j:
                        diff = self.swarm_positions[j] - self.swarm_positions[i]
                        distance = np.linalg.norm(diff)
                        if distance > 0:
                            # Cohesion force
                            center_pull += diff * self.coherence / distance
                
                # Add swirling motion
                angle = time * self.swarm_speed + i * 0.5
                swirl = np.array([
                    np.cos(angle) * 50,
                    np.sin(angle) * 50,
                    np.sin(angle * 0.5) * 30
                ])
                
                # Update swarm position
                self.swarm_positions[i] += (center_pull * 0.01 + swirl * 0.02)
                
                # Interpolate to swarm position
                letter.position = self.original_positions[i] + self.swarm_positions[i] * (1 - swarm_t)
                
                # Random rotation during swarm
                letter.rotation = np.array([
                    np.sin(time * 2 + i) * 0.5,
                    np.cos(time * 3 + i) * 0.5,
                    np.sin(time * 1.5 + i) * 0.3
                ])
            else:
                # Formation phase - settle into final positions
                form_t = (letter_progress - self.formation_time) / (1 - self.formation_time)
                
                # Smooth transition to final position
                current_swarm = self.swarm_positions[i] * (1 - form_t)
                letter.position = self.original_positions[i] + current_swarm
                
                # Reduce rotation
                letter.rotation *= (1 - form_t)
            
            # Fade in
            letter.opacity = min(1.0, letter_progress * 2)