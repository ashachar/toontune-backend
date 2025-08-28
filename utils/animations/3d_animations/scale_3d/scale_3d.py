"""
3D Scale and rotation text animations with individual letter control
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base_3d_text_animation import Base3DTextAnimation, Animation3DConfig, Letter3D


class Zoom3D(Base3DTextAnimation):
    """3D zoom animation with depth effects"""
    
    def __init__(self, config: Animation3DConfig,
                 start_scale: float = 0.1,
                 end_scale: float = 1.0,
                 zoom_from_z: bool = True,  # Zoom from depth
                 spiral_zoom: bool = False,
                 pulsate: bool = False):
        super().__init__(config)
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.zoom_from_z = zoom_from_z
        self.spiral_zoom = spiral_zoom
        self.pulsate = pulsate
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters for 3D zoom effect"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Base scale
            base_scale = self.start_scale + (self.end_scale - self.start_scale) * letter_progress
            
            # Add pulsation if enabled
            if self.pulsate:
                pulse = np.sin(letter_progress * np.pi * 2) * 0.1
                base_scale += pulse * letter_progress
            
            # Apply scale
            letter.scale = np.array([base_scale, base_scale, base_scale])
            
            # Z-depth zoom effect
            if self.zoom_from_z:
                # Letters start far in Z and come forward
                z_start = 500
                z_end = self.original_positions[i][2]
                letter.position[2] = z_start + (z_end - z_start) * letter_progress
            
            # Spiral zoom if enabled
            if self.spiral_zoom:
                angle = letter_progress * np.pi * 4
                radius = (1 - letter_progress) * 100
                letter.position[0] = self.original_positions[i][0] + np.cos(angle + i) * radius
                letter.position[1] = self.original_positions[i][1] + np.sin(angle + i) * radius
                
                # Rotate during spiral
                letter.rotation[2] = angle
            
            # Fade in during zoom
            letter.opacity = letter_progress


class Rotate3DAxis(Base3DTextAnimation):
    """Full 3D rotation around multiple axes"""
    
    def __init__(self, config: Animation3DConfig,
                 rotation_axis: str = "y",  # x, y, z, all
                 rotations: float = 1.0,  # Number of full rotations
                 wobble: bool = False,
                 individual_rotation: bool = True):
        super().__init__(config)
        self.rotation_axis = rotation_axis
        self.rotations = rotations
        self.wobble = wobble
        self.individual_rotation = individual_rotation
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with 3D rotation"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Base rotation
            angle = letter_progress * np.pi * 2 * self.rotations
            
            # Apply rotation based on axis
            if self.rotation_axis == "x":
                letter.rotation[0] = angle
            elif self.rotation_axis == "y":
                letter.rotation[1] = angle
            elif self.rotation_axis == "z":
                letter.rotation[2] = angle
            elif self.rotation_axis == "all":
                # Rotate around all axes
                letter.rotation[0] = angle * 0.5
                letter.rotation[1] = angle
                letter.rotation[2] = angle * 0.3
            
            # Add wobble effect
            if self.wobble:
                wobble_amount = np.sin(letter_progress * np.pi * 3) * 0.1
                letter.rotation[0] += wobble_amount
                letter.rotation[1] += wobble_amount * 0.5
            
            # Individual rotation offsets
            if self.individual_rotation:
                letter.rotation[0] += i * 0.1
                letter.rotation[1] += i * 0.15
                letter.rotation[2] += i * 0.05
            
            # Opacity based on rotation (visible from all angles)
            visibility = 0.5 + 0.5 * abs(np.cos(letter.rotation[1]))
            letter.opacity = visibility * letter_progress


class FlipCard3D(Base3DTextAnimation):
    """3D card flip animation for each letter"""
    
    def __init__(self, config: Animation3DConfig,
                 flip_axis: str = "y",  # x, y
                 flip_sequential: bool = True,
                 flip_direction: int = 1,  # 1 or -1
                 show_back: bool = True):
        super().__init__(config)
        self.flip_axis = flip_axis
        self.flip_sequential = flip_sequential
        self.flip_direction = flip_direction
        self.show_back = show_back
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters for card flip effect"""
        for i, letter in enumerate(self.letters):
            if self.flip_sequential:
                letter_progress = self.get_letter_progress(i, progress)
            else:
                letter_progress = progress
            
            # Calculate flip angle (0 to 180 degrees)
            flip_angle = letter_progress * np.pi * self.flip_direction
            
            # Apply flip rotation
            if self.flip_axis == "y":
                letter.rotation[1] = flip_angle
                # Scale X to create perspective effect
                perspective_scale = abs(np.cos(flip_angle))
                letter.scale[0] = perspective_scale
            elif self.flip_axis == "x":
                letter.rotation[0] = flip_angle
                # Scale Y to create perspective effect
                perspective_scale = abs(np.cos(flip_angle))
                letter.scale[1] = perspective_scale
            
            # Handle visibility and back side
            if self.show_back:
                # Show different color on back
                if abs(flip_angle) > np.pi / 2:
                    # Showing back side
                    letter.color = self.config.depth_color
                else:
                    # Showing front side
                    letter.color = self.config.font_color
            
            # Opacity handling for smooth flip
            letter.opacity = 1.0


class Tumble3D(Base3DTextAnimation):
    """3D tumbling animation with complex rotation"""
    
    def __init__(self, config: Animation3DConfig,
                 tumble_speed: float = 2.0,
                 tumble_chaos: float = 0.5,  # 0 = synchronized, 1 = chaotic
                 gravity_effect: bool = True):
        super().__init__(config)
        self.tumble_speed = tumble_speed
        self.tumble_chaos = tumble_chaos
        self.gravity_effect = gravity_effect
        
        # Generate random tumble parameters for each letter
        self.tumble_params = []
        for i in range(len(self.letters)):
            params = {
                'x_speed': (1 + np.random.random() * tumble_chaos) * tumble_speed,
                'y_speed': (1 + np.random.random() * tumble_chaos) * tumble_speed * 1.5,
                'z_speed': (1 + np.random.random() * tumble_chaos) * tumble_speed * 0.7,
                'phase': np.random.random() * np.pi * 2
            }
            self.tumble_params.append(params)
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters with tumbling motion"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            params = self.tumble_params[i]
            
            # Complex tumbling rotation
            letter.rotation[0] = (letter_progress * np.pi * 2 * params['x_speed'] + params['phase'])
            letter.rotation[1] = (letter_progress * np.pi * 2 * params['y_speed'] + params['phase'] * 0.5)
            letter.rotation[2] = (letter_progress * np.pi * 2 * params['z_speed'] + params['phase'] * 0.3)
            
            # Add gravity effect
            if self.gravity_effect:
                # Letters fall while tumbling
                gravity_offset = letter_progress * letter_progress * 200  # Accelerating fall
                letter.position[1] = self.original_positions[i][1] + gravity_offset
                
                # Slight horizontal drift
                letter.position[0] = self.original_positions[i][0] + \
                                   np.sin(params['phase']) * letter_progress * 30
            
            # Scale varies during tumble
            scale_variation = 1.0 + np.sin(letter_progress * np.pi * 2) * 0.2
            letter.scale = np.array([scale_variation, scale_variation, scale_variation])
            
            # Fade in/out
            if letter_progress < 0.2:
                letter.opacity = letter_progress * 5
            elif letter_progress > 0.8:
                letter.opacity = (1 - letter_progress) * 5
            else:
                letter.opacity = 1.0


class Unfold3D(Base3DTextAnimation):
    """3D unfolding animation like origami"""
    
    def __init__(self, config: Animation3DConfig,
                 unfold_origin: str = "center",  # center, left, right, top, bottom
                 unfold_layers: int = 3,
                 unfold_angle: float = np.pi):
        super().__init__(config)
        self.unfold_origin = unfold_origin
        self.unfold_layers = unfold_layers
        self.unfold_angle = unfold_angle
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters for unfolding effect"""
        total_letters = len(self.letters)
        
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            # Determine fold layer
            if self.unfold_origin == "center":
                distance_from_center = abs(i - total_letters / 2)
                layer = int(distance_from_center / (total_letters / 2) * self.unfold_layers)
            elif self.unfold_origin == "left":
                layer = int(i / total_letters * self.unfold_layers)
            elif self.unfold_origin == "right":
                layer = int((total_letters - i) / total_letters * self.unfold_layers)
            else:
                layer = 0
            
            # Calculate unfold progress for this layer
            layer_delay = layer / self.unfold_layers * 0.5
            layer_progress = max(0, min(1, (letter_progress - layer_delay) / (1 - layer_delay)))
            
            # Unfold rotation
            current_angle = self.unfold_angle * (1 - layer_progress)
            
            if self.unfold_origin in ["center", "left", "right"]:
                # Vertical unfold
                letter.rotation[1] = current_angle
                # Move letters apart as they unfold
                if self.unfold_origin == "center":
                    offset_direction = 1 if i > total_letters / 2 else -1
                else:
                    offset_direction = 1
                letter.position[0] = self.original_positions[i][0] + \
                                   offset_direction * (1 - layer_progress) * 50
            else:
                # Horizontal unfold
                letter.rotation[0] = current_angle
                letter.position[1] = self.original_positions[i][1] + \
                                   (1 - layer_progress) * 50
            
            # Scale increases as unfold progresses
            scale = 0.5 + 0.5 * layer_progress
            letter.scale = np.array([scale, scale, scale])
            
            # Fade in
            letter.opacity = layer_progress


class Explode3D(Base3DTextAnimation):
    """3D explosion effect where letters fly apart"""
    
    def __init__(self, config: Animation3DConfig,
                 explosion_force: float = 300,
                 explosion_center: str = "center",  # center, mouse, random
                 rotation_speed: float = 5.0,
                 scale_during_explosion: float = 1.5,
                 implode: bool = False):  # Reverse explosion
        super().__init__(config)
        self.explosion_force = explosion_force
        self.explosion_center = explosion_center
        self.rotation_speed = rotation_speed
        self.scale_during_explosion = scale_during_explosion
        self.implode = implode
        
        # Calculate explosion vectors
        self.explosion_vectors = []
        self.rotation_axes = []
        
        # Determine explosion center
        if explosion_center == "center":
            center = np.array([config.position[0], config.position[1], config.position[2]])
        else:  # random
            center = np.array([
                config.position[0] + np.random.randn() * 100,
                config.position[1] + np.random.randn() * 100,
                config.position[2]
            ])
        
        for i, letter in enumerate(self.letters):
            # Calculate explosion direction
            direction = self.original_positions[i] - center
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([np.random.randn(), np.random.randn(), np.random.randn()])
                direction = direction / np.linalg.norm(direction)
            
            # Add randomness
            direction += np.random.randn(3) * 0.3
            direction = direction / np.linalg.norm(direction)
            
            self.explosion_vectors.append(direction * explosion_force)
            
            # Random rotation axis
            axis = np.random.randn(3)
            self.rotation_axes.append(axis / np.linalg.norm(axis))
    
    def update_letters(self, progress: float, frame_number: int, fps: float):
        """Update letters for explosion effect"""
        for i, letter in enumerate(self.letters):
            letter_progress = self.get_letter_progress(i, progress)
            
            if self.implode:
                # Reverse explosion (implosion)
                explosion_progress = 1 - letter_progress
            else:
                explosion_progress = letter_progress
            
            # Position based on explosion
            explosion_offset = self.explosion_vectors[i] * explosion_progress
            letter.position = self.original_positions[i] + explosion_offset
            
            # Rotation during explosion
            rotation_amount = explosion_progress * self.rotation_speed
            letter.rotation = self.rotation_axes[i] * rotation_amount
            
            # Scale effect
            if explosion_progress < 0.5:
                # Expand during first half
                scale = 1.0 + (self.scale_during_explosion - 1.0) * (explosion_progress * 2)
            else:
                # Contract during second half
                scale = self.scale_during_explosion - (self.scale_during_explosion - 1.0) * ((explosion_progress - 0.5) * 2)
            
            letter.scale = np.array([scale, scale, scale])
            
            # Fade based on explosion
            if self.implode:
                letter.opacity = letter_progress
            else:
                letter.opacity = 1.0 - explosion_progress * 0.5  # Don't fully fade out