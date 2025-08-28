"""
3D letter burn animation with smoke effects.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .timing import BurnTiming
from .particles import SmokeParticles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BurnState:
    """State information for burning letter."""
    letter: str
    position: Tuple[int, int]
    sprite: np.ndarray
    mask: np.ndarray  # Tracks which parts have burned away
    phase: str
    progress: float
    particles: SmokeParticles


class Letter3DBurn:
    """
    3D letter burn animation with smoke/gas effects.
    Letters burn from edges inward, turning into smoke.
    """
    
    def __init__(
        self,
        duration: float = 2.0,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        text: str = "BURN",
        font_size: int = 100,
        text_color: Tuple[int, int, int] = (255, 220, 0),
        burn_color: Tuple[int, int, int] = (255, 50, 0),
        depth_layers: int = 5,
        depth_offset: int = 2,
        initial_scale: float = 1.0,
        initial_position: Optional[Tuple[int, int]] = None,
        stable_duration: float = 0.1,
        stable_alpha: float = 1.0,
        burn_duration: float = 0.8,
        burn_stagger: float = 0.1,
        smoke_rise_distance: int = 100,
        reverse_order: bool = False,
        random_order: bool = False,
        segment_mask: Optional[np.ndarray] = None,
        is_behind: bool = True,
        shadow_offset: int = 5,
        outline_width: int = 2,
        supersample_factor: int = 2,
        font_path: Optional[str] = None,
        debug: bool = False
    ):
        self.duration = duration
        self.fps = fps
        self.total_frames = int(duration * fps)
        self.resolution = resolution
        self.width, self.height = resolution
        
        # Text properties
        self.text = text
        self.font_size = font_size
        self.text_color = text_color
        self.burn_color = burn_color
        self.depth_layers = depth_layers
        self.depth_offset = depth_offset
        
        # Animation properties
        self.initial_scale = initial_scale
        self.initial_position = initial_position or (self.width // 2, self.height // 2)
        self.smoke_rise_distance = smoke_rise_distance
        
        # Timing
        self.timing = BurnTiming(
            stable_duration=stable_duration,
            burn_duration=burn_duration,
            burn_stagger=burn_stagger,
            reverse_order=reverse_order,
            random_order=random_order
        )
        
        # Rendering properties
        self.stable_alpha = stable_alpha
        self.segment_mask = segment_mask
        self.is_behind = is_behind
        self.shadow_offset = shadow_offset
        self.outline_width = outline_width
        self.supersample_factor = max(1, supersample_factor)
        self.font_path = font_path
        self.debug = debug
        
        # Initialize states
        self.burn_states: Dict[int, BurnState] = {}
        self.letter_positions: List[Tuple[int, int]] = []
        self._initialize_letters()
        
    def _initialize_letters(self):
        """Initialize letter sprites and positions."""
        # Font setup
        if self.font_path:
            font = cv2.FONT_HERSHEY_SIMPLEX  # Fallback for now
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate text dimensions
        text_size = cv2.getTextSize(self.text, font, self.font_size/30, 2)[0]
        
        # Calculate letter positions
        x_start = self.initial_position[0] - text_size[0] // 2
        y_base = self.initial_position[1]
        
        current_x = x_start
        
        for i, letter in enumerate(self.text):
            if letter == ' ':
                current_x += int(self.font_size * 0.3)
                continue
            
            # Get letter size
            letter_size = cv2.getTextSize(letter, font, self.font_size/30, 2)[0]
            
            # Create letter sprite with supersampling
            sprite_size = int(self.font_size * 2 * self.supersample_factor)
            sprite = np.zeros((sprite_size, sprite_size, 4), dtype=np.uint8)
            
            # Draw letter at high resolution
            letter_scale = self.font_size/30 * self.supersample_factor
            thickness = max(1, int(2 * self.supersample_factor))
            
            # Center letter in sprite
            letter_width = letter_size[0] * self.supersample_factor
            letter_height = letter_size[1] * self.supersample_factor
            x_offset = (sprite_size - letter_width) // 2
            y_offset = (sprite_size + letter_height) // 2
            
            # Draw 3D layers
            for layer in range(self.depth_layers):
                depth_shift = (self.depth_layers - layer - 1) * self.depth_offset
                layer_color = tuple(int(c * 0.7) for c in self.text_color) if layer > 0 else self.text_color
                
                cv2.putText(
                    sprite,
                    letter,
                    (x_offset + depth_shift, y_offset + depth_shift),
                    font,
                    letter_scale,
                    (*layer_color, 255),
                    thickness,
                    cv2.LINE_AA
                )
            
            # Downsample if needed
            if self.supersample_factor > 1:
                sprite = cv2.resize(
                    sprite,
                    (sprite_size // self.supersample_factor, sprite_size // self.supersample_factor),
                    interpolation=cv2.INTER_AREA
                )
            
            # Create burn mask (all pixels initially unburned)
            mask = np.ones(sprite.shape[:2], dtype=np.uint8) * 255
            
            # Initialize particle system for this letter
            particles = SmokeParticles(
                max_particles=50,
                smoke_color=(100, 100, 100),
                fire_color=(255, 100, 0),
                rise_speed=2.0
            )
            
            # Store letter state
            self.burn_states[i] = BurnState(
                letter=letter,
                position=(current_x, y_base),
                sprite=sprite,
                mask=mask,
                phase='waiting',
                progress=0.0,
                particles=particles
            )
            
            self.letter_positions.append((current_x, y_base))
            current_x += letter_size[0] + int(self.font_size * 0.05)
    
    def _get_burn_edge(self, mask: np.ndarray, progress: float) -> np.ndarray:
        """
        Get the burning edge pixels for particle emission.
        
        Returns array of (x, y) points along the burning edge.
        """
        if progress <= 0 or progress >= 1:
            return np.array([])
        
        # Create erosion kernel that grows with progress
        kernel_size = max(3, int(progress * 15))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Erode mask to simulate burning from edges
        eroded = cv2.erode(mask, kernel, iterations=1)
        
        # Find edge pixels (difference between current and eroded)
        edge = cv2.absdiff(mask, eroded)
        
        # Get coordinates of edge pixels
        edge_points = np.column_stack(np.where(edge > 128))
        
        # Swap to (x, y) format
        if len(edge_points) > 0:
            edge_points = edge_points[:, [1, 0]]
        
        return edge_points
    
    def _apply_burn_effect(
        self,
        sprite: np.ndarray,
        mask: np.ndarray,
        phase: str,
        progress: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply burn effect to sprite based on phase and progress.
        
        Returns:
            Tuple of (modified_sprite, updated_mask)
        """
        output = sprite.copy()
        new_mask = mask.copy()
        
        if phase == 'ignite':
            # Starting to burn - edges glow orange/red
            kernel_size = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
            
            # Color the edges with fire colors
            fire_color = np.array([0, 50, 255, 255])  # Orange-red in BGR
            edge_mask = (edge > 128).astype(float) * progress
            
            for c in range(3):
                output[:, :, c] = (
                    output[:, :, c] * (1 - edge_mask) +
                    fire_color[c] * edge_mask
                ).astype(np.uint8)
        
        elif phase == 'burn':
            # Main burning - erode from edges inward
            kernel_size = max(3, int(progress * 20))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Erode the mask
            new_mask = cv2.erode(mask, kernel, iterations=1)
            
            # Create burning edge effect
            edge = cv2.absdiff(mask, new_mask)
            edge_mask = (edge > 128).astype(float)
            
            # Apply charred/glowing effect to edges
            glow_intensity = 1.0 - progress * 0.5
            fire_color = np.array([0, 50 * glow_intensity, 255 * glow_intensity, 255])
            
            for c in range(3):
                output[:, :, c] = np.where(
                    edge_mask > 0.5,
                    fire_color[c],
                    output[:, :, c]
                )
            
            # Apply mask to make burned parts transparent
            output[:, :, 3] = (output[:, :, 3] * (new_mask / 255.0)).astype(np.uint8)
        
        elif phase == 'smoke':
            # Almost gone - very faint and mostly transparent
            fade = 1.0 - progress
            output[:, :, 3] = (output[:, :, 3] * fade * 0.3).astype(np.uint8)
            
            # Darken remaining pixels (charred)
            for c in range(3):
                output[:, :, c] = (output[:, :, c] * 0.3).astype(np.uint8)
        
        return output, new_mask
    
    def generate_frame(self, frame_num: int, background: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a frame of the burn animation."""
        if background is None:
            output = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            output = background.copy()
        
        # Process each letter
        for i, state in self.burn_states.items():
            # Get current phase and progress
            phase, progress = self.timing.get_letter_phase(
                frame_num, i, len(self.burn_states), self.fps
            )
            
            state.phase = phase
            state.progress = progress
            
            if phase == 'waiting' or phase == 'gone':
                if phase == 'gone':
                    # Update and render any remaining particles
                    state.particles.update(1/self.fps)
                    output = state.particles.render(output)
                continue
            
            # Apply burn effect
            burned_sprite, new_mask = self._apply_burn_effect(
                state.sprite, state.mask, phase, progress
            )
            
            # Update mask for next frame
            if phase == 'burn':
                state.mask = new_mask
                
                # Emit particles from burning edges
                edge_points = self._get_burn_edge(state.mask, progress)
                if len(edge_points) > 0:
                    # Convert to screen coordinates
                    edge_points[:, 0] += state.position[0] - state.sprite.shape[1] // 2
                    edge_points[:, 1] += state.position[1] - state.sprite.shape[0] // 2
                    
                    # Emit fire and smoke particles
                    state.particles.emit_from_edge(edge_points, num_particles=3, is_fire=True)
                    state.particles.emit_from_edge(edge_points, num_particles=2, is_fire=False)
            
            # Update particles
            state.particles.update(1/self.fps)
            
            # Apply occlusion if needed
            if self.is_behind and self.segment_mask is not None:
                # Extract foreground mask for current frame
                # (This would need integration with segmentation)
                pass
            
            # Render letter
            x, y = state.position
            h, w = burned_sprite.shape[:2]
            
            # Calculate bounds
            x1 = max(0, x - w // 2)
            y1 = max(0, y - h // 2)
            x2 = min(self.width, x1 + w)
            y2 = min(self.height, y1 + h)
            
            # Sprite bounds
            sx1 = x1 - (x - w // 2)
            sy1 = y1 - (y - h // 2)
            sx2 = sx1 + (x2 - x1)
            sy2 = sy1 + (y2 - y1)
            
            if x2 > x1 and y2 > y1:
                # Alpha blending
                sprite_region = burned_sprite[sy1:sy2, sx1:sx2]
                alpha = sprite_region[:, :, 3:4] / 255.0
                
                output[y1:y2, x1:x2] = (
                    output[y1:y2, x1:x2] * (1 - alpha) +
                    sprite_region[:, :, :3] * alpha
                ).astype(np.uint8)
            
            # Render particles
            output = state.particles.render(output, blend_mode='additive' if phase == 'burn' else 'alpha')
        
        if self.debug and frame_num % 10 == 0:
            logger.info(f"Frame {frame_num}: Active particles: "
                       f"{sum(s.particles.get_particle_count() for s in self.burn_states.values())}")
        
        return output
    
    def set_initial_state(self, **kwargs):
        """Set initial state from handoff."""
        # Similar to dissolve, allows continuation from previous animation
        pass