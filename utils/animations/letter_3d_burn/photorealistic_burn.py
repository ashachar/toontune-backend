"""
Photorealistic letter burn animation with real fire and smoke.
Production-quality implementation.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import math

from .fire_generator import FireGenerator, EmberGenerator
from .volumetric_smoke import VolumetricSmoke, SmokeTrail
from .timing import BurnTiming

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LetterBurnState:
    """Complete state for photorealistic letter burning."""
    letter: str
    position: Tuple[int, int]
    original_sprite: np.ndarray
    
    # Burning state
    heat_map: np.ndarray  # Temperature distribution
    char_map: np.ndarray  # How charred each pixel is
    burn_mask: np.ndarray  # What's left of the letter
    
    # Current phase
    phase: str
    progress: float
    
    # Fire components
    is_burning: bool
    flame_intensity: float
    burn_edge_points: List[Tuple[int, int]]
    
    # Timing
    ignition_time: Optional[float]
    
    
class PhotorealisticLetterBurn:
    """
    Production-quality letter burn animation with photorealistic fire and smoke.
    Letters actually catch fire with realistic flames and volumetric smoke.
    """
    
    def __init__(
        self,
        duration: float = 4.0,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        text: str = "FIRE",
        font_size: int = 200,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        
        # Fire properties
        flame_height: int = 150,
        flame_intensity: float = 1.0,
        flame_spread_speed: float = 2.0,
        
        # Timing
        stable_duration: float = 0.3,
        ignite_duration: float = 0.5,
        burn_duration: float = 2.0,
        burn_stagger: float = 0.3,
        
        # Physics
        heat_propagation: float = 0.8,
        char_threshold: float = 0.6,
        
        # Quality
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
        
        # Fire properties
        self.flame_height = flame_height
        self.flame_intensity = flame_intensity
        self.flame_spread_speed = flame_spread_speed
        
        # Physics
        self.heat_propagation = heat_propagation
        self.char_threshold = char_threshold
        
        # Quality
        self.supersample_factor = max(1, supersample_factor)
        self.font_path = font_path
        self.debug = debug
        
        # Timing
        self.timing = BurnTiming(
            stable_duration=stable_duration,
            ignite_duration=ignite_duration,
            burn_duration=burn_duration,
            burn_stagger=burn_stagger
        )
        
        # Initialize systems
        self.fire_gen = FireGenerator(
            width=self.width,
            height=self.height,
            turbulence_scale=0.02
        )
        
        self.smoke_system = VolumetricSmoke(
            resolution=resolution,
            smoke_color=(65, 65, 70),
            turbulence_strength=3.0
        )
        
        self.ember_gen = EmberGenerator(max_embers=100)
        self.smoke_trail = SmokeTrail(self.smoke_system)
        
        # Letter states
        self.letter_states: Dict[int, LetterBurnState] = {}
        self.letter_positions: List[Tuple[int, int]] = []
        
        # Initialize letters
        self._initialize_letters()
        
    def _initialize_letters(self):
        """Initialize letter sprites and burn states."""
        # Font setup
        if self.font_path:
            # Would load custom font here
            font = cv2.FONT_HERSHEY_SIMPLEX
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate text layout
        scale = self.font_size / 30
        thickness = max(2, int(self.font_size / 30))
        
        # Get total text size
        text_size = cv2.getTextSize(self.text, font, scale, thickness)[0]
        
        # Starting position
        x_start = self.width // 2 - text_size[0] // 2
        y_base = self.height // 2
        
        current_x = x_start
        
        for i, letter in enumerate(self.text):
            if letter == ' ':
                current_x += int(self.font_size * 0.3)
                continue
            
            # Get letter dimensions
            letter_size = cv2.getTextSize(letter, font, scale, thickness)[0]
            
            # Create high-res sprite
            sprite_size = max(letter_size[0], letter_size[1]) * 2
            sprite_size = int(sprite_size * self.supersample_factor)
            
            # Create letter sprite
            sprite = np.zeros((sprite_size, sprite_size, 4), dtype=np.uint8)
            
            # Draw letter (white for now, will apply material properties)
            letter_scale = scale * self.supersample_factor
            letter_thickness = thickness * self.supersample_factor
            
            # Center letter in sprite
            text_size_scaled = cv2.getTextSize(letter, font, letter_scale, letter_thickness)[0]
            x_offset = (sprite_size - text_size_scaled[0]) // 2
            y_offset = (sprite_size + text_size_scaled[1]) // 2
            
            cv2.putText(
                sprite,
                letter,
                (x_offset, y_offset),
                font,
                letter_scale,
                (255, 255, 255, 255),
                letter_thickness,
                cv2.LINE_AA
            )
            
            # Downsample if needed
            if self.supersample_factor > 1:
                new_size = sprite_size // self.supersample_factor
                sprite = cv2.resize(sprite, (new_size, new_size), interpolation=cv2.INTER_AREA)
                sprite_size = new_size
            
            # Apply text color
            mask = sprite[:, :, 3] > 0
            sprite[mask, :3] = self.text_color
            
            # Initialize heat map (room temperature)
            heat_map = np.ones((sprite_size, sprite_size), dtype=np.float32) * 20.0
            
            # Initialize char map (0 = unburned, 1 = fully charred)
            char_map = np.zeros((sprite_size, sprite_size), dtype=np.float32)
            
            # Burn mask (what's left of the letter)
            burn_mask = (sprite[:, :, 3] > 0).astype(np.uint8) * 255
            
            # Store state
            self.letter_states[i] = LetterBurnState(
                letter=letter,
                position=(current_x, y_base),
                original_sprite=sprite.copy(),
                heat_map=heat_map,
                char_map=char_map,
                burn_mask=burn_mask,
                phase='waiting',
                progress=0.0,
                is_burning=False,
                flame_intensity=0.0,
                burn_edge_points=[],
                ignition_time=None
            )
            
            self.letter_positions.append((current_x, y_base))
            current_x += letter_size[0] + int(self.font_size * 0.1)
    
    def _propagate_heat(self, state: LetterBurnState, dt: float):
        """Propagate heat through the letter material."""
        # Heat diffusion
        kernel = np.array([[0.05, 0.1, 0.05],
                          [0.1,  0.4, 0.1],
                          [0.05, 0.1, 0.05]], dtype=np.float32)
        
        # Only propagate where there's material
        material_mask = (state.burn_mask > 0).astype(np.float32)
        masked_heat = state.heat_map * material_mask
        
        # Apply heat diffusion
        new_heat = cv2.filter2D(masked_heat, -1, kernel)
        
        # Heat propagation with material conductivity
        state.heat_map = state.heat_map * (1 - self.heat_propagation * dt) + \
                        new_heat * self.heat_propagation * dt
        
        # Cooling (heat loss to environment)
        ambient_temp = 20.0
        cooling_rate = 0.95
        state.heat_map = ambient_temp + (state.heat_map - ambient_temp) * cooling_rate
        
    def _update_burning(self, state: LetterBurnState, frame: int, dt: float):
        """Update the burning process for a letter."""
        
        if state.phase == 'ignite':
            # Start fire at bottom of letter
            h, w = state.burn_mask.shape
            
            # Find bottom edge pixels
            bottom_edge = []
            for x in range(w):
                for y in range(h - 1, h // 2, -1):  # Search from bottom up
                    if state.burn_mask[y, x] > 0:
                        bottom_edge.append((x, y))
                        break
            
            # Heat up bottom edge
            for x, y in bottom_edge:
                state.heat_map[y, x] = 500 + np.random.random() * 200  # Ignition temperature
            
            state.is_burning = True
            state.flame_intensity = state.progress
            
        elif state.phase == 'burn':
            # Propagate heat
            self._propagate_heat(state, dt)
            
            # Update charring based on temperature
            high_temp_mask = state.heat_map > 300
            state.char_map[high_temp_mask] += dt * 0.5
            state.char_map = np.clip(state.char_map, 0, 1)
            
            # Burn away highly charred areas
            burned_away = state.char_map > self.char_threshold
            state.burn_mask[burned_away] = 0
            
            # Find burning edge for particle emission
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges = cv2.morphologyEx(state.burn_mask, cv2.MORPH_GRADIENT, kernel)
            edge_points = np.column_stack(np.where(edges > 0))
            
            if len(edge_points) > 0:
                # Convert to absolute coordinates
                state.burn_edge_points = []
                for py, px in edge_points[::5]:  # Sample every 5th point
                    abs_x = state.position[0] - state.original_sprite.shape[1] // 2 + px
                    abs_y = state.position[1] - state.original_sprite.shape[0] // 2 + py
                    state.burn_edge_points.append((abs_x, abs_y))
                    
                    # Heat these edges
                    state.heat_map[py, px] = 800 + np.random.random() * 200
            
            # Update flame intensity
            state.flame_intensity = 1.0 - (np.sum(state.burn_mask > 0) / 
                                         np.sum(state.original_sprite[:, :, 3] > 0))
            
        elif state.phase == 'smoke':
            # Just smoke, no more fire
            state.is_burning = False
            state.flame_intensity = 0.0
            
            # Cool down
            state.heat_map *= 0.9
            
    def _render_burning_letter(
        self,
        state: LetterBurnState,
        frame: int
    ) -> np.ndarray:
        """Render a single burning letter with all effects."""
        h, w = state.original_sprite.shape[:2]
        rendered = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Start with original sprite
        sprite = state.original_sprite.copy()
        
        # Apply charring (darkening)
        for c in range(3):
            char_effect = 1.0 - state.char_map * 0.8  # Darkens to 20% of original
            sprite[:, :, c] = (sprite[:, :, c] * char_effect).astype(np.uint8)
        
        # Apply burn mask (remove burned parts)
        sprite[:, :, 3] = (sprite[:, :, 3] * (state.burn_mask / 255.0)).astype(np.uint8)
        
        # Add glowing edges where actively burning
        if state.is_burning and len(state.burn_edge_points) > 0:
            # Create glow map from heat
            glow_intensity = np.clip((state.heat_map - 400) / 400, 0, 1)
            
            # Orange glow color
            glow_color = np.array([0, 100, 255], dtype=np.float32)  # BGR
            
            for c in range(3):
                glow_contribution = glow_intensity * glow_color[c]
                sprite[:, :, c] = np.clip(
                    sprite[:, :, c].astype(np.float32) + glow_contribution,
                    0, 255
                ).astype(np.uint8)
        
        return sprite
    
    def generate_frame(self, frame_num: int, background: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a frame of the photorealistic burn animation."""
        
        if background is None:
            # Dark background for fire to stand out
            output = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            output[:, :] = [30, 25, 20]  # Very dark brown
        else:
            output = background.copy()
        
        # Update smoke system
        self.smoke_system.update(1/self.fps)
        self.ember_gen.update(1/self.fps)
        
        # First pass: Update all letter states
        for i, state in self.letter_states.items():
            # Get current phase
            phase, progress = self.timing.get_letter_phase(
                frame_num, i, len(self.letter_states), self.fps
            )
            
            state.phase = phase
            state.progress = progress
            
            # Update burning process
            if phase in ['ignite', 'burn', 'smoke']:
                self._update_burning(state, frame_num, 1/self.fps)
                
                # Emit smoke from burning edges
                if state.is_burning and len(state.burn_edge_points) > 0:
                    for edge_x, edge_y in state.burn_edge_points[::3]:  # Sample points
                        self.smoke_system.emit_smoke(
                            edge_x, edge_y - 20,  # Smoke starts above flames
                            intensity=0.6,
                            temperature=200,
                            initial_radius=25
                        )
                        
                        # Emit embers occasionally
                        if np.random.random() < 0.1:
                            self.ember_gen.emit(edge_x, edge_y, count=2)
        
        # Render smoke first (behind everything)
        output = self.smoke_system.render(output)
        
        # Second pass: Render letters
        for i, state in self.letter_states.items():
            if state.phase == 'waiting' or state.phase == 'gone':
                if state.phase == 'waiting':
                    # Render original letter
                    sprite = state.original_sprite
                else:
                    continue  # Don't render if gone
            else:
                # Render burning letter
                sprite = self._render_burning_letter(state, frame_num)
            
            # Place sprite on canvas
            x, y = state.position
            h, w = sprite.shape[:2]
            
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
                sprite_region = sprite[sy1:sy2, sx1:sx2]
                alpha = sprite_region[:, :, 3:4] / 255.0
                
                output[y1:y2, x1:x2] = (
                    output[y1:y2, x1:x2] * (1 - alpha) +
                    sprite_region[:, :, :3] * alpha
                ).astype(np.uint8)
        
        # Third pass: Render flames on top
        all_flame_positions = []
        for state in self.letter_states.values():
            if state.is_burning and len(state.burn_edge_points) > 0:
                # Add flame positions
                for edge_x, edge_y in state.burn_edge_points[::2]:  # Sample
                    all_flame_positions.append((edge_x, edge_y))
        
        if all_flame_positions:
            # Generate and render flames
            flames = self.fire_gen.generate_flames(
                self.width, self.height,
                np.array(all_flame_positions),
                frame_num,
                intensity=self.flame_intensity
            )
            
            # Additive blending for fire
            flame_alpha = flames[:, :, 3:4] / 255.0
            for c in range(3):
                output[:, :, c] = np.clip(
                    output[:, :, c].astype(np.float32) + 
                    flames[:, :, c].astype(np.float32) * flame_alpha[:, :, 0],
                    0, 255
                ).astype(np.uint8)
        
        # Render embers on top
        output = self.ember_gen.render(output)
        
        if self.debug and frame_num % 10 == 0:
            num_burning = sum(1 for s in self.letter_states.values() if s.is_burning)
            logger.info(f"Frame {frame_num}: {num_burning} letters burning, "
                       f"{len(self.smoke_system.volumes)} smoke volumes, "
                       f"{len(self.ember_gen.embers)} embers")
        
        return output