"""
Photorealistic fire generation using Perlin noise and advanced rendering.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import math


class PerlinNoise:
    """Perlin noise generator for realistic fire movement."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.permutation = np.arange(256, dtype=int)
        np.random.shuffle(self.permutation)
        self.permutation = np.tile(self.permutation, 2)
        
    def fade(self, t):
        """Fade function for smooth interpolation."""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(self, t, a, b):
        """Linear interpolation."""
        return a + t * (b - a)
    
    def grad(self, hash_val, x, y):
        """Gradient function."""
        h = hash_val & 3
        u = x if h < 2 else y
        v = y if h < 2 else x
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
    
    def noise(self, x: float, y: float) -> float:
        """Generate 2D Perlin noise."""
        X = int(np.floor(x)) & 255
        Y = int(np.floor(y)) & 255
        
        x -= np.floor(x)
        y -= np.floor(y)
        
        u = self.fade(x)
        v = self.fade(y)
        
        A = self.permutation[X] + Y
        B = self.permutation[X + 1] + Y
        
        return self.lerp(v, 
            self.lerp(u, self.grad(self.permutation[A], x, y),
                        self.grad(self.permutation[B], x - 1, y)),
            self.lerp(u, self.grad(self.permutation[A + 1], x, y - 1),
                        self.grad(self.permutation[B + 1], x - 1, y - 1)))
    
    def turbulence(self, x: float, y: float, octaves: int = 4) -> float:
        """Generate turbulent noise with multiple octaves."""
        value = 0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0
        
        for _ in range(octaves):
            value += amplitude * abs(self.noise(x * frequency, y * frequency))
            max_value += amplitude
            amplitude *= 0.5
            frequency *= 2
            
        return value / max_value


class FireGenerator:
    """Generate photorealistic fire effects."""
    
    def __init__(
        self,
        width: int = 200,
        height: int = 300,
        base_temperature: float = 1000.0,
        cooling_rate: float = 0.95,
        turbulence_scale: float = 0.02,
        wind_strength: float = 0.1
    ):
        self.width = width
        self.height = height
        self.base_temperature = base_temperature
        self.cooling_rate = cooling_rate
        self.turbulence_scale = turbulence_scale
        self.wind_strength = wind_strength
        
        self.noise_gen = PerlinNoise()
        self.time_offset = 0
        
        # Temperature field
        self.temperature_field = np.zeros((height, width), dtype=np.float32)
        
        # Color gradient for fire (temperature to color mapping)
        self.color_map = self._create_fire_gradient()
        
    def _create_fire_gradient(self) -> np.ndarray:
        """Create realistic fire color gradient based on blackbody radiation."""
        gradient_size = 256
        gradient = np.zeros((gradient_size, 3), dtype=np.uint8)
        
        for i in range(gradient_size):
            t = i / (gradient_size - 1)
            
            if t < 0.2:  # Dark/transparent
                gradient[i] = [0, 0, 0]
            elif t < 0.4:  # Dark red (cooler)
                factor = (t - 0.2) / 0.2
                gradient[i] = [int(139 * factor), 0, 0]
            elif t < 0.6:  # Red to orange
                factor = (t - 0.4) / 0.2
                gradient[i] = [139 + int(116 * factor), int(69 * factor), 0]
            elif t < 0.8:  # Orange to yellow
                factor = (t - 0.6) / 0.2
                gradient[i] = [255, 69 + int(186 * factor), int(165 * factor)]
            else:  # Yellow to white (hottest)
                factor = (t - 0.8) / 0.2
                gradient[i] = [255, 255, 165 + int(90 * factor)]
                
        return gradient
    
    def generate_fire_mask(
        self,
        letter_mask: np.ndarray,
        frame: int,
        intensity: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fire effect for a letter mask.
        
        Returns:
            Tuple of (fire_color_image, fire_alpha_mask)
        """
        h, w = letter_mask.shape
        
        # Create heat source from letter edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(letter_mask, cv2.MORPH_GRADIENT, kernel)
        
        # Initialize temperature field from edges
        heat_source = edges.astype(np.float32) / 255.0 * self.base_temperature * intensity
        
        # Create fire shape using noise
        fire_shape = np.zeros((h, w), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                if heat_source[y, x] > 0:
                    # Sample noise for flame shape
                    noise_x = x * self.turbulence_scale
                    noise_y = y * self.turbulence_scale
                    noise_t = frame * 0.05  # Time evolution
                    
                    # Multiple octaves for realistic turbulence
                    turb = self.noise_gen.turbulence(
                        noise_x + noise_t,
                        noise_y - noise_t * 2,  # Fire rises
                        octaves=4
                    )
                    
                    # Apply wind
                    wind_offset = self.wind_strength * math.sin(noise_t * 2)
                    
                    # Create flame height variation
                    flame_height = 1.0 - (y / h)
                    flame_intensity = heat_source[y, x] * turb * flame_height
                    
                    fire_shape[y, x] = flame_intensity
        
        # Vertical flame propagation (fire rises)
        for y in range(h - 2, -1, -1):
            propagation = fire_shape[y + 1, :] * 0.85  # Heat rises with cooling
            fire_shape[y, :] = np.maximum(fire_shape[y, :], propagation)
        
        # Apply gaussian blur for smooth flames
        fire_shape = cv2.GaussianBlur(fire_shape, (9, 9), 0)
        
        # Normalize and threshold
        if fire_shape.max() > 0:
            fire_shape = fire_shape / fire_shape.max()
        
        # Convert temperature to color
        fire_color = np.zeros((h, w, 3), dtype=np.uint8)
        fire_indices = (fire_shape * 255).astype(np.uint8)
        
        for c in range(3):
            fire_color[:, :, c] = cv2.LUT(fire_indices, self.color_map[:, c])
        
        # Create alpha mask (transparency)
        fire_alpha = (fire_shape * 255).astype(np.uint8)
        fire_alpha = cv2.GaussianBlur(fire_alpha, (5, 5), 0)
        
        return fire_color, fire_alpha
    
    def generate_flames(
        self,
        width: int,
        height: int,
        base_positions: np.ndarray,
        frame: int,
        intensity: float = 1.0
    ) -> np.ndarray:
        """
        Generate standalone flames at specified positions.
        
        Args:
            width: Canvas width
            height: Canvas height
            base_positions: Array of (x, y) positions where flames start
            frame: Current frame number
            intensity: Fire intensity (0-1)
            
        Returns:
            RGBA flame image
        """
        flames = np.zeros((height, width, 4), dtype=np.uint8)
        
        for pos_x, pos_y in base_positions:
            # Generate individual flame
            flame_width = 50
            flame_height = 120
            
            # Create flame shape using noise
            for y in range(max(0, pos_y - flame_height), min(height, pos_y)):
                for x in range(max(0, pos_x - flame_width // 2), 
                             min(width, pos_x + flame_width // 2)):
                    
                    # Distance from flame center
                    dx = (x - pos_x) / (flame_width / 2)
                    dy = (pos_y - y) / flame_height
                    
                    if abs(dx) > 1 or dy < 0 or dy > 1:
                        continue
                    
                    # Flame shape (narrower at top)
                    flame_width_at_y = 1.0 - dy * 0.8
                    if abs(dx) > flame_width_at_y:
                        continue
                    
                    # Sample noise
                    noise_val = self.noise_gen.turbulence(
                        x * 0.05 + frame * 0.1,
                        y * 0.05 - frame * 0.2,
                        octaves=3
                    )
                    
                    # Calculate intensity
                    center_factor = 1.0 - abs(dx) / flame_width_at_y
                    height_factor = dy
                    flame_intensity = center_factor * height_factor * noise_val * intensity
                    
                    if flame_intensity > 0.1:
                        # Map to color
                        color_idx = int(flame_intensity * 255)
                        color_idx = min(255, color_idx)
                        
                        # Blend with existing
                        alpha = flame_intensity
                        flames[y, x, :3] = (
                            flames[y, x, :3] * (1 - alpha) +
                            self.color_map[color_idx] * alpha
                        ).astype(np.uint8)
                        flames[y, x, 3] = min(255, flames[y, x, 3] + int(alpha * 255))
        
        return flames


class EmberGenerator:
    """Generate glowing embers and sparks."""
    
    def __init__(self, max_embers: int = 50):
        self.max_embers = max_embers
        self.embers = []
        
    def emit(self, x: float, y: float, count: int = 5):
        """Emit new embers from position."""
        for _ in range(min(count, self.max_embers - len(self.embers))):
            ember = {
                'x': x + np.random.randn() * 5,
                'y': y + np.random.randn() * 5,
                'vx': np.random.randn() * 2,
                'vy': -np.random.uniform(1, 4),  # Upward velocity
                'life': 1.0,
                'size': np.random.uniform(1, 3),
                'temperature': np.random.uniform(0.7, 1.0)
            }
            self.embers.append(ember)
    
    def update(self, dt: float = 1/30):
        """Update ember positions and properties."""
        embers_to_keep = []
        
        for ember in self.embers:
            # Physics update
            ember['x'] += ember['vx'] * dt * 30
            ember['y'] += ember['vy'] * dt * 30
            
            # Add turbulence
            ember['vx'] += np.random.randn() * 0.5
            ember['vy'] += 0.5  # Gravity/buoyancy
            
            # Cool down
            ember['temperature'] *= 0.98
            ember['life'] -= dt
            
            if ember['life'] > 0 and ember['temperature'] > 0.1:
                embers_to_keep.append(ember)
        
        self.embers = embers_to_keep
    
    def render(self, image: np.ndarray) -> np.ndarray:
        """Render embers onto image."""
        output = image.copy()
        
        for ember in self.embers:
            x, y = int(ember['x']), int(ember['y'])
            
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                # Ember color based on temperature
                temp = ember['temperature']
                if temp > 0.8:  # Very hot - white/yellow
                    color = (255, 255, 200)
                elif temp > 0.5:  # Hot - orange
                    color = (255, 150, 50)
                else:  # Cooling - red
                    color = (200, 50, 0)
                
                size = int(ember['size'])
                if size > 0:
                    cv2.circle(output, (x, y), size, color, -1)
                    
                    # Add glow
                    glow_size = size * 3
                    overlay = output.copy()
                    cv2.circle(overlay, (x, y), glow_size, color, -1)
                    cv2.addWeighted(output, 0.7, overlay, 0.3 * temp, 0, output)
        
        return output