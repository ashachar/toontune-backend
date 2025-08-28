"""
Volumetric smoke rendering for photorealistic smoke effects.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class SmokeVolume:
    """Represents a volume of smoke."""
    x: float
    y: float
    radius: float
    density: float
    temperature: float
    vx: float
    vy: float
    age: float
    turbulence_offset: float


class VolumetricSmoke:
    """Create thick, billowing smoke using volumetric rendering."""
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1280, 720),
        smoke_color: Tuple[int, int, int] = (60, 60, 65),
        ambient_temperature: float = 20.0,
        buoyancy_factor: float = 0.5,
        dissipation_rate: float = 0.02,
        turbulence_strength: float = 2.0
    ):
        self.width, self.height = resolution
        self.smoke_color = np.array(smoke_color)
        self.ambient_temperature = ambient_temperature
        self.buoyancy_factor = buoyancy_factor
        self.dissipation_rate = dissipation_rate
        self.turbulence_strength = turbulence_strength
        
        # Smoke volumes (billowing clouds)
        self.volumes: List[SmokeVolume] = []
        
        # Density field for volumetric rendering
        self.density_field = np.zeros((self.height, self.width), dtype=np.float32)
        self.temperature_field = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Velocity field for fluid dynamics
        self.velocity_field_x = np.zeros((self.height, self.width), dtype=np.float32)
        self.velocity_field_y = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Perlin noise for turbulence
        self._init_noise()
        
    def _init_noise(self):
        """Initialize noise for turbulence."""
        self.noise_scale = 0.01
        self.time_offset = 0
        
    def emit_smoke(
        self,
        x: float,
        y: float,
        intensity: float = 1.0,
        temperature: float = 100.0,
        initial_radius: float = 20.0
    ):
        """
        Emit smoke from a position.
        
        Args:
            x, y: Emission position
            intensity: Smoke density (0-1)
            temperature: Initial temperature (affects rise speed)
            initial_radius: Initial size of smoke volume
        """
        # Create multiple overlapping volumes for thick smoke
        num_volumes = int(3 + intensity * 4)
        
        for i in range(num_volumes):
            volume = SmokeVolume(
                x=x + np.random.randn() * 10,
                y=y + np.random.randn() * 10,
                radius=initial_radius + np.random.uniform(-5, 10),
                density=intensity * np.random.uniform(0.7, 1.0),
                temperature=temperature * np.random.uniform(0.8, 1.2),
                vx=np.random.randn() * 2,
                vy=-np.random.uniform(1, 3),  # Initial upward velocity
                age=0,
                turbulence_offset=np.random.random() * 100
            )
            self.volumes.append(volume)
    
    def _apply_buoyancy(self, volume: SmokeVolume, dt: float):
        """Apply buoyancy force based on temperature difference."""
        temp_diff = volume.temperature - self.ambient_temperature
        if temp_diff > 0:
            # Hot smoke rises
            buoyancy = self.buoyancy_factor * temp_diff / 100
            volume.vy -= buoyancy * dt * 30  # Negative Y is up
    
    def _apply_turbulence(self, volume: SmokeVolume, time: float):
        """Apply turbulent motion to smoke."""
        # Use sine waves for smooth turbulence
        turb_x = math.sin(time * 2 + volume.turbulence_offset) * self.turbulence_strength
        turb_y = math.cos(time * 1.5 + volume.turbulence_offset * 0.7) * self.turbulence_strength
        
        volume.vx += turb_x
        volume.vy += turb_y * 0.5  # Less vertical turbulence
    
    def update(self, dt: float = 1/30):
        """Update smoke simulation."""
        self.time_offset += dt
        volumes_to_keep = []
        
        # Clear fields
        self.density_field.fill(0)
        self.temperature_field.fill(self.ambient_temperature)
        
        for volume in self.volumes:
            # Physics update
            self._apply_buoyancy(volume, dt)
            self._apply_turbulence(volume, self.time_offset)
            
            # Update position
            volume.x += volume.vx * dt * 30
            volume.y += volume.vy * dt * 30
            
            # Expansion (smoke expands as it rises)
            volume.radius *= 1.01
            
            # Cooling and dissipation
            volume.temperature = self.ambient_temperature + \
                              (volume.temperature - self.ambient_temperature) * (1 - self.dissipation_rate)
            volume.density *= (1 - self.dissipation_rate)
            
            # Air resistance
            volume.vx *= 0.98
            volume.vy *= 0.98
            
            # Age
            volume.age += dt
            
            # Keep if still visible
            if volume.density > 0.01 and volume.radius < 200 and \
               0 < volume.x < self.width and -100 < volume.y < self.height:
                volumes_to_keep.append(volume)
                
                # Add to density field
                self._add_to_field(volume)
        
        self.volumes = volumes_to_keep
        
        # Apply diffusion to density field
        self._diffuse_field()
    
    def _add_to_field(self, volume: SmokeVolume):
        """Add smoke volume to density field."""
        x, y = int(volume.x), int(volume.y)
        radius = int(volume.radius)
        
        # Create gaussian distribution
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px, py = x + dx, y + dy
                
                if 0 <= px < self.width and 0 <= py < self.height:
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist <= radius:
                        # Gaussian falloff
                        falloff = math.exp(-(dist / radius) ** 2)
                        self.density_field[py, px] += volume.density * falloff
                        
                        # Temperature contribution
                        temp_contrib = (volume.temperature - self.ambient_temperature) * falloff
                        self.temperature_field[py, px] += temp_contrib
    
    def _diffuse_field(self):
        """Apply diffusion to create smooth smoke."""
        # Simple gaussian blur for diffusion
        self.density_field = cv2.GaussianBlur(self.density_field, (11, 11), 0)
        self.temperature_field = cv2.GaussianBlur(self.temperature_field, (7, 7), 0)
        
        # Clamp density
        self.density_field = np.clip(self.density_field, 0, 1)
    
    def render(
        self,
        background: np.ndarray,
        light_direction: Tuple[float, float, float] = (0.5, -0.7, 0.5)
    ) -> np.ndarray:
        """
        Render volumetric smoke onto background.
        
        Args:
            background: Background image
            light_direction: Direction of lighting for smoke shading
            
        Returns:
            Image with smoke rendered
        """
        output = background.copy()
        h, w = background.shape[:2]
        
        # Create smoke layer
        smoke_layer = np.zeros((h, w, 4), dtype=np.float32)
        
        # Base smoke color with density
        for y in range(h):
            for x in range(w):
                density = self.density_field[y, x]
                if density > 0.01:
                    # Calculate smoke color with shading
                    # Higher areas of smoke are lighter (lit from above)
                    height_factor = 1.0 - (y / h) * 0.3
                    
                    # Temperature affects color slightly (hot smoke is lighter)
                    temp_factor = 1.0 + (self.temperature_field[y, x] - self.ambient_temperature) / 200
                    temp_factor = np.clip(temp_factor, 0.8, 1.2)
                    
                    # Add noise for texture
                    noise = np.random.random() * 0.1 - 0.05
                    
                    # Final color
                    color = self.smoke_color * height_factor * temp_factor + noise * 20
                    color = np.clip(color, 0, 255)
                    
                    smoke_layer[y, x, :3] = color
                    smoke_layer[y, x, 3] = density * 255
        
        # Apply multiple passes for thick smoke
        smoke_layer = smoke_layer.astype(np.uint8)
        
        # First pass - dense smoke
        alpha1 = smoke_layer[:, :, 3:4] / 255.0 * 0.7
        output = output * (1 - alpha1) + smoke_layer[:, :, :3] * alpha1
        output = output.astype(np.uint8)
        
        # Second pass - lighter overlay for volume
        smoke_blur = cv2.GaussianBlur(smoke_layer, (21, 21), 0)
        alpha2 = smoke_blur[:, :, 3:4] / 255.0 * 0.3
        output = output * (1 - alpha2) + smoke_blur[:, :, :3] * alpha2
        output = output.astype(np.uint8)
        
        # Add some glow/scatter for thick smoke areas
        bright_areas = self.density_field > 0.5
        if np.any(bright_areas):
            glow = np.zeros_like(output)
            glow[bright_areas] = [70, 70, 75]
            glow = cv2.GaussianBlur(glow, (31, 31), 0)
            output = cv2.addWeighted(output, 0.9, glow, 0.1, 0)
        
        return output
    
    def clear(self):
        """Clear all smoke."""
        self.volumes.clear()
        self.density_field.fill(0)
        self.temperature_field.fill(self.ambient_temperature)


class SmokeTrail:
    """Create continuous smoke trails from moving sources."""
    
    def __init__(self, smoke_system: VolumetricSmoke):
        self.smoke_system = smoke_system
        self.emission_points = []
        self.last_emission_time = {}
        
    def add_source(self, source_id: str, x: float, y: float):
        """Add or update a smoke source position."""
        self.emission_points.append((source_id, x, y))
        
        # Emit smoke at intervals
        current_time = self.smoke_system.time_offset
        last_time = self.last_emission_time.get(source_id, 0)
        
        if current_time - last_time > 0.05:  # Emit every 0.05 seconds
            self.smoke_system.emit_smoke(
                x, y,
                intensity=0.8,
                temperature=150,
                initial_radius=15
            )
            self.last_emission_time[source_id] = current_time