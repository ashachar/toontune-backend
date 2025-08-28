"""
Particle system for smoke and fire effects.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Particle:
    """Single smoke/fire particle."""
    x: float
    y: float
    vx: float  # velocity x
    vy: float  # velocity y
    life: float  # 0-1, where 1 is birth and 0 is death
    size: float
    color: Tuple[int, int, int]
    opacity: float
    is_fire: bool = False  # True for fire particles, False for smoke


class SmokeParticles:
    """Manages smoke and fire particle effects."""
    
    def __init__(
        self,
        max_particles: int = 100,
        smoke_color: Tuple[int, int, int] = (80, 80, 80),
        fire_color: Tuple[int, int, int] = (255, 100, 0),
        rise_speed: float = 2.0,
        spread_rate: float = 0.5,
        turbulence: float = 0.3
    ):
        self.max_particles = max_particles
        self.smoke_color = smoke_color
        self.fire_color = fire_color
        self.rise_speed = rise_speed
        self.spread_rate = spread_rate
        self.turbulence = turbulence
        self.particles: List[Particle] = []
        
    def emit_from_edge(
        self, 
        edge_points: np.ndarray,
        num_particles: int = 5,
        is_fire: bool = False
    ):
        """
        Emit particles from burning edge points.
        
        Args:
            edge_points: Array of (x, y) points along burning edge
            num_particles: Number of particles to emit
            is_fire: Whether to emit fire or smoke particles
        """
        if len(edge_points) == 0:
            return
        
        for _ in range(min(num_particles, self.max_particles - len(self.particles))):
            # Select random point from edge
            idx = np.random.randint(0, len(edge_points))
            x, y = edge_points[idx]
            
            # Add some position variation
            x += np.random.randn() * 2
            y += np.random.randn() * 2
            
            # Initial velocity (upward with spread)
            vx = np.random.randn() * self.spread_rate
            vy = -self.rise_speed + np.random.randn() * 0.5
            
            # Particle properties
            color = self.fire_color if is_fire else self.smoke_color
            size = np.random.uniform(3, 8) if is_fire else np.random.uniform(5, 15)
            
            particle = Particle(
                x=x, y=y,
                vx=vx, vy=vy,
                life=1.0,
                size=size,
                color=color,
                opacity=0.8 if is_fire else 0.6,
                is_fire=is_fire
            )
            
            self.particles.append(particle)
    
    def update(self, dt: float = 1/30):
        """Update particle positions and properties."""
        particles_to_keep = []
        
        for p in self.particles:
            # Update position
            p.x += p.vx
            p.y += p.vy
            
            # Add turbulence
            p.vx += np.random.randn() * self.turbulence * dt
            
            # Update life (particles fade out)
            p.life -= dt * 0.5  # 2 second lifetime
            
            if p.is_fire:
                # Fire particles rise faster and die quicker
                p.vy -= 0.5 * dt  # Accelerate upward
                p.life -= dt * 0.3  # Extra decay
                
                # Fire gets smaller as it rises
                p.size *= 0.98
                
                # Transition fire to smoke
                if p.life < 0.5 and not p.is_fire:
                    p.is_fire = False
                    p.color = self.smoke_color
                    p.size *= 1.5  # Smoke expands
            else:
                # Smoke particles expand and slow down
                p.size *= 1.02  # Gradual expansion
                p.vx *= 0.98   # Air resistance
                p.vy *= 0.98
            
            # Update opacity based on life
            if p.is_fire:
                p.opacity = p.life * 0.9
            else:
                p.opacity = p.life * 0.4
            
            # Keep particle if still alive
            if p.life > 0 and p.opacity > 0.01:
                particles_to_keep.append(p)
        
        self.particles = particles_to_keep
    
    def render(
        self, 
        frame: np.ndarray,
        blend_mode: str = 'additive'
    ) -> np.ndarray:
        """
        Render particles onto frame.
        
        Args:
            frame: Image to render particles on
            blend_mode: 'additive' for fire, 'alpha' for smoke
        """
        output = frame.copy()
        
        # Sort particles by depth (y-position) for proper layering
        sorted_particles = sorted(self.particles, key=lambda p: p.y, reverse=True)
        
        for p in sorted_particles:
            if p.opacity <= 0:
                continue
            
            # Create particle sprite
            size = int(p.size)
            if size < 1:
                continue
            
            # Create gradient circle for particle
            y, x = np.ogrid[-size:size+1, -size:size+1]
            mask = x**2 + y**2 <= size**2
            dist = np.sqrt(x**2 + y**2)
            dist = dist / (size + 1)
            
            # Gaussian falloff for soft edges
            alpha = np.exp(-(dist**2) * 3) * p.opacity
            alpha[~mask] = 0
            
            # Calculate particle position in frame
            px, py = int(p.x), int(p.y)
            
            # Calculate bounds
            y1 = max(0, py - size)
            y2 = min(frame.shape[0], py + size + 1)
            x1 = max(0, px - size)
            x2 = min(frame.shape[1], px + size + 1)
            
            # Particle bounds in local coordinates
            ly1 = y1 - (py - size)
            ly2 = ly1 + (y2 - y1)
            lx1 = x1 - (px - size)
            lx2 = lx1 + (x2 - x1)
            
            if x2 > x1 and y2 > y1:
                # Extract particle region
                particle_alpha = alpha[ly1:ly2, lx1:lx2]
                
                if p.is_fire and blend_mode == 'additive':
                    # Additive blending for fire (creates glow)
                    for c in range(3):
                        channel = output[y1:y2, x1:x2, c].astype(float)
                        add_value = p.color[c] * particle_alpha
                        output[y1:y2, x1:x2, c] = np.clip(channel + add_value, 0, 255).astype(np.uint8)
                else:
                    # Alpha blending for smoke
                    particle_alpha = particle_alpha[:, :, np.newaxis]
                    output[y1:y2, x1:x2] = (
                        output[y1:y2, x1:x2] * (1 - particle_alpha) +
                        np.array(p.color) * particle_alpha
                    ).astype(np.uint8)
        
        return output
    
    def clear(self):
        """Clear all particles."""
        self.particles.clear()
    
    def get_particle_count(self) -> int:
        """Get current number of particles."""
        return len(self.particles)