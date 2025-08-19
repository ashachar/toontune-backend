"""
Particles animation effect.
Element dissolves into or forms from particles.
"""

import os
import subprocess
import random
import math
from typing import List, Tuple, Optional
from .animate import Animation


class Particles(Animation):
    """
    Animation where element dissolves into or forms from particles.
    
    Creates particle effects like dust, sparkles, or magical dissolution.
    
    Additional Parameters:
    ---------------------
    particle_type : str
        Type: 'dust', 'sparkles', 'bubbles', 'fire', 'snow', 'magic' (default 'sparkles')
    num_particles : int
        Number of particles (10 to 200, default 50)
    particle_size : float
        Size of particles relative to element (0.01 to 0.1, default 0.03)
    particle_direction : str
        Direction: 'up', 'down', 'radial', 'random', 'wind' (default 'up')
    particle_speed : float
        Speed of particle movement (0.5 to 3.0, default 1.0)
    formation_type : str
        'dissolve' (element to particles) or 'form' (particles to element) (default 'dissolve')
    color_variation : bool
        Vary particle colors (default True)
    glow_effect : bool
        Add glow to particles (default True)
    turbulence : float
        Random movement turbulence (0.0 to 1.0, default 0.3)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        particle_type: str = 'sparkles',
        num_particles: int = 50,
        particle_size: float = 0.03,
        particle_direction: str = 'up',
        particle_speed: float = 1.0,
        formation_type: str = 'dissolve',
        color_variation: bool = True,
        glow_effect: bool = True,
        turbulence: float = 0.3,
        direction: float = 0,
        start_frame: int = 0,
        animation_start_frame: int = 0,
        path: Optional[List[Tuple[int, int, int]]] = None,
        fps: int = 30,
        duration: float = 7.0,
        temp_dir: Optional[str] = None,
        remove_background: bool = True,
        background_color: str = '0x000000',
        background_similarity: float = 0.15
    ):
        super().__init__(
            element_path=element_path,
            background_path=background_path,
            position=position,
            direction=direction,
            start_frame=start_frame,
            animation_start_frame=animation_start_frame,
            path=path,
            fps=fps,
            duration=duration,
            temp_dir=temp_dir
        )
        
        self.particle_type = particle_type.lower()
        self.num_particles = max(10, min(200, num_particles))
        self.particle_size = max(0.01, min(0.1, particle_size))
        self.particle_direction = particle_direction.lower()
        self.particle_speed = max(0.5, min(3.0, particle_speed))
        self.formation_type = formation_type.lower()
        self.color_variation = color_variation
        self.glow_effect = glow_effect
        self.turbulence = max(0.0, min(1.0, turbulence))
        self.remove_background = remove_background
        self.background_color = background_color
        self.background_similarity = background_similarity
        
        self.scaled_frames = []
        self.clean_frames = []
        self.particles_data = []
        self.generate_particles()
    
    def generate_particles(self):
        """Generate particle system data."""
        for i in range(self.num_particles):
            # Starting position (relative to element center)
            if self.formation_type == 'form':
                # Particles start spread out
                if self.particle_direction == 'radial':
                    angle = random.uniform(0, 360)
                    distance = random.uniform(100, 300)
                    start_x = math.cos(math.radians(angle)) * distance
                    start_y = math.sin(math.radians(angle)) * distance
                else:
                    start_x = random.uniform(-200, 200)
                    start_y = random.uniform(-200, 200)
            else:
                # Particles start at element position
                start_x = random.uniform(-50, 50)
                start_y = random.uniform(-50, 50)
            
            # Velocity based on direction
            if self.particle_direction == 'up':
                vel_x = random.uniform(-10, 10) * self.particle_speed
                vel_y = -random.uniform(20, 50) * self.particle_speed
            elif self.particle_direction == 'down':
                vel_x = random.uniform(-10, 10) * self.particle_speed
                vel_y = random.uniform(20, 50) * self.particle_speed
            elif self.particle_direction == 'radial':
                angle = math.atan2(start_y, start_x)
                speed = random.uniform(20, 40) * self.particle_speed
                vel_x = math.cos(angle) * speed
                vel_y = math.sin(angle) * speed
            elif self.particle_direction == 'wind':
                vel_x = random.uniform(30, 60) * self.particle_speed
                vel_y = random.uniform(-10, 10) * self.particle_speed
            else:  # random
                vel_x = random.uniform(-30, 30) * self.particle_speed
                vel_y = random.uniform(-30, 30) * self.particle_speed
            
            # Particle properties
            particle = {
                'id': i,
                'start_x': start_x,
                'start_y': start_y,
                'vel_x': vel_x,
                'vel_y': vel_y,
                'size': self.particle_size * random.uniform(0.5, 1.5),
                'lifetime': random.uniform(30, 90),  # frames
                'delay': random.uniform(0, 20),  # start delay
                'rotation': random.uniform(0, 360),
                'rotation_speed': random.uniform(-10, 10),
                'color_shift': random.uniform(-30, 30) if self.color_variation else 0,
                'brightness': random.uniform(0.8, 1.2)
            }
            
            # Type-specific properties
            if self.particle_type == 'bubbles':
                particle['float_speed'] = random.uniform(0.5, 1.5)
                particle['wobble'] = random.uniform(0, math.pi)
            elif self.particle_type == 'fire':
                particle['flicker'] = random.uniform(0, 1)
                particle['color_shift'] = random.uniform(0, 60)  # Red to yellow
            elif self.particle_type == 'snow':
                particle['drift'] = random.uniform(-1, 1)
                particle['fall_speed'] = random.uniform(0.3, 0.7)
            
            self.particles_data.append(particle)
    
    def extract_element_frames(self) -> List[str]:
        """Extract and prepare element frames."""
        print(f"   Extracting and preparing element frames...")
        
        raw_dir = os.path.join(self.temp_dir, "element_raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', self.element_path,
            '-r', str(self.fps),
            os.path.join(raw_dir, 'raw_%04d.png')
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            raw_frames = sorted([
                os.path.join(raw_dir, f)
                for f in os.listdir(raw_dir)
                if f.endswith('.png')
            ])
            print(f"   ✓ Extracted {len(raw_frames)} raw frames")
        except:
            print(f"   ✗ Failed to extract raw frames")
            return []
        
        scaled_dir = os.path.join(self.temp_dir, "element_scaled")
        os.makedirs(scaled_dir, exist_ok=True)
        
        for i, frame in enumerate(raw_frames):
            output_frame = os.path.join(scaled_dir, f'scaled_{i:04d}.png')
            
            cmd = [
                'ffmpeg',
                '-i', frame,
                '-vf', 'scale=200:-1',
                '-y',
                output_frame
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.scaled_frames.append(output_frame)
            except:
                print(f"   ✗ Failed to scale frame {i}")
                return []
        
        print(f"   ✓ Scaled {len(self.scaled_frames)} frames")
        
        if self.remove_background:
            clean_dir = os.path.join(self.temp_dir, "element_clean")
            os.makedirs(clean_dir, exist_ok=True)
            
            for i, frame in enumerate(self.scaled_frames):
                output_frame = os.path.join(clean_dir, f'clean_{i:04d}.png')
                
                cmd = [
                    'ffmpeg',
                    '-i', frame,
                    '-vf', f'colorkey={self.background_color}:{self.background_similarity}:0.05',
                    '-c:v', 'png',
                    '-y',
                    output_frame
                ]
                
                try:
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                    self.clean_frames.append(output_frame)
                except:
                    self.clean_frames.append(frame)
            
            print(f"   ✓ Removed background from frames")
            self.element_frames = self.clean_frames
        else:
            self.element_frames = self.scaled_frames
        
        return self.element_frames
    
    def create_particle_frame(
        self,
        element_frame: str,
        particle: dict,
        time_offset: float,
        output_path: str
    ) -> Tuple[bool, Tuple[int, int]]:
        """Create a single particle frame."""
        
        # Check if particle is active
        if time_offset < particle['delay'] / self.fps:
            return False, (0, 0)
        
        active_time = time_offset - particle['delay'] / self.fps
        
        if active_time * self.fps > particle['lifetime']:
            return False, (0, 0)
        
        # Calculate position
        if self.formation_type == 'form':
            # Particles converge to center
            progress = min(1.0, active_time * 2)
            x = particle['start_x'] * (1 - progress)
            y = particle['start_y'] * (1 - progress)
        else:
            # Particles disperse from center
            x = particle['start_x'] + particle['vel_x'] * active_time
            y = particle['start_y'] + particle['vel_y'] * active_time
        
        # Add turbulence
        if self.turbulence > 0:
            x += math.sin(active_time * 10 + particle['id']) * 10 * self.turbulence
            y += math.cos(active_time * 10 + particle['id']) * 10 * self.turbulence
        
        # Calculate opacity
        lifetime_progress = (active_time * self.fps) / particle['lifetime']
        if self.formation_type == 'form':
            opacity = min(1.0, lifetime_progress * 2)
        else:
            opacity = max(0, 1.0 - lifetime_progress)
        
        # Apply particle type-specific effects
        if self.particle_type == 'bubbles':
            y -= particle['float_speed'] * active_time * 10
            x += math.sin(particle['wobble'] + active_time * 2) * 5
        elif self.particle_type == 'fire':
            opacity *= (0.5 + 0.5 * math.sin(active_time * 20 * particle['flicker']))
        elif self.particle_type == 'snow':
            y += particle['fall_speed'] * active_time * 30
            x += math.sin(active_time) * particle['drift'] * 10
        
        # Create particle visual
        size = int(200 * particle['size'])
        rotation = particle['rotation'] + particle['rotation_speed'] * active_time * 10
        
        filters = [
            f'scale={size}:{size}',
            f'rotate={rotation}*PI/180:c=none',
            f'format=rgba,colorchannelmixer=aa={opacity}'
        ]
        
        # Color and effects
        if particle['color_shift'] != 0:
            filters.append(f'hue=h={particle["color_shift"]}')
        
        if self.glow_effect:
            filters.append(f'gblur=sigma=2,eq=brightness={particle["brightness"]}')
        
        filter_chain = ','.join(filters)
        
        cmd = [
            'ffmpeg',
            '-i', element_frame,
            '-vf', filter_chain,
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True, (int(x), int(y))
        except:
            return False, (0, 0)
    
    def process_frames(self) -> List[str]:
        """Process frames for particles animation."""
        print(f"   Processing particles animation...")
        print(f"   Type: {self.particle_type}")
        print(f"   Particles: {self.num_particles}")
        print(f"   Formation: {self.formation_type}")
        
        output_dir = os.path.join(self.temp_dir, "output_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        output_frames = []
        num_element_frames = len(self.element_frames)
        
        for frame_num in range(self.total_frames):
            if frame_num < self.start_frame:
                output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                if frame_num < len(self.background_frames):
                    subprocess.run(
                        ['cp', self.background_frames[frame_num], output_frame],
                        capture_output=True
                    )
                    output_frames.append(output_frame)
                continue
            
            frame_offset = frame_num - self.start_frame
            time_offset = frame_offset / self.fps
            
            if frame_num < len(self.background_frames):
                # Start with background
                output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                subprocess.run(
                    ['cp', self.background_frames[frame_num], output_frame],
                    capture_output=True
                )
                
                # Calculate element opacity for formation/dissolution
                if self.formation_type == 'form':
                    element_opacity = min(1.0, time_offset * 0.5)
                else:
                    element_opacity = max(0, 1.0 - time_offset * 0.5)
                
                # Add main element if visible
                if element_opacity > 0 and num_element_frames > 0:
                    anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
                    element_frame = self.element_frames[anim_frame_idx]
                    
                    # Apply opacity to element
                    faded_element = os.path.join(self.temp_dir, f'faded_{frame_num}.png')
                    cmd = [
                        'ffmpeg',
                        '-i', element_frame,
                        '-vf', f'format=rgba,colorchannelmixer=aa={element_opacity}',
                        '-y',
                        faded_element
                    ]
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                    
                    temp_output = os.path.join(self.temp_dir, f'temp_main_{frame_num}.png')
                    self.composite_frame(
                        output_frame,
                        faded_element,
                        temp_output,
                        self.position
                    )
                    subprocess.run(['mv', temp_output, output_frame], capture_output=True)
                
                # Add particles (limit for performance)
                if num_element_frames > 0:
                    anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
                    element_frame = self.element_frames[anim_frame_idx]
                    
                    for particle in self.particles_data[:min(20, len(self.particles_data))]:
                        particle_frame = os.path.join(self.temp_dir, f'particle_{frame_num}_{particle["id"]}.png')
                        success, offset = self.create_particle_frame(
                            element_frame,
                            particle,
                            time_offset,
                            particle_frame
                        )
                        
                        if success:
                            part_x = self.position[0] + offset[0]
                            part_y = self.position[1] + offset[1]
                            
                            temp_output = os.path.join(self.temp_dir, f'temp_{frame_num}_{particle["id"]}.png')
                            self.composite_frame(
                                output_frame,
                                particle_frame,
                                temp_output,
                                (part_x, part_y)
                            )
                            subprocess.run(['mv', temp_output, output_frame], capture_output=True)
                
                output_frames.append(output_frame)
                
                if frame_num % 15 == 0:
                    print(f"      Frame {frame_num}: time {time_offset:.2f}s")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames