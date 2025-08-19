"""
Shatter animation effect.
Element breaks into pieces and disperses.
"""

import os
import subprocess
import random
import math
from typing import List, Tuple, Optional
from .animate import Animation


class Shatter(Animation):
    """
    Animation where element shatters into pieces.
    
    Creates a breaking/explosion effect with fragments flying apart.
    
    Additional Parameters:
    ---------------------
    shatter_type : str
        Type: 'glass', 'explosion', 'dissolve', 'crumble' (default 'glass')
    num_pieces : int
        Number of fragments (4 to 50, default 16)
    shatter_point : Tuple[int, int]
        Impact point for shatter origin (default center)
    explosion_force : float
        Force of explosion/dispersion (0.5 to 3.0, default 1.0)
    gravity : float
        Gravity effect on pieces (0.0 to 2.0, default 0.5)
    rotation_speed : float
        Rotation speed of fragments (default 1.0)
    fade_fragments : bool
        Fade out fragments over time (default True)
    shatter_delay : int
        Frame delay before shatter starts (default 0)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        shatter_type: str = 'glass',
        num_pieces: int = 16,
        shatter_point: Optional[Tuple[int, int]] = None,
        explosion_force: float = 1.0,
        gravity: float = 0.5,
        rotation_speed: float = 1.0,
        fade_fragments: bool = True,
        shatter_delay: int = 0,
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
        
        self.shatter_type = shatter_type.lower()
        self.num_pieces = max(4, min(50, num_pieces))
        self.shatter_point = shatter_point if shatter_point else position
        self.explosion_force = max(0.5, min(3.0, explosion_force))
        self.gravity = max(0.0, min(2.0, gravity))
        self.rotation_speed = rotation_speed
        self.fade_fragments = fade_fragments
        self.shatter_delay = max(0, shatter_delay)
        self.remove_background = remove_background
        self.background_color = background_color
        self.background_similarity = background_similarity
        
        self.scaled_frames = []
        self.clean_frames = []
        self.fragments = []
        self.generate_fragments()
    
    def generate_fragments(self):
        """Generate fragment data for shatter animation."""
        grid_size = int(math.sqrt(self.num_pieces))
        
        for i in range(self.num_pieces):
            # Random or grid-based fragment generation
            if self.shatter_type == 'glass':
                # Angular glass-like fragments
                angle = random.uniform(0, 360)
                distance = random.uniform(0, 100)
                velocity_x = math.cos(math.radians(angle)) * distance * self.explosion_force
                velocity_y = math.sin(math.radians(angle)) * distance * self.explosion_force
            elif self.shatter_type == 'explosion':
                # Radial explosion
                angle = (360 / self.num_pieces) * i + random.uniform(-20, 20)
                speed = random.uniform(50, 150) * self.explosion_force
                velocity_x = math.cos(math.radians(angle)) * speed
                velocity_y = math.sin(math.radians(angle)) * speed
            elif self.shatter_type == 'crumble':
                # Downward crumble
                velocity_x = random.uniform(-20, 20)
                velocity_y = random.uniform(10, 50) * self.explosion_force
            else:  # dissolve
                # Random dissolution
                velocity_x = random.uniform(-50, 50) * self.explosion_force
                velocity_y = random.uniform(-50, 50) * self.explosion_force
            
            self.fragments.append({
                'id': i,
                'velocity_x': velocity_x,
                'velocity_y': velocity_y,
                'rotation': random.uniform(-180, 180),
                'rotation_speed': random.uniform(-5, 5) * self.rotation_speed,
                'size': random.uniform(0.3, 1.0),
                'lifetime': random.uniform(30, 60)  # frames
            })
    
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
    
    def create_fragment_frame(
        self,
        element_frame: str,
        fragment: dict,
        time_offset: float,
        output_path: str
    ) -> Tuple[bool, Tuple[int, int]]:
        """Create a single fragment frame with physics simulation."""
        
        # Calculate fragment position
        x = fragment['velocity_x'] * time_offset
        y = fragment['velocity_y'] * time_offset + 0.5 * self.gravity * 9.8 * time_offset * time_offset
        
        # Calculate rotation
        rotation = fragment['rotation'] + fragment['rotation_speed'] * time_offset * 10
        
        # Calculate opacity based on lifetime
        if self.fade_fragments:
            opacity = max(0, 1.0 - (time_offset * 30 / fragment['lifetime']))
        else:
            opacity = 1.0
        
        if opacity <= 0:
            return False, (0, 0)
        
        # Create fragment using crop and transform
        # Simplified: use scaling and rotation to simulate fragment
        size = fragment['size']
        
        filters = [
            f'scale=iw*{size}:ih*{size}',
            f'rotate={rotation}*PI/180:c=none',
            f'format=rgba,colorchannelmixer=aa={opacity}'
        ]
        
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
        """Process frames for shatter animation."""
        print(f"   Processing shatter animation...")
        print(f"   Type: {self.shatter_type}")
        print(f"   Pieces: {self.num_pieces}")
        print(f"   Force: {self.explosion_force}")
        
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
            
            # Check if shatter has started
            if frame_offset < self.shatter_delay:
                # Show intact element
                if frame_num >= self.animation_start_frame and num_element_frames > 0:
                    anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
                else:
                    anim_frame_idx = 0
                
                element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
                
                if element_frame and frame_num < len(self.background_frames):
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    if self.composite_frame(
                        background_frame,
                        element_frame,
                        output_frame,
                        self.position
                    ):
                        output_frames.append(output_frame)
            else:
                # Shatter in progress
                shatter_time = (frame_offset - self.shatter_delay) / self.fps
                
                if frame_num < len(self.background_frames):
                    # Start with background
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    subprocess.run(
                        ['cp', self.background_frames[frame_num], output_frame],
                        capture_output=True
                    )
                    
                    # Add fragments
                    if num_element_frames > 0:
                        anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
                        element_frame = self.element_frames[anim_frame_idx]
                        
                        # Simplified: show fading/dispersing effect
                        if shatter_time < 2.0:  # 2 seconds of shatter
                            for i, fragment in enumerate(self.fragments[:min(5, len(self.fragments))]):
                                fragment_frame = os.path.join(self.temp_dir, f'frag_{frame_num}_{i}.png')
                                success, offset = self.create_fragment_frame(
                                    element_frame,
                                    fragment,
                                    shatter_time,
                                    fragment_frame
                                )
                                
                                if success:
                                    frag_x = self.position[0] + offset[0]
                                    frag_y = self.position[1] + offset[1]
                                    
                                    # Composite fragment
                                    temp_output = os.path.join(self.temp_dir, f'temp_{frame_num}_{i}.png')
                                    self.composite_frame(
                                        output_frame,
                                        fragment_frame,
                                        temp_output,
                                        (frag_x, frag_y)
                                    )
                                    subprocess.run(['mv', temp_output, output_frame], capture_output=True)
                    
                    output_frames.append(output_frame)
                    
                    if frame_num % 15 == 0:
                        print(f"      Frame {frame_num}: shatter time {shatter_time:.2f}s")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames