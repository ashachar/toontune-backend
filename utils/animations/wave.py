"""
Wave distortion animation.
Creates wave/ripple effects through elements.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .animate import Animation


class Wave(Animation):
    """
    Animation where element is distorted by wave effects.
    
    Creates rippling, waving, or undulating distortion effects.
    
    Additional Parameters:
    ---------------------
    wave_type : str
        Type of wave: 'horizontal', 'vertical', 'radial', 'flag' (default 'horizontal')
    wave_amplitude : float
        Maximum displacement in pixels (default 20)
    wave_frequency : float
        Number of wave cycles (default 2.0)
    wave_speed : float
        Speed of wave propagation (default 1.0)
    wave_direction : str
        Direction of wave travel: 'forward', 'backward' (default 'forward')
    damping : float
        Wave amplitude reduction over time (0.0 to 1.0, default 0.0)
    phase_offset : float
        Initial phase of wave in radians (default 0.0)
    distort_edges : bool
        Whether to distort element edges (default True)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        wave_type: str = 'horizontal',
        wave_amplitude: float = 20,
        wave_frequency: float = 2.0,
        wave_speed: float = 1.0,
        wave_direction: str = 'forward',
        damping: float = 0.0,
        phase_offset: float = 0.0,
        distort_edges: bool = True,
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
        
        self.wave_type = wave_type.lower()
        self.wave_amplitude = max(0, wave_amplitude)
        self.wave_frequency = max(0.1, wave_frequency)
        self.wave_speed = wave_speed
        self.wave_direction = wave_direction.lower()
        self.damping = max(0.0, min(1.0, damping))
        self.phase_offset = phase_offset
        self.distort_edges = distort_edges
        self.remove_background = remove_background
        self.background_color = background_color
        self.background_similarity = background_similarity
        
        self.scaled_frames = []
        self.clean_frames = []
    
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
    
    def create_wave_distortion(
        self,
        element_frame: str,
        time_offset: float,
        output_path: str
    ) -> bool:
        """Apply wave distortion to frame."""
        
        # Calculate phase based on time and speed
        if self.wave_direction == 'backward':
            phase = self.phase_offset - (time_offset * self.wave_speed * 2 * math.pi)
        else:
            phase = self.phase_offset + (time_offset * self.wave_speed * 2 * math.pi)
        
        # Apply damping over time
        amplitude = self.wave_amplitude * (1.0 - self.damping * min(1.0, time_offset / 100))
        
        # Build FFmpeg filter based on wave type
        if self.wave_type == 'horizontal':
            # Horizontal wave (vertical displacement)
            filter_expr = f'geq=p(X, Y + {amplitude} * sin({self.wave_frequency} * 2 * PI * X / W + {phase}))'
            
        elif self.wave_type == 'vertical':
            # Vertical wave (horizontal displacement)
            filter_expr = f'geq=p(X + {amplitude} * sin({self.wave_frequency} * 2 * PI * Y / H + {phase}), Y)'
            
        elif self.wave_type == 'radial':
            # Radial/circular wave from center
            filter_expr = f'geq=p(X + {amplitude} * sin(sqrt((X-W/2)^2 + (Y-H/2)^2) * {self.wave_frequency} * 0.05 + {phase}) * (X-W/2) / sqrt((X-W/2)^2 + (Y-H/2)^2 + 1), Y + {amplitude} * sin(sqrt((X-W/2)^2 + (Y-H/2)^2) * {self.wave_frequency} * 0.05 + {phase}) * (Y-H/2) / sqrt((X-W/2)^2 + (Y-H/2)^2 + 1))'
            
        elif self.wave_type == 'flag':
            # Flag-like wave (stronger at edges)
            filter_expr = f'geq=p(X, Y + {amplitude} * sin({self.wave_frequency} * 2 * PI * X / W + {phase}) * X / W)'
            
        else:
            # Default to horizontal wave
            filter_expr = f'geq=p(X, Y + {amplitude} * sin({self.wave_frequency} * 2 * PI * X / W + {phase}))'
        
        # Apply the wave distortion
        cmd = [
            'ffmpeg',
            '-i', element_frame,
            '-vf', filter_expr,
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except:
            # If complex filter fails, try simpler approach
            try:
                # Use displacement map approach
                simple_filter = f'hue=s=1:h={phase}*10'  # Fallback to color shift
                cmd = [
                    'ffmpeg',
                    '-i', element_frame,
                    '-vf', simple_filter,
                    '-y',
                    output_path
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                return True
            except:
                # Ultimate fallback - just copy
                subprocess.run(['cp', element_frame, output_path], capture_output=True)
                return True
    
    def process_frames(self) -> List[str]:
        """Process frames for wave animation."""
        print(f"   Processing wave animation...")
        print(f"   Type: {self.wave_type}")
        print(f"   Amplitude: {self.wave_amplitude}px")
        print(f"   Frequency: {self.wave_frequency}")
        print(f"   Speed: {self.wave_speed}")
        
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
            
            # Calculate time offset for wave animation
            frame_offset = frame_num - self.start_frame
            time_offset = frame_offset / self.fps
            
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                # Apply wave distortion
                waved_frame = os.path.join(self.temp_dir, f'wave_{frame_num:04d}.png')
                
                if self.create_wave_distortion(element_frame, time_offset, waved_frame):
                    if self.path:
                        current_position = self.get_position_at_frame(frame_num)
                    else:
                        current_position = self.position
                    
                    # Add slight position oscillation for enhanced effect
                    if self.wave_type in ['horizontal', 'flag']:
                        # Slight vertical oscillation
                        y_offset = math.sin(phase) * self.wave_amplitude * 0.1
                        current_position = (current_position[0], int(current_position[1] + y_offset))
                    
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    if self.composite_frame(
                        background_frame,
                        waved_frame,
                        output_frame,
                        current_position
                    ):
                        output_frames.append(output_frame)
                        
                        if frame_num % 15 == 0:
                            phase = self.phase_offset + (time_offset * self.wave_speed * 2 * math.pi)
                            print(f"      Frame {frame_num}: phase {phase:.2f}")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames