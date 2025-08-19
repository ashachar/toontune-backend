"""
Roll in/out animation.
Element rolls like a wheel or barrel into/out of view.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .animate import Animation


class Roll(Animation):
    """
    Animation where element rolls into or out of view.
    
    The element rotates while moving, simulating rolling motion like a wheel.
    
    Additional Parameters:
    ---------------------
    roll_direction : str
        Direction to roll: 'left', 'right' (default 'right')
    roll_type : str
        'in' for roll entry, 'out' for roll exit (default 'in')
    roll_duration : int
        Number of frames for roll animation (default 45)
    rotations : float
        Number of complete rotations during roll (default 2.0)
    bounce_effect : bool
        Add slight bounce when rolling (default True)
    bounce_amplitude : float
        Height of bounce in pixels (default 10)
    easing : str
        Easing function: 'linear', 'ease_in', 'ease_out', 'ease_in_out' (default 'ease_out')
    deformation : bool
        Apply slight squash during ground contact (default True)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        roll_direction: str = 'right',
        roll_type: str = 'in',
        roll_duration: int = 45,
        rotations: float = 2.0,
        bounce_effect: bool = True,
        bounce_amplitude: float = 10,
        easing: str = 'ease_out',
        deformation: bool = True,
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
        
        self.roll_direction = roll_direction.lower()
        self.roll_type = roll_type.lower()
        self.roll_duration = max(10, roll_duration)
        self.rotations = max(0.5, rotations)
        self.bounce_effect = bounce_effect
        self.bounce_amplitude = max(0, bounce_amplitude)
        self.easing = easing
        self.deformation = deformation
        self.remove_background = remove_background
        self.background_color = background_color
        self.background_similarity = background_similarity
        
        self.scaled_frames = []
        self.clean_frames = []
        self.start_position = None
        self.end_position = position
    
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
    
    def calculate_start_position(self) -> Tuple[int, int]:
        """Calculate off-screen starting position for roll-in."""
        if not self.background_frames:
            return (0, 0)
        
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=s=x:p=0',
            self.background_frames[0]
        ]
        
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            video_width, video_height = map(int, result.stdout.strip().split('x'))
        except:
            video_width, video_height = 1920, 1080
        
        element_size = 200
        target_x, target_y = self.end_position
        
        if self.roll_direction == 'left':
            return (video_width + element_size, target_y)
        else:  # right
            return (-element_size, target_y)
    
    def ease_function(self, t: float) -> float:
        """Apply easing function."""
        if self.easing == 'linear':
            return t
        elif self.easing == 'ease_in':
            return t * t
        elif self.easing == 'ease_out':
            return 1 - (1 - t) * (1 - t)
        elif self.easing == 'ease_in_out':
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - pow(-2 * t + 2, 2) / 2
        else:
            return t
    
    def create_rolled_frame(
        self,
        element_frame: str,
        rotation: float,
        squash: float,
        output_path: str
    ) -> bool:
        """Create a rotated and optionally squashed frame."""
        
        filters = []
        
        # Apply rotation
        if self.roll_direction == 'left':
            rotation = -rotation  # Roll counter-clockwise when going left
        filters.append(f'rotate={rotation}*PI/180:c=none')
        
        # Apply deformation (squash) if touching ground
        if self.deformation and squash > 0:
            scale_y = 1.0 - squash * 0.1  # Max 10% squash
            scale_x = 1.0 + squash * 0.05  # Slight horizontal expansion
            filters.append(f'scale=iw*{scale_x}:ih*{scale_y}')
        
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
            return True
        except:
            return False
    
    def calculate_roll_params(self, frame_offset: int) -> Tuple[Tuple[int, int], float, float, float]:
        """Calculate position, rotation, bounce height, and squash for rolling motion."""
        
        if self.roll_type == 'in':
            if frame_offset >= self.roll_duration:
                return (self.end_position, 360 * self.rotations, 0, 0)
            
            progress = frame_offset / self.roll_duration
            eased_progress = self.ease_function(progress)
            
            # Calculate position
            if not self.start_position:
                self.start_position = self.calculate_start_position()
            
            start_x, start_y = self.start_position
            end_x, end_y = self.end_position
            
            current_x = start_x + (end_x - start_x) * eased_progress
            
            # Calculate rotation (proportional to distance traveled)
            rotation = 360 * self.rotations * eased_progress
            
            # Calculate bounce
            if self.bounce_effect:
                # Multiple bounces that decay
                bounce_freq = 4  # Number of bounce peaks
                bounce_phase = progress * bounce_freq * math.pi
                bounce_decay = 1.0 - progress  # Decay over time
                bounce_height = abs(math.sin(bounce_phase)) * self.bounce_amplitude * bounce_decay
                current_y = end_y - bounce_height
            else:
                current_y = end_y
            
            # Calculate squash (when touching ground)
            if self.deformation and bounce_height < 2:
                squash = 1.0 - bounce_height / 2
            else:
                squash = 0
            
        elif self.roll_type == 'out':
            out_start = self.total_frames - self.roll_duration - self.start_frame
            
            if frame_offset < out_start:
                return (self.end_position, 0, 0, 0)
            
            progress = (frame_offset - out_start) / self.roll_duration
            eased_progress = self.ease_function(progress)
            
            # Calculate end position off-screen
            if self.roll_direction == 'left':
                final_x = -200
            else:
                final_x = 2000  # Assuming typical screen width
            
            start_x, start_y = self.end_position
            current_x = start_x + (final_x - start_x) * eased_progress
            
            # Rotation continues in same direction
            rotation = 360 * self.rotations * eased_progress
            
            # Bounce effect for roll out
            if self.bounce_effect:
                bounce_phase = progress * 3 * math.pi
                bounce_height = abs(math.sin(bounce_phase)) * self.bounce_amplitude * (1 - progress * 0.5)
                current_y = start_y - bounce_height
            else:
                current_y = start_y
            
            squash = 0
            
        else:
            return (self.end_position, 0, 0, 0)
        
        return ((int(current_x), int(current_y)), rotation, bounce_height, squash)
    
    def process_frames(self) -> List[str]:
        """Process frames for roll animation."""
        print(f"   Processing roll animation...")
        print(f"   Direction: {self.roll_direction}")
        print(f"   Type: {self.roll_type}")
        print(f"   Rotations: {self.rotations}")
        print(f"   Bounce: {self.bounce_effect}")
        
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
            current_position, rotation, bounce_height, squash = self.calculate_roll_params(frame_offset)
            
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                rolled_frame = os.path.join(self.temp_dir, f'roll_{frame_num:04d}.png')
                
                if self.create_rolled_frame(element_frame, rotation, squash, rolled_frame):
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    if self.composite_frame(
                        background_frame,
                        rolled_frame,
                        output_frame,
                        current_position
                    ):
                        output_frames.append(output_frame)
                        
                        if frame_num % 15 == 0:
                            print(f"      Frame {frame_num}: rotation {rotation:.1f}°")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames