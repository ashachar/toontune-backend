"""
Spin/Rotate animation.
Element spins or rotates with various effects.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .animate import Animation


class Spin(Animation):
    """
    Animation where element spins or rotates.
    
    The element rotates around its center with optional scaling and motion blur effects.
    
    Additional Parameters:
    ---------------------
    spin_degrees : float
        Total degrees to rotate (360 = full rotation, default 360)
    spin_duration : int
        Number of frames for spin animation (default 30)
    spin_type : str
        'in' for spin on entry, 'out' for spin on exit, 'continuous' (default 'in')
    spin_direction : str
        'clockwise' or 'counter-clockwise' (default 'clockwise')
    easing : str
        Easing function: 'linear', 'ease_in', 'ease_out', 'ease_in_out' (default 'linear')
    scale_during_spin : bool
        Whether to scale during spin for depth effect (default True)
    scale_factor : float
        Maximum scale change during spin (0.5 to 1.5, default 0.8)
    motion_blur : bool
        Apply motion blur effect during fast rotation (default False)
    remove_background : bool
        Whether to remove black background (default True)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        spin_degrees: float = 360,
        spin_duration: int = 30,
        spin_type: str = 'in',
        spin_direction: str = 'clockwise',
        easing: str = 'linear',
        scale_during_spin: bool = True,
        scale_factor: float = 0.8,
        motion_blur: bool = False,
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
        
        self.spin_degrees = spin_degrees
        self.spin_duration = max(1, spin_duration)
        self.spin_type = spin_type.lower()
        self.spin_direction = spin_direction.lower()
        self.easing = easing
        self.scale_during_spin = scale_during_spin
        self.scale_factor = max(0.5, min(1.5, scale_factor))
        self.motion_blur = motion_blur
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
    
    def ease_function(self, t: float) -> float:
        """Apply easing function."""
        if self.easing == 'linear':
            return t
        elif self.easing == 'ease_in':
            return t * t * t
        elif self.easing == 'ease_out':
            return 1 - pow(1 - t, 3)
        elif self.easing == 'ease_in_out':
            if t < 0.5:
                return 4 * t * t * t
            else:
                return 1 - pow(-2 * t + 2, 3) / 2
        else:
            return t
    
    def create_rotated_frame(
        self,
        element_frame: str,
        rotation: float,
        scale: float,
        blur_amount: float,
        output_path: str
    ) -> bool:
        """Create a rotated and optionally scaled/blurred frame."""
        
        filters = []
        
        # Apply rotation
        if self.spin_direction == 'counter-clockwise':
            rotation = -rotation
        filters.append(f'rotate={rotation}*PI/180:c=none')
        
        # Apply scaling if needed
        if scale != 1.0:
            filters.append(f'scale=iw*{scale}:ih*{scale}')
        
        # Apply motion blur if needed
        if self.motion_blur and blur_amount > 0:
            # Simple motion blur using boxblur
            blur_radius = min(10, int(blur_amount * 5))
            if blur_radius > 0:
                filters.append(f'boxblur={blur_radius}:1')
        
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
    
    def calculate_spin_params(self, frame_offset: int) -> Tuple[float, float, float]:
        """Calculate rotation, scale, and blur for given frame."""
        
        if self.spin_type == 'in':
            if frame_offset >= self.spin_duration:
                return (self.spin_degrees, 1.0, 0.0)
            
            progress = frame_offset / self.spin_duration
            eased_progress = self.ease_function(progress)
            
            rotation = self.spin_degrees * eased_progress
            
            # Calculate scale if enabled
            if self.scale_during_spin:
                # Start small, grow to full size
                scale = self.scale_factor + (1.0 - self.scale_factor) * eased_progress
            else:
                scale = 1.0
            
            # Calculate blur based on rotation speed
            if self.motion_blur:
                # Maximum blur at fastest rotation point
                speed = abs(self.spin_degrees / self.spin_duration)
                blur = min(1.0, speed / 30) * (1 - eased_progress)
            else:
                blur = 0.0
                
        elif self.spin_type == 'out':
            out_start = self.total_frames - self.spin_duration - self.start_frame
            
            if frame_offset < out_start:
                return (0, 1.0, 0.0)
            if frame_offset >= out_start + self.spin_duration:
                return (self.spin_degrees, self.scale_factor, 0.0)
            
            progress = (frame_offset - out_start) / self.spin_duration
            eased_progress = self.ease_function(progress)
            
            rotation = self.spin_degrees * eased_progress
            
            if self.scale_during_spin:
                # Shrink during spin out
                scale = 1.0 - (1.0 - self.scale_factor) * eased_progress
            else:
                scale = 1.0
            
            if self.motion_blur:
                speed = abs(self.spin_degrees / self.spin_duration)
                blur = min(1.0, speed / 30) * eased_progress
            else:
                blur = 0.0
                
        elif self.spin_type == 'continuous':
            total_frames = self.total_frames - self.start_frame
            progress = frame_offset / max(1, total_frames)
            
            rotation = self.spin_degrees * progress
            
            # Pulse scale if enabled
            if self.scale_during_spin:
                # Create a pulsing effect
                pulse = math.sin(progress * math.pi * 4)
                scale = 1.0 + (self.scale_factor - 1.0) * abs(pulse) * 0.5
            else:
                scale = 1.0
            
            blur = 0.0
        else:
            rotation = 0
            scale = 1.0
            blur = 0.0
        
        return (rotation, scale, blur)
    
    def process_frames(self) -> List[str]:
        """Process frames for spin animation."""
        print(f"   Processing spin animation...")
        print(f"   Degrees: {self.spin_degrees}")
        print(f"   Type: {self.spin_type}")
        print(f"   Direction: {self.spin_direction}")
        print(f"   Scale effect: {self.scale_during_spin}")
        
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
            rotation, scale, blur = self.calculate_spin_params(frame_offset)
            
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                rotated_frame = os.path.join(self.temp_dir, f'rotate_{frame_num:04d}.png')
                
                if self.create_rotated_frame(
                    element_frame,
                    rotation,
                    scale,
                    blur,
                    rotated_frame
                ):
                    if self.path:
                        current_position = self.get_position_at_frame(frame_num)
                    else:
                        current_position = self.position
                    
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    if self.composite_frame(
                        background_frame,
                        rotated_frame,
                        output_frame,
                        current_position
                    ):
                        output_frames.append(output_frame)
                        
                        if frame_num % 10 == 0:
                            print(f"      Frame {frame_num}: rotation {rotation:.1f}°, scale {scale:.2f}")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames