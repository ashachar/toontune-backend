"""
Skew distortion animation.
Element skews/tilts diagonally for dynamic effects.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .animate import Animation


class Skew(Animation):
    """
    Animation with skew/shear distortion effects.
    
    Creates diagonal tilting and perspective-like transformations.
    
    Additional Parameters:
    ---------------------
    skew_type : str
        Type: 'horizontal', 'vertical', 'both', 'perspective' (default 'horizontal')
    skew_angle : float
        Maximum skew angle in degrees (-45 to 45, default 30)
    oscillate : bool
        Oscillate skew back and forth (default True)
    frequency : float
        Oscillation frequency (default 1.0)
    easing : str
        Easing function: 'linear', 'sine', 'elastic' (default 'sine')
    anchor_point : str
        Anchor for skew: 'center', 'top', 'bottom', 'left', 'right' (default 'center')
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        skew_type: str = 'horizontal',
        skew_angle: float = 30,
        oscillate: bool = True,
        frequency: float = 1.0,
        easing: str = 'sine',
        anchor_point: str = 'center',
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
        
        self.skew_type = skew_type.lower()
        self.skew_angle = max(-45, min(45, skew_angle))
        self.oscillate = oscillate
        self.frequency = max(0.1, frequency)
        self.easing = easing
        self.anchor_point = anchor_point.lower()
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
    
    def calculate_skew(self, time_offset: float) -> Tuple[float, float]:
        """Calculate skew angles for current time."""
        
        # Calculate base angle
        if self.oscillate:
            if self.easing == 'sine':
                angle = self.skew_angle * math.sin(time_offset * self.frequency * 2 * math.pi)
            elif self.easing == 'elastic':
                t = time_offset * self.frequency
                angle = self.skew_angle * math.sin(t * 2 * math.pi) * math.exp(-t * 0.3)
            else:  # linear
                t = (time_offset * self.frequency) % 1.0
                if t < 0.5:
                    angle = self.skew_angle * (t * 4 - 1)
                else:
                    angle = self.skew_angle * (3 - t * 4)
        else:
            # One-way skew
            angle = self.skew_angle
        
        # Apply to appropriate axis
        if self.skew_type == 'horizontal':
            return (angle, 0)
        elif self.skew_type == 'vertical':
            return (0, angle)
        elif self.skew_type == 'both':
            return (angle, angle * 0.7)  # Different ratios for visual interest
        elif self.skew_type == 'perspective':
            # Simulate perspective with varying skew
            return (angle, -angle * 0.5)
        else:
            return (0, 0)
    
    def apply_skew(
        self,
        element_frame: str,
        skew_x: float,
        skew_y: float,
        output_path: str
    ) -> bool:
        """Apply skew transformation to frame."""
        
        # Convert skew angles to radians for transformation
        skew_x_rad = math.radians(skew_x)
        skew_y_rad = math.radians(skew_y)
        
        # Create perspective transformation matrix values
        # Using perspective filter for more control
        if abs(skew_x) > 0 or abs(skew_y) > 0:
            # Calculate transformation based on anchor point
            if self.anchor_point == 'top':
                y_offset = -50
            elif self.anchor_point == 'bottom':
                y_offset = 50
            elif self.anchor_point == 'left':
                x_offset = -50
            elif self.anchor_point == 'right':
                x_offset = 50
            else:
                x_offset = 0
                y_offset = 0
            
            # Build perspective transformation
            # Simplified approach using rotate and scale
            filters = []
            
            if abs(skew_x) > 0:
                # Horizontal skew effect
                filters.append(f'perspective=x0=0:y0=0:x1=w:y1={int(skew_x)}:x2=0:y2=h:x3=w:y3=h+{int(skew_x)}')
            
            if abs(skew_y) > 0:
                # Vertical skew effect (simplified with rotation)
                filters.append(f'rotate={skew_y * 0.5}*PI/180:c=none')
            
            if not filters:
                subprocess.run(['cp', element_frame, output_path], capture_output=True)
                return True
            
            filter_chain = ','.join(filters)
        else:
            # No skew
            subprocess.run(['cp', element_frame, output_path], capture_output=True)
            return True
        
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
            # Fallback to simpler transformation
            try:
                simple_filter = f'rotate={skew_x * 0.3}*PI/180:c=none'
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
                subprocess.run(['cp', element_frame, output_path], capture_output=True)
                return True
    
    def process_frames(self) -> List[str]:
        """Process frames for skew animation."""
        print(f"   Processing skew animation...")
        print(f"   Type: {self.skew_type}")
        print(f"   Angle: {self.skew_angle}°")
        print(f"   Oscillate: {self.oscillate}")
        
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
            
            # Calculate skew for this frame
            skew_x, skew_y = self.calculate_skew(time_offset)
            
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                skewed_frame = os.path.join(self.temp_dir, f'skew_{frame_num:04d}.png')
                
                if self.apply_skew(element_frame, skew_x, skew_y, skewed_frame):
                    if self.path:
                        current_position = self.get_position_at_frame(frame_num)
                    else:
                        current_position = self.position
                    
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    if self.composite_frame(
                        background_frame,
                        skewed_frame,
                        output_frame,
                        current_position
                    ):
                        output_frames.append(output_frame)
                        
                        if frame_num % 15 == 0:
                            print(f"      Frame {frame_num}: skew ({skew_x:.1f}°, {skew_y:.1f}°)")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames