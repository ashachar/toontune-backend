"""
Stretch and Squash distortion animation.
Classic animation principle for elastic deformation.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .scale_transform import ScaleTransformAnimation


class StretchSquash(ScaleTransformAnimation):
    """
    Animation with stretch and squash deformation effects.
    
    Creates elastic, cartoon-like deformations commonly used in animation.
    
    Additional Parameters:
    ---------------------
    deform_type : str
        Type: 'stretch', 'squash', 'bounce', 'elastic', 'breathe' (default 'elastic')
    intensity : float
        Maximum deformation amount (0.1 to 2.0, default 0.5)
    frequency : float
        Oscillation frequency for cyclic deformations (default 1.0)
    axis : str
        Deformation axis: 'vertical', 'horizontal', 'both' (default 'vertical')
    preserve_volume : bool
        Maintain volume during deformation (default True)
    anticipation : bool
        Add anticipation squash before stretch (default True)
    follow_through : bool
        Add follow-through oscillation (default True)
    easing : str
        Easing function: 'linear', 'elastic', 'bounce' (default 'elastic')
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        deform_type: str = 'elastic',
        intensity: float = 0.5,
        frequency: float = 1.0,
        axis: str = 'vertical',
        preserve_volume: bool = True,
        anticipation: bool = True,
        follow_through: bool = True,
        easing: str = 'elastic',
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
            start_width=None,  # Will vary based on deformation
            start_height=None,  # Will vary based on deformation
            end_width=None,  # Will vary based on deformation
            end_height=None,  # Will vary based on deformation
            maintain_aspect_ratio=False,  # Stretch/squash intentionally distorts aspect ratio
            direction=direction,
            start_frame=start_frame,
            animation_start_frame=animation_start_frame,
            path=path,
            fps=fps,
            duration=duration,
            temp_dir=temp_dir
        )
        
        self.deform_type = deform_type.lower()
        self.intensity = max(0.1, min(2.0, intensity))
        self.frequency = max(0.1, frequency)
        self.axis = axis.lower()
        self.preserve_volume = preserve_volume
        self.anticipation = anticipation
        self.follow_through = follow_through
        self.easing = easing
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
    
    def calculate_deformation(self, time_offset: float) -> Tuple[float, float]:
        """Calculate stretch/squash deformation factors."""
        
        if self.deform_type == 'stretch':
            # Simple stretch
            phase = time_offset * self.frequency * 2 * math.pi
            stretch_factor = 1.0 + self.intensity * math.sin(phase)
            
            if self.preserve_volume:
                squash_factor = 1.0 / math.sqrt(stretch_factor)
            else:
                squash_factor = 1.0
                
        elif self.deform_type == 'squash':
            # Simple squash
            phase = time_offset * self.frequency * 2 * math.pi
            squash_factor = 1.0 - self.intensity * 0.5 * abs(math.sin(phase))
            
            if self.preserve_volume:
                stretch_factor = 1.0 / squash_factor
            else:
                stretch_factor = 1.0
                
        elif self.deform_type == 'bounce':
            # Bouncing ball effect
            phase = time_offset * self.frequency * 2 * math.pi
            impact = abs(math.sin(phase))
            
            if impact > 0.9:  # Near ground impact
                squash_factor = 0.7
                stretch_factor = 1.4
            else:
                # In air
                stretch_factor = 1.0 + self.intensity * 0.3 * (1 - impact)
                squash_factor = 1.0 / stretch_factor if self.preserve_volume else 1.0
                
        elif self.deform_type == 'elastic':
            # Elastic oscillation with damping
            phase = time_offset * self.frequency * 2 * math.pi
            damping = math.exp(-time_offset * 0.5)
            
            if self.anticipation and time_offset < 0.5:
                # Anticipation squash
                anticipation = math.sin(time_offset * 4 * math.pi) * 0.2
                stretch_factor = 1.0 - anticipation
            else:
                # Main elastic motion
                stretch_factor = 1.0 + self.intensity * math.sin(phase) * damping
            
            if self.preserve_volume:
                squash_factor = 1.0 / stretch_factor
            else:
                squash_factor = 1.0
                
        elif self.deform_type == 'breathe':
            # Breathing/pulsing effect
            phase = time_offset * self.frequency * math.pi
            breathe = (math.sin(phase) + 1) / 2  # 0 to 1
            
            scale = 1.0 + self.intensity * 0.2 * breathe
            stretch_factor = scale
            squash_factor = scale
            
        else:
            stretch_factor = 1.0
            squash_factor = 1.0
        
        # Apply axis constraints
        if self.axis == 'horizontal':
            return (stretch_factor, 1.0)
        elif self.axis == 'vertical':
            return (1.0, stretch_factor)
        else:  # both
            return (squash_factor, stretch_factor)
    
    def apply_deformation(
        self,
        element_frame: str,
        scale_x: float,
        scale_y: float,
        output_path: str
    ) -> bool:
        """Apply stretch/squash deformation to frame."""
        
        # Calculate new dimensions
        width = int(200 * scale_x)
        height = int(200 * scale_y)
        
        # Build filter
        filter_str = f'scale={width}:{height}'
        
        # Add follow-through wobble if enabled
        if self.follow_through and abs(scale_x - 1.0) < 0.1 and abs(scale_y - 1.0) < 0.1:
            # Small random wobble near rest position
            import random
            wobble = random.uniform(-0.02, 0.02)
            filter_str += f',rotate={wobble}:c=none'
        
        cmd = [
            'ffmpeg',
            '-i', element_frame,
            '-vf', filter_str,
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except:
            return False
    
    def process_frames(self) -> List[str]:
        """Process frames for stretch/squash animation."""
        print(f"   Processing stretch/squash animation...")
        print(f"   Type: {self.deform_type}")
        print(f"   Intensity: {self.intensity}")
        print(f"   Axis: {self.axis}")
        print(f"   Volume preservation: {self.preserve_volume}")
        
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
            
            # Calculate time offset
            frame_offset = frame_num - self.start_frame
            time_offset = frame_offset / self.fps
            
            # Calculate deformation
            scale_x, scale_y = self.calculate_deformation(time_offset)
            
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                deformed_frame = os.path.join(self.temp_dir, f'deform_{frame_num:04d}.png')
                
                if self.apply_deformation(element_frame, scale_x, scale_y, deformed_frame):
                    if self.path:
                        current_position = self.get_position_at_frame(frame_num)
                    else:
                        current_position = self.position
                    
                    # Adjust position for deformation center
                    # Keep bottom or center anchored depending on deformation
                    if self.deform_type == 'bounce' and scale_y < 1.0:
                        # Anchor to bottom during squash
                        y_offset = int((1.0 - scale_y) * 100)
                        current_position = (current_position[0], current_position[1] + y_offset)
                    
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    if self.composite_frame(
                        background_frame,
                        deformed_frame,
                        output_frame,
                        current_position
                    ):
                        output_frames.append(output_frame)
                        
                        if frame_num % 15 == 0:
                            print(f"      Frame {frame_num}: scale ({scale_x:.2f}, {scale_y:.2f})")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames