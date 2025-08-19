"""
Flip animation.
Element flips like a card with 3D perspective.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .animate import Animation


class Flip(Animation):
    """
    Animation where element flips like a card.
    
    The element rotates around an axis with perspective transformation,
    creating a 3D card flip effect.
    
    Additional Parameters:
    ---------------------
    flip_axis : str
        Axis to flip around: 'horizontal', 'vertical', 'diagonal' (default 'horizontal')
    flip_duration : int
        Number of frames for flip animation (default 30)
    flip_direction : str
        Direction of flip: 'forward' or 'backward' (default 'forward')
    flip_type : str
        'in' for flip on entry, 'out' for flip on exit, 'continuous' (default 'in')
    num_flips : float
        Number of complete flips (default 1.0, can be 0.5 for half flip)
    perspective_strength : float
        Strength of 3D perspective effect (0.0 to 1.0, default 0.7)
    show_back : bool
        Whether to show back side during flip (default False)
    remove_background : bool
        Whether to remove black background (default True)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        flip_axis: str = 'horizontal',
        flip_duration: int = 30,
        flip_direction: str = 'forward',
        flip_type: str = 'in',
        num_flips: float = 1.0,
        perspective_strength: float = 0.7,
        show_back: bool = False,
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
        
        self.flip_axis = flip_axis.lower()
        self.flip_duration = max(10, flip_duration)
        self.flip_direction = flip_direction.lower()
        self.flip_type = flip_type.lower()
        self.num_flips = max(0.5, min(3.0, num_flips))
        self.perspective_strength = max(0.0, min(1.0, perspective_strength))
        self.show_back = show_back
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
    
    def create_flipped_frame(
        self,
        element_frame: str,
        flip_angle: float,
        output_path: str
    ) -> bool:
        """Create a flipped version with perspective transformation."""
        
        # Calculate perspective based on flip angle
        angle_rad = math.radians(flip_angle)
        
        if self.flip_axis == 'horizontal':
            # Horizontal flip (around Y axis)
            scale_x = abs(math.cos(angle_rad))
            
            # Apply perspective compression
            if self.perspective_strength > 0:
                scale_x = scale_x ** (1 + self.perspective_strength)
            
            # Check if showing back side
            if abs(flip_angle % 360) > 90 and abs(flip_angle % 360) < 270:
                if not self.show_back:
                    return False
                # Mirror the image for back side
                filter_str = f'scale=iw*{scale_x}:ih,hflip'
            else:
                filter_str = f'scale=iw*{scale_x}:ih'
                
        elif self.flip_axis == 'vertical':
            # Vertical flip (around X axis)
            scale_y = abs(math.cos(angle_rad))
            
            if self.perspective_strength > 0:
                scale_y = scale_y ** (1 + self.perspective_strength)
            
            if abs(flip_angle % 360) > 90 and abs(flip_angle % 360) < 270:
                if not self.show_back:
                    return False
                filter_str = f'scale=iw:ih*{scale_y},vflip'
            else:
                filter_str = f'scale=iw:ih*{scale_y}'
                
        elif self.flip_axis == 'diagonal':
            # Diagonal flip (around diagonal axis)
            scale = abs(math.cos(angle_rad))
            
            if self.perspective_strength > 0:
                scale = scale ** (1 + self.perspective_strength)
            
            rotation = flip_angle
            filter_str = f'scale=iw*{scale}:ih*{scale},rotate={rotation}*PI/180:c=none'
        else:
            filter_str = 'copy'
        
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
    
    def calculate_flip_angle(self, frame_offset: int) -> float:
        """Calculate flip angle for given frame."""
        if self.flip_type == 'in':
            if frame_offset >= self.flip_duration:
                return 0
            progress = frame_offset / self.flip_duration
            
            if self.flip_direction == 'backward':
                angle = 360 * self.num_flips * (1 - progress)
            else:
                angle = 360 * self.num_flips * progress
                
        elif self.flip_type == 'out':
            out_start = self.total_frames - self.flip_duration - self.start_frame
            if frame_offset < out_start:
                return 0
            if frame_offset >= out_start + self.flip_duration:
                return 360 * self.num_flips
                
            progress = (frame_offset - out_start) / self.flip_duration
            
            if self.flip_direction == 'backward':
                angle = 360 * self.num_flips * (1 - progress)
            else:
                angle = 360 * self.num_flips * progress
                
        elif self.flip_type == 'continuous':
            total_frames = self.total_frames - self.start_frame
            progress = frame_offset / max(1, total_frames)
            angle = 360 * self.num_flips * progress
        else:
            angle = 0
            
        return angle
    
    def process_frames(self) -> List[str]:
        """Process frames for flip animation."""
        print(f"   Processing flip animation...")
        print(f"   Axis: {self.flip_axis}")
        print(f"   Type: {self.flip_type}")
        print(f"   Flips: {self.num_flips}")
        print(f"   Perspective: {self.perspective_strength}")
        
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
            flip_angle = self.calculate_flip_angle(frame_offset)
            
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                flipped_frame = os.path.join(self.temp_dir, f'flip_{frame_num:04d}.png')
                
                if self.create_flipped_frame(element_frame, flip_angle, flipped_frame):
                    if self.path:
                        current_position = self.get_position_at_frame(frame_num)
                    else:
                        current_position = self.position
                    
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    if self.composite_frame(
                        background_frame,
                        flipped_frame,
                        output_frame,
                        current_position
                    ):
                        output_frames.append(output_frame)
                else:
                    # Element is not visible (back side without show_back)
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    if frame_num < len(self.background_frames):
                        subprocess.run(
                            ['cp', self.background_frames[frame_num], output_frame],
                            capture_output=True
                        )
                        output_frames.append(output_frame)
                
                if frame_num % 10 == 0:
                    print(f"      Frame {frame_num}: angle {flip_angle:.1f}°")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames