"""
3D Rotation animation.
Element rotates in 3D space around X, Y, or Z axes.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .animate import Animation


class Rotate3D(Animation):
    """
    Animation with 3D rotation effects around multiple axes.
    
    Creates perspective 3D rotations with depth and lighting effects.
    
    Additional Parameters:
    ---------------------
    rotation_axis : str
        Axis: 'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz' (default 'y')
    rotation_speed : float
        Degrees per frame (default 2.0)
    perspective_distance : float
        Distance for perspective effect (100 to 1000, default 500)
    lighting : bool
        Apply lighting/shading based on rotation (default True)
    depth_blur : bool
        Apply depth-based blur (default True)
    wobble : bool
        Add slight wobble for organic feel (default False)
    rotation_direction : str
        'forward' or 'reverse' (default 'forward')
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        rotation_axis: str = 'y',
        rotation_speed: float = 2.0,
        perspective_distance: float = 500,
        lighting: bool = True,
        depth_blur: bool = True,
        wobble: bool = False,
        rotation_direction: str = 'forward',
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
        
        self.rotation_axis = rotation_axis.lower()
        self.rotation_speed = rotation_speed
        self.perspective_distance = max(100, min(1000, perspective_distance))
        self.lighting = lighting
        self.depth_blur = depth_blur
        self.wobble = wobble
        self.rotation_direction = rotation_direction.lower()
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
    
    def calculate_3d_rotation(self, frame_offset: int) -> Tuple[float, float, float, float]:
        """Calculate 3D rotation angles and perspective scale."""
        
        # Calculate base rotation
        if self.rotation_direction == 'reverse':
            angle = -frame_offset * self.rotation_speed
        else:
            angle = frame_offset * self.rotation_speed
        
        # Add wobble if enabled
        wobble_offset = 0
        if self.wobble:
            wobble_offset = math.sin(frame_offset * 0.1) * 5
        
        # Apply rotation to specified axes
        rot_x = 0
        rot_y = 0
        rot_z = 0
        
        if 'x' in self.rotation_axis:
            rot_x = angle + wobble_offset
        if 'y' in self.rotation_axis:
            rot_y = angle + wobble_offset * 0.7
        if 'z' in self.rotation_axis:
            rot_z = angle + wobble_offset * 0.5
        
        # Calculate perspective scale based on rotation
        # Simulates depth when rotating
        if 'y' in self.rotation_axis:
            # For Y rotation, scale based on viewing angle
            scale = 1.0 + 0.2 * math.cos(math.radians(rot_y))
        elif 'x' in self.rotation_axis:
            scale = 1.0 + 0.2 * math.cos(math.radians(rot_x))
        else:
            scale = 1.0
        
        return (rot_x, rot_y, rot_z, scale)
    
    def apply_3d_rotation(
        self,
        element_frame: str,
        rot_x: float,
        rot_y: float,
        rot_z: float,
        scale: float,
        output_path: str
    ) -> bool:
        """Apply 3D rotation transformation to frame."""
        
        filters = []
        
        # Simulate 3D rotation using 2D transformations
        # This is a simplified approach since FFmpeg doesn't have true 3D
        
        if abs(rot_y) > 0:
            # Y-axis rotation (horizontal flip effect)
            y_scale = abs(math.cos(math.radians(rot_y)))
            
            if y_scale < 0.01:
                # Edge-on view
                y_scale = 0.01
            
            # Check if we're viewing the back
            if (rot_y % 360) > 90 and (rot_y % 360) < 270:
                filters.append(f'scale=iw*{y_scale}:ih,hflip')
            else:
                filters.append(f'scale=iw*{y_scale}:ih')
            
            # Add perspective skew
            skew = math.sin(math.radians(rot_y)) * 20
            if abs(skew) > 0.5:
                filters.append(f'rotate={skew * 0.5}*PI/180:c=none')
        
        if abs(rot_x) > 0:
            # X-axis rotation (vertical flip effect)
            x_scale = abs(math.cos(math.radians(rot_x)))
            
            if x_scale < 0.01:
                x_scale = 0.01
            
            # Check if we're viewing from below
            if (rot_x % 360) > 90 and (rot_x % 360) < 270:
                filters.append(f'scale=iw:ih*{x_scale},vflip')
            else:
                filters.append(f'scale=iw:ih*{x_scale}')
        
        if abs(rot_z) > 0:
            # Z-axis rotation (standard 2D rotation)
            filters.append(f'rotate={rot_z}*PI/180:c=none')
        
        # Apply overall scale
        if scale != 1.0:
            filters.append(f'scale=iw*{scale}:ih*{scale}')
        
        # Apply lighting effect
        if self.lighting:
            # Simulate lighting based on rotation angle
            brightness_y = 0.8 + 0.2 * math.cos(math.radians(rot_y))
            brightness_x = 0.8 + 0.2 * math.cos(math.radians(rot_x))
            brightness = (brightness_y + brightness_x) / 2
            
            filters.append(f'eq=brightness={brightness - 1}')
        
        # Apply depth blur
        if self.depth_blur:
            # Blur more when rotated away
            blur_amount = 0
            if 'y' in self.rotation_axis:
                blur_amount = abs(math.sin(math.radians(rot_y))) * 2
            if 'x' in self.rotation_axis:
                blur_amount = max(blur_amount, abs(math.sin(math.radians(rot_x))) * 2)
            
            if blur_amount > 0.1:
                filters.append(f'gblur=sigma={blur_amount}')
        
        if not filters:
            subprocess.run(['cp', element_frame, output_path], capture_output=True)
            return True
        
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
            # Fallback to simple rotation
            try:
                simple_filter = f'rotate={rot_z}*PI/180:c=none'
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
        """Process frames for 3D rotation animation."""
        print(f"   Processing 3D rotation animation...")
        print(f"   Axes: {self.rotation_axis}")
        print(f"   Speed: {self.rotation_speed}°/frame")
        print(f"   Lighting: {self.lighting}")
        print(f"   Depth blur: {self.depth_blur}")
        
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
            
            # Calculate 3D rotation
            rot_x, rot_y, rot_z, scale = self.calculate_3d_rotation(frame_offset)
            
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                rotated_frame = os.path.join(self.temp_dir, f'rot3d_{frame_num:04d}.png')
                
                if self.apply_3d_rotation(element_frame, rot_x, rot_y, rot_z, scale, rotated_frame):
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
                        
                        if frame_num % 15 == 0:
                            print(f"      Frame {frame_num}: rotation ({rot_x:.1f}°, {rot_y:.1f}°, {rot_z:.1f}°)")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames