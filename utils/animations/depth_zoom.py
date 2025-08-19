"""
Depth Zoom animation.
Element moves along Z-axis with perspective depth effects.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .scale_transform import ScaleTransformAnimation


class DepthZoom(ScaleTransformAnimation):
    """
    Animation where element zooms along the Z-axis with depth effects.
    
    Creates perspective depth movement, like moving through space.
    
    Additional Parameters:
    ---------------------
    zoom_type : str
        Type: 'approach', 'recede', 'fly_through', 'dolly' (default 'approach')
    start_depth : float
        Starting Z position (-10 to 10, default -5)
    end_depth : float
        Ending Z position (-10 to 10, default 2)
    focal_length : float
        Camera focal length simulation (20 to 200, default 50)
    depth_blur : bool
        Apply depth of field blur (default True)
    motion_blur : bool
        Apply motion blur during fast movement (default True)
    parallax_layers : int
        Number of parallax background layers (0 to 5, default 0)
    fog_effect : bool
        Add atmospheric fog with distance (default True)
    camera_shake : bool
        Add camera shake during movement (default False)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        zoom_type: str = 'approach',
        start_depth: float = -5,
        end_depth: float = 2,
        focal_length: float = 50,
        depth_blur: bool = True,
        motion_blur: bool = True,
        parallax_layers: int = 0,
        fog_effect: bool = True,
        camera_shake: bool = False,
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
        
        self.zoom_type = zoom_type.lower()
        self.start_depth = max(-10, min(10, start_depth))
        self.end_depth = max(-10, min(10, end_depth))
        self.focal_length = max(20, min(200, focal_length))
        self.depth_blur = depth_blur
        self.motion_blur = motion_blur
        self.parallax_layers = max(0, min(5, parallax_layers))
        self.fog_effect = fog_effect
        self.camera_shake = camera_shake
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
    
    def calculate_depth_parameters(self, frame_offset: int) -> Tuple[float, float, float, float, Tuple[int, int]]:
        """Calculate depth-based parameters for current frame."""
        
        # Calculate progress through animation
        total_frames = self.total_frames - self.start_frame
        progress = min(1.0, frame_offset / max(1, total_frames - 30))
        
        # Apply easing based on zoom type
        if self.zoom_type == 'approach':
            # Ease out for approach
            eased_progress = 1 - (1 - progress) ** 2
        elif self.zoom_type == 'recede':
            # Ease in for receding
            eased_progress = progress ** 2
        elif self.zoom_type == 'fly_through':
            # S-curve for fly-through
            eased_progress = progress * progress * (3 - 2 * progress)
        else:  # dolly
            # Linear for dolly zoom
            eased_progress = progress
        
        # Calculate current depth
        current_depth = self.start_depth + (self.end_depth - self.start_depth) * eased_progress
        
        # Convert depth to scale using perspective projection
        # Simulate camera at z=0, object at z=current_depth
        # Scale = focal_length / (focal_length + depth * 100)
        if current_depth < -9.9:
            scale = 0.01  # Very far
        else:
            scale = self.focal_length / (self.focal_length + current_depth * 20)
        
        # Ensure reasonable scale limits
        scale = max(0.01, min(5.0, scale))
        
        # Calculate opacity based on fog effect
        if self.fog_effect:
            if current_depth < -3:
                # Fade in from distance
                opacity = max(0.2, 1.0 + (current_depth + 3) / 3)
            elif current_depth > 3:
                # Fade out when too close
                opacity = max(0.2, 1.0 - (current_depth - 3) / 3)
            else:
                opacity = 1.0
        else:
            opacity = 1.0
        
        # Calculate blur amount for depth of field
        if self.depth_blur:
            # Blur when far or very close
            if abs(current_depth) > 2:
                blur_amount = min(10, abs(current_depth) - 2)
            else:
                blur_amount = 0
        else:
            blur_amount = 0
        
        # Calculate motion blur based on speed
        if self.motion_blur and frame_offset > 0:
            depth_change = abs((self.end_depth - self.start_depth) / max(1, total_frames))
            motion_blur_amount = min(5, depth_change * 100)
        else:
            motion_blur_amount = 0
        
        # Calculate position with camera shake
        position = list(self.position)
        if self.camera_shake:
            import random
            shake_intensity = abs(current_depth - self.start_depth) * 2
            position[0] += random.randint(-int(shake_intensity), int(shake_intensity))
            position[1] += random.randint(-int(shake_intensity), int(shake_intensity))
        
        # Dolly zoom effect (counter-zoom)
        if self.zoom_type == 'dolly':
            # Keep size constant while changing perspective
            scale = 1.0
            # Adjust position to simulate dolly effect
            position[0] += int((current_depth - self.start_depth) * 10)
        
        return scale, opacity, blur_amount, motion_blur_amount, tuple(position)
    
    def apply_depth_effects(
        self,
        element_frame: str,
        scale: float,
        opacity: float,
        blur: float,
        motion_blur: float,
        output_path: str
    ) -> bool:
        """Apply depth-based effects to frame."""
        
        filters = []
        
        # Apply scale
        if scale != 1.0:
            scaled_size = int(200 * scale)
            if scaled_size > 0:
                filters.append(f'scale={scaled_size}:-1')
        
        # Apply depth blur
        if blur > 0:
            filters.append(f'gblur=sigma={blur}')
        
        # Apply motion blur
        if motion_blur > 0:
            filters.append(f'boxblur={int(motion_blur)}:1:1:1')
        
        # Apply opacity
        if opacity < 1.0:
            filters.append(f'format=rgba,colorchannelmixer=aa={opacity}')
        
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
            return False
    
    def process_frames(self) -> List[str]:
        """Process frames for depth zoom animation."""
        print(f"   Processing depth zoom animation...")
        print(f"   Type: {self.zoom_type}")
        print(f"   Depth: {self.start_depth} → {self.end_depth}")
        print(f"   Focal length: {self.focal_length}")
        
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
            
            # Calculate depth parameters
            scale, opacity, blur, motion_blur, position = self.calculate_depth_parameters(frame_offset)
            
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                # Apply depth effects
                depth_frame = os.path.join(self.temp_dir, f'depth_{frame_num:04d}.png')
                
                if self.apply_depth_effects(
                    element_frame,
                    scale,
                    opacity,
                    blur,
                    motion_blur,
                    depth_frame
                ):
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    # Add parallax layers if enabled
                    if self.parallax_layers > 0 and frame_offset > 0:
                        # Create simple parallax by shifting background
                        parallax_shift = int((frame_offset / 30) * 5)
                        parallax_bg = os.path.join(self.temp_dir, f'parallax_{frame_num}.png')
                        
                        cmd = [
                            'ffmpeg',
                            '-i', background_frame,
                            '-vf', f'crop=iw:ih:{parallax_shift}:0',
                            '-y',
                            parallax_bg
                        ]
                        
                        try:
                            subprocess.run(cmd, capture_output=True, text=True, check=True)
                            background_frame = parallax_bg
                        except:
                            pass
                    
                    if self.composite_frame(
                        background_frame,
                        depth_frame,
                        output_frame,
                        position
                    ):
                        output_frames.append(output_frame)
                        
                        if frame_num % 15 == 0:
                            current_depth = self.start_depth + (self.end_depth - self.start_depth) * (frame_offset / max(1, self.total_frames - self.start_frame - 30))
                            print(f"      Frame {frame_num}: depth {current_depth:.2f}, scale {scale:.2f}")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames