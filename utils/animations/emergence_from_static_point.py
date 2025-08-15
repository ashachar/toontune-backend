"""
Emergence from Static Point animation.
Element emerges pixel-by-pixel from a fixed point.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .animate import Animation


class EmergenceFromStaticPoint(Animation):
    """
    Animation where element emerges pixel-by-pixel from a static point.
    
    The element rises/emerges in the specified direction, revealing one
    pixel row at a time while also advancing through its animation frames.
    
    Additional Parameters:
    ---------------------
    emergence_speed : float
        Pixels per frame to reveal (default 1.0)
    remove_background : bool
        Whether to remove black background (default True)
    background_color : str
        Background color to remove (default '0x000000')
    background_similarity : float
        Similarity threshold for background removal (default 0.15)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        direction: float = 0,
        start_frame: int = 0,
        animation_start_frame: int = 0,
        path: Optional[List[Tuple[int, int, int]]] = None,
        fps: int = 30,
        duration: float = 7.0,
        temp_dir: Optional[str] = None,
        emergence_speed: float = 1.0,
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
        
        self.emergence_speed = emergence_speed
        self.remove_background = remove_background
        self.background_color = background_color
        self.background_similarity = background_similarity
        
        # Storage for processed frames
        self.scaled_frames = []
        self.clean_frames = []
        self.cropped_frames = []
    
    def extract_element_frames(self) -> List[str]:
        """Override to handle scaling and background removal."""
        print(f"   Extracting and preparing element frames...")
        print(f"   DEBUG: Element path: {self.element_path}")
        print(f"   DEBUG: Temp dir: {self.temp_dir}")
        
        # First extract raw frames
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
        except subprocess.TimeoutExpired:
            print(f"   ✗ Timeout extracting frames")
            return []
        except:
            print(f"   ✗ Failed to extract raw frames")
            return []
        
        # Scale frames to 200px wide (or configurable)
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
        
        # Remove background if requested
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
                    # If colorkey fails, use original
                    self.clean_frames.append(frame)
            
            print(f"   ✓ Removed background from frames")
            self.element_frames = self.clean_frames
        else:
            self.element_frames = self.scaled_frames
        
        return self.element_frames
    
    def create_cropped_frame(
        self,
        element_frame: str,
        pixels_to_show: int,
        direction: float,
        output_path: str
    ) -> bool:
        """
        Create a cropped version of element frame based on emergence direction.
        
        Direction determines which edge to reveal from:
        - 0° (up): Reveal from top
        - 90° (right): Reveal from right edge
        - 180° (down): Reveal from bottom
        - 270° (left): Reveal from left edge
        """
        
        # Get frame dimensions
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=s=x:p=0',
            element_frame
        ]
        
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            width, height = map(int, result.stdout.strip().split('x'))
        except:
            return False
        
        # Determine crop parameters based on direction
        # Normalize direction to 0-360
        dir_norm = direction % 360
        
        if 45 <= dir_norm < 135:  # Right
            # Reveal from right edge
            crop_w = min(pixels_to_show, width)
            crop_h = height
            crop_x = width - crop_w
            crop_y = 0
        elif 135 <= dir_norm < 225:  # Down
            # Reveal from bottom
            crop_w = width
            crop_h = min(pixels_to_show, height)
            crop_x = 0
            crop_y = height - crop_h
        elif 225 <= dir_norm < 315:  # Left
            # Reveal from left edge
            crop_w = min(pixels_to_show, width)
            crop_h = height
            crop_x = 0
            crop_y = 0
        else:  # Up (315-45, including 0)
            # Reveal from top
            crop_w = width
            crop_h = min(pixels_to_show, height)
            crop_x = 0
            crop_y = 0
        
        # Apply crop
        cmd = [
            'ffmpeg',
            '-i', element_frame,
            '-vf', f'crop={crop_w}:{crop_h}:{crop_x}:{crop_y}',
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except:
            return False
    
    def calculate_emergence_position(
        self,
        base_position: Tuple[int, int],
        pixels_revealed: int,
        direction: float
    ) -> Tuple[int, int]:
        """
        Calculate position for emerged element based on direction.
        
        The element moves opposite to reveal direction:
        - Revealing from top (0°) → element moves up
        - Revealing from right (90°) → element moves left
        - Revealing from bottom (180°) → element moves down
        - Revealing from left (270°) → element moves right
        """
        x, y = base_position
        
        # Convert direction to radians
        angle_rad = math.radians(direction)
        
        # Calculate movement (opposite to reveal direction)
        # For emergence, we move in the direction specified
        dx = math.sin(angle_rad) * pixels_revealed
        dy = -math.cos(angle_rad) * pixels_revealed  # Negative because y increases downward
        
        return (int(x + dx), int(y + dy))
    
    def process_frames(self) -> List[str]:
        """
        Process frames for pixel-by-pixel emergence.
        """
        print(f"   Processing emergence animation...")
        
        output_dir = os.path.join(self.temp_dir, "output_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        output_frames = []
        num_element_frames = len(self.element_frames)
        
        for frame_num in range(self.total_frames):
            # Skip frames before start_frame
            if frame_num < self.start_frame:
                # Just use background
                output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                if frame_num < len(self.background_frames):
                    subprocess.run(
                        ['cp', self.background_frames[frame_num], output_frame],
                        capture_output=True
                    )
                    output_frames.append(output_frame)
                continue
            
            # Calculate emergence progress
            emergence_frame = frame_num - self.start_frame
            pixels_to_reveal = int((emergence_frame + 1) * self.emergence_speed)
            
            # Get animation frame (with looping)
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            # Create cropped element for this frame
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                # Create cropped version
                cropped_frame = os.path.join(self.temp_dir, f'crop_{frame_num:04d}.png')
                
                if self.create_cropped_frame(
                    element_frame,
                    pixels_to_reveal,
                    self.direction,
                    cropped_frame
                ):
                    # Calculate position (considering movement path if defined)
                    base_pos = self.get_position_at_frame(frame_num)
                    
                    # Adjust position based on emergence
                    final_pos = self.calculate_emergence_position(
                        base_pos,
                        pixels_to_reveal,
                        self.direction
                    )
                    
                    # Composite onto background
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    if self.composite_frame(
                        background_frame,
                        cropped_frame,
                        output_frame,
                        final_pos
                    ):
                        output_frames.append(output_frame)
                        
                        # Progress indicator
                        if frame_num % 30 == 0:
                            print(f"      Frame {frame_num}: {pixels_to_reveal}px revealed")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames