"""
Submerge to Static Point animation.
Element submerges pixel-by-pixel into a fixed point (opposite of emergence).
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .animate import Animation


class SubmergeToStaticPoint(Animation):
    """
    Animation where element submerges pixel-by-pixel into a static point.
    
    The element disappears/submerges in the specified direction, hiding one
    pixel row at a time while also advancing through its animation frames.
    This is the opposite of EmergenceFromStaticPoint.
    
    Additional Parameters:
    ---------------------
    submerge_speed : float
        Pixels per frame to hide (default 1.0)
    submerge_start_frame : int
        Frame at which to start submerging (relative to start_frame)
        Default is 0, meaning submerge starts immediately when element appears
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
        submerge_start_frame: int = 0,
        path: Optional[List[Tuple[int, int, int]]] = None,
        fps: int = 30,
        duration: float = 7.0,
        temp_dir: Optional[str] = None,
        submerge_speed: float = 1.0,
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
        
        self.submerge_speed = submerge_speed
        self.submerge_start_frame = max(0, submerge_start_frame)
        self.remove_background = remove_background
        self.background_color = background_color
        self.background_similarity = background_similarity
        
        # Storage for processed frames
        self.scaled_frames = []
        self.clean_frames = []
        self.element_dimensions = None
    
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
        
        # Get dimensions of first scaled frame
        if self.scaled_frames:
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0',
                self.scaled_frames[0]
            ]
            
            try:
                result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                width, height = map(int, result.stdout.strip().split('x'))
                self.element_dimensions = (width, height)
                print(f"   Element dimensions: {width}x{height}")
            except:
                print(f"   ✗ Failed to get element dimensions")
        
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
        pixels_to_hide: int,
        direction: float,
        output_path: str
    ) -> bool:
        """
        Create a cropped version of element frame for submerging.
        
        Direction determines which edge to hide into:
        - 0° (up): Hide into top edge
        - 90° (right): Hide into right edge
        - 180° (down): Hide into bottom edge
        - 270° (left): Hide into left edge
        """
        
        if not self.element_dimensions:
            return False
        
        width, height = self.element_dimensions
        
        # Calculate how many pixels remain visible
        # Normalize direction to 0-360
        dir_norm = direction % 360
        
        if 45 <= dir_norm < 135:  # Right - hide from left
            # Hide into right edge, so crop from right side
            pixels_visible = max(0, width - pixels_to_hide)
            if pixels_visible <= 0:
                return False  # Fully hidden
            crop_w = pixels_visible
            crop_h = height
            crop_x = width - pixels_visible  # Start from right side
            crop_y = 0
        elif 135 <= dir_norm < 225:  # Down - hide from top
            # Hide into bottom edge, so crop from bottom
            pixels_visible = max(0, height - pixels_to_hide)
            if pixels_visible <= 0:
                return False  # Fully hidden
            crop_w = width
            crop_h = pixels_visible
            crop_x = 0
            crop_y = height - pixels_visible  # Start from bottom
        elif 225 <= dir_norm < 315:  # Left - hide from right
            # Hide into left edge, so crop from left side
            pixels_visible = max(0, width - pixels_to_hide)
            if pixels_visible <= 0:
                return False  # Fully hidden
            crop_w = pixels_visible
            crop_h = height
            crop_x = 0  # Start from left
            crop_y = 0
        else:  # Up (315-45, including 0) - hide from bottom
            # Hide into top edge, so crop from top
            pixels_visible = max(0, height - pixels_to_hide)
            if pixels_visible <= 0:
                return False  # Fully hidden
            crop_w = width
            crop_h = pixels_visible
            crop_x = 0
            crop_y = 0  # Start from top
        
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
    
    def calculate_submerge_position(
        self,
        base_position: Tuple[int, int],
        pixels_hidden: int,
        direction: float
    ) -> Tuple[int, int]:
        """
        Calculate position for submerging element based on direction.
        
        The visible part of the element moves toward the submerge point:
        - Submerging into top (0°) → visible part moves up
        - Submerging into right (90°) → visible part moves right
        - Submerging into bottom (180°) → visible part moves down
        - Submerging into left (270°) → visible part moves left
        """
        x, y = base_position
        
        # Convert direction to radians
        angle_rad = math.radians(direction)
        
        # Calculate movement toward submerge point
        # The visible part moves in the direction of submersion
        dx = math.sin(angle_rad) * pixels_hidden
        dy = -math.cos(angle_rad) * pixels_hidden  # Negative because y increases downward
        
        return (int(x + dx), int(y + dy))
    
    def process_frames(self) -> List[str]:
        """
        Process frames for pixel-by-pixel submerging.
        """
        print(f"   Processing submerge animation...")
        print(f"   Submerge direction: {self.direction}°")
        print(f"   Submerge starts at frame: {self.start_frame + self.submerge_start_frame}")
        
        output_dir = os.path.join(self.temp_dir, "output_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        output_frames = []
        num_element_frames = len(self.element_frames)
        
        # Calculate max pixels to hide based on dimensions
        if self.element_dimensions:
            width, height = self.element_dimensions
            dir_norm = self.direction % 360
            if 45 <= dir_norm < 135 or 225 <= dir_norm < 315:  # Horizontal
                max_pixels = width
            else:  # Vertical
                max_pixels = height
        else:
            max_pixels = 200  # Default fallback
        
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
            
            # Calculate submerge progress
            submerge_trigger_frame = self.start_frame + self.submerge_start_frame
            
            if frame_num < submerge_trigger_frame:
                # Full visibility before submerge starts
                pixels_to_hide = 0
            else:
                # Calculate submerging progress
                submerge_frame = frame_num - submerge_trigger_frame
                pixels_to_hide = int(submerge_frame * self.submerge_speed)
            
            # Check if fully submerged
            if pixels_to_hide >= max_pixels:
                # Just use background after element is fully submerged
                output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                if frame_num < len(self.background_frames):
                    subprocess.run(
                        ['cp', self.background_frames[frame_num], output_frame],
                        capture_output=True
                    )
                    output_frames.append(output_frame)
                continue
            
            # Get animation frame (with looping)
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            # Create cropped element for this frame
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                if pixels_to_hide == 0:
                    # Use full frame before submerging starts
                    cropped_frame = element_frame
                    base_pos = self.get_position_at_frame(frame_num)
                    final_pos = base_pos
                else:
                    # Create cropped version
                    cropped_frame = os.path.join(self.temp_dir, f'crop_{frame_num:04d}.png')
                    
                    if not self.create_cropped_frame(
                        element_frame,
                        pixels_to_hide,
                        self.direction,
                        cropped_frame
                    ):
                        # Fully hidden, use background only
                        output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                        subprocess.run(
                            ['cp', self.background_frames[frame_num], output_frame],
                            capture_output=True
                        )
                        output_frames.append(output_frame)
                        continue
                    
                    # Calculate position (considering movement path if defined)
                    base_pos = self.get_position_at_frame(frame_num)
                    
                    # Adjust position based on submerging
                    final_pos = self.calculate_submerge_position(
                        base_pos,
                        pixels_to_hide,
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
                        print(f"      Frame {frame_num}: {pixels_to_hide}px hidden")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames