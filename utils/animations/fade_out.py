"""
Fade Out animation.
Element fades out at a specified center point with configurable speed.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .animate import Animation


class FadeOut(Animation):
    """
    Animation where element fades out at a specified center point.
    
    The element gradually decreases opacity from 1 to 0 at the specified
    location, while also advancing through its animation frames.
    
    Additional Parameters:
    ---------------------
    center_point : Tuple[int, int]
        The center point where the object should fade out (overrides position)
    fade_speed : float
        Opacity change per frame (0.0 to 1.0, default 0.05)
        - 0.05 = slow fade (20 frames to full transparency)
        - 0.1 = medium fade (10 frames to full transparency)
        - 0.2 = fast fade (5 frames to full transparency)
    fade_start_frame : int
        Frame at which to start fading out (relative to start_frame)
        Default is 0, meaning fade starts immediately when element appears
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
        center_point: Optional[Tuple[int, int]] = None,
        fade_speed: float = 0.05,
        fade_start_frame: int = 0,
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
        # Use center_point if provided, otherwise use position
        final_position = center_point if center_point is not None else position
        
        super().__init__(
            element_path=element_path,
            background_path=background_path,
            position=final_position,
            direction=direction,
            start_frame=start_frame,
            animation_start_frame=animation_start_frame,
            path=path,
            fps=fps,
            duration=duration,
            temp_dir=temp_dir
        )
        
        self.center_point = center_point if center_point is not None else position
        self.fade_speed = max(0.01, min(1.0, fade_speed))  # Clamp between 0.01 and 1.0
        self.fade_start_frame = max(0, fade_start_frame)
        self.remove_background = remove_background
        self.background_color = background_color
        self.background_similarity = background_similarity
        
        # Storage for processed frames
        self.scaled_frames = []
        self.clean_frames = []
    
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
    
    def apply_fade(
        self,
        element_frame: str,
        opacity: float,
        output_path: str
    ) -> bool:
        """
        Apply fade/opacity effect to an element frame.
        
        Parameters:
        -----------
        element_frame : str
            Path to the element frame
        opacity : float
            Opacity value (0.0 to 1.0)
        output_path : str
            Output path for the faded frame
        """
        
        # Use FFmpeg's format filter to set alpha channel opacity
        cmd = [
            'ffmpeg',
            '-i', element_frame,
            '-vf', f'format=rgba,colorchannelmixer=aa={opacity}',
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except:
            return False
    
    def process_frames(self) -> List[str]:
        """
        Process frames for fade-out animation.
        """
        print(f"   Processing fade-out animation...")
        print(f"   Center point: {self.center_point}")
        print(f"   Fade speed: {self.fade_speed} per frame")
        print(f"   Fade starts at frame: {self.start_frame + self.fade_start_frame}")
        
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
            
            # Calculate fade progress
            fade_trigger_frame = self.start_frame + self.fade_start_frame
            
            if frame_num < fade_trigger_frame:
                # Full opacity before fade starts
                current_opacity = 1.0
            else:
                # Calculate fading opacity
                fade_frame = frame_num - fade_trigger_frame
                current_opacity = max(0.0, 1.0 - (fade_frame * self.fade_speed))
            
            # Skip if completely faded out
            if current_opacity <= 0:
                # Just use background after element is fully faded
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
            
            # Get element frame
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                # Apply fade effect
                faded_frame = os.path.join(self.temp_dir, f'fade_{frame_num:04d}.png')
                
                if self.apply_fade(element_frame, current_opacity, faded_frame):
                    # Get position (use center_point, considering movement path if defined)
                    if self.path:
                        # If there's a path, use it but center around center_point
                        base_pos = self.get_position_at_frame(frame_num)
                    else:
                        # Static position at center_point
                        base_pos = self.center_point
                    
                    # Composite onto background
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    if self.composite_frame(
                        background_frame,
                        faded_frame,
                        output_frame,
                        base_pos
                    ):
                        output_frames.append(output_frame)
                        
                        # Progress indicator
                        if frame_num % 30 == 0:
                            print(f"      Frame {frame_num}: opacity {current_opacity:.2f}")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames