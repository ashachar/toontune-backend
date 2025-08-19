"""
Slide Out animation.
Element slides out of view to specified direction.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .animate import Animation


class SlideOut(Animation):
    """
    Animation where element slides out of view to off-screen.
    
    The element starts at target position and slides out to off-screen
    with customizable speed and easing.
    
    Additional Parameters:
    ---------------------
    slide_direction : str
        Direction to slide to: 'left', 'right', 'top', 'bottom' (default 'right')
    slide_duration : int
        Number of frames for the slide animation (default 30)
    slide_start_frame : int
        Frame at which to start sliding out (relative to start_frame)
    easing : str
        Easing function: 'linear', 'ease_in', 'ease_out', 'ease_in_out' (default 'ease_in')
    remove_background : bool
        Whether to remove black background (default True)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        slide_direction: str = 'right',
        slide_duration: int = 30,
        slide_start_frame: int = 0,
        easing: str = 'ease_in',
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
        
        self.slide_direction = slide_direction.lower()
        self.slide_duration = max(1, slide_duration)
        self.slide_start_frame = max(0, slide_start_frame)
        self.easing = easing
        self.remove_background = remove_background
        self.background_color = background_color
        self.background_similarity = background_similarity
        
        self.scaled_frames = []
        self.clean_frames = []
        self.start_position = position
        self.end_position = None
    
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
    
    def calculate_end_position(self) -> Tuple[int, int]:
        """Calculate off-screen ending position based on slide direction."""
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
        
        start_x, start_y = self.start_position
        element_size = 200
        
        if self.slide_direction == 'left':
            return (-element_size, start_y)
        elif self.slide_direction == 'right':
            return (video_width + element_size, start_y)
        elif self.slide_direction == 'top':
            return (start_x, -element_size)
        elif self.slide_direction == 'bottom':
            return (start_x, video_height + element_size)
        else:
            return self.start_position
    
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
    
    def calculate_slide_position(self, frame_offset: int) -> Tuple[int, int]:
        """Calculate position during slide animation."""
        slide_trigger = self.slide_start_frame
        
        if frame_offset < slide_trigger:
            return self.start_position
        
        slide_progress = frame_offset - slide_trigger
        if slide_progress >= self.slide_duration:
            return self.end_position
        
        progress = slide_progress / self.slide_duration
        eased_progress = self.ease_function(progress)
        
        start_x, start_y = self.start_position
        end_x, end_y = self.end_position
        
        current_x = start_x + (end_x - start_x) * eased_progress
        current_y = start_y + (end_y - start_y) * eased_progress
        
        return (int(current_x), int(current_y))
    
    def process_frames(self) -> List[str]:
        """Process frames for slide-out animation."""
        print(f"   Processing slide-out animation...")
        print(f"   Slide direction: {self.slide_direction}")
        print(f"   Duration: {self.slide_duration} frames")
        print(f"   Start at frame: {self.slide_start_frame}")
        
        self.end_position = self.calculate_end_position()
        print(f"   Start position: {self.start_position}")
        print(f"   End position: {self.end_position}")
        
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
            current_position = self.calculate_slide_position(frame_offset)
            
            # Check if element has slid completely off-screen
            if frame_offset >= self.slide_start_frame + self.slide_duration:
                output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                if frame_num < len(self.background_frames):
                    subprocess.run(
                        ['cp', self.background_frames[frame_num], output_frame],
                        capture_output=True
                    )
                    output_frames.append(output_frame)
                continue
            
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                background_frame = self.background_frames[frame_num]
                
                if self.composite_frame(
                    background_frame,
                    element_frame,
                    output_frame,
                    current_position
                ):
                    output_frames.append(output_frame)
                    
                    if frame_num % 15 == 0:
                        print(f"      Frame {frame_num}: position {current_position}")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames