"""
Zoom Out animation.
Element zooms/scales out of view to a point.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .scale_transform import ScaleTransformAnimation


class ZoomOut(ScaleTransformAnimation):
    """
    Animation where element zooms out of view to a point.
    
    The element starts at full size and scales down to very small or invisible
    at the specified position with customizable speed and easing.
    
    Additional Parameters:
    ---------------------
    start_scale : float
        Initial scale factor (default 1.0)
    end_scale : float
        Final scale factor (0.0 to 1.0, default 0.0)
    zoom_duration : int
        Number of frames for zoom animation (default 20)
    zoom_start_frame : int
        Frame at which to start zooming out (relative to start_frame)
    zoom_center : Tuple[int, int]
        Center point for zoom effect (default uses position)
    easing : str
        Easing function: 'linear', 'ease_in', 'ease_out', 'ease_in_out' (default 'ease_in')
    rotation_during_zoom : float
        Degrees to rotate during zoom (default 0)
    remove_background : bool
        Whether to remove black background (default True)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        start_scale: float = 1.0,
        end_scale: float = 0.0,
        zoom_duration: int = 20,
        zoom_start_frame: int = 0,
        zoom_center: Optional[Tuple[int, int]] = None,
        easing: str = 'ease_in',
        rotation_during_zoom: float = 0,
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
        
        self.start_scale = max(0.1, min(2.0, start_scale))
        self.end_scale = max(0.0, min(2.0, end_scale))
        self.zoom_duration = max(1, zoom_duration)
        self.zoom_start_frame = max(0, zoom_start_frame)
        self.zoom_center = zoom_center if zoom_center else position
        self.easing = easing
        self.rotation_during_zoom = rotation_during_zoom
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
    
    def create_scaled_frame(
        self,
        element_frame: str,
        scale: float,
        rotation: float,
        output_path: str
    ) -> bool:
        """Create a scaled and rotated version of the element frame."""
        
        filters = []
        
        scaled_size = int(200 * scale)
        if scaled_size > 0:
            filters.append(f'scale={scaled_size}:-1')
        else:
            return False
        
        if rotation != 0:
            filters.append(f'rotate={rotation}*PI/180:c=none')
        
        filter_chain = ','.join(filters) if filters else 'copy'
        
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
        """Process frames for zoom-out animation."""
        print(f"   Processing zoom-out animation...")
        print(f"   Scale: {self.start_scale} → {self.end_scale}")
        print(f"   Duration: {self.zoom_duration} frames")
        print(f"   Start at frame: {self.zoom_start_frame}")
        
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
            zoom_trigger = self.zoom_start_frame
            
            if frame_offset < zoom_trigger:
                current_scale = self.start_scale
                current_rotation = 0
            elif frame_offset < zoom_trigger + self.zoom_duration:
                zoom_progress = (frame_offset - zoom_trigger) / self.zoom_duration
                eased_progress = self.ease_function(zoom_progress)
                
                current_scale = self.start_scale + (self.end_scale - self.start_scale) * eased_progress
                current_rotation = self.rotation_during_zoom * eased_progress
            else:
                current_scale = self.end_scale
                current_rotation = self.rotation_during_zoom
            
            if current_scale < 0.01:
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
                scaled_frame = os.path.join(self.temp_dir, f'scaled_{frame_num:04d}.png')
                
                if self.create_scaled_frame(
                    element_frame,
                    current_scale,
                    current_rotation,
                    scaled_frame
                ):
                    if self.path:
                        current_position = self.get_position_at_frame(frame_num)
                    else:
                        current_position = self.zoom_center
                    
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    if self.composite_frame(
                        background_frame,
                        scaled_frame,
                        output_frame,
                        current_position
                    ):
                        output_frames.append(output_frame)
                        
                        if frame_num % 10 == 0:
                            print(f"      Frame {frame_num}: scale {current_scale:.2f}")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames