"""
Base Animation class for all animation types.
"""

import os
import subprocess
import tempfile
import shutil
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import math


class Animation(ABC):
    """
    Base class for all animations.
    
    Parameters:
    -----------
    element_path : str
        Path to the element (image/video) to animate
    background_path : str
        Path to the background video/image
    position : Tuple[int, int]
        Starting (x, y) position for the element
    direction : float
        Direction in degrees (0=up, 90=right, 180=down, 270=left)
    start_frame : int
        Frame index where element should first appear
    animation_start_frame : int
        Frame where element starts animating (default 0)
    path : List[Tuple[int, int, int]]
        List of (frame, x, y) keypoints for movement path
    fps : int
        Frames per second (default 30)
    duration : float
        Total duration in seconds
    temp_dir : str
        Temporary directory for processing (auto-created if None)
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
        temp_dir: Optional[str] = None
    ):
        # Validate inputs
        if not os.path.exists(element_path):
            raise FileNotFoundError(f"Element not found: {element_path}")
        if not os.path.exists(background_path):
            raise FileNotFoundError(f"Background not found: {background_path}")
        
        self.element_path = element_path
        self.background_path = background_path
        self.position = position
        self.direction = direction
        self.start_frame = start_frame
        self.animation_start_frame = animation_start_frame
        self.path = path or []
        self.fps = fps
        self.duration = duration
        self.total_frames = int(fps * duration)
        
        # Create temp directory
        if temp_dir:
            self.temp_dir = temp_dir
            self.cleanup_temp = False
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="animation_")
            self.cleanup_temp = True
        
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Storage for extracted frames
        self.element_frames = []
        self.background_frames = []
        self.output_frames = []
    
    def extract_element_frames(self) -> List[str]:
        """Extract and prepare frames from element."""
        print(f"   Extracting frames from element...")
        
        element_dir = os.path.join(self.temp_dir, "element_frames")
        os.makedirs(element_dir, exist_ok=True)
        
        # Extract frames from element
        cmd = [
            'ffmpeg',
            '-i', self.element_path,
            '-r', str(self.fps),
            os.path.join(element_dir, 'element_%04d.png')
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Get list of extracted frames
            frames = sorted([
                os.path.join(element_dir, f) 
                for f in os.listdir(element_dir) 
                if f.endswith('.png')
            ])
            
            print(f"   ✓ Extracted {len(frames)} element frames")
            self.element_frames = frames
            return frames
            
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed to extract element frames")
            return []
    
    def extract_background_frames(self) -> List[str]:
        """Extract frames from background."""
        print(f"   Extracting frames from background...")
        
        bg_dir = os.path.join(self.temp_dir, "background_frames")
        os.makedirs(bg_dir, exist_ok=True)
        
        # Check if we need to loop the background
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            self.background_path
        ]
        
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            bg_duration = float(result.stdout.strip())
            
            # If background is shorter than needed, loop it
            if bg_duration < self.duration:
                print(f"   Background is {bg_duration}s, looping to {self.duration}s")
                looped_bg = os.path.join(self.temp_dir, "background_looped.mp4")
                
                loop_count = int(math.ceil(self.duration / bg_duration))
                loop_cmd = [
                    'ffmpeg',
                    '-stream_loop', str(loop_count),
                    '-i', self.background_path,
                    '-t', str(self.duration),
                    '-c', 'copy',
                    '-y',
                    looped_bg
                ]
                
                subprocess.run(loop_cmd, capture_output=True, text=True, check=True)
                bg_source = looped_bg
            else:
                bg_source = self.background_path
            
            # Extract frames
            cmd = [
                'ffmpeg',
                '-i', bg_source,
                '-r', str(self.fps),
                '-frames:v', str(self.total_frames),
                os.path.join(bg_dir, 'bg_%04d.png')
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            frames = sorted([
                os.path.join(bg_dir, f)
                for f in os.listdir(bg_dir)
                if f.endswith('.png')
            ])
            
            print(f"   ✓ Extracted {len(frames)} background frames")
            self.background_frames = frames
            return frames
            
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed to extract background frames")
            return []
    
    def apply_rotation(self, frame_path: str, angle: float, output_path: str) -> bool:
        """Apply rotation to a frame."""
        if angle == 0:
            # No rotation needed
            if frame_path != output_path:
                shutil.copy(frame_path, output_path)
            return True
        
        cmd = [
            'ffmpeg',
            '-i', frame_path,
            '-vf', f'rotate={math.radians(angle)}:c=none',
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except:
            return False
    
    def get_position_at_frame(self, frame_num: int) -> Tuple[int, int]:
        """
        Calculate position at given frame based on path keypoints.
        Interpolates between keypoints.
        """
        if not self.path:
            # No path defined, use static position
            return self.position
        
        # Find surrounding keypoints
        prev_kp = None
        next_kp = None
        
        for kp in self.path:
            kp_frame, kp_x, kp_y = kp
            
            if kp_frame <= frame_num:
                prev_kp = kp
            if kp_frame >= frame_num and next_kp is None:
                next_kp = kp
        
        # Handle edge cases
        if prev_kp is None:
            # Before first keypoint, use starting position
            return self.position
        
        if next_kp is None:
            # After last keypoint, use last position
            return (prev_kp[1], prev_kp[2])
        
        if prev_kp == next_kp:
            # Exactly on a keypoint
            return (prev_kp[1], prev_kp[2])
        
        # Interpolate between keypoints
        prev_frame, prev_x, prev_y = prev_kp
        next_frame, next_x, next_y = next_kp
        
        # Calculate interpolation factor
        t = (frame_num - prev_frame) / (next_frame - prev_frame)
        
        # Linear interpolation
        x = int(prev_x + (next_x - prev_x) * t)
        y = int(prev_y + (next_y - prev_y) * t)
        
        return (x, y)
    
    def composite_frame(
        self,
        background_frame: str,
        element_frame: str,
        output_frame: str,
        position: Tuple[int, int],
        **kwargs
    ) -> bool:
        """
        Composite element frame onto background at position.
        Can be overridden by subclasses for custom compositing.
        """
        x, y = position
        
        cmd = [
            'ffmpeg',
            '-i', background_frame,
            '-i', element_frame,
            '-filter_complex',
            f'[0:v][1:v]overlay={x}:{y}',
            '-y',
            output_frame
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except:
            return False
    
    def create_video_from_frames(self, output_path: str) -> bool:
        """Create final video from output frames."""
        print(f"   Creating final video...")
        
        if not self.output_frames:
            print(f"   ✗ No output frames to combine")
            return False
        
        # Create input file list for FFmpeg
        list_file = os.path.join(self.temp_dir, "frames.txt")
        with open(list_file, 'w') as f:
            for frame in self.output_frames:
                f.write(f"file '{frame}'\n")
                f.write(f"duration {1.0/self.fps}\n")
            # Add last frame again without duration
            f.write(f"file '{self.output_frames[-1]}'\n")
        
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   ✓ Video created: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed to create video")
            return False
    
    @abstractmethod
    def process_frames(self) -> List[str]:
        """
        Process frames according to animation type.
        Must be implemented by subclasses.
        Returns list of output frame paths.
        """
        pass
    
    def render(self, output_path: str) -> bool:
        """
        Main render method to create the animation.
        """
        print(f"\n🎬 Rendering {self.__class__.__name__} animation...")
        
        try:
            # Extract frames
            if not self.extract_element_frames():
                return False
            
            if not self.extract_background_frames():
                return False
            
            # Process frames according to animation type
            self.output_frames = self.process_frames()
            
            if not self.output_frames:
                print("   ✗ No frames processed")
                return False
            
            # Create final video
            success = self.create_video_from_frames(output_path)
            
            # Cleanup if needed
            if self.cleanup_temp and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"   🧹 Cleaned up temp directory")
            
            return success
            
        except Exception as e:
            print(f"   ✗ Render failed: {str(e)}")
            return False
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'cleanup_temp') and self.cleanup_temp:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                except:
                    pass