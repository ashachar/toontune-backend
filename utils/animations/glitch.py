"""
Glitch animation effect.
Digital glitch/distortion effects like screen interference.
"""

import os
import subprocess
import random
import math
from typing import List, Tuple, Optional
from .animate import Animation


class Glitch(Animation):
    """
    Animation with digital glitch effects.
    
    Creates various types of digital distortion including RGB shifts,
    scan lines, pixel sorting, and data corruption effects.
    
    Additional Parameters:
    ---------------------
    glitch_intensity : float
        Overall intensity of glitch effects (0.0 to 1.0, default 0.5)
    glitch_frequency : float
        How often glitches occur (0.0 to 1.0, default 0.3)
    glitch_duration : int
        Frames per glitch burst (default 3)
    rgb_shift : bool
        Enable RGB channel separation (default True)
    scan_lines : bool
        Add horizontal scan line distortion (default True)
    pixel_sort : bool
        Enable pixel sorting effect (default True)
    color_invert : bool
        Random color inversion (default True)
    displacement : bool
        Image displacement/offset (default True)
    noise : bool
        Add digital noise (default True)
    glitch_pattern : str
        'random', 'rhythmic', 'escalating' (default 'random')
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        glitch_intensity: float = 0.5,
        glitch_frequency: float = 0.3,
        glitch_duration: int = 3,
        rgb_shift: bool = True,
        scan_lines: bool = True,
        pixel_sort: bool = True,
        color_invert: bool = True,
        displacement: bool = True,
        noise: bool = True,
        glitch_pattern: str = 'random',
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
        
        self.glitch_intensity = max(0.0, min(1.0, glitch_intensity))
        self.glitch_frequency = max(0.0, min(1.0, glitch_frequency))
        self.glitch_duration = max(1, glitch_duration)
        self.rgb_shift = rgb_shift
        self.scan_lines = scan_lines
        self.pixel_sort = pixel_sort
        self.color_invert = color_invert
        self.displacement = displacement
        self.noise = noise
        self.glitch_pattern = glitch_pattern.lower()
        self.remove_background = remove_background
        self.background_color = background_color
        self.background_similarity = background_similarity
        
        self.scaled_frames = []
        self.clean_frames = []
        
        # Pre-calculate glitch timeline
        self.glitch_timeline = self.generate_glitch_timeline()
    
    def generate_glitch_timeline(self) -> List[dict]:
        """Generate timeline of when and what type of glitches occur."""
        timeline = []
        total_frames = self.total_frames
        
        if self.glitch_pattern == 'rhythmic':
            # Regular intervals
            interval = int(1.0 / self.glitch_frequency) if self.glitch_frequency > 0 else total_frames
            for frame in range(0, total_frames, interval):
                timeline.append({
                    'frame': frame,
                    'duration': self.glitch_duration,
                    'intensity': self.glitch_intensity,
                    'effects': self.random_effects()
                })
                
        elif self.glitch_pattern == 'escalating':
            # Increasing frequency and intensity
            frame = 0
            intensity = 0.1
            while frame < total_frames:
                timeline.append({
                    'frame': frame,
                    'duration': self.glitch_duration,
                    'intensity': min(1.0, intensity),
                    'effects': self.random_effects()
                })
                intensity += 0.1
                interval = max(5, int(30 * (1 - intensity)))
                frame += interval
                
        else:  # random
            # Random glitch occurrences
            frame = 0
            while frame < total_frames:
                if random.random() < self.glitch_frequency:
                    timeline.append({
                        'frame': frame,
                        'duration': random.randint(1, self.glitch_duration),
                        'intensity': random.uniform(0.3, self.glitch_intensity),
                        'effects': self.random_effects()
                    })
                frame += random.randint(5, 20)
        
        return timeline
    
    def random_effects(self) -> dict:
        """Generate random combination of glitch effects."""
        return {
            'rgb_shift': self.rgb_shift and random.random() > 0.3,
            'scan_lines': self.scan_lines and random.random() > 0.4,
            'pixel_sort': self.pixel_sort and random.random() > 0.6,
            'color_invert': self.color_invert and random.random() > 0.7,
            'displacement': self.displacement and random.random() > 0.5,
            'noise': self.noise and random.random() > 0.3
        }
    
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
    
    def apply_glitch_effects(
        self,
        element_frame: str,
        effects: dict,
        intensity: float,
        output_path: str
    ) -> bool:
        """Apply combination of glitch effects to frame."""
        
        filters = []
        
        # RGB channel shift
        if effects.get('rgb_shift', False):
            shift_amount = int(10 * intensity)
            # Split and shift RGB channels
            filters.append(f'split[r][g][b];[r]crop=iw-{shift_amount}:ih:0:0[r1];[g]crop=iw-{shift_amount}:ih:{shift_amount}:0[g1];[b]crop=iw-{shift_amount}:ih:{shift_amount*2}:0[b1];[r1][g1][b1]mergergb')
        
        # Scan lines
        if effects.get('scan_lines', False):
            line_height = max(1, int(5 * (1 - intensity)))
            filters.append(f'drawbox=x=0:y=0:w=iw:h=ih:color=black@0.3:t={line_height}')
        
        # Color inversion
        if effects.get('color_invert', False):
            filters.append('negate')
        
        # Displacement/offset
        if effects.get('displacement', False):
            x_offset = random.randint(-20, 20) * intensity
            y_offset = random.randint(-10, 10) * intensity
            filters.append(f'crop=iw:ih:{x_offset}:{y_offset}')
        
        # Digital noise
        if effects.get('noise', False):
            noise_amount = int(100 * intensity)
            filters.append(f'noise=alls={noise_amount}:allf=t')
        
        # Pixel sorting effect (simplified)
        if effects.get('pixel_sort', False):
            # Simulate with horizontal blur
            blur_amount = int(20 * intensity)
            filters.append(f'boxblur={blur_amount}:0')
        
        if not filters:
            # No effects, just copy
            subprocess.run(['cp', element_frame, output_path], capture_output=True)
            return True
        
        # Build filter chain
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
            # Fallback to original if glitch fails
            subprocess.run(['cp', element_frame, output_path], capture_output=True)
            return True
    
    def is_glitching(self, frame_num: int) -> Tuple[bool, dict, float]:
        """Check if current frame should have glitch effects."""
        for glitch in self.glitch_timeline:
            if glitch['frame'] <= frame_num < glitch['frame'] + glitch['duration']:
                return True, glitch['effects'], glitch['intensity']
        return False, {}, 0.0
    
    def process_frames(self) -> List[str]:
        """Process frames for glitch animation."""
        print(f"   Processing glitch animation...")
        print(f"   Pattern: {self.glitch_pattern}")
        print(f"   Intensity: {self.glitch_intensity}")
        print(f"   Frequency: {self.glitch_frequency}")
        print(f"   Glitch events: {len(self.glitch_timeline)}")
        
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
            
            # Check if glitching
            is_glitch, effects, intensity = self.is_glitching(frame_num)
            
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                if is_glitch:
                    # Apply glitch effects
                    glitched_frame = os.path.join(self.temp_dir, f'glitch_{frame_num:04d}.png')
                    self.apply_glitch_effects(element_frame, effects, intensity, glitched_frame)
                    final_element = glitched_frame
                else:
                    final_element = element_frame
                
                if self.path:
                    current_position = self.get_position_at_frame(frame_num)
                else:
                    current_position = self.position
                
                # Add random position jitter during glitch
                if is_glitch and effects.get('displacement', False):
                    jitter_x = random.randint(-5, 5) * intensity
                    jitter_y = random.randint(-5, 5) * intensity
                    current_position = (
                        current_position[0] + jitter_x,
                        current_position[1] + jitter_y
                    )
                
                output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                background_frame = self.background_frames[frame_num]
                
                if self.composite_frame(
                    background_frame,
                    final_element,
                    output_frame,
                    current_position
                ):
                    output_frames.append(output_frame)
                    
                    if is_glitch and frame_num % 10 == 0:
                        print(f"      Frame {frame_num}: GLITCH intensity {intensity:.2f}")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames