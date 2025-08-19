"""
Lens Flare animation effect.
Camera lens flare and light streak effects.
"""

import os
import subprocess
import math
import random
from typing import List, Tuple, Optional
from .animate import Animation


class LensFlare(Animation):
    """
    Animation with lens flare and optical light effects.
    
    Creates realistic camera lens flares, light streaks, and optical artifacts.
    
    Additional Parameters:
    ---------------------
    flare_type : str
        Type: 'anamorphic', 'circular', 'starburst', 'streaks', 'bokeh' (default 'anamorphic')
    flare_position : Tuple[int, int]
        Light source position (default top-right)
    flare_intensity : float
        Brightness intensity (0.1 to 2.0, default 1.0)
    flare_color : str
        Primary flare color in hex (default '#FFFF88')
    num_artifacts : int
        Number of lens artifacts (1 to 10, default 5)
    movement : bool
        Animate flare movement (default True)
    rainbow_effect : bool
        Add chromatic aberration/rainbow (default True)
    bloom : bool
        Add bloom/glow effect (default True)
    flicker : bool
        Add realistic flicker (default False)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        flare_type: str = 'anamorphic',
        flare_position: Optional[Tuple[int, int]] = None,
        flare_intensity: float = 1.0,
        flare_color: str = '#FFFF88',
        num_artifacts: int = 5,
        movement: bool = True,
        rainbow_effect: bool = True,
        bloom: bool = True,
        flicker: bool = False,
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
        
        self.flare_type = flare_type.lower()
        self.flare_position = flare_position if flare_position else (150, 50)  # Top-right
        self.flare_intensity = max(0.1, min(2.0, flare_intensity))
        self.flare_color = flare_color
        self.num_artifacts = max(1, min(10, num_artifacts))
        self.movement = movement
        self.rainbow_effect = rainbow_effect
        self.bloom = bloom
        self.flicker = flicker
        self.remove_background = remove_background
        self.background_color = background_color
        self.background_similarity = background_similarity
        
        self.scaled_frames = []
        self.clean_frames = []
        self.artifacts = self.generate_artifacts()
    
    def generate_artifacts(self) -> List[dict]:
        """Generate lens flare artifacts configuration."""
        artifacts = []
        
        for i in range(self.num_artifacts):
            # Position artifacts along flare axis
            t = (i + 1) / (self.num_artifacts + 1)
            
            artifact = {
                'id': i,
                'position_ratio': t,  # Position along flare line
                'size': random.uniform(0.5, 2.0) * (1 - t * 0.5),  # Smaller farther away
                'opacity': random.uniform(0.3, 0.8) * (1 - t * 0.3),
                'color_shift': random.uniform(-30, 30) if self.rainbow_effect else 0,
                'shape': random.choice(['circle', 'hexagon', 'star']) if self.flare_type == 'bokeh' else 'circle'
            }
            
            artifacts.append(artifact)
        
        return artifacts
    
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
    
    def create_lens_flare_overlay(
        self,
        time_offset: float,
        frame_size: Tuple[int, int],
        output_path: str
    ) -> bool:
        """Create lens flare overlay frame."""
        
        width, height = frame_size
        
        # Calculate flare position (with movement if enabled)
        if self.movement:
            fx = self.flare_position[0] + math.sin(time_offset) * 20
            fy = self.flare_position[1] + math.cos(time_offset * 0.7) * 15
        else:
            fx, fy = self.flare_position
        
        # Calculate intensity (with flicker if enabled)
        intensity = self.flare_intensity
        if self.flicker:
            intensity *= (0.8 + 0.2 * random.random())
        
        # Create flare based on type
        filters = []
        
        if self.flare_type == 'anamorphic':
            # Horizontal streaks
            filters.append(f'drawbox=x=0:y={int(fy-2)}:w={width}:h=4:color=white@{intensity * 0.5}:t=fill')
            filters.append(f'boxblur=10:1')
            
        elif self.flare_type == 'starburst':
            # Star-shaped rays
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                x2 = fx + math.cos(rad) * width
                y2 = fy + math.sin(rad) * height
                # Simplified: draw lines as boxes
                filters.append(f'drawbox=x={int(fx-1)}:y={int(fy-1)}:w=2:h=2:color=white@{intensity}:t=fill')
            
        elif self.flare_type == 'circular':
            # Circular halos
            for r in [30, 60, 90]:
                opacity = intensity * (1 - r / 100)
                # Draw circle using multiple small boxes (simplified)
                filters.append(f'drawbox=x={int(fx-r/2)}:y={int(fy-r/2)}:w={r}:h={r}:color=white@{opacity * 0.3}:t=fill')
            
        elif self.flare_type == 'streaks':
            # Light streaks
            filters.append(f'drawbox=x={int(fx-100)}:y={int(fy-1)}:w=200:h=2:color=white@{intensity * 0.7}:t=fill')
            filters.append(f'drawbox=x={int(fx-1)}:y={int(fy-100)}:w=2:h=200:color=white@{intensity * 0.7}:t=fill')
            
        else:  # bokeh
            # Bokeh circles
            for artifact in self.artifacts[:3]:  # Limit for performance
                ax = fx + (width/2 - fx) * artifact['position_ratio']
                ay = fy + (height/2 - fy) * artifact['position_ratio']
                size = int(20 * artifact['size'])
                opacity = artifact['opacity'] * intensity
                filters.append(f'drawbox=x={int(ax-size/2)}:y={int(ay-size/2)}:w={size}:h={size}:color=white@{opacity}:t=fill')
        
        # Add bloom effect
        if self.bloom:
            filters.append(f'gblur=sigma={5 * intensity}')
        
        # Add color tint
        color_hex = self.flare_color.lstrip('#')
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0
        filters.append(f'colorbalance=rs={r-0.5}:gs={g-0.5}:bs={b-0.5}')
        
        # Create blank canvas and apply filters
        filter_chain = ','.join(filters) if filters else 'null'
        
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'color=c=black:s={width}x{height}:d=1',
            '-vf', filter_chain,
            '-frames:v', '1',
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except:
            # Create simple white glow fallback
            try:
                cmd = [
                    'ffmpeg',
                    '-f', 'lavfi',
                    '-i', f'color=c=white:s={width}x{height}:d=1',
                    '-vf', f'format=rgba,colorchannelmixer=aa={intensity * 0.3},gblur=sigma=10',
                    '-frames:v', '1',
                    '-y',
                    output_path
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                return True
            except:
                return False
    
    def process_frames(self) -> List[str]:
        """Process frames for lens flare animation."""
        print(f"   Processing lens flare animation...")
        print(f"   Type: {self.flare_type}")
        print(f"   Intensity: {self.flare_intensity}")
        print(f"   Position: {self.flare_position}")
        
        output_dir = os.path.join(self.temp_dir, "output_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        output_frames = []
        num_element_frames = len(self.element_frames)
        
        # Get frame dimensions
        frame_width = 1920
        frame_height = 1080
        if self.background_frames:
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
                frame_width, frame_height = map(int, result.stdout.strip().split('x'))
            except:
                pass
        
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
            time_offset = frame_offset / self.fps
            
            # Start with background
            if frame_num < len(self.background_frames):
                output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                subprocess.run(
                    ['cp', self.background_frames[frame_num], output_frame],
                    capture_output=True
                )
                
                # Add element if available
                if frame_num >= self.animation_start_frame and num_element_frames > 0:
                    anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
                    element_frame = self.element_frames[anim_frame_idx]
                    
                    temp_output = os.path.join(self.temp_dir, f'temp_elem_{frame_num}.png')
                    if self.path:
                        current_position = self.get_position_at_frame(frame_num)
                    else:
                        current_position = self.position
                    
                    self.composite_frame(
                        output_frame,
                        element_frame,
                        temp_output,
                        current_position
                    )
                    subprocess.run(['mv', temp_output, output_frame], capture_output=True)
                
                # Create and add lens flare overlay
                flare_overlay = os.path.join(self.temp_dir, f'flare_{frame_num:04d}.png')
                if self.create_lens_flare_overlay(time_offset, (frame_width, frame_height), flare_overlay):
                    # Composite flare using screen blend mode
                    temp_output = os.path.join(self.temp_dir, f'temp_flare_{frame_num}.png')
                    cmd = [
                        'ffmpeg',
                        '-i', output_frame,
                        '-i', flare_overlay,
                        '-filter_complex', '[0][1]blend=all_mode=screen',
                        '-y',
                        temp_output
                    ]
                    
                    try:
                        subprocess.run(cmd, capture_output=True, text=True, check=True)
                        subprocess.run(['mv', temp_output, output_frame], capture_output=True)
                    except:
                        pass
                
                output_frames.append(output_frame)
                
                if frame_num % 15 == 0:
                    print(f"      Frame {frame_num}: flare active")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames