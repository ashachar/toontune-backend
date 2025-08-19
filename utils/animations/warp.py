"""
Warp distortion animation.
Flexible, elastic warping effects.
"""

import os
import subprocess
import math
import random
from typing import List, Tuple, Optional
from .animate import Animation


class Warp(Animation):
    """
    Animation with flexible warping and distortion effects.
    
    Creates rubber-like, liquid, or space-time warp distortions.
    
    Additional Parameters:
    ---------------------
    warp_type : str
        Type: 'rubber', 'liquid', 'twist', 'vortex', 'bulge', 'pinch' (default 'rubber')
    warp_intensity : float
        Intensity of warping (0.1 to 2.0, default 0.5)
    warp_center : Tuple[int, int]
        Center point for warp effect (default element center)
    warp_radius : float
        Radius of warp effect in pixels (default 100)
    oscillate : bool
        Oscillate warp effect (default True)
    frequency : float
        Oscillation frequency (default 1.0)
    randomize : bool
        Add random warping variations (default False)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        warp_type: str = 'rubber',
        warp_intensity: float = 0.5,
        warp_center: Optional[Tuple[int, int]] = None,
        warp_radius: float = 100,
        oscillate: bool = True,
        frequency: float = 1.0,
        randomize: bool = False,
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
        
        self.warp_type = warp_type.lower()
        self.warp_intensity = max(0.1, min(2.0, warp_intensity))
        self.warp_center = warp_center if warp_center else (100, 100)  # Center of 200px element
        self.warp_radius = max(10, warp_radius)
        self.oscillate = oscillate
        self.frequency = max(0.1, frequency)
        self.randomize = randomize
        self.remove_background = remove_background
        self.background_color = background_color
        self.background_similarity = background_similarity
        
        self.scaled_frames = []
        self.clean_frames = []
        self.warp_seed = random.randint(0, 1000)
    
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
    
    def apply_warp(
        self,
        element_frame: str,
        time_offset: float,
        output_path: str
    ) -> bool:
        """Apply warp distortion to frame."""
        
        # Calculate warp parameters
        if self.oscillate:
            phase = time_offset * self.frequency * 2 * math.pi
            intensity = self.warp_intensity * (0.5 + 0.5 * math.sin(phase))
        else:
            intensity = self.warp_intensity
        
        # Add randomization
        if self.randomize:
            intensity *= (0.8 + 0.4 * random.random())
        
        # Build warp filter based on type
        cx, cy = self.warp_center
        radius = self.warp_radius
        
        if self.warp_type == 'rubber':
            # Rubber sheet distortion
            filter_expr = f'geq=p(X + {intensity * 20} * sin(2*PI*Y/{radius}), Y + {intensity * 20} * sin(2*PI*X/{radius}))'
            
        elif self.warp_type == 'liquid':
            # Liquid ripple effect
            t = time_offset * 5
            filter_expr = f'geq=p(X + {intensity * 10} * sin(sqrt((X-{cx})^2 + (Y-{cy})^2)/{radius}*2*PI - {t}), Y)'
            
        elif self.warp_type == 'twist':
            # Twisting/spiral warp
            angle = intensity * math.pi
            filter_expr = f'geq=p((X-{cx})*cos({angle}*sqrt((X-{cx})^2+(Y-{cy})^2)/{radius})-(Y-{cy})*sin({angle}*sqrt((X-{cx})^2+(Y-{cy})^2)/{radius})+{cx}, (X-{cx})*sin({angle}*sqrt((X-{cx})^2+(Y-{cy})^2)/{radius})+(Y-{cy})*cos({angle}*sqrt((X-{cx})^2+(Y-{cy})^2)/{radius})+{cy})'
            
        elif self.warp_type == 'vortex':
            # Vortex/whirlpool effect
            spin = intensity * 5
            filter_expr = f'geq=p(X*cos({spin}*(1-sqrt((X-{cx})^2+(Y-{cy})^2)/{radius}))-Y*sin({spin}*(1-sqrt((X-{cx})^2+(Y-{cy})^2)/{radius})), X*sin({spin}*(1-sqrt((X-{cx})^2+(Y-{cy})^2)/{radius}))+Y*cos({spin}*(1-sqrt((X-{cx})^2+(Y-{cy})^2)/{radius})))'
            
        elif self.warp_type == 'bulge':
            # Bulge/fisheye effect
            factor = 1 + intensity
            filter_expr = f'lenscorrection=k1={factor}:k2={factor}'
            
        elif self.warp_type == 'pinch':
            # Pinch/implode effect
            factor = -intensity
            filter_expr = f'lenscorrection=k1={factor}:k2={factor}'
            
        else:
            # Default rubber warp
            filter_expr = f'geq=p(X + {intensity * 20} * sin(2*PI*Y/{radius}), Y + {intensity * 20} * sin(2*PI*X/{radius}))'
        
        cmd = [
            'ffmpeg',
            '-i', element_frame,
            '-vf', filter_expr,
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except:
            # Fallback to simpler distortion
            try:
                simple_filter = f'lenscorrection=k1={intensity * 0.5}:k2={intensity * 0.5}'
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
        """Process frames for warp animation."""
        print(f"   Processing warp animation...")
        print(f"   Type: {self.warp_type}")
        print(f"   Intensity: {self.warp_intensity}")
        print(f"   Radius: {self.warp_radius}px")
        
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
            time_offset = frame_offset / self.fps
            
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                warped_frame = os.path.join(self.temp_dir, f'warp_{frame_num:04d}.png')
                
                if self.apply_warp(element_frame, time_offset, warped_frame):
                    if self.path:
                        current_position = self.get_position_at_frame(frame_num)
                    else:
                        current_position = self.position
                    
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    if self.composite_frame(
                        background_frame,
                        warped_frame,
                        output_frame,
                        current_position
                    ):
                        output_frames.append(output_frame)
                        
                        if frame_num % 15 == 0:
                            print(f"      Frame {frame_num}: warp phase {time_offset:.2f}")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames