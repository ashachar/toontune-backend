"""
Neon Glow animation effect.
Creates fluorescent/neon glow effects around elements.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .animate import Animation


class NeonGlow(Animation):
    """
    Animation with neon/fluorescent glow effects.
    
    Creates glowing outlines and pulsing neon-like effects around elements.
    
    Additional Parameters:
    ---------------------
    glow_color : str
        Color of glow in hex format (default '#00FF00' - green)
    glow_intensity : float
        Intensity of glow effect (0.0 to 1.0, default 0.8)
    glow_radius : int
        Radius of glow in pixels (default 10)
    pulse : bool
        Enable pulsing glow effect (default True)
    pulse_speed : float
        Speed of pulsing (default 1.0)
    edge_detect : bool
        Apply glow only to edges (default True)
    multi_color : bool
        Enable multi-color rainbow glow (default False)
    flicker : bool
        Add realistic neon flicker (default True)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        glow_color: str = '#00FF00',
        glow_intensity: float = 0.8,
        glow_radius: int = 10,
        pulse: bool = True,
        pulse_speed: float = 1.0,
        edge_detect: bool = True,
        multi_color: bool = False,
        flicker: bool = True,
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
        
        self.glow_color = glow_color
        self.glow_intensity = max(0.0, min(1.0, glow_intensity))
        self.glow_radius = max(1, min(50, glow_radius))
        self.pulse = pulse
        self.pulse_speed = max(0.1, pulse_speed)
        self.edge_detect = edge_detect
        self.multi_color = multi_color
        self.flicker = flicker
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
    
    def apply_neon_glow(
        self,
        element_frame: str,
        time_offset: float,
        output_path: str
    ) -> bool:
        """Apply neon glow effect to frame."""
        
        # Calculate dynamic glow parameters
        if self.pulse:
            pulse_factor = 0.5 + 0.5 * math.sin(time_offset * self.pulse_speed * 2 * math.pi)
            current_intensity = self.glow_intensity * pulse_factor
            current_radius = int(self.glow_radius * (0.7 + 0.3 * pulse_factor))
        else:
            current_intensity = self.glow_intensity
            current_radius = self.glow_radius
        
        # Add flicker effect
        if self.flicker and time_offset > 0:
            import random
            if random.random() < 0.05:  # 5% chance of flicker
                current_intensity *= random.uniform(0.3, 0.7)
        
        # Convert hex color to RGB
        color_hex = self.glow_color.lstrip('#')
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0
        
        # Build filter chain
        filters = []
        
        if self.edge_detect:
            # Detect edges first
            filters.append('edgedetect=low=0.1:high=0.3:mode=colormix')
        
        # Apply glow using multiple passes of blur
        for i in range(3):
            blur_radius = current_radius * (i + 1) / 3
            filters.append(f'boxblur={blur_radius}:{blur_radius}')
        
        # Color the glow
        if self.multi_color:
            # Rainbow effect based on time
            hue_shift = (time_offset * 100) % 360
            filters.append(f'hue=h={hue_shift}:s=2')
        else:
            # Single color glow
            filters.append(f'colorbalance=rs={r}:gs={g}:bs={b}')
        
        # Adjust brightness for glow intensity
        brightness = 0.5 + current_intensity
        filters.append(f'eq=brightness={brightness}:saturation=2')
        
        # Combine filters
        filter_chain = ','.join(filters)
        
        # Create glow layer
        glow_frame = os.path.join(self.temp_dir, f'glow_layer_{time_offset:.2f}.png')
        cmd = [
            'ffmpeg',
            '-i', element_frame,
            '-vf', filter_chain,
            '-y',
            glow_frame
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Composite glow with original
            cmd = [
                'ffmpeg',
                '-i', element_frame,
                '-i', glow_frame,
                '-filter_complex', f'[1][0]overlay=0:0',
                '-y',
                output_path
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except:
            # Fallback to simpler glow
            try:
                simple_filter = f'gblur=sigma={current_radius}:steps=2,eq=brightness={brightness}'
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
        """Process frames for neon glow animation."""
        print(f"   Processing neon glow animation...")
        print(f"   Glow color: {self.glow_color}")
        print(f"   Intensity: {self.glow_intensity}")
        print(f"   Radius: {self.glow_radius}px")
        print(f"   Effects: {'Pulse' if self.pulse else ''} {'Flicker' if self.flicker else ''}")
        
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
            
            # Calculate time offset for effects
            frame_offset = frame_num - self.start_frame
            time_offset = frame_offset / self.fps
            
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                # Apply neon glow
                glowing_frame = os.path.join(self.temp_dir, f'neon_{frame_num:04d}.png')
                
                if self.apply_neon_glow(element_frame, time_offset, glowing_frame):
                    if self.path:
                        current_position = self.get_position_at_frame(frame_num)
                    else:
                        current_position = self.position
                    
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    if self.composite_frame(
                        background_frame,
                        glowing_frame,
                        output_frame,
                        current_position
                    ):
                        output_frames.append(output_frame)
                        
                        if frame_num % 15 == 0:
                            if self.pulse:
                                pulse_factor = 0.5 + 0.5 * math.sin(time_offset * self.pulse_speed * 2 * math.pi)
                                print(f"      Frame {frame_num}: pulse {pulse_factor:.2f}")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames