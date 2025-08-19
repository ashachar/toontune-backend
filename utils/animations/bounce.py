"""
Bounce animation.
Element bounces when entering or exiting.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .animate import Animation


class Bounce(Animation):
    """
    Animation where element bounces into position.
    
    The element drops from above and bounces several times before settling,
    or can bounce out when exiting.
    
    Additional Parameters:
    ---------------------
    bounce_height : float
        Initial bounce height in pixels (default 200)
    num_bounces : int
        Number of bounces (default 3)
    bounce_duration : int
        Total frames for bounce animation (default 45)
    bounce_type : str
        'in' for bounce on entry, 'out' for bounce on exit, 'both' (default 'in')
    gravity : float
        Gravity strength affecting bounce physics (default 0.8)
    damping : float
        Energy loss per bounce (0.0 to 1.0, default 0.6)
    squash_stretch : bool
        Apply squash and stretch effect (default True)
    remove_background : bool
        Whether to remove black background (default True)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        bounce_height: float = 200,
        num_bounces: int = 3,
        bounce_duration: int = 45,
        bounce_type: str = 'in',
        gravity: float = 0.8,
        damping: float = 0.6,
        squash_stretch: bool = True,
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
        
        self.bounce_height = max(10, bounce_height)
        self.num_bounces = max(1, min(10, num_bounces))
        self.bounce_duration = max(10, bounce_duration)
        self.bounce_type = bounce_type.lower()
        self.gravity = max(0.1, min(2.0, gravity))
        self.damping = max(0.1, min(0.9, damping))
        self.squash_stretch = squash_stretch
        self.remove_background = remove_background
        self.background_color = background_color
        self.background_similarity = background_similarity
        
        # Storage for processed frames
        self.scaled_frames = []
        self.clean_frames = []
        
        # Physics calculations
        self.bounce_keyframes = []
        self.calculate_bounce_physics()
    
    def calculate_bounce_physics(self):
        """Pre-calculate bounce keyframes using physics simulation."""
        keyframes = []
        current_height = self.bounce_height
        current_velocity = 0
        frame = 0
        bounce_count = 0
        
        while bounce_count < self.num_bounces and frame < self.bounce_duration:
            # Apply gravity
            current_velocity += self.gravity
            current_height -= current_velocity
            
            # Check for ground collision
            if current_height <= 0:
                current_height = 0
                current_velocity = -current_velocity * self.damping
                bounce_count += 1
            
            keyframes.append({
                'frame': frame,
                'height': max(0, current_height),
                'velocity': current_velocity,
                'bounce_num': bounce_count
            })
            
            frame += 1
            
            # Stop if velocity is too small
            if abs(current_velocity) < 0.5 and current_height < 1:
                break
        
        self.bounce_keyframes = keyframes
    
    def extract_element_frames(self) -> List[str]:
        """Override to handle scaling and background removal."""
        print(f"   Extracting and preparing element frames...")
        
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
        except:
            print(f"   ✗ Failed to extract raw frames")
            return []
        
        # Scale frames
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
                    self.clean_frames.append(frame)
            
            print(f"   ✓ Removed background from frames")
            self.element_frames = self.clean_frames
        else:
            self.element_frames = self.scaled_frames
        
        return self.element_frames
    
    def apply_squash_stretch(
        self,
        element_frame: str,
        velocity: float,
        output_path: str
    ) -> bool:
        """Apply squash and stretch deformation based on velocity."""
        
        if not self.squash_stretch:
            # Just copy the frame
            subprocess.run(['cp', element_frame, output_path], capture_output=True)
            return True
        
        # Calculate deformation based on velocity
        max_velocity = 20
        velocity_ratio = min(abs(velocity) / max_velocity, 1.0)
        
        if abs(velocity) < 1:
            # At rest or slow - slight squash
            scale_x = 1.05
            scale_y = 0.95
        elif velocity > 0:
            # Falling - stretch vertically
            scale_x = 1.0 - (0.2 * velocity_ratio)
            scale_y = 1.0 + (0.3 * velocity_ratio)
        else:
            # Rising - compress vertically
            scale_x = 1.0 + (0.15 * velocity_ratio)
            scale_y = 1.0 - (0.2 * velocity_ratio)
        
        # Apply scaling
        cmd = [
            'ffmpeg',
            '-i', element_frame,
            '-vf', f'scale=iw*{scale_x}:ih*{scale_y}',
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except:
            # Fallback to original
            subprocess.run(['cp', element_frame, output_path], capture_output=True)
            return True
    
    def get_bounce_position(self, frame_offset: int) -> Tuple[int, float]:
        """Get height offset for bounce at given frame."""
        if frame_offset >= len(self.bounce_keyframes):
            return (0, 0)
        
        keyframe = self.bounce_keyframes[frame_offset]
        return (keyframe['height'], keyframe['velocity'])
    
    def process_frames(self) -> List[str]:
        """Process frames for bounce animation."""
        print(f"   Processing bounce animation...")
        print(f"   Type: {self.bounce_type}")
        print(f"   Bounces: {self.num_bounces}")
        print(f"   Height: {self.bounce_height}px")
        print(f"   Squash/Stretch: {self.squash_stretch}")
        
        output_dir = os.path.join(self.temp_dir, "output_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        output_frames = []
        num_element_frames = len(self.element_frames)
        
        for frame_num in range(self.total_frames):
            # Skip frames before start_frame
            if frame_num < self.start_frame:
                output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                if frame_num < len(self.background_frames):
                    subprocess.run(
                        ['cp', self.background_frames[frame_num], output_frame],
                        capture_output=True
                    )
                    output_frames.append(output_frame)
                continue
            
            # Calculate bounce position
            frame_offset = frame_num - self.start_frame
            
            if self.bounce_type == 'in' and frame_offset < len(self.bounce_keyframes):
                # Bounce in animation
                height_offset, velocity = self.get_bounce_position(frame_offset)
                base_x, base_y = self.position
                current_position = (base_x, base_y - int(height_offset))
                apply_effect = True
            elif self.bounce_type == 'out':
                # Bounce out animation (reverse of bounce in)
                out_start = self.total_frames - self.bounce_duration - self.start_frame
                if frame_offset >= out_start:
                    bounce_frame = frame_offset - out_start
                    if bounce_frame < len(self.bounce_keyframes):
                        # Reverse the bounce animation
                        reversed_frame = len(self.bounce_keyframes) - 1 - bounce_frame
                        height_offset, velocity = self.get_bounce_position(reversed_frame)
                        base_x, base_y = self.position
                        current_position = (base_x, base_y - int(height_offset))
                        apply_effect = True
                    else:
                        # Element has bounced away
                        output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                        if frame_num < len(self.background_frames):
                            subprocess.run(
                                ['cp', self.background_frames[frame_num], output_frame],
                                capture_output=True
                            )
                            output_frames.append(output_frame)
                        continue
                else:
                    current_position = self.position
                    velocity = 0
                    apply_effect = False
            else:
                # No bounce or animation complete
                current_position = self.position
                velocity = 0
                apply_effect = False
            
            # Get animation frame (with looping)
            if frame_num >= self.animation_start_frame and num_element_frames > 0:
                anim_frame_idx = (frame_num - self.animation_start_frame) % num_element_frames
            else:
                anim_frame_idx = 0
            
            element_frame = self.element_frames[anim_frame_idx] if self.element_frames else None
            
            if element_frame and frame_num < len(self.background_frames):
                # Apply squash/stretch if needed
                if apply_effect and self.squash_stretch:
                    deformed_frame = os.path.join(self.temp_dir, f'deform_{frame_num:04d}.png')
                    self.apply_squash_stretch(element_frame, velocity, deformed_frame)
                    final_element = deformed_frame
                else:
                    final_element = element_frame
                
                # Composite onto background
                output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                background_frame = self.background_frames[frame_num]
                
                if self.composite_frame(
                    background_frame,
                    final_element,
                    output_frame,
                    current_position
                ):
                    output_frames.append(output_frame)
                    
                    if frame_num % 15 == 0:
                        print(f"      Frame {frame_num}: position {current_position}")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames