"""
Carousel 3D rotation animation.
Elements rotate in a circular 3D carousel pattern.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from .animate import Animation


class Carousel(Animation):
    """
    Animation where elements rotate in a 3D carousel/merry-go-round pattern.
    
    Creates a circular rotation with perspective depth effects.
    
    Additional Parameters:
    ---------------------
    num_items : int
        Number of carousel items (2 to 12, default 6)
    radius : float
        Carousel radius in pixels (default 150)
    rotation_speed : float
        Degrees per frame (default 1.5)
    tilt_angle : float
        Tilt of carousel plane in degrees (default 15)
    perspective_scale : bool
        Scale items based on depth (default True)
    fade_back : bool
        Fade items when in back (default True)
    vertical_carousel : bool
        Rotate vertically instead of horizontally (default False)
    oscillate : bool
        Add slight oscillation/wobble (default False)
    item_rotation : bool
        Rotate individual items as they orbit (default True)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        num_items: int = 6,
        radius: float = 150,
        rotation_speed: float = 1.5,
        tilt_angle: float = 15,
        perspective_scale: bool = True,
        fade_back: bool = True,
        vertical_carousel: bool = False,
        oscillate: bool = False,
        item_rotation: bool = True,
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
        
        self.num_items = max(2, min(12, num_items))
        self.radius = max(50, radius)
        self.rotation_speed = rotation_speed
        self.tilt_angle = max(-45, min(45, tilt_angle))
        self.perspective_scale = perspective_scale
        self.fade_back = fade_back
        self.vertical_carousel = vertical_carousel
        self.oscillate = oscillate
        self.item_rotation = item_rotation
        self.remove_background = remove_background
        self.background_color = background_color
        self.background_similarity = background_similarity
        
        self.scaled_frames = []
        self.clean_frames = []
        
        # Calculate item angles
        self.item_angles = []
        angle_step = 360 / self.num_items
        for i in range(self.num_items):
            self.item_angles.append(i * angle_step)
    
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
                '-vf', 'scale=100:-1',  # Smaller for carousel items
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
    
    def calculate_carousel_position(
        self,
        item_index: int,
        rotation_angle: float
    ) -> Tuple[Tuple[int, int], float, float, float]:
        """Calculate 3D position and properties for carousel item."""
        
        # Current angle for this item
        current_angle = self.item_angles[item_index] + rotation_angle
        
        # Add oscillation if enabled
        if self.oscillate:
            wobble = math.sin(rotation_angle * 0.1 + item_index) * 5
            current_angle += wobble
        
        # Convert to radians
        angle_rad = math.radians(current_angle)
        
        if self.vertical_carousel:
            # Vertical carousel rotation
            x = self.radius * math.sin(angle_rad)
            y = self.radius * math.cos(angle_rad)
            z = 0
        else:
            # Horizontal carousel with tilt
            tilt_rad = math.radians(self.tilt_angle)
            
            x = self.radius * math.sin(angle_rad)
            z = self.radius * math.cos(angle_rad)
            y = z * math.sin(tilt_rad)  # Apply tilt
            z = z * math.cos(tilt_rad)
        
        # Calculate depth for perspective
        # z ranges from -radius to +radius, normalize to 0-1
        depth = (z + self.radius) / (2 * self.radius)
        
        # Calculate scale based on depth
        if self.perspective_scale:
            scale = 0.6 + 0.4 * depth  # Scale from 0.6 to 1.0
        else:
            scale = 1.0
        
        # Calculate opacity based on depth
        if self.fade_back:
            opacity = 0.4 + 0.6 * depth  # Opacity from 0.4 to 1.0
        else:
            opacity = 1.0
        
        # Calculate item rotation if enabled
        if self.item_rotation:
            item_rot = current_angle
        else:
            item_rot = 0
        
        # Final position
        final_x = int(self.position[0] + x)
        final_y = int(self.position[1] + y)
        
        return ((final_x, final_y), scale, opacity, item_rot)
    
    def create_carousel_item(
        self,
        element_frame: str,
        scale: float,
        opacity: float,
        rotation: float,
        output_path: str
    ) -> bool:
        """Create transformed carousel item."""
        
        filters = []
        
        # Apply scale
        if scale != 1.0:
            scaled_size = int(100 * scale)
            filters.append(f'scale={scaled_size}:-1')
        
        # Apply rotation
        if rotation != 0:
            filters.append(f'rotate={rotation}*PI/180:c=none')
        
        # Apply opacity
        if opacity < 1.0:
            filters.append(f'format=rgba,colorchannelmixer=aa={opacity}')
        
        if not filters:
            subprocess.run(['cp', element_frame, output_path], capture_output=True)
            return True
        
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
            return False
    
    def process_frames(self) -> List[str]:
        """Process frames for carousel animation."""
        print(f"   Processing carousel animation...")
        print(f"   Items: {self.num_items}")
        print(f"   Radius: {self.radius}px")
        print(f"   Speed: {self.rotation_speed}°/frame")
        
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
            rotation_angle = frame_offset * self.rotation_speed
            
            # Start with background
            if frame_num < len(self.background_frames):
                output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                subprocess.run(
                    ['cp', self.background_frames[frame_num], output_frame],
                    capture_output=True
                )
                
                # Sort items by depth for proper rendering order
                items_with_depth = []
                for i in range(self.num_items):
                    pos, scale, opacity, item_rot = self.calculate_carousel_position(i, rotation_angle)
                    # Calculate actual z-depth for sorting
                    angle_rad = math.radians(self.item_angles[i] + rotation_angle)
                    if self.vertical_carousel:
                        z_depth = math.cos(angle_rad)
                    else:
                        z_depth = math.cos(angle_rad) * math.cos(math.radians(self.tilt_angle))
                    
                    items_with_depth.append((i, pos, scale, opacity, item_rot, z_depth))
                
                # Sort by depth (back to front)
                items_with_depth.sort(key=lambda x: x[5])
                
                # Render items back to front
                for item_data in items_with_depth:
                    i, pos, scale, opacity, item_rot, _ = item_data
                    
                    if num_element_frames > 0:
                        # Use different frame for each item or loop
                        anim_frame_idx = ((frame_num - self.animation_start_frame) + i * 5) % num_element_frames
                        element_frame = self.element_frames[anim_frame_idx]
                        
                        # Create transformed item
                        item_frame = os.path.join(self.temp_dir, f'item_{frame_num}_{i}.png')
                        
                        if self.create_carousel_item(element_frame, scale, opacity, item_rot, item_frame):
                            # Composite item
                            temp_output = os.path.join(self.temp_dir, f'temp_{frame_num}_{i}.png')
                            self.composite_frame(
                                output_frame,
                                item_frame,
                                temp_output,
                                pos
                            )
                            subprocess.run(['mv', temp_output, output_frame], capture_output=True)
                
                output_frames.append(output_frame)
                
                if frame_num % 15 == 0:
                    print(f"      Frame {frame_num}: rotation {rotation_angle:.1f}°")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames