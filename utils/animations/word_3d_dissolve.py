"""
Word 3D Dissolve animation.
3D text with depth dissolves word by word with particle effects.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import random

class Word3DDissolve:
    """
    Animation where 3D text with depth dissolves word by word or letter by letter.
    
    The dissolution includes the depth layers, creating a volumetric dissolve effect
    where particles from different depth layers dissolve at slightly different rates.
    
    Additional Parameters:
    ---------------------
    text : str
        Text to animate (default 'START')
    segment_mask : Union[np.ndarray, str]
        Either a numpy array of the segment mask or path to mask image
    font_size : int
        Base font size (default 100)
    font_path : str
        Path to font file (optional)
    text_color : Tuple[int, int, int]
        RGB color for text face (default (255, 220, 0))
    depth_color : Tuple[int, int, int]
        RGB color for text depth/sides (default (180, 150, 0))
    depth_layers : int
        Number of depth layers for 3D effect (default 8)
    depth_offset : int
        Pixel offset between depth layers (default 2)
    dissolve_mode : str
        'word' or 'letter' dissolution (default 'letter')
    dissolve_duration : float
        Duration of each word/letter dissolve in seconds (default 0.5)
    dissolve_overlap : float
        Overlap between consecutive dissolves (0-1, default 0.3)
    particle_size : int
        Size of dissolve particles (default 3)
    particle_velocity : Tuple[float, float]
        Base velocity for particles (x, y) in pixels/frame (default (0, -3))
    particle_acceleration : Tuple[float, float]
        Acceleration for particles (default (0, -0.1))
    particle_spread : float
        Random spread factor for particles (default 2.0)
    fade_start : float
        When to start fading (0-1 through dissolve, default 0.3)
    stable_duration : float
        Duration text stays stable before dissolving (default 0.5)
    random_order : bool
        If True, dissolve in random order (default True)
    depth_dissolve_delay : float
        Delay between depth layer dissolves in seconds (default 0.02)
    perspective_angle : float
        Angle for 3D perspective effect in degrees (default 25)
    """
    
    def __init__(
        self,
        duration: float = 3.0,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        text: str = 'START',
        segment_mask: Optional[Union[np.ndarray, str]] = None,
        font_size: int = 100,
        font_path: Optional[str] = None,
        text_color: Tuple[int, int, int] = (255, 220, 0),
        depth_color: Tuple[int, int, int] = (180, 150, 0),
        depth_layers: int = 8,
        depth_offset: int = 2,
        dissolve_mode: str = 'letter',
        dissolve_duration: float = 0.5,
        dissolve_overlap: float = 0.3,
        particle_size: int = 3,
        particle_velocity: Tuple[float, float] = (0, -3),
        particle_acceleration: Tuple[float, float] = (0, -0.1),
        particle_spread: float = 2.0,
        fade_start: float = 0.3,
        stable_duration: float = 0.5,
        random_order: bool = True,
        depth_dissolve_delay: float = 0.02,
        perspective_angle: float = 25,
        **kwargs
    ):
        self.duration = duration
        self.fps = fps
        self.resolution = resolution
        self.total_frames = int(fps * duration)
        
        self.text = text
        self.font_size = font_size
        self.font_path = font_path
        self.text_color = text_color
        self.depth_color = depth_color
        self.depth_layers = depth_layers
        self.depth_offset = depth_offset
        self.dissolve_mode = dissolve_mode
        self.dissolve_duration = dissolve_duration
        self.dissolve_overlap = dissolve_overlap
        self.particle_size = particle_size
        self.particle_velocity = particle_velocity
        self.particle_acceleration = particle_acceleration
        self.particle_spread = particle_spread
        self.fade_start = fade_start
        self.stable_duration = stable_duration
        self.random_order = random_order
        self.depth_dissolve_delay = depth_dissolve_delay
        self.perspective_angle = perspective_angle
        
        # Load or create segment mask
        if segment_mask is None:
            self.segment_mask = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)
        elif isinstance(segment_mask, str):
            mask_img = Image.open(segment_mask).convert('L')
            mask_img = mask_img.resize(resolution, Image.Resampling.LANCZOS)
            self.segment_mask = np.array(mask_img)
        else:
            self.segment_mask = segment_mask
            if self.segment_mask.shape[:2] != (resolution[1], resolution[0]):
                self.segment_mask = cv2.resize(
                    self.segment_mask, 
                    resolution, 
                    interpolation=cv2.INTER_LINEAR
                )
        
        self.segment_mask = (self.segment_mask > 128).astype(np.uint8) * 255
        
        # Create font
        if self.font_path and os.path.exists(self.font_path):
            self.font = ImageFont.truetype(self.font_path, self.font_size)
        else:
            try:
                self.font = ImageFont.truetype(
                    "/System/Library/Fonts/Helvetica.ttc", 
                    self.font_size
                )
            except:
                self.font = ImageFont.load_default()
        
        # Parse text into units
        if self.dissolve_mode == 'word':
            self.text_units = self.text.split()
        else:  # letter mode
            self.text_units = [char for char in self.text if char != ' ']
        
        # Randomize order if requested
        self.dissolve_order = list(range(len(self.text_units)))
        if self.random_order:
            random.shuffle(self.dissolve_order)
        
        # Calculate timing
        self.stable_frames = int(self.stable_duration * fps)
        self.dissolve_frames = int(self.dissolve_duration * fps)
        
        # Calculate start frame for each unit's dissolve
        overlap_frames = int(self.dissolve_frames * self.dissolve_overlap)
        self.unit_start_frames = []
        
        current_frame = self.stable_frames
        for i in range(len(self.text_units)):
            self.unit_start_frames.append(current_frame)
            current_frame += self.dissolve_frames - overlap_frames
        
        # Initialize particles storage for each depth layer
        self.particles = {}  # {unit_index: {depth_layer: [(x, y, vx, vy, age, alpha)]}}
        self.dissolved_pixels = {}  # {unit_index: {depth_layer: set of (x, y)}}
        
        # Pre-render 3D text units
        self.prerender_text_units()
    
    def prerender_text_units(self):
        """Pre-render each text unit as a 3D element with position info."""
        self.text_unit_images = []
        self.text_unit_positions = []
        self.text_unit_masks = []
        
        # First, render the complete 3D text to get positioning
        full_text_img = self.render_3d_text(self.text, apply_perspective=True)
        
        # Calculate center position
        center_x = self.resolution[0] // 2
        center_y = self.resolution[1] // 2
        
        # Position for full text
        full_w, full_h = full_text_img.size
        full_x = center_x - full_w // 2
        full_y = center_y - full_h // 2
        
        # Now render each unit and calculate its position
        if self.dissolve_mode == 'word':
            current_text = ""
            for i, word in enumerate(self.text_units):
                # Calculate position of this word in the full text
                if i > 0:
                    current_text += " "
                word_start = len(current_text)
                current_text += word
                
                # Render just this word with 3D effect
                word_img = self.render_3d_text(word, apply_perspective=True)
                
                # Calculate approximate position (simplified - assumes monospace-ish)
                char_width = full_w / len(self.text)
                word_x = full_x + int(word_start * char_width)
                word_y = full_y
                
                self.text_unit_images.append(word_img)
                self.text_unit_positions.append((word_x, word_y))
                
                # Create mask for this word
                word_mask = np.array(word_img)[:, :, 3] > 0
                self.text_unit_masks.append(word_mask)
        
        else:  # letter mode
            char_index = 0
            for i, char in enumerate(self.text):
                if char == ' ':
                    continue
                    
                # Render just this letter with 3D effect
                char_img = self.render_3d_text(char, apply_perspective=True)
                
                # Calculate position
                char_width = full_w / len(self.text)
                char_x = full_x + int(i * char_width)
                char_y = full_y
                
                self.text_unit_images.append(char_img)
                self.text_unit_positions.append((char_x, char_y))
                
                # Create mask for this character
                char_mask = np.array(char_img)[:, :, 3] > 0
                self.text_unit_masks.append(char_mask)
    
    def render_3d_text(
        self, 
        text: str,
        scale: float = 1.0,
        alpha: float = 1.0,
        apply_perspective: bool = True
    ) -> Image.Image:
        """
        Render 3D text with depth effect.
        """
        # Calculate scaled font size
        scaled_font_size = int(self.font_size * scale)
        if self.font_path and os.path.exists(self.font_path):
            scaled_font = ImageFont.truetype(self.font_path, scaled_font_size)
        else:
            try:
                scaled_font = ImageFont.truetype(
                    "/System/Library/Fonts/Helvetica.ttc", 
                    scaled_font_size
                )
            except:
                scaled_font = ImageFont.load_default()
        
        # Get text dimensions
        temp_img = Image.new('RGBA', self.resolution, (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=scaled_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create larger canvas for 3D effect
        canvas_width = text_width + self.depth_layers * self.depth_offset * 2
        canvas_height = text_height + self.depth_layers * self.depth_offset * 2
        
        # Create 3D text image
        text_3d = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_3d)
        
        # Calculate base position
        base_x = (canvas_width - text_width) // 2
        base_y = (canvas_height - text_height) // 2
        
        # Draw depth layers
        depth_offset_scaled = int(self.depth_offset * scale)
        for i in range(self.depth_layers, 0, -1):
            offset_x = base_x + i * depth_offset_scaled
            offset_y = base_y - i * depth_offset_scaled
            
            # Interpolate color
            t = (self.depth_layers - i) / max(self.depth_layers - 1, 1)
            color_r = int(self.depth_color[0] * (1 - t) + self.text_color[0] * t * 0.7)
            color_g = int(self.depth_color[1] * (1 - t) + self.text_color[1] * t * 0.7)
            color_b = int(self.depth_color[2] * (1 - t) + self.text_color[2] * t * 0.7)
            layer_alpha = int(255 * alpha)
            
            draw.text(
                (offset_x, offset_y),
                text,
                font=scaled_font,
                fill=(color_r, color_g, color_b, layer_alpha)
            )
        
        # Draw front face with outline
        for ox in range(-2, 3):
            for oy in range(-2, 3):
                if ox != 0 or oy != 0:
                    draw.text(
                        (base_x + ox, base_y + oy),
                        text,
                        font=scaled_font,
                        fill=(0, 0, 0, int(255 * alpha))
                    )
        
        draw.text(
            (base_x, base_y),
            text,
            font=scaled_font,
            fill=(*self.text_color, int(255 * alpha))
        )
        
        # Apply perspective if requested
        if apply_perspective and self.perspective_angle > 0:
            text_3d_np = np.array(text_3d)
            h, w = text_3d_np.shape[:2]
            
            angle_rad = np.radians(self.perspective_angle)
            src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            perspective_offset = int(h * np.tan(angle_rad) * 0.2)
            dst_pts = np.float32([
                [perspective_offset, 0],
                [w - perspective_offset, 0],
                [w, h],
                [0, h]
            ])
            
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            text_3d_np = cv2.warpPerspective(
                text_3d_np, 
                matrix, 
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT
            )
            
            text_3d = Image.fromarray(text_3d_np)
        
        return text_3d
    
    def create_particles(self, unit_index: int, depth_layer: int, frame_number: int):
        """Create particles for dissolving a text unit's specific depth layer."""
        if unit_index not in self.particles:
            self.particles[unit_index] = {}
            self.dissolved_pixels[unit_index] = {}
        
        if depth_layer not in self.particles[unit_index]:
            self.particles[unit_index][depth_layer] = []
            self.dissolved_pixels[unit_index][depth_layer] = set()
        
        # Get the unit's image and position
        unit_img = self.text_unit_images[unit_index]
        unit_x, unit_y = self.text_unit_positions[unit_index]
        unit_mask = self.text_unit_masks[unit_index]
        
        # Calculate which pixels to dissolve this frame
        unit_start = self.unit_start_frames[self.dissolve_order[unit_index]]
        
        # Add delay for depth layers
        depth_delay_frames = int(depth_layer * self.depth_dissolve_delay * self.fps)
        effective_start = unit_start + depth_delay_frames
        
        if frame_number < effective_start:
            return
        
        frames_into_dissolve = frame_number - effective_start
        if frames_into_dissolve >= self.dissolve_frames:
            return
        
        dissolve_progress = frames_into_dissolve / max(self.dissolve_frames - 1, 1)
        
        # Sample pixels to convert to particles
        unit_np = np.array(unit_img)
        
        # Focus on pixels that correspond to this depth layer
        # Simplified: use vertical bands for depth layers
        h, w = unit_np.shape[:2]
        layer_band_width = w // self.depth_layers
        x_start = depth_layer * layer_band_width
        x_end = min((depth_layer + 1) * layer_band_width, w)
        
        # Find non-transparent pixels in this band
        for y in range(h):
            for x in range(x_start, x_end):
                if unit_np[y, x, 3] > 0:  # Non-transparent
                    pixel_key = (x, y)
                    
                    # Check if already dissolved
                    if pixel_key in self.dissolved_pixels[unit_index][depth_layer]:
                        continue
                    
                    # Probability of dissolving increases with progress
                    if random.random() < dissolve_progress * 0.3:  # 30% chance at full progress
                        # Mark as dissolved
                        self.dissolved_pixels[unit_index][depth_layer].add(pixel_key)
                        
                        # Create particle
                        world_x = unit_x + x
                        world_y = unit_y + y
                        
                        # Random velocity
                        vx = self.particle_velocity[0] + random.uniform(-self.particle_spread, self.particle_spread)
                        vy = self.particle_velocity[1] + random.uniform(-self.particle_spread, 0)
                        
                        # Get pixel color
                        color = tuple(unit_np[y, x, :3])
                        alpha = unit_np[y, x, 3]
                        
                        particle = {
                            'x': world_x,
                            'y': world_y,
                            'vx': vx,
                            'vy': vy,
                            'age': 0,
                            'color': color,
                            'alpha': alpha,
                            'size': self.particle_size
                        }
                        
                        self.particles[unit_index][depth_layer].append(particle)
    
    def update_particles(self, unit_index: int, depth_layer: int):
        """Update particle positions and properties."""
        if unit_index not in self.particles:
            return
        if depth_layer not in self.particles[unit_index]:
            return
        
        updated_particles = []
        for particle in self.particles[unit_index][depth_layer]:
            # Update position
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            # Update velocity (acceleration)
            particle['vx'] += self.particle_acceleration[0]
            particle['vy'] += self.particle_acceleration[1]
            
            # Update age
            particle['age'] += 1
            
            # Fade out over time
            max_age = self.dissolve_frames
            age_factor = 1.0 - (particle['age'] / max_age)
            particle['current_alpha'] = particle['alpha'] * age_factor
            
            # Keep particle if still visible and on screen
            if (particle['current_alpha'] > 0 and 
                0 <= particle['x'] < self.resolution[0] and 
                0 <= particle['y'] < self.resolution[1]):
                updated_particles.append(particle)
        
        self.particles[unit_index][depth_layer] = updated_particles
    
    def generate_frame(self, frame_number: int, background: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a single frame of the 3D word dissolve animation."""
        
        # Create base frame
        if background is not None:
            frame = background.copy()
            if frame.shape[2] == 3:
                frame = np.concatenate([frame, np.ones((*frame.shape[:2], 1), dtype=np.uint8) * 255], axis=2)
        else:
            frame = np.zeros((*self.resolution[::-1], 4), dtype=np.uint8)
            frame[:, :, 3] = 255
        
        # Convert to PIL for compositing
        frame_pil = Image.fromarray(frame)
        
        # Render stable text units (not yet dissolving)
        for i, unit_index in enumerate(self.dissolve_order):
            unit_start = self.unit_start_frames[unit_index]
            
            if frame_number < unit_start:
                # Unit hasn't started dissolving yet - render normally
                unit_img = self.text_unit_images[i]
                unit_x, unit_y = self.text_unit_positions[i]
                
                # Apply segment mask for occlusion
                unit_np = np.array(unit_img)
                h, w = unit_np.shape[:2]
                
                # Check bounds
                y1 = max(0, unit_y)
                y2 = min(unit_y + h, self.resolution[1])
                x1 = max(0, unit_x)
                x2 = min(unit_x + w, self.resolution[0])
                
                if y2 > y1 and x2 > x1:
                    # Get mask region
                    mask_region = self.segment_mask[y1:y2, x1:x2]
                    
                    # Apply mask to alpha channel
                    ty1 = y1 - unit_y
                    ty2 = ty1 + (y2 - y1)
                    tx1 = x1 - unit_x
                    tx2 = tx1 + (x2 - x1)
                    
                    unit_alpha = unit_np[ty1:ty2, tx1:tx2, 3].astype(float)
                    unit_alpha = unit_alpha * (1 - mask_region / 255.0) * 0.5  # 50% opacity behind
                    unit_np[ty1:ty2, tx1:tx2, 3] = unit_alpha.astype(np.uint8)
                
                # Composite onto frame
                unit_pil = Image.fromarray(unit_np)
                temp_frame = Image.new('RGBA', self.resolution, (0, 0, 0, 0))
                temp_frame.paste(unit_pil, (unit_x, unit_y), unit_pil)
                frame_pil = Image.alpha_composite(frame_pil, temp_frame)
            
            else:
                # Unit is dissolving - handle per depth layer
                for depth_layer in range(self.depth_layers):
                    # Create particles for this depth layer
                    self.create_particles(i, depth_layer, frame_number)
                    
                    # Update existing particles
                    self.update_particles(i, depth_layer)
                
                # Render remaining parts of the unit (not yet dissolved)
                unit_img_copy = self.text_unit_images[i].copy()
                unit_np = np.array(unit_img_copy)
                
                # Remove dissolved pixels
                if i in self.dissolved_pixels:
                    for depth_layer in self.dissolved_pixels[i]:
                        for (x, y) in self.dissolved_pixels[i][depth_layer]:
                            if 0 <= y < unit_np.shape[0] and 0 <= x < unit_np.shape[1]:
                                unit_np[y, x, 3] = 0  # Make transparent
                
                # Apply fade based on dissolve progress
                unit_start = self.unit_start_frames[unit_index]
                frames_into_dissolve = frame_number - unit_start
                if frames_into_dissolve < self.dissolve_frames:
                    dissolve_progress = frames_into_dissolve / max(self.dissolve_frames - 1, 1)
                    if dissolve_progress > self.fade_start:
                        fade_factor = 1.0 - (dissolve_progress - self.fade_start) / (1.0 - self.fade_start)
                        unit_np[:, :, 3] = (unit_np[:, :, 3] * fade_factor).astype(np.uint8)
                
                # Composite remaining unit
                unit_x, unit_y = self.text_unit_positions[i]
                unit_pil = Image.fromarray(unit_np)
                temp_frame = Image.new('RGBA', self.resolution, (0, 0, 0, 0))
                temp_frame.paste(unit_pil, (unit_x, unit_y), unit_pil)
                frame_pil = Image.alpha_composite(frame_pil, temp_frame)
        
        # Render particles
        particle_layer = Image.new('RGBA', self.resolution, (0, 0, 0, 0))
        particle_draw = ImageDraw.Draw(particle_layer)
        
        for unit_index in self.particles:
            for depth_layer in self.particles[unit_index]:
                for particle in self.particles[unit_index][depth_layer]:
                    if particle['current_alpha'] > 0:
                        x = int(particle['x'])
                        y = int(particle['y'])
                        size = particle['size']
                        alpha = int(particle['current_alpha'])
                        color = (*particle['color'], alpha)
                        
                        # Draw particle as a small circle
                        particle_draw.ellipse(
                            [x - size, y - size, x + size, y + size],
                            fill=color
                        )
        
        # Composite particles onto frame
        frame_pil = Image.alpha_composite(frame_pil, particle_layer)
        
        return np.array(frame_pil)