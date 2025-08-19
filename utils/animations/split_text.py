"""
Split Text animation.
Text splits apart and moves in different directions.
"""

import os
import subprocess
import math
import random
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from .animate import Animation


class SplitText(Animation):
    """
    Animation where text splits apart into pieces.
    
    Text can split by characters, words, or lines and move in various patterns.
    
    Additional Parameters:
    ---------------------
    text : str
        Text to animate
    split_mode : str
        How to split: 'character', 'word', 'line', 'half' (default 'word')
    split_direction : str
        Direction: 'horizontal', 'vertical', 'explode', 'random' (default 'horizontal')
    split_timing : str
        Timing: 'simultaneous', 'sequential', 'cascade' (default 'simultaneous')
    split_distance : int
        Distance pieces travel in pixels (default 200)
    rotation_on_split : bool
        Rotate pieces as they split (default True)
    fade_on_split : bool
        Fade pieces as they move (default False)
    rejoin : bool
        Rejoin pieces after splitting (default False)
    font_size : int
        Font size (default 48)
    font_color : str
        Text color in hex (default '#FFFFFF')
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        text: Optional[str] = None,
        split_mode: str = 'word',
        split_direction: str = 'horizontal',
        split_timing: str = 'simultaneous',
        split_distance: int = 200,
        rotation_on_split: bool = True,
        fade_on_split: bool = False,
        rejoin: bool = False,
        font_size: int = 48,
        font_color: str = '#FFFFFF',
        direction: float = 0,
        start_frame: int = 0,
        animation_start_frame: int = 0,
        path: Optional[List[Tuple[int, int, int]]] = None,
        fps: int = 30,
        duration: float = 7.0,
        temp_dir: Optional[str] = None
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
        
        self.text = text if text else "SPLIT TEXT"
        self.split_mode = split_mode.lower()
        self.split_direction = split_direction.lower()
        self.split_timing = split_timing.lower()
        self.split_distance = max(50, split_distance)
        self.rotation_on_split = rotation_on_split
        self.fade_on_split = fade_on_split
        self.rejoin = rejoin
        self.font_size = max(10, font_size)
        self.font_color = font_color
        
        self.font = None
        self.text_pieces = []
        self.prepare_text_pieces()
    
    def prepare_text_pieces(self):
        """Split text into pieces based on mode."""
        
        if self.split_mode == 'character':
            # Split into individual characters
            for i, char in enumerate(self.text):
                if char != ' ':
                    self.text_pieces.append({
                        'text': char,
                        'index': i,
                        'type': 'character',
                        'delay': i * 2 if self.split_timing == 'sequential' else 0
                    })
                    
        elif self.split_mode == 'word':
            # Split into words
            words = self.text.split()
            for i, word in enumerate(words):
                self.text_pieces.append({
                    'text': word,
                    'index': i,
                    'type': 'word',
                    'delay': i * 5 if self.split_timing == 'sequential' else 0
                })
                
        elif self.split_mode == 'line':
            # Split into lines
            lines = self.text.split('\n')
            for i, line in enumerate(lines):
                self.text_pieces.append({
                    'text': line,
                    'index': i,
                    'type': 'line',
                    'delay': i * 10 if self.split_timing == 'sequential' else 0
                })
                
        elif self.split_mode == 'half':
            # Split in half
            mid = len(self.text) // 2
            self.text_pieces.append({
                'text': self.text[:mid],
                'index': 0,
                'type': 'half',
                'delay': 0
            })
            self.text_pieces.append({
                'text': self.text[mid:],
                'index': 1,
                'type': 'half',
                'delay': 0
            })
        
        # Calculate movement vectors for each piece
        for i, piece in enumerate(self.text_pieces):
            if self.split_direction == 'horizontal':
                # Move left and right
                if i % 2 == 0:
                    piece['velocity'] = (-self.split_distance / 30, 0)
                else:
                    piece['velocity'] = (self.split_distance / 30, 0)
                    
            elif self.split_direction == 'vertical':
                # Move up and down
                if piece['type'] == 'half':
                    if i == 0:
                        piece['velocity'] = (0, -self.split_distance / 30)
                    else:
                        piece['velocity'] = (0, self.split_distance / 30)
                else:
                    piece['velocity'] = (0, (i - len(self.text_pieces) / 2) * 10)
                    
            elif self.split_direction == 'explode':
                # Explode from center
                angle = (360 / len(self.text_pieces)) * i
                rad = math.radians(angle)
                speed = self.split_distance / 30
                piece['velocity'] = (
                    math.cos(rad) * speed,
                    math.sin(rad) * speed
                )
                
            else:  # random
                piece['velocity'] = (
                    random.uniform(-self.split_distance, self.split_distance) / 30,
                    random.uniform(-self.split_distance, self.split_distance) / 30
                )
            
            # Add rotation if enabled
            if self.rotation_on_split:
                piece['rotation_speed'] = random.uniform(-10, 10)
            else:
                piece['rotation_speed'] = 0
    
    def extract_element_frames(self) -> List[str]:
        """Generate text piece frames instead of extracting from video."""
        print(f"   Generating split text frames...")
        print(f"   Text: {self.text[:50]}...")
        print(f"   Split mode: {self.split_mode}")
        
        # Load font
        try:
            font_paths = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ]
            
            font_loaded = False
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        self.font = ImageFont.truetype(font_path, self.font_size)
                        font_loaded = True
                        break
                    except:
                        continue
            
            if not font_loaded:
                self.font = ImageFont.load_default()
        except:
            self.font = ImageFont.load_default()
        
        # Create individual frames for each text piece
        pieces_dir = os.path.join(self.temp_dir, "text_pieces")
        os.makedirs(pieces_dir, exist_ok=True)
        
        for i, piece in enumerate(self.text_pieces):
            # Create transparent image with text piece
            img = Image.new('RGBA', (800, 200), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Calculate base position for piece
            if self.split_mode == 'character':
                x_offset = i * (self.font_size * 0.6)
                y_offset = 0
            elif self.split_mode == 'word':
                x_offset = i * (self.font_size * 3)
                y_offset = 0
            elif self.split_mode == 'half':
                x_offset = i * (self.font_size * 5)
                y_offset = 0
            else:
                x_offset = 0
                y_offset = i * (self.font_size * 1.5)
            
            draw.text(
                (50 + x_offset, 50 + y_offset),
                piece['text'],
                font=self.font,
                fill=self.font_color
            )
            
            piece_path = os.path.join(pieces_dir, f'piece_{i}.png')
            img.save(piece_path, 'PNG')
            piece['frame_path'] = piece_path
            piece['base_position'] = (50 + x_offset, 50 + y_offset)
        
        print(f"   ✓ Generated {len(self.text_pieces)} text pieces")
        
        # For compatibility, set element_frames to empty
        self.element_frames = []
        return self.element_frames
    
    def process_frames(self) -> List[str]:
        """Process frames for split text animation."""
        print(f"   Processing split text animation...")
        print(f"   Pieces: {len(self.text_pieces)}")
        print(f"   Direction: {self.split_direction}")
        
        output_dir = os.path.join(self.temp_dir, "output_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        output_frames = []
        split_duration = 60  # 2 seconds at 30fps
        
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
            
            # Start with background
            if frame_num < len(self.background_frames):
                output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                subprocess.run(
                    ['cp', self.background_frames[frame_num], output_frame],
                    capture_output=True
                )
                
                # Add text pieces
                for piece in self.text_pieces:
                    # Check if piece should be visible
                    if frame_offset < piece['delay']:
                        continue
                    
                    piece_time = frame_offset - piece['delay']
                    
                    # Calculate position
                    if piece_time < split_duration:
                        # Splitting apart
                        progress = piece_time / split_duration
                        if self.rejoin:
                            progress = min(1.0, progress * 2)  # Split faster if rejoining
                    elif self.rejoin and piece_time < split_duration * 2:
                        # Rejoining
                        progress = 1.0 - (piece_time - split_duration) / split_duration
                    else:
                        # Stay in position or disappeared
                        if self.rejoin:
                            progress = 0
                        else:
                            progress = 1.0
                    
                    # Calculate offset
                    x_offset = piece['velocity'][0] * progress * 30
                    y_offset = piece['velocity'][1] * progress * 30
                    
                    # Calculate rotation
                    rotation = piece['rotation_speed'] * piece_time
                    
                    # Calculate opacity
                    if self.fade_on_split and not self.rejoin:
                        opacity = max(0, 1.0 - progress)
                    else:
                        opacity = 1.0
                    
                    if opacity > 0:
                        # Apply transformations to piece
                        transformed_piece = os.path.join(self.temp_dir, f'trans_{frame_num}_{piece["index"]}.png')
                        
                        filters = []
                        if rotation != 0:
                            filters.append(f'rotate={rotation}*PI/180:c=none')
                        if opacity < 1.0:
                            filters.append(f'format=rgba,colorchannelmixer=aa={opacity}')
                        
                        if filters:
                            filter_chain = ','.join(filters)
                            cmd = [
                                'ffmpeg',
                                '-i', piece['frame_path'],
                                '-vf', filter_chain,
                                '-y',
                                transformed_piece
                            ]
                            subprocess.run(cmd, capture_output=True, text=True, check=True)
                        else:
                            transformed_piece = piece['frame_path']
                        
                        # Composite piece
                        piece_x = self.position[0] + piece['base_position'][0] + int(x_offset)
                        piece_y = self.position[1] + piece['base_position'][1] + int(y_offset)
                        
                        temp_output = os.path.join(self.temp_dir, f'temp_{frame_num}_{piece["index"]}.png')
                        self.composite_frame(
                            output_frame,
                            transformed_piece,
                            temp_output,
                            (piece_x, piece_y)
                        )
                        subprocess.run(['mv', temp_output, output_frame], capture_output=True)
                
                output_frames.append(output_frame)
                
                if frame_num % 15 == 0:
                    print(f"      Frame {frame_num}: split progress")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames