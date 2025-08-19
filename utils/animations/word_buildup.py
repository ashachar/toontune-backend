"""
Word Build-up animation.
Words appear one by one to build complete text.
"""

import os
import subprocess
import math
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from .animate import Animation


class WordBuildup(Animation):
    """
    Animation where words appear sequentially to build up text.
    
    Words can appear with various effects and timings.
    
    Additional Parameters:
    ---------------------
    text : str
        Text to animate
    buildup_mode : str
        How words appear: 'fade', 'slide', 'pop', 'typewriter' (default 'fade')
    word_delay : int
        Frames between words (default 5)
    entrance_direction : str
        For slide mode: 'left', 'right', 'top', 'bottom', 'random' (default 'bottom')
    emphasis_effect : bool
        Add emphasis to newly appearing words (default True)
    hold_duration : int
        Frames to hold complete text (default 30)
    font_size : int
        Font size (default 48)
    font_color : str
        Text color in hex (default '#FFFFFF')
    highlight_color : str
        Color for emphasis effect (default '#FFFF00')
    alignment : str
        Text alignment: 'left', 'center', 'right' (default 'center')
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        text: Optional[str] = None,
        buildup_mode: str = 'fade',
        word_delay: int = 5,
        entrance_direction: str = 'bottom',
        emphasis_effect: bool = True,
        hold_duration: int = 30,
        font_size: int = 48,
        font_color: str = '#FFFFFF',
        highlight_color: str = '#FFFF00',
        alignment: str = 'center',
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
        
        self.text = text if text else "Word by word text animation"
        self.buildup_mode = buildup_mode.lower()
        self.word_delay = max(1, word_delay)
        self.entrance_direction = entrance_direction.lower()
        self.emphasis_effect = emphasis_effect
        self.hold_duration = max(0, hold_duration)
        self.font_size = max(10, font_size)
        self.font_color = font_color
        self.highlight_color = highlight_color
        self.alignment = alignment.lower()
        
        self.font = None
        self.words = []
        self.word_positions = []
        self.prepare_words()
    
    def prepare_words(self):
        """Split text into words and calculate positions."""
        
        # Split text into lines and words
        lines = self.text.split('\n')
        all_words = []
        
        for line_idx, line in enumerate(lines):
            words = line.split()
            for word_idx, word in enumerate(words):
                all_words.append({
                    'text': word,
                    'line': line_idx,
                    'word_in_line': word_idx,
                    'global_index': len(all_words),
                    'appear_frame': len(all_words) * self.word_delay
                })
        
        self.words = all_words
        
        # Load font
        try:
            font_paths = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
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
        
        # Calculate word positions
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        
        line_height = self.font_size * 1.5
        y_offset = 0
        
        for line_idx in range(len(lines)):
            line_words = [w for w in self.words if w['line'] == line_idx]
            line_text = ' '.join([w['text'] for w in line_words])
            
            # Calculate line width for alignment
            try:
                line_width = draw.textlength(line_text, font=self.font)
            except:
                line_width = len(line_text) * self.font_size * 0.6
            
            if self.alignment == 'center':
                x_start = -line_width / 2
            elif self.alignment == 'right':
                x_start = -line_width
            else:  # left
                x_start = 0
            
            x_offset = x_start
            for word in line_words:
                word['position'] = (x_offset, y_offset)
                try:
                    word_width = draw.textlength(word['text'] + ' ', font=self.font)
                except:
                    word_width = len(word['text'] + ' ') * self.font_size * 0.6
                x_offset += word_width
            
            y_offset += line_height
    
    def extract_element_frames(self) -> List[str]:
        """Generate word frames for animation."""
        print(f"   Generating word build-up frames...")
        print(f"   Text: {self.text[:50]}...")
        print(f"   Words: {len(self.words)}")
        print(f"   Mode: {self.buildup_mode}")
        
        # Create frames directory
        frames_dir = os.path.join(self.temp_dir, "word_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Generate animation frames
        total_frames = len(self.words) * self.word_delay + self.hold_duration + 30
        self.word_frames = []
        
        for frame_idx in range(total_frames):
            # Create frame with accumulated words
            img = Image.new('RGBA', (1200, 400), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Determine which words are visible
            for word in self.words:
                if frame_idx >= word['appear_frame']:
                    word_age = frame_idx - word['appear_frame']
                    
                    # Calculate word appearance based on mode
                    if self.buildup_mode == 'fade':
                        # Fade in over 10 frames
                        opacity = min(1.0, word_age / 10.0)
                        alpha = int(255 * opacity)
                    else:
                        alpha = 255
                    
                    # Position offset for entrance effects
                    x_offset = 0
                    y_offset = 0
                    
                    if self.buildup_mode == 'slide':
                        slide_progress = min(1.0, word_age / 10.0)
                        
                        if self.entrance_direction == 'left':
                            x_offset = -50 * (1 - slide_progress)
                        elif self.entrance_direction == 'right':
                            x_offset = 50 * (1 - slide_progress)
                        elif self.entrance_direction == 'top':
                            y_offset = -30 * (1 - slide_progress)
                        elif self.entrance_direction == 'bottom':
                            y_offset = 30 * (1 - slide_progress)
                        else:  # random
                            import random
                            random.seed(word['global_index'])
                            x_offset = random.uniform(-30, 30) * (1 - slide_progress)
                            y_offset = random.uniform(-30, 30) * (1 - slide_progress)
                    
                    elif self.buildup_mode == 'pop':
                        # Scale effect
                        pop_progress = min(1.0, word_age / 5.0)
                        if pop_progress < 1.0:
                            # Create scaled version (simplified - use position offset)
                            scale = 0.5 + 0.5 * pop_progress
                            x_offset = word['position'][0] * (1 - scale)
                            y_offset = word['position'][1] * (1 - scale)
                    
                    # Determine color (emphasis for new words)
                    if self.emphasis_effect and word_age < 10:
                        # Blend from highlight to normal color
                        blend = word_age / 10.0
                        color_hex = self.highlight_color if blend < 0.5 else self.font_color
                    else:
                        color_hex = self.font_color
                    
                    # Convert hex to RGBA
                    color_hex = color_hex.lstrip('#')
                    r = int(color_hex[0:2], 16)
                    g = int(color_hex[2:4], 16)
                    b = int(color_hex[4:6], 16)
                    
                    # Draw word
                    draw.text(
                        (600 + word['position'][0] + x_offset, 
                         100 + word['position'][1] + y_offset),
                        word['text'],
                        font=self.font,
                        fill=(r, g, b, alpha)
                    )
            
            # Save frame
            frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
            img.save(frame_path, 'PNG')
            self.word_frames.append(frame_path)
        
        print(f"   ✓ Generated {len(self.word_frames)} frames")
        
        # Set element_frames for compatibility
        self.element_frames = self.word_frames
        return self.element_frames
    
    def process_frames(self) -> List[str]:
        """Process frames for word build-up animation."""
        print(f"   Processing word build-up animation...")
        
        output_dir = os.path.join(self.temp_dir, "output_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        output_frames = []
        num_word_frames = len(self.word_frames)
        
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
            
            # Get word frame
            if frame_offset < num_word_frames:
                word_frame_idx = frame_offset
            else:
                # Keep showing complete text
                word_frame_idx = num_word_frames - 1 if num_word_frames > 0 else 0
            
            if word_frame_idx < num_word_frames and frame_num < len(self.background_frames):
                word_frame = self.word_frames[word_frame_idx]
                output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                background_frame = self.background_frames[frame_num]
                
                # Composite word frame onto background
                if self.composite_frame(
                    background_frame,
                    word_frame,
                    output_frame,
                    self.position
                ):
                    output_frames.append(output_frame)
                    
                    if frame_num % 30 == 0:
                        words_shown = min(len(self.words), frame_offset // self.word_delay)
                        print(f"      Frame {frame_num}: {words_shown}/{len(self.words)} words")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames