"""
Typewriter text animation.
Text appears letter by letter like being typed.
"""

import os
import subprocess
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from .animate import Animation


class Typewriter(Animation):
    """
    Animation where text appears character by character like typewriting.
    
    Can work with text strings or extract text from images using OCR.
    
    Additional Parameters:
    ---------------------
    text : str
        Text to animate (if not using image with text)
    font_size : int
        Font size for rendered text (default 48)
    font_color : str
        Color of text in hex format (default '#FFFFFF')
    typing_speed : int
        Frames per character (default 2)
    cursor_visible : bool
        Show blinking cursor (default True)
    cursor_blink_rate : int
        Frames between cursor blinks (default 15)
    text_position : Tuple[int, int]
        Position for text (default uses main position)
    sound_effect : bool
        Add typing sound markers for post-processing (default False)
    line_height : float
        Line height multiplier for multi-line text (default 1.2)
    """
    
    def __init__(
        self,
        element_path: str,
        background_path: str,
        position: Tuple[int, int],
        text: Optional[str] = None,
        font_size: int = 48,
        font_color: str = '#FFFFFF',
        typing_speed: int = 2,
        cursor_visible: bool = True,
        cursor_blink_rate: int = 15,
        text_position: Optional[Tuple[int, int]] = None,
        sound_effect: bool = False,
        line_height: float = 1.2,
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
        
        self.text = text
        self.font_size = max(10, font_size)
        self.font_color = font_color
        self.typing_speed = max(1, typing_speed)
        self.cursor_visible = cursor_visible
        self.cursor_blink_rate = max(1, cursor_blink_rate)
        self.text_position = text_position if text_position else position
        self.sound_effect = sound_effect
        self.line_height = max(1.0, line_height)
        
        # Text frames storage
        self.text_frames = []
        self.font = None
        
        # Sound markers for post-processing
        self.sound_markers = []
    
    def extract_element_frames(self) -> List[str]:
        """Generate text frames instead of extracting from video."""
        print(f"   Generating typewriter text frames...")
        
        if not self.text:
            # If no text provided, try to use element_path as text source
            if os.path.exists(self.element_path) and self.element_path.endswith('.txt'):
                with open(self.element_path, 'r') as f:
                    self.text = f.read()
            else:
                self.text = "Hello World!"  # Default text
        
        print(f"   Text to animate: {self.text[:50]}...")
        
        # Create frames directory
        text_frames_dir = os.path.join(self.temp_dir, "text_frames")
        os.makedirs(text_frames_dir, exist_ok=True)
        
        # Try to use a system font
        try:
            # Try to find a monospace font for typewriter effect
            font_paths = [
                "/System/Library/Fonts/Courier.dfont",
                "/System/Library/Fonts/Monaco.dfont",
                "/System/Library/Fonts/Menlo.ttc",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
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
                # Fallback to default font
                self.font = ImageFont.load_default()
        except:
            self.font = ImageFont.load_default()
        
        # Generate frames for each stage of typing
        lines = self.text.split('\n')
        total_chars = len(self.text.replace('\n', ''))
        
        char_count = 0
        frame_count = 0
        
        # Generate frames progressively showing more text
        for num_chars in range(total_chars + 1):
            # Build text up to current character count
            current_text = ""
            chars_added = 0
            
            for line in lines:
                if chars_added >= num_chars:
                    break
                    
                chars_to_add = min(len(line), num_chars - chars_added)
                if chars_to_add > 0:
                    current_text += line[:chars_to_add]
                chars_added += chars_to_add
                
                if chars_added < num_chars and chars_added < total_chars:
                    current_text += "\n"
            
            # Create frame with current text
            for _ in range(self.typing_speed):
                frame_path = os.path.join(text_frames_dir, f'text_{frame_count:04d}.png')
                
                # Create transparent image
                img = Image.new('RGBA', (800, 600), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                
                # Draw text
                y_offset = 0
                for line in current_text.split('\n'):
                    draw.text(
                        (50, 50 + y_offset),
                        line,
                        font=self.font,
                        fill=self.font_color
                    )
                    y_offset += int(self.font_size * self.line_height)
                
                # Add cursor if visible
                if self.cursor_visible and num_chars == total_chars:
                    # Blinking cursor at end
                    if (frame_count // self.cursor_blink_rate) % 2 == 0:
                        cursor_x = 50 + draw.textlength(current_text.split('\n')[-1], font=self.font)
                        cursor_y = 50 + (len(current_text.split('\n')) - 1) * int(self.font_size * self.line_height)
                        draw.text(
                            (cursor_x, cursor_y),
                            "|",
                            font=self.font,
                            fill=self.font_color
                        )
                elif self.cursor_visible and frame_count % 2 == 0:
                    # Cursor during typing
                    last_line = current_text.split('\n')[-1] if current_text else ""
                    cursor_x = 50 + draw.textlength(last_line, font=self.font)
                    cursor_y = 50 + (len(current_text.split('\n')) - 1) * int(self.font_size * self.line_height)
                    draw.text(
                        (cursor_x, cursor_y),
                        "_",
                        font=self.font,
                        fill=self.font_color
                    )
                
                img.save(frame_path, 'PNG')
                self.text_frames.append(frame_path)
                
                # Add sound marker for typing sound
                if self.sound_effect and frame_count % self.typing_speed == 0 and num_chars > 0:
                    self.sound_markers.append({
                        'frame': frame_count,
                        'type': 'keystroke'
                    })
                
                frame_count += 1
        
        # Add some frames with complete text (with blinking cursor)
        for i in range(30):  # 1 second at 30fps
            frame_path = os.path.join(text_frames_dir, f'text_{frame_count:04d}.png')
            
            img = Image.new('RGBA', (800, 600), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw complete text
            y_offset = 0
            for line in self.text.split('\n'):
                draw.text(
                    (50, 50 + y_offset),
                    line,
                    font=self.font,
                    fill=self.font_color
                )
                y_offset += int(self.font_size * self.line_height)
            
            # Blinking cursor
            if self.cursor_visible and (i // self.cursor_blink_rate) % 2 == 0:
                last_line = self.text.split('\n')[-1]
                cursor_x = 50 + draw.textlength(last_line, font=self.font)
                cursor_y = 50 + (len(self.text.split('\n')) - 1) * int(self.font_size * self.line_height)
                draw.text(
                    (cursor_x, cursor_y),
                    "|",
                    font=self.font,
                    fill=self.font_color
                )
            
            img.save(frame_path, 'PNG')
            self.text_frames.append(frame_path)
            frame_count += 1
        
        print(f"   ✓ Generated {len(self.text_frames)} text frames")
        
        if self.sound_effect and self.sound_markers:
            # Save sound markers for post-processing
            markers_file = os.path.join(self.temp_dir, "typing_sound_markers.txt")
            with open(markers_file, 'w') as f:
                for marker in self.sound_markers:
                    f.write(f"{marker['frame']},{marker['type']}\n")
            print(f"   ✓ Saved {len(self.sound_markers)} sound markers")
        
        self.element_frames = self.text_frames
        return self.element_frames
    
    def process_frames(self) -> List[str]:
        """Process frames for typewriter animation."""
        print(f"   Processing typewriter animation...")
        print(f"   Text length: {len(self.text)} characters")
        print(f"   Typing speed: {self.typing_speed} frames/char")
        
        output_dir = os.path.join(self.temp_dir, "output_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        output_frames = []
        num_text_frames = len(self.text_frames)
        
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
            
            # Calculate which text frame to use
            frame_offset = frame_num - self.start_frame
            
            if frame_offset < num_text_frames:
                text_frame_idx = frame_offset
            else:
                # Keep showing complete text
                text_frame_idx = num_text_frames - 1
            
            if text_frame_idx >= 0 and text_frame_idx < num_text_frames:
                text_frame = self.text_frames[text_frame_idx]
                
                if frame_num < len(self.background_frames):
                    output_frame = os.path.join(output_dir, f'out_{frame_num:04d}.png')
                    background_frame = self.background_frames[frame_num]
                    
                    # Use text_position for compositing
                    if self.composite_frame(
                        background_frame,
                        text_frame,
                        output_frame,
                        self.text_position
                    ):
                        output_frames.append(output_frame)
                        
                        if frame_num % 30 == 0:
                            chars_shown = min(len(self.text), text_frame_idx // self.typing_speed)
                            print(f"      Frame {frame_num}: {chars_shown} characters")
        
        print(f"   ✓ Processed {len(output_frames)} frames")
        return output_frames