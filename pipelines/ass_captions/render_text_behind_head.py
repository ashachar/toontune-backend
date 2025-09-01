#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render text behind person's head using true video compositing.

This script takes a video, mask video, and enriched transcript to create
a new video where text appears BEHIND the person (not visible through them).
Uses OpenCV for frame-by-frame processing with proper masking.

Unlike ASS subtitles which can only overlay text, this creates true
depth-layered text that is properly occluded by the person.
"""

import os
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
import argparse


@dataclass
class SubPhrase:
    """Represents a phrase with all its properties."""
    text: str
    words: List[str]
    start_time: float
    end_time: float
    importance: float
    emphasis_type: str
    font_size_multiplier: float
    bold: bool
    color_tint: Optional[List[int]]
    position: str  # top/bottom
    appearance_index: int
    opacity_boost: float


class TextRenderer:
    """Handles text rendering with proper styling and animation."""
    
    def __init__(self, video_width: int, video_height: int):
        self.W = video_width
        self.H = video_height
        self.base_font_size = 48
        
        # Try to load system font
        self.font_path = self._find_font_path()
    
    def _find_font_path(self) -> str:
        """Find suitable font path on system."""
        # Common font paths
        paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/arial.ttf"
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
        
        # Fallback - PIL will use default
        return ""
    
    def get_font(self, size: int, bold: bool = False) -> ImageFont.ImageFont:
        """Get PIL font object."""
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        return font
    
    def measure_text(self, text: str, font_size: int) -> Tuple[int, int]:
        """Measure text dimensions."""
        font = self.get_font(font_size)
        try:
            bbox = font.getbbox(text)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return width, height
        except:
            # Fallback estimation
            return len(text) * int(font_size * 0.6), int(font_size * 1.2)
    
    def split_text_into_lines(self, words: List[str], max_words: int = 6) -> List[str]:
        """Split words into lines if > max_words."""
        if len(words) <= max_words:
            return [" ".join(words)]
        
        # Split roughly in half
        mid = len(words) // 2
        line1 = " ".join(words[:mid])
        line2 = " ".join(words[mid:])
        return [line1, line2]
    
    def render_text_with_border(self, text: str, font_size: int, bold: bool, 
                               color_tint: Optional[List[int]] = None) -> np.ndarray:
        """Render text with black border on transparent background."""
        font = self.get_font(font_size, bold)
        
        # Measure text
        text_w, text_h = self.measure_text(text, font_size)
        
        # Add padding for border
        border_width = 3
        padding = border_width * 2
        canvas_w = text_w + padding * 2
        canvas_h = text_h + padding * 2
        
        # Create transparent image
        img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Text color
        if color_tint:
            text_color = tuple(color_tint) + (255,)  # Add alpha
        else:
            text_color = (255, 255, 255, 255)  # White
        
        border_color = (0, 0, 0, 255)  # Black border
        
        # Text position (accounting for padding)
        text_x = padding
        text_y = padding
        
        # Draw text border (black outline)
        for dx in range(-border_width, border_width + 1):
            for dy in range(-border_width, border_width + 1):
                if dx != 0 or dy != 0:  # Skip center
                    draw.text((text_x + dx, text_y + dy), text, 
                             font=font, fill=border_color)
        
        # Draw main text
        draw.text((text_x, text_y), text, font=font, fill=text_color)
        
        # Convert to numpy array (BGRA for OpenCV)
        img_array = np.array(img)
        img_bgra = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
        
        return img_bgra


class TextBehindHeadRenderer:
    """Main renderer for text-behind-head effect."""
    
    def __init__(self, video_path: str, mask_video_path: str, transcript_path: str):
        self.video_path = video_path
        self.mask_video_path = mask_video_path
        self.transcript_path = transcript_path
        
        # Load video properties
        self.cap = cv2.VideoCapture(video_path)
        self.mask_cap = cv2.VideoCapture(mask_video_path)
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {self.W}x{self.H}, {self.fps} fps, {self.total_frames} frames")
        
        # Initialize text renderer
        self.text_renderer = TextRenderer(self.W, self.H)
        
        # Load transcript
        self.phrases = self._load_transcript()
        print(f"Loaded {len(self.phrases)} phrases")
    
    def _load_transcript(self) -> List[SubPhrase]:
        """Load enriched transcript."""
        with open(self.transcript_path, 'r') as f:
            data = json.load(f)
        
        phrases = []
        for p in data["phrases"]:
            phrase = SubPhrase(
                text=p["text"],
                words=p["words"],
                start_time=p["start_time"],
                end_time=p["end_time"],
                importance=p["importance"],
                emphasis_type=p["emphasis_type"],
                font_size_multiplier=p["visual_style"]["font_size_multiplier"],
                bold=p["visual_style"]["bold"],
                color_tint=p["visual_style"]["color_tint"],
                position=p["position"],
                appearance_index=p["appearance_index"],
                opacity_boost=p["visual_style"]["opacity_boost"]
            )
            phrases.append(phrase)
        
        return phrases
    
    def get_active_words_at_time(self, current_time: float) -> List[Dict]:
        """Get all words that should be visible at current time with their animation state."""
        active_words = []
        
        for phrase in self.phrases:
            if phrase.start_time <= current_time <= phrase.end_time:
                # Calculate word-by-word timing
                total_duration = phrase.end_time - phrase.start_time
                time_per_word = total_duration / len(phrase.words)
                
                for word_idx, word in enumerate(phrase.words):
                    word_start = phrase.start_time + (word_idx * time_per_word)
                    word_appear_duration = min(0.3, time_per_word * 0.5)  # 300ms max
                    
                    if current_time >= word_start:
                        # Calculate animation progress (0-1)
                        animation_progress = min(1.0, (current_time - word_start) / word_appear_duration)
                        
                        active_words.append({
                            'word': word,
                            'phrase': phrase,
                            'word_index': word_idx,
                            'animation_progress': animation_progress,
                            'word_start': word_start
                        })
        
        return active_words
    
    def calculate_layout(self, active_words: List[Dict]) -> Dict:
        """Calculate position and styling for each active word."""
        if not active_words:
            return {}
        
        # Group words by phrase and calculate layouts
        phrase_layouts = {}
        phrase_groups = {}
        
        # Group by phrase
        for word_data in active_words:
            phrase = word_data['phrase']
            phrase_id = id(phrase)
            
            if phrase_id not in phrase_groups:
                phrase_groups[phrase_id] = {
                    'phrase': phrase,
                    'words': []
                }
            
            phrase_groups[phrase_id]['words'].append(word_data)
        
        # Calculate layout for each phrase
        for phrase_id, group in phrase_groups.items():
            phrase = group['phrase']
            phrase_words = group['words']
            
            # Calculate font size
            base_font_size = 48
            font_size = int(base_font_size * phrase.font_size_multiplier)
            
            # Split into lines
            all_words = phrase.words
            lines = self.text_renderer.split_text_into_lines(all_words)
            
            # Calculate total text dimensions
            max_line_width = max(self.text_renderer.measure_text(line, font_size)[0] for line in lines)
            total_height = len(lines) * int(font_size * 1.3)
            
            # Position based on preference
            if phrase.position == "top":
                y_base = int(self.H * 0.25)  # Top zone
            else:
                y_base = int(self.H * 0.75)  # Bottom zone
            
            x_base = (self.W - max_line_width) // 2  # Center horizontally
            
            # Calculate position for each word
            word_positions = {}
            current_line = 0
            current_x = x_base
            word_index_in_phrase = 0
            
            for line_idx, line_text in enumerate(lines):
                line_words = line_text.split()
                line_y = y_base + line_idx * int(font_size * 1.3)
                
                # Reset x for each line
                line_width = self.text_renderer.measure_text(line_text, font_size)[0]
                line_x_start = (self.W - line_width) // 2
                current_x = line_x_start
                
                for word in line_words:
                    # Find this word in our active words
                    for word_data in phrase_words:
                        if word_data['word_index'] == word_index_in_phrase:
                            word_w = self.text_renderer.measure_text(word, font_size)[0]
                            
                            word_positions[word_data['word_index']] = {
                                'x': current_x,
                                'y': line_y,
                                'width': word_w,
                                'height': int(font_size * 1.2),
                                'font_size': font_size,
                                'animation_progress': word_data['animation_progress']
                            }
                            
                            current_x += word_w + int(font_size * 0.3)  # Add spacing
                            break
                    
                    word_index_in_phrase += 1
            
            phrase_layouts[phrase_id] = word_positions
        
        return phrase_layouts
    
    def apply_slide_animation(self, word_pos: Dict, animation_progress: float) -> Dict:
        """Apply slide-from-above animation to word position."""
        slide_distance = 40
        
        # Animate Y position (slide from above)
        original_y = word_pos['y']
        start_y = original_y - slide_distance
        current_y = start_y + (slide_distance * animation_progress)
        
        # Apply fade-in
        opacity = animation_progress
        
        return {
            **word_pos,
            'y': int(current_y),
            'opacity': opacity
        }
    
    def render_frame(self, frame: np.ndarray, mask: np.ndarray, 
                    current_time: float) -> np.ndarray:
        """Render frame with text behind head."""
        # Get active words
        active_words = self.get_active_words_at_time(current_time)
        if not active_words:
            return frame
        
        # Calculate layout
        layouts = self.calculate_layout(active_words)
        if not layouts:
            return frame
        
        # Create text layer (same size as video)
        text_layer = np.zeros((self.H, self.W, 4), dtype=np.uint8)  # BGRA
        
        # Render each word
        for phrase_id, word_positions in layouts.items():
            # Find the phrase for styling
            phrase = None
            for word_data in active_words:
                if id(word_data['phrase']) == phrase_id:
                    phrase = word_data['phrase']
                    break
            
            if not phrase:
                continue
            
            for word_data in active_words:
                if id(word_data['phrase']) != phrase_id:
                    continue
                
                word_idx = word_data['word_index']
                if word_idx not in word_positions:
                    continue
                
                word_pos = word_positions[word_idx]
                
                # Apply animation
                animated_pos = self.apply_slide_animation(word_pos, word_pos['animation_progress'])
                
                if animated_pos['opacity'] <= 0:
                    continue
                
                # Render word
                word_text = word_data['word']
                font_size = animated_pos['font_size']
                
                word_img = self.text_renderer.render_text_with_border(
                    word_text, font_size, phrase.bold, phrase.color_tint
                )
                
                # Apply opacity
                if animated_pos['opacity'] < 1.0:
                    word_img[:, :, 3] = (word_img[:, :, 3] * animated_pos['opacity']).astype(np.uint8)
                
                # Position on text layer
                x = animated_pos['x']
                y = animated_pos['y']
                
                # Ensure text fits in frame
                word_h, word_w = word_img.shape[:2]
                
                # Clip to frame bounds
                x_start = max(0, x)
                y_start = max(0, y)
                x_end = min(self.W, x + word_w)
                y_end = min(self.H, y + word_h)
                
                if x_end <= x_start or y_end <= y_start:
                    continue
                
                # Calculate source region
                src_x_start = x_start - x
                src_y_start = y_start - y
                src_x_end = src_x_start + (x_end - x_start)
                src_y_end = src_y_start + (y_end - y_start)
                
                # Extract regions
                dst_region = text_layer[y_start:y_end, x_start:x_end]
                src_region = word_img[src_y_start:src_y_end, src_x_start:src_x_end]
                
                # Alpha blend
                alpha = src_region[:, :, 3:4].astype(np.float32) / 255.0
                
                # Blend RGB channels
                for c in range(3):
                    dst_region[:, :, c] = (
                        (1.0 - alpha[:, :, 0]) * dst_region[:, :, c] + 
                        alpha[:, :, 0] * src_region[:, :, c]
                    ).astype(np.uint8)
                
                # Combine alpha channels
                dst_region[:, :, 3] = np.maximum(
                    dst_region[:, :, 3],
                    src_region[:, :, 3]
                )
        
        # Now composite with mask
        # Mask: white = person (foreground), black = background
        # We want text to appear only in background areas
        
        # Convert mask to 0-1 range
        mask_norm = mask.astype(np.float32) / 255.0
        
        # Background mask (where text should appear)
        # Invert mask: black areas (background) = 1, white areas (person) = 0
        bg_mask = 1.0 - mask_norm
        
        # Apply background mask to text layer alpha
        text_alpha = text_layer[:, :, 3:4].astype(np.float32) / 255.0
        masked_text_alpha = text_alpha * bg_mask[:, :, np.newaxis]
        
        # Composite text behind person
        frame_float = frame.astype(np.float32)
        text_float = text_layer[:, :, :3].astype(np.float32)
        
        # Blend text with original frame
        result = frame_float.copy()
        for c in range(3):
            result[:, :, c] = (
                (1.0 - masked_text_alpha[:, :, 0]) * frame_float[:, :, c] + 
                masked_text_alpha[:, :, 0] * text_float[:, :, c]
            )
        
        return result.astype(np.uint8)
    
    def process_video(self, output_path: str):
        """Process entire video with text-behind-head effect."""
        print(f"Processing video: {self.video_path}")
        print(f"Output: {output_path}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.W, self.H))
        
        frame_count = 0
        
        try:
            while True:
                # Read frames
                ret1, frame = self.cap.read()
                ret2, mask_frame = self.mask_cap.read()
                
                if not ret1 or not ret2:
                    break
                
                # Current time in seconds
                current_time = frame_count / self.fps
                
                # Convert mask to grayscale
                if len(mask_frame.shape) == 3:
                    mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                else:
                    mask = mask_frame
                
                # Render frame with text
                result_frame = self.render_frame(frame, mask, current_time)
                
                # Write frame
                out.write(result_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Progress every second
                    print(f"Processed {frame_count}/{self.total_frames} frames ({current_time:.1f}s)")
        
        finally:
            # Cleanup
            self.cap.release()
            self.mask_cap.release()
            out.release()
        
        print(f"Video processing complete: {output_path}")
        
        # Convert to H.264 for compatibility
        h264_output = output_path.replace('.mp4', '_h264.mp4')
        self._convert_to_h264(output_path, h264_output)
    
    def _convert_to_h264(self, input_path: str, output_path: str):
        """Convert video to H.264 format for compatibility."""
        import subprocess
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-c:a', 'copy',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"H.264 conversion complete: {output_path}")
            
            # Remove intermediate file
            os.remove(input_path)
            
        except subprocess.CalledProcessError as e:
            print(f"H.264 conversion failed: {e}")
            print(f"Keeping original file: {input_path}")


def main():
    parser = argparse.ArgumentParser(description='Render text behind person using mask')
    parser.add_argument('video_path', help='Input video file')
    parser.add_argument('mask_video_path', help='Mask video file (white=person, black=background)')
    parser.add_argument('transcript_path', help='Enriched transcript JSON file')
    parser.add_argument('-o', '--output', help='Output video file', 
                       default='output_text_behind_head.mp4')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    if not os.path.exists(args.mask_video_path):
        print(f"Error: Mask video file not found: {args.mask_video_path}")
        return 1
    
    if not os.path.exists(args.transcript_path):
        print(f"Error: Transcript file not found: {args.transcript_path}")
        return 1
    
    # Process video
    renderer = TextBehindHeadRenderer(args.video_path, args.mask_video_path, args.transcript_path)
    renderer.process_video(args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())