"""
Create video with fixed word positions - words appear in predetermined locations
No shifting or reflow when new words enter
"""

import cv2
import numpy as np
import sys
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class WordLayout:
    """Stores the predetermined position and dimensions of a word"""
    word: str
    x: int
    y: int
    width: int
    height: int
    start_time: float
    end_time: float

class FixedPositionRenderer:
    """Renders words at fixed, predetermined positions"""
    
    def __init__(self, font_size=55, color=(255, 255, 255)):
        self.font_size = font_size
        self.color = color
        
        # Load font
        try:
            self.font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
        except:
            self.font = ImageFont.load_default()
        
        # Cache for rendered word images
        self.word_cache = {}
    
    def calculate_sentence_layout(self, sentence: str, frame_width: int = 1280, 
                                 frame_height: int = 720) -> List[WordLayout]:
        """
        Calculate fixed positions for all words in a sentence.
        Returns list of WordLayout objects with predetermined positions.
        """
        words = sentence.split()
        layouts = []
        
        # Create temporary image for text measurements
        temp_img = Image.new('RGBA', (frame_width, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        
        # Calculate total width needed for the sentence
        word_widths = []
        space_width = draw.textbbox((0, 0), " ", font=self.font)[2]
        
        total_width = 0
        for word in words:
            bbox = draw.textbbox((0, 0), word, font=self.font)
            word_width = bbox[2] - bbox[0]
            word_height = bbox[3] - bbox[1]
            word_widths.append((word_width, word_height))
            total_width += word_width
        
        # Add spaces between words
        total_width += space_width * (len(words) - 1)
        
        # Center the sentence horizontally
        start_x = (frame_width - total_width) // 2
        y_position = 360  # Center vertically
        
        # Calculate fixed position for each word
        current_x = start_x
        for i, word in enumerate(words):
            word_width, word_height = word_widths[i]
            
            layout = WordLayout(
                word=word,
                x=current_x,
                y=y_position,
                width=word_width,
                height=word_height,
                start_time=0,  # Will be set later
                end_time=0     # Will be set later
            )
            layouts.append(layout)
            
            # Move x position for next word
            current_x += word_width + space_width
        
        return layouts
    
    def render_word(self, word: str, opacity: float = 1.0) -> np.ndarray:
        """Render a single word as a transparent sprite"""
        # Check cache
        cache_key = f"{word}_{opacity:.2f}"
        if cache_key in self.word_cache:
            return self.word_cache[cache_key].copy()
        
        # Create image just big enough for the word
        temp_img = Image.new('RGBA', (500, 150), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        
        # Get word dimensions
        bbox = draw.textbbox((0, 0), word, font=self.font)
        word_width = bbox[2] - bbox[0] + 20  # Add padding
        word_height = bbox[3] - bbox[1] + 20
        
        # Create properly sized image
        img = Image.new('RGBA', (word_width, word_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw word with opacity
        text_color = (*self.color, int(255 * opacity))
        draw.text((10, 10), word, fill=text_color, font=self.font)
        
        # Convert to numpy array (RGBA)
        sprite = np.array(img)
        
        # Convert RGBA to BGRA for OpenCV
        sprite_bgr = sprite.copy()
        sprite_bgr[:, :, 0] = sprite[:, :, 2]  # B = R
        sprite_bgr[:, :, 2] = sprite[:, :, 0]  # R = B
        
        # Cache the result
        self.word_cache[cache_key] = sprite_bgr.copy()
        
        return sprite_bgr
    
    def composite_word(self, frame: np.ndarray, word_sprite: np.ndarray, 
                      x: int, y: int, y_offset: int = 0) -> np.ndarray:
        """Composite a word sprite onto the frame at a fixed position"""
        # Adjust y position for animation (rise/dissolve)
        actual_y = y + y_offset - word_sprite.shape[0] // 2
        actual_x = x
        
        # Ensure we're within frame bounds
        y_start = max(0, actual_y)
        y_end = min(frame.shape[0], actual_y + word_sprite.shape[0])
        x_start = max(0, actual_x)
        x_end = min(frame.shape[1], actual_x + word_sprite.shape[1])
        
        # Calculate sprite region to use
        sprite_y_start = max(0, -actual_y)
        sprite_y_end = sprite_y_start + (y_end - y_start)
        sprite_x_start = max(0, -actual_x)
        sprite_x_end = sprite_x_start + (x_end - x_start)
        
        if y_end > y_start and x_end > x_start:
            # Get alpha channel
            alpha = word_sprite[sprite_y_start:sprite_y_end, 
                              sprite_x_start:sprite_x_end, 3] / 255.0
            
            # Composite using alpha blending
            for c in range(3):
                frame[y_start:y_end, x_start:x_end, c] = (
                    frame[y_start:y_end, x_start:x_end, c] * (1.0 - alpha) +
                    word_sprite[sprite_y_start:sprite_y_end, 
                              sprite_x_start:sprite_x_end, c] * alpha
                ).astype(np.uint8)
        
        return frame

def group_segments_into_sentences(segments):
    """Group segments into sentences based on punctuation"""
    sentences = []
    current_sentence = []
    
    for segment in segments:
        text = segment['text'].strip()
        if not text:
            continue
            
        current_sentence.append(segment)
        
        # Check if segment ends with sentence-ending punctuation
        if text[-1] in '.?!':
            sentences.append(current_sentence)
            current_sentence = []
    
    # Add any remaining segments as a sentence
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

def create_fixed_position_animation():
    """Create video with words appearing at fixed, predetermined positions"""
    
    print("Creating Fixed Position Sentence Animation")
    print("=" * 60)
    print("Words appear at predetermined positions - no shifting!")
    print()
    
    # Load the actual transcript
    transcript_path = "uploads/assets/videos/ai_math1/transcript.json"
    with open(transcript_path, 'r') as f:
        transcript = json.load(f)
    
    # Get segments for first 5 seconds
    segments_in_range = [s for s in transcript['segments'] 
                        if s['start'] < 5.0]
    
    # Group into sentences
    sentences = group_segments_into_sentences(segments_in_range)
    
    print(f"Sentences found in first 5 seconds:")
    for i, sentence_segments in enumerate(sentences):
        full_text = ' '.join([s['text'].strip() for s in sentence_segments])
        start = sentence_segments[0]['start']
        end = sentence_segments[-1]['end']
        print(f"  Sentence {i+1}: {full_text[:50]}...")
        print(f"    Timing: {start:.2f}s - {end:.2f}s")
    
    # Load input video
    input_video = "outputs/ai_math1_5sec.mp4"
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Error: Could not open {input_video}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {width}x{height} @ {fps:.2f} fps")
    print(f"Duration: {total_frames/fps:.2f} seconds")
    
    # Create output video
    output_path = "outputs/ai_math1_5sec_fixed_positions.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create renderer
    renderer = FixedPositionRenderer(font_size=55)
    
    # Process each sentence and calculate layouts
    sentence_data = []
    
    for sentence_idx, sentence_segments in enumerate(sentences):
        # Combine segments into full sentence text
        full_text = ' '.join([s['text'].strip() for s in sentence_segments])
        sentence_start = sentence_segments[0]['start']
        sentence_end = sentence_segments[-1]['end']
        
        # Calculate fixed layout for all words in sentence
        layouts = renderer.calculate_sentence_layout(full_text, width, height)
        
        # Assign timing to each word based on segments
        word_index = 0
        for segment in sentence_segments:
            segment_words = segment['text'].strip().split()
            segment_duration = segment['end'] - segment['start']
            word_duration = segment_duration / len(segment_words) if segment_words else 0
            
            for i, word in enumerate(segment_words):
                if word_index < len(layouts):
                    layouts[word_index].start_time = segment['start'] + (i * word_duration)
                    layouts[word_index].end_time = segment['start'] + ((i + 1) * word_duration)
                    word_index += 1
        
        # Calculate dissolve timing
        if sentence_idx < len(sentences) - 1:
            next_sentence_start = sentences[sentence_idx + 1][0]['start']
            dissolve_start = max(sentence_end, next_sentence_start - 0.5)
        else:
            dissolve_start = min(sentence_end + 1.0, 5.0)
        
        sentence_data.append({
            'layouts': layouts,
            'start': sentence_start,
            'end': sentence_end,
            'dissolve_start': dissolve_start,
            'dissolve_end': dissolve_start + 0.5,
            'sentence_idx': sentence_idx
        })
        
        print(f"\nSentence {sentence_idx + 1} layout:")
        for layout in layouts[:5]:  # Show first 5 words
            print(f"  '{layout.word}' at x={layout.x}, y={layout.y}")
        if len(layouts) > 5:
            print(f"  ... and {len(layouts) - 5} more words")
    
    print(f"\nRendering frames...")
    
    # Process frames
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        animated_frame = frame.copy()
        time_seconds = frame_num / fps
        
        # Process each sentence
        for sent in sentence_data:
            if sent['start'] <= time_seconds < sent['dissolve_end']:
                
                # Render each word at its fixed position
                for layout in sent['layouts']:
                    # Check if this word should be visible
                    if time_seconds >= layout.start_time:
                        # Calculate word state
                        opacity = 1.0
                        y_offset = 0
                        
                        if time_seconds < layout.end_time:
                            # Word is appearing (rising)
                            word_progress = (time_seconds - layout.start_time) / (layout.end_time - layout.start_time)
                            # Smooth easing for rise
                            eased = (1 - np.cos(word_progress * np.pi)) / 2
                            opacity = eased
                            y_offset = int((1 - eased) * 50)  # Rise from 50 pixels below
                        elif time_seconds < sent['dissolve_start']:
                            # Word is fully visible at its fixed position
                            opacity = 1.0
                            y_offset = 0
                        elif time_seconds < sent['dissolve_end']:
                            # Word is dissolving upward from its fixed position
                            dissolve_progress = (time_seconds - sent['dissolve_start']) / 0.5
                            opacity = 1.0 - dissolve_progress
                            y_offset = int(-dissolve_progress * 100)  # Rise upward
                        else:
                            # Word has dissolved
                            continue
                        
                        # Render word at its FIXED position with animation offset
                        if opacity > 0.01:
                            word_sprite = renderer.render_word(layout.word, opacity)
                            animated_frame = renderer.composite_word(
                                animated_frame, word_sprite, 
                                layout.x, layout.y, y_offset
                            )
                
                break  # Only render one sentence at a time
        
        # Add info overlay
        cv2.putText(animated_frame, 
                   f"Fixed Position Animation | {time_seconds:.2f}s / 5.0s", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1)
        
        out.write(animated_frame)
        
        if frame_num % 25 == 0:
            progress = frame_num / total_frames
            print(f"  Frame {frame_num}/{total_frames} ({progress*100:.1f}%)")
    
    out.release()
    cap.release()
    
    # Convert to H.264
    print("\nConverting to H.264...")
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -c:a copy -movflags +faststart {h264_output} -y"
    result = os.system(convert_cmd)
    
    if result == 0:
        os.remove(output_path)
        print(f"✅ Success! Video created: {h264_output}")
    else:
        print(f"⚠️ H.264 conversion failed, keeping original: {output_path}")
        h264_output = output_path
    
    print("\n" + "=" * 60)
    print("FIXED POSITION ANIMATION COMPLETE!")
    print(f"Output: {h264_output}")
    print("\nKey Features:")
    print("  ✓ Each word has a FIXED, predetermined position")
    print("  ✓ Words appear in their final locations - no shifting!")
    print("  ✓ New words don't affect existing word positions")
    print("  ✓ Clean rendering without artifacts")
    print("  ✓ Smooth rise and dissolve animations")
    print("=" * 60)
    
    return h264_output


if __name__ == "__main__":
    print("FIXED POSITION WORD ANIMATION")
    print("=" * 60)
    print("Implementation:")
    print("- Pre-calculates layout for entire sentence")
    print("- Each word knows its final position from the start")
    print("- Words appear at their designated spots")
    print("- No reflow or shifting when new words enter")
    print("- Existing words remain perfectly stable")
    print()
    
    output = create_fixed_position_animation()
    
    if output:
        print(f"\n✨ Success! Final video: {output}")
        print("\nWords now appear at predetermined positions without")
        print("affecting already-visible words!")