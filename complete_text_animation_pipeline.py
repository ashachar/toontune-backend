"""
Complete Text Animation Pipeline
Combines word rise animation with fog dissolve for sentences
"""

import cv2
import numpy as np
import sys
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/word_3d')

# Import our animations
from fog_dissolve import CleanFogDissolve
from base_3d_text_animation import Animation3DConfig
from word_3d import WordRiseSequence3D

@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    x_position: int
    y_position: int

@dataclass
class SentenceAnimation:
    full_text: str
    words: List[WordTiming]
    sentence_start: float
    sentence_end: float
    fog_dissolve_start: float
    fog_dissolve_end: float
    position: Tuple[int, int]

class CompleteTextPipeline:
    """
    Complete pipeline for text animations:
    1. Words rise/slide in individually
    2. Full sentence displays
    3. Fog dissolve transition
    4. Next sentence begins
    """
    
    def __init__(self, font_size=55):
        self.font_size = font_size
        self.fog_effect = CleanFogDissolve(font_size=font_size)
        
        # Font for measurements
        try:
            self.font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
        except:
            self.font = ImageFont.load_default()
    
    def calculate_word_positions(self, sentence: str, center: Tuple[int, int]) -> List[Dict]:
        """Calculate fixed positions for all words in sentence"""
        words = sentence.split()
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        
        # Get total width
        total_width = 0
        word_widths = []
        space_width = draw.textbbox((0, 0), " ", font=self.font)[2]
        
        for word in words:
            bbox = draw.textbbox((0, 0), word, font=self.font)
            width = bbox[2] - bbox[0]
            word_widths.append(width)
            total_width += width
        
        total_width += space_width * (len(words) - 1)
        
        # Calculate positions
        start_x = center[0] - total_width // 2
        current_x = start_x
        positions = []
        
        for i, word in enumerate(words):
            positions.append({
                'word': word,
                'x': current_x,
                'y': center[1],
                'width': word_widths[i]
            })
            current_x += word_widths[i] + space_width
        
        return positions
    
    def render_word_rise(self, word: str, frame: np.ndarray, x: int, y: int, 
                        progress: float, from_below: bool = True) -> np.ndarray:
        """Render a single word with rise animation"""
        if progress <= 0:
            return frame
        
        # Create word image
        padding = 50
        word_img = Image.new('RGBA', (300, 150), (0, 0, 0, 0))
        draw = ImageDraw.Draw(word_img)
        
        # Draw word
        bbox = draw.textbbox((0, 0), word, font=self.font)
        word_width = bbox[2] - bbox[0]
        word_height = bbox[3] - bbox[1]
        
        # Smooth easing for rise
        eased_progress = (1 - np.cos(progress * np.pi)) / 2
        
        # Calculate opacity
        opacity = eased_progress
        
        # Calculate position offset (rise from below or above)
        if from_below:
            y_offset = int((1 - eased_progress) * 50)  # Rise from 50px below
        else:
            y_offset = int((eased_progress - 1) * 50)  # Drop from 50px above
        
        # Draw with opacity
        text_color = (255, 255, 255, int(255 * opacity))
        draw.text((padding, padding), word, fill=text_color, font=self.font)
        
        # Convert to numpy array
        word_array = np.array(word_img)
        
        # Convert RGBA to BGRA for OpenCV
        word_bgr = word_array.copy()
        word_bgr[:, :, 0] = word_array[:, :, 2]
        word_bgr[:, :, 2] = word_array[:, :, 0]
        
        # Apply to frame at position
        actual_y = y + y_offset - padding
        actual_x = x - padding
        
        y_start = max(0, actual_y)
        y_end = min(frame.shape[0], actual_y + word_bgr.shape[0])
        x_start = max(0, actual_x)
        x_end = min(frame.shape[1], actual_x + word_bgr.shape[1])
        
        if y_end > y_start and x_end > x_start:
            sprite_region = word_bgr[:y_end-actual_y, :x_end-actual_x]
            alpha = sprite_region[:, :, 3] / 255.0
            
            for c in range(3):
                frame[y_start:y_end, x_start:x_end, c] = (
                    frame[y_start:y_end, x_start:x_end, c] * (1.0 - alpha) +
                    sprite_region[:, :, c] * alpha
                ).astype(np.uint8)
        
        return frame
    
    def render_full_sentence(self, sentence: str, frame: np.ndarray, 
                           position: Tuple[int, int]) -> np.ndarray:
        """Render complete sentence (used after all words have risen)"""
        # Simply render all words at their positions
        positions = self.calculate_word_positions(sentence, position)
        
        for word_pos in positions:
            frame = self.render_word_rise(
                word_pos['word'], frame, 
                word_pos['x'], word_pos['y'], 
                1.0  # Full opacity
            )
        
        return frame
    
    def process_frame(self, frame: np.ndarray, time_seconds: float, 
                     sentences: List[SentenceAnimation]) -> np.ndarray:
        """Process a single frame with all animations"""
        result = frame.copy()
        
        for sentence in sentences:
            # Check if this sentence is active
            if time_seconds < sentence.sentence_start or \
               time_seconds > sentence.fog_dissolve_end:
                continue
            
            # Phase 1: Words rise individually
            if time_seconds < sentence.sentence_end:
                # Calculate word positions once
                word_positions = self.calculate_word_positions(
                    sentence.full_text, sentence.position
                )
                
                # Render each word based on its timing
                for i, word_timing in enumerate(sentence.words):
                    if time_seconds >= word_timing.start:
                        # Calculate word animation progress
                        if time_seconds < word_timing.end:
                            progress = (time_seconds - word_timing.start) / \
                                     (word_timing.end - word_timing.start)
                        else:
                            progress = 1.0
                        
                        # Alternate between rising from below and above
                        from_below = (i % 2 == 0)
                        
                        # Render word at its position
                        if i < len(word_positions):
                            result = self.render_word_rise(
                                word_timing.word, result,
                                word_positions[i]['x'],
                                word_positions[i]['y'],
                                progress,
                                from_below
                            )
            
            # Phase 2: Fog dissolve out
            elif time_seconds >= sentence.fog_dissolve_start:
                fog_progress = (time_seconds - sentence.fog_dissolve_start) / \
                              (sentence.fog_dissolve_end - sentence.fog_dissolve_start)
                fog_progress = min(1.0, fog_progress)
                
                # Apply fog dissolve to entire sentence
                result = self.fog_effect.apply_clean_fog(
                    sentence.full_text, result, 
                    sentence.position, fog_progress
                )
        
        return result


def create_complete_pipeline_video():
    """Create 6-second video with complete text animation pipeline"""
    
    print("Creating Complete Text Animation Pipeline")
    print("=" * 60)
    print("Effects:")
    print("  1. Words rise/slide in individually")
    print("  2. Complete sentence displays")
    print("  3. Fog dissolve transition")
    print("  4. Next sentence begins")
    print()
    
    # Extract 6-second segment
    input_video = "uploads/assets/videos/ai_math1.mp4"
    temp_segment = "outputs/ai_math1_6sec_segment.mp4"
    
    print("Extracting 6-second segment...")
    os.system(f"ffmpeg -i {input_video} -t 6 -c:v libx264 -preset fast -crf 18 {temp_segment} -y 2>/dev/null")
    
    # Load transcript
    transcript_path = "uploads/assets/videos/ai_math1/transcript.json"
    with open(transcript_path, 'r') as f:
        transcript = json.load(f)
    
    # Create sentence animations from transcript
    sentences_data = []
    
    # Sentence 1: "Yes, AI created new math." (0-2.8s)
    sentence1 = SentenceAnimation(
        full_text="Yes, AI created new math.",
        words=[
            WordTiming("Yes,", 0.0, 0.4, 0, 0),
            WordTiming("AI", 0.4, 0.8, 0, 0),
            WordTiming("created", 0.8, 1.4, 0, 0),
            WordTiming("new", 1.4, 1.8, 0, 0),
            WordTiming("math.", 1.8, 2.4, 0, 0),
        ],
        sentence_start=0.0,
        sentence_end=2.4,
        fog_dissolve_start=2.4,
        fog_dissolve_end=2.8,
        position=(640, 360)
    )
    sentences_data.append(sentence1)
    
    # Sentence 2: Part of "Would you be surprised..." (2.8-6s)
    sentence2 = SentenceAnimation(
        full_text="Would you be surprised if AI",
        words=[
            WordTiming("Would", 2.8, 3.2, 0, 0),
            WordTiming("you", 3.2, 3.5, 0, 0),
            WordTiming("be", 3.5, 3.8, 0, 0),
            WordTiming("surprised", 3.8, 4.4, 0, 0),
            WordTiming("if", 4.4, 4.6, 0, 0),
            WordTiming("AI", 4.6, 5.0, 0, 0),
        ],
        sentence_start=2.8,
        sentence_end=5.0,
        fog_dissolve_start=5.0,
        fog_dissolve_end=5.8,
        position=(640, 360)
    )
    sentences_data.append(sentence2)
    
    print(f"\nConfigured {len(sentences_data)} sentences with animations")
    
    # Open video
    cap = cv2.VideoCapture(temp_segment)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} fps")
    print(f"Total frames: {total_frames}")
    
    # Create output video
    output_path = "outputs/complete_text_pipeline_6sec.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create pipeline
    pipeline = CompleteTextPipeline(font_size=55)
    
    print("\nRendering animation pipeline...")
    
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        time_seconds = frame_num / fps
        
        # Process frame with animations
        animated_frame = pipeline.process_frame(frame, time_seconds, sentences_data)
        
        # Add timeline info
        cv2.putText(animated_frame, 
                   f"Complete Pipeline | {time_seconds:.2f}s / 6.0s", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1)
        
        # Add current phase indicator
        phase = "Waiting"
        for sentence in sentences_data:
            if sentence.sentence_start <= time_seconds < sentence.sentence_end:
                phase = f"Words Rising: {sentence.full_text[:20]}..."
            elif sentence.fog_dissolve_start <= time_seconds < sentence.fog_dissolve_end:
                phase = "Fog Dissolve"
        
        cv2.putText(animated_frame, phase, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (100, 255, 100), 2)
        
        out.write(animated_frame)
        
        if frame_num % 25 == 0:
            progress = frame_num / total_frames
            print(f"  Frame {frame_num}/{total_frames} ({progress*100:.1f}%)")
    
    out.release()
    cap.release()
    
    # Convert to H.264
    print("\nConverting to H.264...")
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    os.system(f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -c:a copy -movflags +faststart {h264_output} -y 2>/dev/null")
    os.remove(output_path)
    
    # Clean up temp file
    os.remove(temp_segment)
    
    print(f"\n✅ Complete pipeline video created: {h264_output}")
    print("\nAnimation sequence:")
    print("  0.0-2.4s: 'Yes, AI created new math.' - words rise in")
    print("  2.4-2.8s: Fog dissolve transition")
    print("  2.8-5.0s: 'Would you be surprised if AI' - words rise in")
    print("  5.0-5.8s: Fog dissolve transition")
    print("  5.8-6.0s: Ready for next sentence")
    
    return h264_output


if __name__ == "__main__":
    print("COMPLETE TEXT ANIMATION PIPELINE")
    print("=" * 60)
    print("6-second demonstration combining:")
    print("  • Word rise animations (alternating from below/above)")
    print("  • Fixed word positions in sentences")
    print("  • Fog dissolve transitions")
    print("  • Smooth sentence-to-sentence flow")
    print()
    
    output = create_complete_pipeline_video()
    
    if output:
        print(f"\n✨ Success! Pipeline video ready: {output}")
        print("\nThe video demonstrates the full text animation workflow")
        print("with professional transitions between sentences.")