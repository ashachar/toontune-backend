"""
Full Word-Level Text Animation Pipeline for AI Math Video
Processes entire transcript with word-by-word animation and fog dissolve
"""

import cv2
import numpy as np
import sys
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
import random

@dataclass
class WordObject:
    """Individual word with persistent position and state"""
    text: str
    x: int  # Fixed X position (never changes)
    y: int  # Fixed Y position (never changes)
    width: int
    height: int
    start_time: float
    end_time: float
    rise_duration: float
    from_below: bool  # Direction for this word's sentence
    # Fog parameters (randomized once, then fixed)
    blur_x: float
    blur_y: float
    fog_density: float
    dissolve_speed: float

@dataclass
class SentenceData:
    """Data for a complete sentence"""
    text: str
    start_time: float
    end_time: float
    fog_start: float
    fog_end: float
    from_below: bool
    words: List[WordObject] = None

class WordLevelPipeline:
    """
    Pipeline that maintains word-level objects throughout all animations
    """
    
    def __init__(self, font_size=48):
        self.font_size = font_size
        self.color = (255, 255, 255)
        
        # Font for measurements
        try:
            self.font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
        except:
            self.font = ImageFont.load_default()
        
        # Store all word objects
        self.word_objects: List[WordObject] = []
        self.sentences: List[SentenceData] = []
    
    def create_word_objects(self, word_text: str, start_time: float, duration: float,
                           center: Tuple[int, int], from_below: bool) -> WordObject:
        """Create a single word object with timing"""
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        
        bbox = draw.textbbox((0, 0), word_text, font=self.font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        return WordObject(
            text=word_text,
            x=0,  # Will be set when positioning sentence
            y=center[1],
            width=width,
            height=height,
            start_time=start_time,
            end_time=start_time + duration,
            rise_duration=0.8,
            from_below=from_below,
            blur_x=random.uniform(0.8, 1.2),
            blur_y=random.uniform(0.8, 1.2),
            fog_density=random.uniform(0.9, 1.1),
            dissolve_speed=random.uniform(0.95, 1.05)
        )
    
    def position_sentence_words(self, words: List[WordObject], center: Tuple[int, int]):
        """Calculate and set fixed positions for all words in a sentence"""
        if not words:
            return
        
        # Calculate total width including spaces
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        space_width = draw.textbbox((0, 0), " ", font=self.font)[2]
        
        total_width = sum(w.width for w in words) + space_width * (len(words) - 1)
        
        # Position words
        start_x = center[0] - total_width // 2
        current_x = start_x
        
        for word in words:
            word.x = current_x
            word.y = center[1]
            current_x += word.width + space_width
    
    def parse_transcript_to_sentences(self, transcript_path: str) -> List[SentenceData]:
        """Parse transcript and create sentence data with proper timing"""
        with open(transcript_path, 'r') as f:
            transcript = json.load(f)
        
        sentences = []
        segments = transcript['segments']
        
        # Process each segment as a sentence
        for i, segment in enumerate(segments):
            text = segment['text'].strip()
            if not text:
                continue
            
            # Alternate direction for visual variety
            from_below = (i % 2 == 0)
            
            # Calculate fog timing (0.8s after sentence ends)
            fog_start = segment['end'] + 0.5
            fog_end = fog_start + 0.8
            
            sentence = SentenceData(
                text=text,
                start_time=segment['start'],
                end_time=segment['end'],
                fog_start=fog_start,
                fog_end=fog_end,
                from_below=from_below
            )
            sentences.append(sentence)
        
        return sentences
    
    def create_words_for_sentence(self, sentence: SentenceData, center: Tuple[int, int]):
        """Create word objects for a sentence with proper timing"""
        words_text = sentence.text.split()
        if not words_text:
            return []
        
        # Calculate timing for each word
        total_duration = sentence.end_time - sentence.start_time
        word_duration = total_duration / len(words_text) * 0.8  # 80% for words, 20% gaps
        
        word_objects = []
        current_time = sentence.start_time
        
        for word in words_text:
            word_obj = self.create_word_objects(
                word, current_time, word_duration,
                center, sentence.from_below
            )
            word_objects.append(word_obj)
            current_time += word_duration * 1.25  # Add small gap between words
        
        # Position all words
        self.position_sentence_words(word_objects, center)
        
        return word_objects
    
    def render_word(self, word_obj: WordObject, frame: np.ndarray, 
                   time_seconds: float, fog_progress: float = 0.0, 
                   is_dissolved: bool = False) -> np.ndarray:
        """Render a single word with its current animation state"""
        
        # If word has fully dissolved, don't render it
        if is_dissolved:
            return frame
        
        # Determine word visibility and animation progress
        if time_seconds < word_obj.start_time:
            return frame  # Word hasn't started yet
        
        # Calculate rise animation progress
        rise_progress = 1.0
        if time_seconds < word_obj.start_time + word_obj.rise_duration:
            rise_progress = (time_seconds - word_obj.start_time) / word_obj.rise_duration
        
        # Create word image with padding for effects
        padding = 100
        canvas_width = word_obj.width + padding * 2
        canvas_height = word_obj.height + padding * 2
        word_img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(word_img)
        
        # Calculate rise offset (only during rise animation)
        y_offset = 0
        opacity = 1.0
        if rise_progress < 1.0:
            # Smooth easing
            eased_progress = (1 - np.cos(rise_progress * np.pi)) / 2
            opacity = eased_progress
            
            # Rise from below or above
            if word_obj.from_below:
                y_offset = int((1 - eased_progress) * 50)
            else:
                y_offset = int((eased_progress - 1) * 50)
        
        # Draw word with opacity
        text_color = (255, 255, 255, int(255 * opacity))
        draw.text((padding, padding), word_obj.text, fill=text_color, font=self.font)
        
        # Convert to numpy
        word_array = np.array(word_img)
        
        # Apply fog effect if needed (but NOT position change!)
        if fog_progress > 0:
            word_array = self.apply_fog_to_word(word_array, word_obj, fog_progress)
        
        # Convert RGBA to BGRA for OpenCV
        word_bgr = np.zeros_like(word_array)
        word_bgr[:, :, 0] = word_array[:, :, 2]  # B = R
        word_bgr[:, :, 1] = word_array[:, :, 1]  # G = G
        word_bgr[:, :, 2] = word_array[:, :, 0]  # R = B
        word_bgr[:, :, 3] = word_array[:, :, 3]  # A = A
        
        # Apply to frame at FIXED position (only y_offset during rise)
        actual_x = word_obj.x - padding
        actual_y = word_obj.y + y_offset - padding
        
        # Ensure within bounds
        y_start = max(0, actual_y)
        y_end = min(frame.shape[0], actual_y + word_bgr.shape[0])
        x_start = max(0, actual_x)
        x_end = min(frame.shape[1], actual_x + word_bgr.shape[1])
        
        if y_end > y_start and x_end > x_start:
            # Calculate sprite region
            sprite_y_start = max(0, -actual_y)
            sprite_y_end = sprite_y_start + (y_end - y_start)
            sprite_x_start = max(0, -actual_x)
            sprite_x_end = sprite_x_start + (x_end - x_start)
            
            sprite_region = word_bgr[sprite_y_start:sprite_y_end, 
                                    sprite_x_start:sprite_x_end]
            
            if sprite_region.shape[2] == 4:
                alpha = sprite_region[:, :, 3].astype(np.float32) / 255.0
                
                for c in range(3):
                    frame[y_start:y_end, x_start:x_end, c] = (
                        frame[y_start:y_end, x_start:x_end, c].astype(np.float32) * (1.0 - alpha) +
                        sprite_region[:, :, c].astype(np.float32) * alpha
                    ).astype(np.uint8)
        
        return frame
    
    def apply_fog_to_word(self, word_img: np.ndarray, word_obj: WordObject, 
                         progress: float) -> np.ndarray:
        """Apply fog effect to word image without changing position"""
        if progress <= 0:
            return word_img
        
        result = word_img.copy()
        
        # Adjust progress with word's dissolve speed
        adjusted_progress = min(1.0, progress * word_obj.dissolve_speed)
        
        # Phase 1: Progressive blur
        if adjusted_progress > 0:
            blur_amount = adjusted_progress * 15
            blur_x = blur_amount * word_obj.blur_x
            blur_y = blur_amount * word_obj.blur_y
            
            if blur_x > 0 or blur_y > 0:
                result = cv2.GaussianBlur(result, (0, 0), 
                                         sigmaX=blur_x, sigmaY=blur_y)
        
        # Phase 2: Fog texture
        if adjusted_progress > 0.3:
            fog_progress = (adjusted_progress - 0.3) / 0.5
            h, w = result.shape[:2]
            fog = np.random.randn(h, w) * 20 * fog_progress
            fog = gaussian_filter(fog, sigma=3)
            
            if result.shape[2] == 4:
                alpha = result[:, :, 3].astype(np.float32)
                alpha = alpha * (1.0 - fog_progress * 0.5)
                alpha = np.clip(alpha + fog * word_obj.fog_density, 0, 255)
                result[:, :, 3] = alpha.astype(np.uint8)
        
        # Phase 3: Final fade
        if adjusted_progress > 0.6:
            fade_progress = (adjusted_progress - 0.6) / 0.4
            fade_amount = 1.0 - (fade_progress * 0.9)
            
            if result.shape[2] == 4:
                result[:, :, 3] = (result[:, :, 3] * fade_amount).astype(np.uint8)
        
        return result
    
    def process_frame(self, frame: np.ndarray, time_seconds: float) -> np.ndarray:
        """Process frame by rendering all active words"""
        result = frame.copy()
        
        # Render each word based on its sentence timing
        for sentence in self.sentences:
            # Check if sentence should be dissolved
            is_dissolved = time_seconds > sentence.fog_end
            
            # Calculate fog progress for this sentence
            fog_progress = 0.0
            if sentence.fog_start <= time_seconds <= sentence.fog_end:
                fog_progress = (time_seconds - sentence.fog_start) / (sentence.fog_end - sentence.fog_start)
            
            # Render all words in this sentence
            if sentence.words:
                for word in sentence.words:
                    result = self.render_word(word, result, time_seconds, 
                                            fog_progress, is_dissolved)
        
        return result


def create_full_video_pipeline():
    """Create full video with word-level tracking throughout"""
    
    print("Creating Full Word-Level Animation Pipeline")
    print("=" * 60)
    print("Processing entire AI Math video with all sentences")
    print()
    
    # Input/output paths
    input_video = "uploads/assets/videos/ai_math1.mp4"
    transcript_path = "uploads/assets/videos/ai_math1/transcript.json"
    output_path = "outputs/ai_math1_full_word_animation.mp4"
    
    # Create pipeline
    pipeline = WordLevelPipeline(font_size=48)
    
    # Parse transcript into sentences
    print("Parsing transcript...")
    pipeline.sentences = pipeline.parse_transcript_to_sentences(transcript_path)
    print(f"Found {len(pipeline.sentences)} sentences")
    
    # Create word objects for each sentence
    print("Creating word objects...")
    center_position = (640, 360)  # Center of 1280x720 video
    
    for i, sentence in enumerate(pipeline.sentences):
        sentence.words = pipeline.create_words_for_sentence(sentence, center_position)
        pipeline.word_objects.extend(sentence.words)
        
        # Show first few sentences for verification
        if i < 5:
            direction = "below" if sentence.from_below else "above"
            print(f"  Sentence {i+1} ({len(sentence.words)} words, from {direction}): "
                  f"{sentence.text[:50]}...")
    
    print(f"Total word objects created: {len(pipeline.word_objects)}")
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Total frames: {total_frames}")
    
    # Create output video (temporary, without audio)
    temp_output = "outputs/temp_full_word_animation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    print("\nRendering animation...")
    print("This may take a few minutes for the full video...")
    
    frame_count = 0
    last_progress = -1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        time_seconds = frame_count / fps
        
        # Process frame with word animations
        animated_frame = pipeline.process_frame(frame, time_seconds)
        
        # Add subtle progress indicator
        progress_text = f"Time: {time_seconds:.1f}s / {duration:.1f}s"
        cv2.putText(animated_frame, progress_text,
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, (150, 150, 150), 1)
        
        out.write(animated_frame)
        
        # Progress reporting
        progress = int(frame_count * 100 / total_frames)
        if progress != last_progress and progress % 5 == 0:
            print(f"  Progress: {progress}% ({frame_count}/{total_frames} frames)")
            last_progress = progress
        
        frame_count += 1
    
    out.release()
    cap.release()
    
    print("\nMerging with audio and converting to H.264...")
    
    # Merge with audio and convert to H.264
    final_output = output_path.replace('.mp4', '_h264.mp4')
    os.system(f"""ffmpeg -i {temp_output} -i {input_video} \
        -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
        -c:a copy -map 0:v:0 -map 1:a:0? \
        -movflags +faststart {final_output} -y 2>/dev/null""")
    
    # Clean up temp file
    os.remove(temp_output)
    
    print(f"\n✅ Full video created: {final_output}")
    print("\nAnimation features:")
    print("  • All sentences from transcript animated")
    print("  • Words enter from alternating directions")
    print("  • Fixed positions maintained throughout")
    print("  • Fog dissolve after each sentence")
    print("  • Original audio preserved")
    print(f"  • Total processing: {len(pipeline.sentences)} sentences, {len(pipeline.word_objects)} words")
    
    return final_output


if __name__ == "__main__":
    print("FULL WORD-LEVEL TEXT ANIMATION PIPELINE")
    print("=" * 60)
    print("Complete transcript animation with professional effects")
    print()
    
    output = create_full_video_pipeline()
    
    if output:
        print(f"\n✨ Success! Full video ready: {output}")
        print("\nThis demonstrates the complete word-level animation system")
        print("applied to an entire video with perfect synchronization.")