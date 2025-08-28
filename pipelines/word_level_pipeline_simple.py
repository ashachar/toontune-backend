"""
Word-Level Text Animation Pipeline
Maintains individual word objects throughout all animation phases
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

class WordLevelPipeline:
    """
    Pipeline that maintains word-level objects throughout all animations
    """
    
    def __init__(self, font_size=55):
        self.font_size = font_size
        self.color = (255, 255, 255)
        
        # Font for measurements
        try:
            self.font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
        except:
            self.font = ImageFont.load_default()
        
        # Store all word objects
        self.word_objects: List[WordObject] = []
    
    def create_sentence_words(self, text: str, word_timings: List[Dict], 
                             center: Tuple[int, int], from_below: bool) -> List[WordObject]:
        """Create word objects for a sentence with fixed positions"""
        words = text.split()
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        
        # Calculate total width
        total_width = 0
        word_measurements = []
        space_width = draw.textbbox((0, 0), " ", font=self.font)[2]
        
        for word in words:
            bbox = draw.textbbox((0, 0), word, font=self.font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            word_measurements.append((width, height))
            total_width += width
        
        total_width += space_width * (len(words) - 1)
        
        # Create word objects with fixed positions
        start_x = center[0] - total_width // 2
        current_x = start_x
        word_objects = []
        
        for i, (word, timing) in enumerate(zip(words, word_timings)):
            width, height = word_measurements[i]
            
            # Create word object with all parameters
            word_obj = WordObject(
                text=word,
                x=current_x,  # Fixed position
                y=center[1],  # Fixed position
                width=width,
                height=height,
                start_time=timing['start'],
                end_time=timing['end'],
                rise_duration=0.8,  # Gentle rise
                from_below=from_below,  # Same direction for entire sentence
                # Fog parameters (randomized once per word)
                blur_x=random.uniform(0.8, 1.2),
                blur_y=random.uniform(0.8, 1.2),
                fog_density=random.uniform(0.9, 1.1),
                dissolve_speed=random.uniform(0.95, 1.05)
            )
            
            word_objects.append(word_obj)
            current_x += width + space_width
        
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
    
    def process_frame(self, frame: np.ndarray, time_seconds: float,
                     sentence_fog_times: List[Tuple[float, float]]) -> np.ndarray:
        """Process frame by rendering all active words"""
        result = frame.copy()
        
        # Determine fog progress and dissolved state for each sentence
        fog_progress_by_sentence = []
        dissolved_by_sentence = []
        for fog_start, fog_end in sentence_fog_times:
            if fog_start <= time_seconds <= fog_end:
                progress = (time_seconds - fog_start) / (fog_end - fog_start)
                fog_progress_by_sentence.append(progress)
                dissolved_by_sentence.append(False)
            elif time_seconds > fog_end:
                fog_progress_by_sentence.append(1.0)
                dissolved_by_sentence.append(True)  # Sentence has fully dissolved
            else:
                fog_progress_by_sentence.append(0.0)
                dissolved_by_sentence.append(False)
        
        # Render each word
        for word_obj in self.word_objects:
            # Determine which sentence this word belongs to
            sentence_index = -1
            for i, (fog_start, fog_end) in enumerate(sentence_fog_times):
                # Word belongs to sentence i if it starts before this fog period
                # and (if there's a next sentence) before the next sentence starts
                if i == len(sentence_fog_times) - 1:
                    # Last sentence
                    if word_obj.start_time < fog_start:
                        sentence_index = i
                        break
                else:
                    next_fog_start = sentence_fog_times[i+1][0]
                    # Check if word is in the time window for this sentence
                    if word_obj.start_time < fog_start and word_obj.start_time < next_fog_start - 1:
                        sentence_index = i
                        break
            
            # Get fog progress and dissolved state for this word's sentence
            fog_progress = 0.0
            is_dissolved = False
            if 0 <= sentence_index < len(fog_progress_by_sentence):
                fog_progress = fog_progress_by_sentence[sentence_index]
                is_dissolved = dissolved_by_sentence[sentence_index]
            
            # Render the word (will skip if dissolved)
            result = self.render_word(word_obj, result, time_seconds, 
                                    fog_progress, is_dissolved)
        
        return result


def create_word_level_video():
    """Create video with word-level tracking throughout"""
    
    print("Creating Word-Level Animation Pipeline")
    print("=" * 60)
    print("Features:")
    print("  • Words tracked individually throughout")
    print("  • Same direction for all words in a sentence")
    print("  • Fixed positions maintained during fog")
    print("  • No position shifts between phases")
    print()
    
    # Extract 6-second segment
    input_video = "uploads/assets/videos/ai_math1.mp4"
    temp_segment = "outputs/ai_math1_6sec_word_level.mp4"
    
    print("Extracting 6-second segment with audio...")
    os.system(f"ffmpeg -i {input_video} -t 6 -c:v libx264 -preset fast -crf 18 -c:a copy {temp_segment} -y 2>/dev/null")
    
    # Create pipeline
    pipeline = WordLevelPipeline(font_size=55)
    
    # Sentence 1: "Yes, AI created new math." (from below)
    sentence1_timings = [
        {'start': 0.0, 'end': 0.4},  # Yes,
        {'start': 0.4, 'end': 0.8},  # AI
        {'start': 0.8, 'end': 1.4},  # created
        {'start': 1.4, 'end': 1.8},  # new
        {'start': 1.8, 'end': 2.4},  # math.
    ]
    
    words1 = pipeline.create_sentence_words(
        "Yes, AI created new math.",
        sentence1_timings,
        (640, 360),
        from_below=True  # All words from below
    )
    pipeline.word_objects.extend(words1)
    
    # Sentence 2: "Would you be surprised if AI" (from above)
    sentence2_timings = [
        {'start': 2.8, 'end': 3.2},  # Would
        {'start': 3.2, 'end': 3.5},  # you
        {'start': 3.5, 'end': 3.8},  # be
        {'start': 3.8, 'end': 4.4},  # surprised
        {'start': 4.4, 'end': 4.6},  # if
        {'start': 4.6, 'end': 5.0},  # AI
    ]
    
    words2 = pipeline.create_sentence_words(
        "Would you be surprised if AI",
        sentence2_timings,
        (640, 360),
        from_below=False  # All words from above
    )
    pipeline.word_objects.extend(words2)
    
    # Fog dissolve times for each sentence
    sentence_fog_times = [
        (2.4, 2.8),  # Sentence 1 fog
        (5.0, 5.8),  # Sentence 2 fog
    ]
    
    print(f"Created {len(pipeline.word_objects)} word objects")
    print("  Sentence 1: from below")
    print("  Sentence 2: from above")
    
    # Open video
    cap = cv2.VideoCapture(temp_segment)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {width}x{height} @ {fps:.2f} fps")
    print(f"Total frames: {total_frames}")
    
    # Create output video (temporary, without audio)
    temp_output = "outputs/word_level_pipeline_temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    print("\nRendering word-level animation...")
    
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        time_seconds = frame_num / fps
        
        # Process frame with word-level rendering
        animated_frame = pipeline.process_frame(frame, time_seconds, sentence_fog_times)
        
        # Add info overlay
        cv2.putText(animated_frame, 
                   f"Word-Level Pipeline | {time_seconds:.2f}s / 6.0s", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1)
        
        # Phase indicator
        phase = "Waiting"
        if 0.0 <= time_seconds < 2.4:
            phase = "Sentence 1: Rising from below"
        elif 2.4 <= time_seconds < 2.8:
            phase = "Sentence 1: Fog dissolve"
        elif 2.8 <= time_seconds < 5.0:
            phase = "Sentence 2: Dropping from above"
        elif 5.0 <= time_seconds < 5.8:
            phase = "Sentence 2: Fog dissolve"
        
        cv2.putText(animated_frame, phase, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (100, 255, 100), 2)
        
        out.write(animated_frame)
        
        if frame_num % 25 == 0:
            progress = frame_num / total_frames
            print(f"  Frame {frame_num}/{total_frames} ({progress*100:.1f}%)")
    
    out.release()
    cap.release()
    
    # Merge with audio and convert to H.264
    print("\nMerging with audio and converting to H.264...")
    final_output = "outputs/word_level_pipeline_h264.mp4"
    
    # Copy audio from original segment and video from processed version
    os.system(f"ffmpeg -i {temp_output} -i {temp_segment} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -c:a copy -map 0:v:0 -map 1:a:0? -movflags +faststart {final_output} -y 2>/dev/null")
    
    # Clean up temp files
    os.remove(temp_output)
    os.remove(temp_segment)
    
    print(f"\n✅ Word-level pipeline video created: {final_output}")
    print("\nKey improvements:")
    print("  ✓ All words in sentence from SAME direction")
    print("  ✓ Word objects maintained throughout")
    print("  ✓ Fixed positions - no movement during fog")
    print("  ✓ Smooth transitions between all phases")
    
    return final_output


if __name__ == "__main__":
    print("WORD-LEVEL TEXT ANIMATION PIPELINE")
    print("=" * 60)
    print("Maintains individual word tracking throughout all phases")
    print()
    
    output = create_word_level_video()
    
    if output:
        print(f"\n✨ Success! Video ready: {output}")
        print("\nThis version maintains word-level control throughout,")
        print("ensuring consistent positions and smooth animations.")