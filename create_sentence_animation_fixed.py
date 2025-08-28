"""
Create video with sentence-by-sentence display - FIXED VERSION
No blue artifacts, clean rendering
"""

import cv2
import numpy as np
import sys
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/word_3d')

from base_3d_text_animation import Animation3DConfig
from word_3d import WordRiseSequence3D

class SimpleWordRise:
    """Simplified word rise animation without blue artifacts"""
    
    def __init__(self, text, font_size=55, color=(255, 255, 255)):
        self.text = text
        self.words = text.split()
        self.font_size = font_size
        self.color = color
        
        # Load font
        try:
            self.font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
        except:
            self.font = ImageFont.load_default()
    
    def render_text(self, text, opacity=1.0, y_offset=0):
        """Render text with clean alpha blending"""
        # Create image with text
        img_width = 1280
        img_height = 200
        img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text horizontally
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2 + y_offset
        
        # Draw white text with proper opacity
        text_color = (*self.color, int(255 * opacity))
        draw.text((x, y), text, fill=text_color, font=self.font)
        
        # Convert to numpy array (RGBA)
        sprite = np.array(img)
        
        # Convert RGBA to BGRA for OpenCV
        sprite_bgr = sprite.copy()
        sprite_bgr[:, :, 0] = sprite[:, :, 2]  # B = R
        sprite_bgr[:, :, 2] = sprite[:, :, 0]  # R = B
        
        return sprite_bgr
    
    def apply_to_frame(self, frame, words_to_show, word_opacities, word_y_offsets):
        """Apply word animation to frame"""
        if not words_to_show:
            return frame
        
        # Join words that should be visible
        text = ' '.join(words_to_show)
        
        # Calculate average opacity and y_offset
        avg_opacity = sum(word_opacities) / len(word_opacities) if word_opacities else 1.0
        avg_y_offset = sum(word_y_offsets) / len(word_y_offsets) if word_y_offsets else 0
        
        # Render text
        text_sprite = self.render_text(text, avg_opacity, int(avg_y_offset))
        
        # Find text bounds
        alpha = text_sprite[:, :, 3]
        if np.max(alpha) == 0:
            return frame
        
        # Calculate position (centered)
        y_pos = 360 - 100  # Center vertically with offset
        x_pos = 0
        
        # Ensure we're within frame bounds
        y_end = min(y_pos + text_sprite.shape[0], frame.shape[0])
        x_end = min(x_pos + text_sprite.shape[1], frame.shape[1])
        
        # Composite text onto frame using clean alpha blending
        for c in range(3):
            frame[y_pos:y_end, x_pos:x_end, c] = (
                frame[y_pos:y_end, x_pos:x_end, c] * (1.0 - alpha[:y_end-y_pos, :x_end-x_pos] / 255.0) +
                text_sprite[:y_end-y_pos, :x_end-x_pos, c] * (alpha[:y_end-y_pos, :x_end-x_pos] / 255.0)
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

def create_clean_sentence_animation():
    """Create video with clean sentence-by-sentence animation"""
    
    print("Creating Clean Sentence Animation (No Blue Artifacts)")
    print("=" * 60)
    
    # Load the actual transcript
    transcript_path = "uploads/assets/videos/ai_math1/transcript.json"
    with open(transcript_path, 'r') as f:
        transcript = json.load(f)
    
    # Get segments for first 5 seconds
    segments_in_range = [s for s in transcript['segments'] 
                        if s['start'] < 5.0]
    
    # Group into sentences
    sentences = group_segments_into_sentences(segments_in_range)
    
    print(f"\nSentences found in first 5 seconds:")
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
    output_path = "outputs/ai_math1_5sec_clean_sentences.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each sentence
    sentence_animations = []
    
    for sentence_idx, sentence_segments in enumerate(sentences):
        # Combine segments into full sentence text
        full_text = ' '.join([s['text'].strip() for s in sentence_segments])
        sentence_start = sentence_segments[0]['start']
        sentence_end = sentence_segments[-1]['end']
        
        # Calculate word timings within the sentence
        words = full_text.split()
        
        # Create word timing based on segments
        word_timings = []
        word_index = 0
        
        for segment in sentence_segments:
            segment_words = segment['text'].strip().split()
            segment_duration = segment['end'] - segment['start']
            word_duration = segment_duration / len(segment_words) if segment_words else 0
            
            for i, word in enumerate(segment_words):
                if word_index < len(words):
                    word_start = segment['start'] + (i * word_duration)
                    word_timings.append({
                        'word': word,
                        'start': word_start,
                        'end': word_start + word_duration,
                        'index': word_index
                    })
                    word_index += 1
        
        # Calculate when to start dissolve
        if sentence_idx < len(sentences) - 1:
            next_sentence_start = sentences[sentence_idx + 1][0]['start']
            dissolve_start = max(sentence_end, next_sentence_start - 0.5)
        else:
            dissolve_start = min(sentence_end + 1.0, 5.0)
        
        sentence_animations.append({
            'text': full_text,
            'words': words,
            'word_timings': word_timings,
            'start': sentence_start,
            'end': sentence_end,
            'dissolve_start': dissolve_start,
            'dissolve_end': dissolve_start + 0.5,
            'sentence_idx': sentence_idx
        })
    
    print(f"\nCreated {len(sentence_animations)} sentence animations")
    print("\nRendering frames...")
    
    # Create animation renderer
    text_renderer = SimpleWordRise("", font_size=55)
    
    # Track current sentence
    current_sentence_idx = -1
    
    # Process frames
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        animated_frame = frame.copy()
        time_seconds = frame_num / fps
        
        # Find which sentence should be active
        for sent_data in sentence_animations:
            if sent_data['start'] <= time_seconds < sent_data['dissolve_end']:
                
                # Calculate which words should be visible
                words_to_show = []
                word_opacities = []
                word_y_offsets = []
                
                for word_timing in sent_data['word_timings']:
                    if time_seconds >= word_timing['start']:
                        words_to_show.append(word_timing['word'])
                        
                        # Calculate word state
                        if time_seconds < word_timing['end']:
                            # Word is appearing (rising)
                            word_progress = (time_seconds - word_timing['start']) / (word_timing['end'] - word_timing['start'])
                            # Smooth easing for rise
                            eased = (1 - np.cos(word_progress * np.pi)) / 2
                            word_opacities.append(eased)
                            word_y_offsets.append((1 - eased) * 50)  # Rise from 50 pixels below
                        elif time_seconds < sent_data['dissolve_start']:
                            # Word is fully visible
                            word_opacities.append(1.0)
                            word_y_offsets.append(0)
                        elif time_seconds < sent_data['dissolve_end']:
                            # Word is dissolving upward
                            dissolve_progress = (time_seconds - sent_data['dissolve_start']) / 0.5
                            word_opacities.append(1.0 - dissolve_progress)
                            word_y_offsets.append(-dissolve_progress * 100)  # Rise upward
                        else:
                            # Word has dissolved
                            word_opacities.append(0.0)
                            word_y_offsets.append(-100)
                
                # Apply text to frame
                if words_to_show:
                    animated_frame = text_renderer.apply_to_frame(
                        animated_frame, words_to_show, word_opacities, word_y_offsets
                    )
                
                break
        
        # Add info overlay
        cv2.putText(animated_frame, 
                   f"Clean Sentence Animation | {time_seconds:.2f}s / 5.0s", 
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
    print("CLEAN SENTENCE ANIMATION COMPLETE!")
    print(f"Output: {h264_output}")
    print("\nFeatures:")
    print("  ✓ NO BLUE ARTIFACTS - Clean rendering")
    print("  ✓ Sentences displayed one at a time")
    print("  ✓ Words appear in sync with speech timing")
    print("  ✓ Smooth upward dissolve transitions")
    print("=" * 60)
    
    return h264_output


if __name__ == "__main__":
    print("CLEAN SENTENCE ANIMATION (FIXED)")
    print("=" * 60)
    print("Fixed issues:")
    print("- Removed blue flickering artifacts")
    print("- Clean alpha blending without color bleeding")
    print("- Simplified rendering pipeline")
    print("- Proper BGR/RGB handling for OpenCV")
    print()
    
    output = create_clean_sentence_animation()
    
    if output:
        print(f"\n✨ Success! Final video: {output}")
        print("\nThe video now has clean, artifact-free text animations!")