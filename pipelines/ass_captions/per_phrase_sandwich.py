#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-phrase sandwich compositing - renders and composites each phrase independently.
This eliminates flickering by allowing each phrase to have its own layering decision.
"""

import cv2
import numpy as np
import subprocess
import os
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import math

@dataclass
class FaceRegion:
    """Represents a detected face region"""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def left(self) -> int:
        return self.x
    
    @property
    def right(self) -> int:
        return self.x + self.width
    
    @property
    def top(self) -> int:
        return self.y
    
    @property
    def bottom(self) -> int:
        return self.y + self.height


class FaceDetector:
    """Detects faces in frames using OpenCV's Haar Cascade"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self._face_cache = {}
    
    def detect_faces(self, frame: np.ndarray, frame_idx: int) -> List[FaceRegion]:
        """Detect faces in a frame"""
        if frame_idx in self._face_cache:
            return self._face_cache[frame_idx]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50)
        )
        
        face_regions = []
        for (x, y, w, h) in faces:
            margin = int(w * 0.2)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = w + (margin * 2)
            h = h + (margin * 2)
            face_regions.append(FaceRegion(x=x, y=y, width=w, height=h))
        
        self._face_cache[frame_idx] = face_regions
        return face_regions


class PhraseRenderer:
    """Renders individual phrases with animation effects"""
    
    def __init__(self):
        # Try to load a good font
        self.fonts = {}
        
    def get_font(self, size: int):
        """Get or create font at specified size"""
        if size not in self.fonts:
            try:
                self.fonts[size] = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', size)
            except:
                self.fonts[size] = ImageFont.load_default()
        return self.fonts[size]
    
    def render_phrase(self, phrase: Dict, current_time: float, frame_shape: Tuple[int, int], 
                     scene_end_time: float = None, y_override: int = None) -> Optional[np.ndarray]:
        """
        Render a phrase with fade-in and slide-from-above animation.
        Returns RGBA image with transparent background.
        """
        # Use scene end time if provided, otherwise use phrase end time
        actual_end_time = scene_end_time if scene_end_time else phrase["end_time"]
        
        # Check if phrase should be visible
        if not (phrase["start_time"] <= current_time <= actual_end_time):
            return None
        
        # Calculate animation progress
        time_since_start = current_time - phrase["start_time"]
        fade_duration = 0.3  # 300ms fade in
        slide_duration = 0.3  # 300ms slide
        
        # Calculate opacity (fade in)
        if time_since_start < fade_duration:
            opacity = time_since_start / fade_duration
        else:
            opacity = 1.0
        
        # Calculate Y offset (slide from above)
        if time_since_start < slide_duration:
            slide_progress = time_since_start / slide_duration
            # Ease-out cubic
            slide_progress = 1 - pow(1 - slide_progress, 3)
            y_offset = int((1 - slide_progress) * 30)  # Start 30 pixels above
        else:
            y_offset = 0
        
        # Get text properties
        text = phrase["text"]
        font_size = int(48 * phrase["visual_style"]["font_size_multiplier"])
        font = self.get_font(font_size)
        
        # Create transparent image
        img = Image.new('RGBA', (frame_shape[1], frame_shape[0]), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Calculate position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (frame_shape[1] - text_width) // 2
        
        # Use override Y position if provided (for stacking)
        if y_override is not None:
            y = y_override - y_offset
        else:
            if phrase["position"] == "top":
                y = 180 - y_offset
            else:
                y = 540 - y_offset
        
        # Apply word-by-word animation if we have word timings
        if "word_timings" in phrase and phrase["word_timings"]:
            words = text.split()
            current_x = x
            
            for word, timing in zip(words, phrase["word_timings"]):
                word_start = timing["start"]
                word_end = timing["end"]
                
                # Check if word should be visible
                if current_time >= word_start:
                    # Calculate word's individual fade
                    word_time = current_time - word_start
                    word_fade_duration = 0.2
                    
                    if word_time < word_fade_duration:
                        word_opacity = (word_time / word_fade_duration) * opacity
                    else:
                        word_opacity = opacity
                    
                    # Draw word with black outline
                    word_bbox = draw.textbbox((0, 0), word + " ", font=font)
                    word_width = word_bbox[2] - word_bbox[0]
                    
                    # Black outline (draw multiple times with offset)
                    outline_color = (0, 0, 0, int(255 * word_opacity))
                    for dx in [-2, -1, 0, 1, 2]:
                        for dy in [-2, -1, 0, 1, 2]:
                            if dx != 0 or dy != 0:
                                draw.text((current_x + dx, y + dy), word + " ", 
                                         font=font, fill=outline_color)
                    
                    # White text
                    text_color = (255, 255, 255, int(255 * word_opacity))
                    draw.text((current_x, y), word + " ", font=font, fill=text_color)
                    
                    current_x += word_width
        else:
            # Fallback: render entire phrase at once
            # Black outline
            outline_color = (0, 0, 0, int(255 * opacity))
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
            
            # White text
            text_color = (255, 255, 255, int(255 * opacity))
            draw.text((x, y), text, font=font, fill=text_color)
        
        # Convert to numpy array
        return np.array(img)


def check_phrase_face_overlap(phrase: Dict, face_regions: List[FaceRegion], frame_width: int) -> bool:
    """Check if a phrase's bounding box overlaps with any face"""
    font_size = int(48 * phrase["visual_style"]["font_size_multiplier"])
    
    if phrase["position"] == "top":
        y = 180
    else:
        y = 540
    
    text = phrase["text"]
    estimated_width = len(text) * int(font_size * 0.6)
    x = (frame_width - estimated_width) // 2
    
    text_left = x
    text_right = x + estimated_width
    text_top = y - font_size
    text_bottom = y + int(font_size * 0.5)
    
    for face in face_regions:
        if (text_left < face.right and text_right > face.left and
            text_top < face.bottom and text_bottom > face.top):
            return True
    
    return False


def extract_word_timings(enriched_phrases: List[Dict], original_transcript_path: str) -> None:
    """Extract word-level timings from original transcript and add to enriched phrases"""
    # Load original transcript
    with open(original_transcript_path, 'r') as f:
        original_data = json.load(f)
    
    # Get all words with timings
    all_words = [w for w in original_data["words"] if w["type"] == "word"]
    word_index = 0
    
    for phrase in enriched_phrases:
        phrase_words = phrase["text"].split()
        word_timings = []
        
        for word in phrase_words:
            # Find matching word in original transcript
            while word_index < len(all_words):
                orig_word = all_words[word_index]
                # Clean up word text for comparison
                clean_orig = orig_word["text"].strip().rstrip(',.:;!?')
                clean_phrase = word.strip().rstrip(',.:;!?')
                
                if clean_orig.lower() == clean_phrase.lower():
                    word_timings.append({
                        "text": word,
                        "start": orig_word["start"],
                        "end": orig_word["end"]
                    })
                    word_index += 1
                    break
                word_index += 1
        
        if len(word_timings) == len(phrase_words):
            phrase["word_timings"] = word_timings
        else:
            print(f"Warning: Could not extract word timings for phrase: {phrase['text'][:30]}...")


def apply_per_phrase_sandwich(
    original_video: str,
    mask_video: str,
    transcript_path: str,
    output_path: str
):
    """
    Apply per-phrase sandwich compositing with independent layering decisions.
    Each phrase is rendered separately and composited based on face overlap.
    """
    # Load transcript
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # Extract word timings from original transcript if available
    if "source_transcript" in transcript_data:
        orig_path = "../../" + transcript_data["source_transcript"]
        if os.path.exists(orig_path):
            print("Extracting word-level timings from original transcript...")
            extract_word_timings(transcript_data["phrases"], orig_path)
    
    # Initialize components
    face_detector = FaceDetector()
    phrase_renderer = PhraseRenderer()
    
    # Open videos
    cap_orig = cv2.VideoCapture(original_video)
    cap_mask = cv2.VideoCapture(mask_video)
    
    # Get video properties
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Pre-analyze which phrases need face checking
    print("Pre-analyzing phrases for face overlap...")
    phrases_behind = set()
    
    for phrase in transcript_data.get("phrases", []):
        start_frame = int(phrase["start_time"] * fps)
        end_frame = int(phrase["end_time"] * fps)
        
        # Sample frames during phrase lifetime
        for check_frame in range(start_frame, min(end_frame + 1, total_frames), 5):
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, check_frame)
            ret, frame = cap_orig.read()
            if not ret:
                continue
            
            face_regions = face_detector.detect_faces(frame, check_frame)
            if face_regions:
                # Debug: show face regions detected
                if check_frame == start_frame:  # Only log first frame
                    for face in face_regions:
                        print(f"    Frame {check_frame}: Face at x={face.x}, y={face.y}, w={face.width}, h={face.height}")
                
                if check_phrase_face_overlap(phrase, face_regions, width):
                    phrase_key = f"{phrase['start_time']:.2f}_{phrase['text'][:20]}"
                    phrases_behind.add(phrase_key)
                    print(f"  Phrase '{phrase['text'][:30]}...' will go behind (overlaps with face)")
                    break
    
    print(f"Marked {len(phrases_behind)} phrases for behind-face rendering")
    
    # Reset video captures
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_mask.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Group phrases by appearance_index (scene)
    from collections import defaultdict
    scenes = defaultdict(list)
    for phrase in transcript_data.get("phrases", []):
        scenes[phrase.get("appearance_index", 0)].append(phrase)
    
    # Calculate scene end times
    scene_end_times = {}
    for scene_idx, scene_phrases in scenes.items():
        scene_end = max(p["end_time"] for p in scene_phrases)
        for phrase in scene_phrases:
            phrase_key = f"{phrase['start_time']:.2f}_{phrase['text'][:20]}"
            scene_end_times[phrase_key] = scene_end
    
    print(f"Processing {total_frames} frames with per-phrase compositing...")
    print(f"Found {len(scenes)} scenes with grouped timing")
    
    # Process each frame
    for frame_idx in range(total_frames):
        ret_orig, frame_original = cap_orig.read()
        ret_mask, mask_frame = cap_mask.read()
        
        if not ret_orig or not ret_mask:
            break
        
        current_time = frame_idx / fps
        
        # Track vertical positions for stacking
        top_phrases = []
        bottom_phrases = []
        
        # First pass: collect visible phrases and organize by position
        for phrase in transcript_data.get("phrases", []):
            phrase_key = f"{phrase['start_time']:.2f}_{phrase['text'][:20]}"
            scene_end = scene_end_times.get(phrase_key, phrase["end_time"])
            
            # Check if phrase is visible at current time
            if phrase["start_time"] <= current_time <= scene_end:
                if phrase.get("position") == "top":
                    top_phrases.append((phrase, phrase_key, scene_end))
                else:
                    bottom_phrases.append((phrase, phrase_key, scene_end))
        
        # Start with original frame
        composite = frame_original.copy()
        
        # Extract foreground mask
        green_screen_color = np.array([154, 254, 119], dtype=np.uint8)
        tolerance = 25
        
        diff = np.abs(mask_frame.astype(np.int16) - green_screen_color.astype(np.int16))
        is_green_screen = np.all(diff <= tolerance, axis=2)
        
        # Person mask: NOT green screen
        person_mask = (~is_green_screen).astype(np.uint8)
        
        # Light erosion
        kernel = np.ones((2,2), np.uint8)
        person_mask = cv2.erode(person_mask, kernel, iterations=1)
        
        # Calculate Y positions for stacking
        top_y_positions = []
        bottom_y_positions = []
        
        # Stack top phrases (starting at y=180, going down)
        current_y = 180
        for phrase, _, _ in top_phrases:
            font_size = int(48 * phrase["visual_style"]["font_size_multiplier"])
            top_y_positions.append(current_y)
            current_y += int(font_size * 1.5)  # Add spacing between lines
        
        # Stack bottom phrases (starting at y=540, going down)
        current_y = 540
        for phrase, _, _ in bottom_phrases:
            font_size = int(48 * phrase["visual_style"]["font_size_multiplier"])
            bottom_y_positions.append(current_y)
            current_y += int(font_size * 1.5)  # Add spacing between lines
        
        # Process phrases in two passes: behind first, then in front
        
        # Pass 1: Render phrases that go behind
        for i, (phrase, phrase_key, scene_end) in enumerate(top_phrases):
            if phrase_key in phrases_behind:
                y_pos = top_y_positions[i]
                phrase_img = phrase_renderer.render_phrase(phrase, current_time, (height, width), scene_end, y_pos)
                if phrase_img is not None:
                    alpha = phrase_img[:, :, 3:4] / 255.0
                    composite = composite * (1 - alpha) + phrase_img[:, :, :3] * alpha
        
        for i, (phrase, phrase_key, scene_end) in enumerate(bottom_phrases):
            if phrase_key in phrases_behind:
                y_pos = bottom_y_positions[i]
                phrase_img = phrase_renderer.render_phrase(phrase, current_time, (height, width), scene_end, y_pos)
                if phrase_img is not None:
                    alpha = phrase_img[:, :, 3:4] / 255.0
                    composite = composite * (1 - alpha) + phrase_img[:, :, :3] * alpha
        
        # Apply foreground (person) on top of background+behind-phrases
        person_mask_3ch = np.stack([person_mask, person_mask, person_mask], axis=2)
        composite = np.where(person_mask_3ch == 1, frame_original, composite)
        
        # Pass 2: Render phrases that go in front
        for i, (phrase, phrase_key, scene_end) in enumerate(top_phrases):
            if phrase_key not in phrases_behind:
                y_pos = top_y_positions[i]
                phrase_img = phrase_renderer.render_phrase(phrase, current_time, (height, width), scene_end, y_pos)
                if phrase_img is not None:
                    alpha = phrase_img[:, :, 3:4] / 255.0
                    composite = composite * (1 - alpha) + phrase_img[:, :, :3] * alpha
        
        for i, (phrase, phrase_key, scene_end) in enumerate(bottom_phrases):
            if phrase_key not in phrases_behind:
                y_pos = bottom_y_positions[i]
                phrase_img = phrase_renderer.render_phrase(phrase, current_time, (height, width), scene_end, y_pos)
                if phrase_img is not None:
                    alpha = phrase_img[:, :, 3:4] / 255.0
                    composite = composite * (1 - alpha) + phrase_img[:, :, :3] * alpha
        
        composite = composite.astype(np.uint8)
        out.write(composite)
        
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
            if len(top_phrases) > 1:
                print(f"  Stacking {len(top_phrases)} phrases at top")
            if len(bottom_phrases) > 1:
                print(f"  Stacking {len(bottom_phrases)} phrases at bottom")
    
    # Clean up
    cap_orig.release()
    cap_mask.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved: {output_path}")
    
    # Convert to H.264
    output_h264 = output_path.replace('.mp4', '_h264.mp4')
    cmd = [
        "ffmpeg", "-y",
        "-i", output_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_h264
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"H.264 version: {output_h264}")
    
    # Remove temp file
    os.remove(output_path)
    return output_h264


def main():
    input_video = "ai_math1_6sec.mp4"
    mask_video = "../../uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4"
    transcript_path = "../../uploads/assets/videos/ai_math1/transcript_enriched_partial.json"
    output_video = "ai_math1_per_phrase_sandwich.mp4"
    
    final_video = apply_per_phrase_sandwich(
        input_video,
        mask_video,
        transcript_path,
        output_video
    )
    
    print(f"\n✅ Per-phrase sandwich video created: {final_video}")
    print("\nFeatures:")
    print("  • Each phrase rendered and composited independently")
    print("  • No flickering - each phrase has consistent layering")
    print("  • Phrases can be simultaneously front AND behind")
    print("  • Maintains all animations (fade-in, slide-from-above)")
    print("  • Word-by-word animation preserved")


if __name__ == "__main__":
    main()