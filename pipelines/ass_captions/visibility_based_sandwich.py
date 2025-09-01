#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visibility-based sandwich compositing for ASS captions.
Places text behind foreground only if it remains >90% visible.
"""

import cv2
import numpy as np
import subprocess
import os
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict


class PhraseRenderer:
    """Renders individual phrases with animation effects"""
    
    def __init__(self):
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
        
        # Store bounding box info for visibility checking
        phrase["_render_bbox"] = (x, y, x + text_width, y + text_height)
        
        # Convert to numpy array
        return np.array(img)


def calculate_text_visibility(text_bbox: Tuple[int, int, int, int], 
                             person_mask: np.ndarray) -> float:
    """
    Calculate what percentage of text area would be visible if placed behind foreground.
    
    Args:
        text_bbox: (x1, y1, x2, y2) bounding box of text
        person_mask: Binary mask where 1 = foreground (person), 0 = background
        
    Returns:
        Visibility ratio (0.0 to 1.0)
    """
    x1, y1, x2, y2 = text_bbox
    
    # Ensure bbox is within frame bounds
    h, w = person_mask.shape
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return 1.0  # Invalid bbox, assume fully visible
    
    # Extract text region from mask
    text_region_mask = person_mask[y1:y2, x1:x2]
    
    # Calculate visibility
    total_pixels = text_region_mask.size
    if total_pixels == 0:
        return 1.0
    
    # Count background pixels (where text would be visible)
    background_pixels = np.sum(text_region_mask == 0)
    visibility_ratio = background_pixels / total_pixels
    
    return visibility_ratio


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


def apply_visibility_based_sandwich(
    original_video: str,
    mask_video: str,
    transcript_path: str,
    output_path: str,
    visibility_threshold: float = 0.9
):
    """
    Apply visibility-based sandwich compositing.
    Text goes behind foreground only if it remains >90% visible.
    
    Args:
        original_video: Path to original video
        mask_video: Path to green screen mask video
        transcript_path: Path to enriched transcript JSON
        output_path: Output video path
        visibility_threshold: Minimum visibility ratio to place text behind (default 0.9 = 90%)
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
    print(f"Visibility threshold: {visibility_threshold:.0%} (text must be >{visibility_threshold:.0%} visible to go behind)")
    
    # Group phrases by appearance_index (scene)
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
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames with visibility-based compositing...")
    print(f"Found {len(scenes)} scenes with grouped timing")
    
    # Track statistics
    behind_count = 0
    front_count = 0
    
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
        
        # Render all phrases to check their visibility
        phrases_to_render = []
        
        for i, (phrase, phrase_key, scene_end) in enumerate(top_phrases):
            y_pos = top_y_positions[i]
            phrase_img = phrase_renderer.render_phrase(phrase, current_time, (height, width), scene_end, y_pos)
            if phrase_img is not None and "_render_bbox" in phrase:
                visibility = calculate_text_visibility(phrase["_render_bbox"], person_mask)
                should_be_behind = visibility > visibility_threshold
                phrases_to_render.append((phrase_img, should_be_behind, phrase["text"][:20], visibility))
        
        for i, (phrase, phrase_key, scene_end) in enumerate(bottom_phrases):
            y_pos = bottom_y_positions[i]
            phrase_img = phrase_renderer.render_phrase(phrase, current_time, (height, width), scene_end, y_pos)
            if phrase_img is not None and "_render_bbox" in phrase:
                visibility = calculate_text_visibility(phrase["_render_bbox"], person_mask)
                should_be_behind = visibility > visibility_threshold
                phrases_to_render.append((phrase_img, should_be_behind, phrase["text"][:20], visibility))
        
        # Process phrases in two passes: behind first, then in front
        
        # Pass 1: Render phrases that go behind (high visibility)
        for phrase_img, should_be_behind, phrase_text, visibility in phrases_to_render:
            if should_be_behind:
                alpha = phrase_img[:, :, 3:4] / 255.0
                composite = composite * (1 - alpha) + phrase_img[:, :, :3] * alpha
                behind_count += 1
        
        # Apply foreground (person) on top of background+behind-phrases
        person_mask_3ch = np.stack([person_mask, person_mask, person_mask], axis=2)
        composite = np.where(person_mask_3ch == 1, frame_original, composite)
        
        # Pass 2: Render phrases that go in front (low visibility)
        for phrase_img, should_be_behind, phrase_text, visibility in phrases_to_render:
            if not should_be_behind:
                alpha = phrase_img[:, :, 3:4] / 255.0
                composite = composite * (1 - alpha) + phrase_img[:, :, :3] * alpha
                front_count += 1
        
        composite = composite.astype(np.uint8)
        out.write(composite)
        
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
            if len(phrases_to_render) > 0:
                for _, should_be_behind, phrase_text, visibility in phrases_to_render:
                    position = "behind" if should_be_behind else "front"
                    print(f"  '{phrase_text}': {visibility:.1%} visible → {position}")
    
    # Clean up
    cap_orig.release()
    cap_mask.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nCompositing statistics:")
    print(f"  Phrases placed behind: {behind_count}")
    print(f"  Phrases placed in front: {front_count}")
    
    print(f"\nVideo saved: {output_path}")
    
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
    output_video = "ai_math1_visibility_sandwich.mp4"
    
    final_video = apply_visibility_based_sandwich(
        input_video,
        mask_video,
        transcript_path,
        output_video,
        visibility_threshold=0.9  # Text must be >90% visible to go behind
    )
    
    print(f"\n✅ Visibility-based sandwich video created: {final_video}")
    print("\nFeatures:")
    print("  • Text goes behind only if >90% visible")
    print("  • Text stays in front if it would be >10% occluded")
    print("  • Per-phrase independent decisions")
    print("  • Vertical stacking for same-position phrases")
    print("  • All animations preserved (fade-in, slide-from-above)")


if __name__ == "__main__":
    main()