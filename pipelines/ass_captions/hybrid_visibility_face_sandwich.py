#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid visibility + face detection sandwich compositing.
Places text behind foreground when either:
1. Text would be >90% visible behind foreground, OR
2. Text overlaps with detected face/head regions
"""

import cv2
import numpy as np
import subprocess
import os
import json
import time
import replicate
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))


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


def detect_head_with_sam2_replicate(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Detect head/face using SAM2 via Replicate API.
    Returns binary mask (255 for head, 0 for background).
    """
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print(f"  Frame {frame_idx}: No REPLICATE_API_TOKEN, using fallback")
        return detect_face_opencv_fallback(frame)
    
    try:
        # Save frame to temporary file
        temp_path = f"/tmp/frame_{frame_idx}.jpg"
        cv2.imwrite(temp_path, frame)
        
        with open(temp_path, 'rb') as f:
            # Run SAM2 segmentation
            prediction = replicate.run(
                'meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83',
                input={
                    'image': f,
                    'points_per_side': 16,  # Less dense for faster processing
                    'pred_iou_thresh': 0.88,  # Higher threshold for better quality
                    'stability_score_thresh': 0.92,  # Higher stability requirement
                    'use_m2m': True,
                    'multimask_output': False
                }
            )
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Get masks
        individual_masks_urls = prediction.get('individual_masks', [])
        
        if not individual_masks_urls:
            print(f"  Frame {frame_idx}: No masks found")
            return detect_face_opencv_fallback(frame)
        
        # Look for face-like masks (in upper portion of frame)
        h, w = frame.shape[:2]
        best_mask = None
        best_score = -1
        
        for mask_url in individual_masks_urls[:10]:  # Check top 10 masks
            response = requests.get(str(mask_url))
            mask_img = Image.open(BytesIO(response.content))
            mask_array = np.array(mask_img)
            
            if len(mask_array.shape) == 3:
                mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
            
            if mask_array.shape[:2] != frame.shape[:2]:
                mask_array = cv2.resize(mask_array, (w, h))
            
            binary_mask = (mask_array > 128).astype(np.uint8)
            
            # Calculate mask properties
            coords = np.where(binary_mask > 0)
            if len(coords[0]) == 0:
                continue
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Score based on:
            # 1. Position (prefer upper portion)
            # 2. Size (not too small, not too large)
            # 3. Aspect ratio (face-like)
            
            center_y = (y_min + y_max) / 2
            center_x = (x_min + x_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            area = np.sum(binary_mask)
            
            # Position score (prefer upper half)
            position_score = 1.0 - (center_y / h) if center_y < h/2 else 0.3
            
            # Size score (prefer reasonable face size)
            ideal_area = h * w * 0.05  # ~5% of image
            size_score = 1.0 - abs(area - ideal_area) / ideal_area
            size_score = max(0, min(1, size_score))
            
            # Aspect ratio score (faces are usually 0.7-1.3 ratio)
            if height > 0:
                aspect = width / height
                aspect_score = 1.0 - abs(aspect - 1.0) / 2.0
                aspect_score = max(0, min(1, aspect_score))
            else:
                aspect_score = 0
            
            # Centered score (prefer centered objects)
            center_score = 1.0 - abs(center_x - w/2) / (w/2)
            
            # Combined score
            score = position_score * 0.4 + size_score * 0.3 + aspect_score * 0.2 + center_score * 0.1
            
            if score > best_score:
                best_score = score
                best_mask = binary_mask
        
        if best_mask is not None:
            # Expand mask slightly for safety
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            best_mask = cv2.dilate(best_mask, kernel, iterations=1)
            return best_mask * 255
        
    except Exception as e:
        print(f"  Frame {frame_idx}: SAM2 API error: {e}")
    
    # Fallback to OpenCV
    return detect_face_opencv_fallback(frame)


def detect_face_opencv_fallback(frame: np.ndarray) -> np.ndarray:
    """Fallback face detection using OpenCV Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(50, 50)
    )
    
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for (x, y, w, h) in faces:
        # Add 20% margin
        margin = int(w * 0.2)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = w + (margin * 2)
        h = h + (margin * 2)
        # Draw filled rectangle for face region
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    return mask


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


def check_text_face_overlap(text_bbox: Tuple[int, int, int, int], 
                           face_mask: np.ndarray) -> bool:
    """
    Check if text overlaps with face region.
    
    Args:
        text_bbox: (x1, y1, x2, y2) bounding box of text
        face_mask: Binary mask where 255 = face, 0 = background
        
    Returns:
        True if text overlaps with face
    """
    x1, y1, x2, y2 = text_bbox
    
    # Ensure bbox is within frame bounds
    h, w = face_mask.shape
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    # Check if any face pixels in text region
    text_region = face_mask[y1:y2, x1:x2]
    return np.any(text_region > 0)


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


def apply_hybrid_sandwich(
    original_video: str,
    mask_video: str,
    transcript_path: str,
    output_path: str,
    visibility_threshold: float = 0.9,
    detect_faces: bool = True,
    sample_face_every: int = 10
):
    """
    Apply hybrid visibility + face detection sandwich compositing.
    
    Args:
        original_video: Path to original video
        mask_video: Path to green screen mask video  
        transcript_path: Path to enriched transcript JSON
        output_path: Output video path
        visibility_threshold: Minimum visibility ratio to place text behind (default 0.9 = 90%)
        detect_faces: Whether to use face detection (default True)
        sample_face_every: Sample faces every N frames for efficiency
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
    print(f"Visibility threshold: {visibility_threshold:.0%}")
    print(f"Face detection: {'Enabled' if detect_faces else 'Disabled'}")
    
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
    
    # Pre-detect faces if enabled
    face_masks_cache = {}
    if detect_faces:
        print(f"\nPre-detecting faces (sampling every {sample_face_every} frames)...")
        for frame_idx in range(0, total_frames, sample_face_every):
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap_orig.read()
            if ret:
                print(f"  Detecting faces in frame {frame_idx}...")
                face_mask = detect_head_with_sam2_replicate(frame, frame_idx)
                face_masks_cache[frame_idx] = face_mask
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\nProcessing {total_frames} frames with hybrid compositing...")
    print(f"Found {len(scenes)} scenes with grouped timing")
    
    # Track statistics
    behind_visibility = 0
    behind_face = 0
    front_count = 0
    
    # Process each frame
    for frame_idx in range(total_frames):
        ret_orig, frame_original = cap_orig.read()
        ret_mask, mask_frame = cap_mask.read()
        
        if not ret_orig or not ret_mask:
            break
        
        current_time = frame_idx / fps
        
        # Get or interpolate face mask
        face_mask = None
        if detect_faces:
            if frame_idx in face_masks_cache:
                face_mask = face_masks_cache[frame_idx]
            else:
                # Find nearest cached masks for interpolation
                before_idx = max([i for i in face_masks_cache.keys() if i < frame_idx], default=-1)
                after_idx = min([i for i in face_masks_cache.keys() if i > frame_idx], default=total_frames)
                
                if before_idx >= 0 and after_idx < total_frames:
                    # Interpolate between masks
                    alpha = (frame_idx - before_idx) / (after_idx - before_idx)
                    face_mask = cv2.addWeighted(
                        face_masks_cache[before_idx], 1 - alpha,
                        face_masks_cache[after_idx], alpha, 0
                    )
                    face_mask = (face_mask > 127).astype(np.uint8) * 255
                elif before_idx >= 0:
                    face_mask = face_masks_cache[before_idx]
                else:
                    face_mask = np.zeros((height, width), dtype=np.uint8)
        
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
        
        # Render all phrases and decide placement
        phrases_to_render = []
        
        for i, (phrase, phrase_key, scene_end) in enumerate(top_phrases):
            y_pos = top_y_positions[i]
            phrase_img = phrase_renderer.render_phrase(phrase, current_time, (height, width), scene_end, y_pos)
            if phrase_img is not None and "_render_bbox" in phrase:
                visibility = calculate_text_visibility(phrase["_render_bbox"], person_mask)
                face_overlap = check_text_face_overlap(phrase["_render_bbox"], face_mask) if face_mask is not None else False
                
                # Hybrid decision: behind if high visibility OR face overlap
                should_be_behind = (visibility > visibility_threshold) or face_overlap
                
                reason = "face" if face_overlap else ("visibility" if visibility > visibility_threshold else "none")
                phrases_to_render.append((phrase_img, should_be_behind, phrase["text"][:20], visibility, reason))
        
        for i, (phrase, phrase_key, scene_end) in enumerate(bottom_phrases):
            y_pos = bottom_y_positions[i]
            phrase_img = phrase_renderer.render_phrase(phrase, current_time, (height, width), scene_end, y_pos)
            if phrase_img is not None and "_render_bbox" in phrase:
                visibility = calculate_text_visibility(phrase["_render_bbox"], person_mask)
                face_overlap = check_text_face_overlap(phrase["_render_bbox"], face_mask) if face_mask is not None else False
                
                # Hybrid decision: behind if high visibility OR face overlap
                should_be_behind = (visibility > visibility_threshold) or face_overlap
                
                reason = "face" if face_overlap else ("visibility" if visibility > visibility_threshold else "none")
                phrases_to_render.append((phrase_img, should_be_behind, phrase["text"][:20], visibility, reason))
        
        # Process phrases in two passes: behind first, then in front
        
        # Pass 1: Render phrases that go behind
        for phrase_img, should_be_behind, phrase_text, visibility, reason in phrases_to_render:
            if should_be_behind:
                alpha = phrase_img[:, :, 3:4] / 255.0
                composite = composite * (1 - alpha) + phrase_img[:, :, :3] * alpha
                if reason == "face":
                    behind_face += 1
                else:
                    behind_visibility += 1
        
        # Apply foreground (person) on top of background+behind-phrases
        person_mask_3ch = np.stack([person_mask, person_mask, person_mask], axis=2)
        composite = np.where(person_mask_3ch == 1, frame_original, composite)
        
        # Pass 2: Render phrases that go in front
        for phrase_img, should_be_behind, phrase_text, visibility, reason in phrases_to_render:
            if not should_be_behind:
                alpha = phrase_img[:, :, 3:4] / 255.0
                composite = composite * (1 - alpha) + phrase_img[:, :, :3] * alpha
                front_count += 1
        
        composite = composite.astype(np.uint8)
        out.write(composite)
        
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
            if len(phrases_to_render) > 0:
                for _, should_be_behind, phrase_text, visibility, reason in phrases_to_render:
                    position = f"behind ({reason})" if should_be_behind else "front"
                    print(f"  '{phrase_text}': {visibility:.1%} visible → {position}")
    
    # Clean up
    cap_orig.release()
    cap_mask.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nCompositing statistics:")
    print(f"  Phrases placed behind (visibility): {behind_visibility}")
    print(f"  Phrases placed behind (face): {behind_face}")
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
    output_video = "ai_math1_hybrid_sandwich.mp4"
    
    # Check for Replicate API token
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("\n⚠️  WARNING: No REPLICATE_API_TOKEN found in environment")
        print("   Face detection will use OpenCV fallback (less accurate)")
        print("   To use SAM2 for better face detection:")
        print("   export REPLICATE_API_TOKEN='your_token_here'\n")
    
    final_video = apply_hybrid_sandwich(
        input_video,
        mask_video,
        transcript_path,
        output_video,
        visibility_threshold=0.9,  # Text must be >90% visible for visibility rule
        detect_faces=True,  # Also detect faces
        sample_face_every=10  # Sample faces every 10 frames
    )
    
    print(f"\n✅ Hybrid visibility + face detection video created: {final_video}")
    print("\nFeatures:")
    print("  • Text goes behind if >90% visible (visibility rule)")
    print("  • Text goes behind if overlapping with face (face rule)")
    print("  • Hybrid approach: either condition triggers behind placement")
    print("  • Per-phrase independent decisions")
    print("  • Vertical stacking for same-position phrases")
    print("  • All animations preserved")


if __name__ == "__main__":
    main()