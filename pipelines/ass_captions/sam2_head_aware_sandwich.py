#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 head-aware sandwich compositing for ASS captions.
Uses SAM2 video tracking to detect head throughout video.
Text ALWAYS goes behind if it touches the head, even by 1 pixel.

New merging logic:
- If 2+ subsentences at same position (top/bottom) go behind face,
  they are merged into a single line with scaled-down font to fit
- Single behind-face phrases are enlarged up to 1.8x
- Merged behind-face phrases are scaled down as needed (down to 0.3x)
"""

import cv2
import numpy as np
import subprocess
import os
import sys
import json
import random
import time
import replicate
import requests
import random
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Add paths for SAM2 imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'sam2_api'))

try:
    from video_segmentation import SAM2VideoSegmenter, ClickPoint, SegmentationConfig
    SAM2_API_AVAILABLE = True
except ImportError:
    print("Warning: SAM2 API not available")
    SAM2_API_AVAILABLE = False


class EntranceEffect(Enum):
    """Entrance effect types for subsentences"""
    FADE_WORD_BY_WORD = "fade_word_by_word"
    FADE_SLIDE_FROM_TOP = "fade_slide_top"
    FADE_SLIDE_FROM_BOTTOM = "fade_slide_bottom"
    FADE_SLIDE_FROM_LEFT = "fade_slide_left"  # For short phrases only
    FADE_SLIDE_FROM_RIGHT = "fade_slide_right"  # For short phrases only
    FADE_SLIDE_WHOLE_TOP = "fade_slide_whole_top"  # Whole phrase slides
    FADE_SLIDE_WHOLE_BOTTOM = "fade_slide_whole_bottom"  # Whole phrase slides
    
    @staticmethod
    def get_random_effect(exclude_side_effects: bool = True) -> 'EntranceEffect':
        """Get a random entrance effect.
        
        Args:
            exclude_side_effects: If True, exclude left/right slide effects
                                (those are reserved for head-tracking phrases)
        """
        import random
        effects = list(EntranceEffect)
        if exclude_side_effects:
            # Exclude side slides which are for head-tracking phrases
            effects = [e for e in effects if e not in [
                EntranceEffect.FADE_SLIDE_FROM_LEFT,
                EntranceEffect.FADE_SLIDE_FROM_RIGHT
            ]]
        return random.choice(effects)


class DisappearanceEffect(Enum):
    """Disappearance effect types for subsentences"""
    FADE_OUT = "fade_out"  # Simple fade to transparent
    BLUR_DISSOLVE = "blur_dissolve"  # Fade out with increasing blur
    COLOR_FADE_TO_BLACK = "color_fade_black"  # Fade to black then transparent
    COLOR_FADE_TO_WHITE = "color_fade_white"  # Fade to white then transparent
    GLOW_DISSOLVE = "glow_dissolve"  # Add glow effect while fading
    SLIDE_OUT_LEFT = "slide_out_left"  # Slide off to the left
    SLIDE_OUT_RIGHT = "slide_out_right"  # Slide off to the right
    SLIDE_OUT_UP = "slide_out_up"  # Slide up and out
    SLIDE_OUT_DOWN = "slide_out_down"  # Slide down and out
    SHRINK_AND_FADE = "shrink_fade"  # Scale down while fading
    
    @staticmethod
    def get_random_effect() -> 'DisappearanceEffect':
        """Get a random disappearance effect."""
        import random
        return random.choice(list(DisappearanceEffect))


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
                     scene_end_time: float = None, y_override: int = None, 
                     size_multiplier: float = 1.0, merged_text: str = None,
                     entrance_effect: EntranceEffect = None,
                     disappearance_effect: DisappearanceEffect = None) -> Optional[np.ndarray]:
        """
        Render a phrase with specified entrance and disappearance animations.
        Returns RGBA image with transparent background.
        
        Key timing invariants:
          - logical_end = scene_end_time (if provided) else phrase['end_time']
          - draw_end = logical_end + disappear_duration
          
          Gate visibility using draw_end.
          Compute disappearance progress from logical_end.
        
        Args:
            merged_text: If provided, render this text instead of phrase["text"]
            entrance_effect: The entrance animation to apply
            disappearance_effect: The disappearance animation to apply
        """
        # CRITICAL: Separate logical times from draw times
        logical_start = phrase["start_time"]
        logical_end = scene_end_time if scene_end_time else phrase["end_time"]
        entrance_duration = 0.4  # 400ms for entrance
        disappear_duration = 0.5  # 500ms for disappearance
        
        # Drawing windows extend beyond logical times for animations
        draw_start = logical_start - entrance_duration
        draw_end = logical_end + disappear_duration
        
        # Single visibility gate using extended draw window
        if not (draw_start <= current_time <= draw_end):
            return None
        
        # Calculate animation timings relative to logical start
        time_since_start = current_time - phrase["start_time"]
        
        # Default effects if not specified
        if entrance_effect is None:
            entrance_effect = EntranceEffect.FADE_WORD_BY_WORD
        if disappearance_effect is None:
            disappearance_effect = DisappearanceEffect.FADE_OUT
        
        # Calculate base opacity and offsets
        base_opacity = 1.0
        color_shift = (255, 255, 255)  # Default white
        scale_factor = 1.0
        blur_amount = 0
        
        # Determine animation phase and progress
        if current_time < logical_start:
            # ENTRANCE PHASE (before logical start)
            animation_phase = "ENTRANCE"
            time_until_start = logical_start - current_time
            progress = 1.0 - (time_until_start / entrance_duration)
            progress = max(0, min(1, progress))  # Clamp to [0, 1]
            # Ease-out cubic for smooth deceleration
            eased_progress = 1 - pow(1 - progress, 3)
            base_opacity = progress
            
            # Debug logging for entrance
            debug_text = merged_text if merged_text else phrase["text"]
            if progress < 0.02:  # Just started
                print(f"    🎭 ENTRANCE START: '{debug_text[:20]}...' | effect={entrance_effect.value} | t={current_time:.2f}s | start={logical_start:.2f}s")
            elif 0.48 < progress < 0.52:  # Halfway
                print(f"    🎭 ENTRANCE MID: '{debug_text[:20]}...' | progress={progress:.0%} | opacity={base_opacity:.2f}")
        elif current_time < logical_end:
            # STEADY PHASE
            animation_phase = "STEADY"
            progress = 1.0
            eased_progress = 1.0
            base_opacity = 1.0
        else:
            # DISAPPEARANCE PHASE (current_time >= logical_end)
            animation_phase = "DISAPPEAR"
            # CRITICAL: Calculate progress from logical_end, not draw_end
            disappear_progress = min((current_time - logical_end) / disappear_duration, 1.0)
            # Ease-in for acceleration
            eased_disappear = disappear_progress * disappear_progress
            
            # Enhanced debug logging for phase transitions
            debug_text = merged_text if merged_text else phrase["text"]
            
            # Log phase transitions (only at key moments)
            if disappear_progress < 0.02:  # Just started
                print(f"    🎬 DISAPPEAR START: '{debug_text[:20]}...' | effect={disappearance_effect.value} | t={current_time:.2f}s | logical_end={logical_end:.2f}s")
            elif 0.48 < disappear_progress < 0.52:  # Halfway
                print(f"    🎬 DISAPPEAR MID: '{debug_text[:20]}...' | progress={disappear_progress:.0%} | opacity={base_opacity:.2f}")
            elif disappear_progress > 0.98:  # Almost done
                print(f"    🎬 DISAPPEAR END: '{debug_text[:20]}...' | complete at t={current_time:.2f}s")
            
            # Now handle the disappearance phase
            progress = disappear_progress
            eased_progress = eased_disappear
            
            # Apply disappearance effect
            if disappearance_effect == DisappearanceEffect.FADE_OUT:
                base_opacity = 1 - disappear_progress
            elif disappearance_effect == DisappearanceEffect.BLUR_DISSOLVE:
                base_opacity = 1 - disappear_progress
                blur_amount = int(disappear_progress * 5)  # Max 5px blur
            elif disappearance_effect == DisappearanceEffect.COLOR_FADE_TO_BLACK:
                base_opacity = 1 - disappear_progress
                fade_val = int(255 * (1 - disappear_progress))
                color_shift = (fade_val, fade_val, fade_val)
            elif disappearance_effect == DisappearanceEffect.COLOR_FADE_TO_WHITE:
                base_opacity = 1 - disappear_progress
                fade_val = 255 - int(127 * disappear_progress)  # Start white, fade to gray-white
                color_shift = (fade_val, fade_val, fade_val)
            elif disappearance_effect == DisappearanceEffect.GLOW_DISSOLVE:
                base_opacity = 1 - disappear_progress
                # Glow would need special rendering (simplified here)
            elif disappearance_effect == DisappearanceEffect.SHRINK_AND_FADE:
                base_opacity = 1 - disappear_progress
                scale_factor = 1 - (disappear_progress * 0.3)  # Shrink to 70%
            else:
                # Slide effects handled separately
                base_opacity = 1 - disappear_progress
        
        # Calculate position offsets based on effects
        x_offset = 0
        y_offset = 0
        slide_distance = 40  # Pixels to slide
        
        # Entrance position offsets
        if time_since_start < animation_duration:
            if entrance_effect in [EntranceEffect.FADE_SLIDE_FROM_TOP, EntranceEffect.FADE_SLIDE_WHOLE_TOP]:
                y_offset = int((1 - eased_progress) * -slide_distance)  # Negative = from above
            elif entrance_effect in [EntranceEffect.FADE_SLIDE_FROM_BOTTOM, EntranceEffect.FADE_SLIDE_WHOLE_BOTTOM]:
                y_offset = int((1 - eased_progress) * slide_distance)  # Positive = from below
            elif entrance_effect == EntranceEffect.FADE_SLIDE_FROM_LEFT:
                x_offset = int((1 - eased_progress) * -slide_distance)  # Negative = from left
            elif entrance_effect == EntranceEffect.FADE_SLIDE_FROM_RIGHT:
                x_offset = int((1 - eased_progress) * slide_distance)  # Positive = from right
        
        # Disappearance position offsets
        if animation_phase == "DISAPPEAR":
            if disappearance_effect == DisappearanceEffect.SLIDE_OUT_LEFT:
                x_offset = int(-eased_disappear * slide_distance * 2)  # Slide left
            elif disappearance_effect == DisappearanceEffect.SLIDE_OUT_RIGHT:
                x_offset = int(eased_disappear * slide_distance * 2)  # Slide right
            elif disappearance_effect == DisappearanceEffect.SLIDE_OUT_UP:
                y_offset = int(-eased_disappear * slide_distance * 2)  # Slide up
            elif disappearance_effect == DisappearanceEffect.SLIDE_OUT_DOWN:
                y_offset = int(eased_disappear * slide_distance * 2)  # Slide down
        
        # For whole-phrase effects, apply same opacity to all words
        whole_phrase_effects = [
            EntranceEffect.FADE_SLIDE_WHOLE_TOP,
            EntranceEffect.FADE_SLIDE_WHOLE_BOTTOM,
            EntranceEffect.FADE_SLIDE_FROM_LEFT,
            EntranceEffect.FADE_SLIDE_FROM_RIGHT
        ]
        use_whole_phrase_animation = entrance_effect in whole_phrase_effects
        
        # Get text properties
        text = merged_text if merged_text else phrase["text"]
        # When enlarging text that goes behind, we want it to be 1.5x the BASE size,
        # not 1.5x the already-reduced size. So if size_multiplier > 1, we ignore
        # the visual_style multiplier and use our target size directly.
        if size_multiplier > 1.0:
            # Use size_multiplier directly on base font size, ignoring visual_style reduction
            base_font_size = int(48 * size_multiplier)
        else:
            # Normal case: apply both multipliers
            base_font_size = int(48 * phrase["visual_style"]["font_size_multiplier"] * size_multiplier)
        
        # Apply scale factor for shrink effect
        font_size = int(base_font_size * scale_factor)
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
            y = y_override
        else:
            if phrase["position"] == "top":
                y = 180
            else:
                y = 540
        
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
                    
                    if use_whole_phrase_animation:
                        # For whole-phrase effects, all words use same base opacity
                        word_opacity = base_opacity
                    elif entrance_effect == EntranceEffect.FADE_WORD_BY_WORD and time_since_start < animation_duration:
                        # Word-by-word fade in during entrance
                        if word_time < word_fade_duration:
                            word_opacity = word_time / word_fade_duration * base_opacity
                        else:
                            word_opacity = base_opacity
                    else:
                        # For other effects or after entrance, use base opacity
                        word_opacity = base_opacity
                    
                    # Draw word with black outline
                    word_bbox = draw.textbbox((0, 0), word + " ", font=font)
                    word_width = word_bbox[2] - word_bbox[0]
                    
                    # Apply position offset (for slide effects)
                    word_x = current_x + x_offset
                    word_y = y + y_offset
                    
                    # Apply color shift for disappearance effects
                    text_r, text_g, text_b = color_shift
                    
                    # Black outline (draw multiple times with offset)
                    outline_opacity = max(0, int(word_opacity * 255 * 0.8))  # Outline slightly more transparent
                    outline_color = (0, 0, 0, outline_opacity)
                    for dx in [-2, -1, 0, 1, 2]:
                        for dy in [-2, -1, 0, 1, 2]:
                            if dx != 0 or dy != 0:
                                draw.text((word_x + dx, word_y + dy), word + " ", 
                                         font=font, fill=outline_color)
                    
                    # Colored text with opacity
                    text_color = (text_r, text_g, text_b, int(255 * word_opacity))
                    draw.text((word_x, word_y), word + " ", font=font, fill=text_color)
                    
                    current_x += word_width
        else:
            # Fallback: render entire phrase at once
            # Apply position offsets
            final_x = x + x_offset
            final_y = y + y_offset
            
            # Apply color shift for disappearance effects
            text_r, text_g, text_b = color_shift
            
            # Black outline
            outline_opacity = max(0, int(base_opacity * 255 * 0.8))  # Outline slightly more transparent
            outline_color = (0, 0, 0, outline_opacity)
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx != 0 or dy != 0:
                        draw.text((final_x + dx, final_y + dy), text, font=font, fill=outline_color)
            
            # Colored text with opacity
            text_color = (text_r, text_g, text_b, int(255 * base_opacity))
            draw.text((final_x, final_y), text, font=font, fill=text_color)
        
        # Store bounding box info for visibility checking
        phrase["_render_bbox"] = (x, y, x + text_width, y + text_height)
        
        # Convert to numpy array
        return np.array(img)


def generate_sam2_head_mask(video_path: str, output_mask_path: str) -> bool:
    """
    Generate SAM2 head tracking mask for the video.
    
    Args:
        video_path: Path to input video
        output_mask_path: Path to save the mask video
        
    Returns:
        True if successful, False otherwise
    """
    if not SAM2_API_AVAILABLE:
        print("❌ SAM2 API not available")
        return False
    
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("❌ No REPLICATE_API_TOKEN found")
        return False
    
    print("\n🎯 Generating SAM2 head tracking mask...")
    
    # Read first frame to determine click position
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Could not read video")
        cap.release()
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = first_frame.shape[:2]
    cap.release()
    
    # Click points for head tracking (upper-center region)
    center_x = w // 2
    center_y = h // 3
    
    click_points = [
        (center_x, center_y, 0),  # Center of face
        (center_x - 30, center_y - 30, 0),  # Top-left for hair
        (center_x + 30, center_y - 30, 0),  # Top-right for hair
        (center_x, center_y - 50, 0),  # Top for hair/forehead
    ]
    
    print(f"  Running SAM2 with {len(click_points)} click points...")
    
    # Initialize SAM2 video segmenter
    segmenter = SAM2VideoSegmenter()
    
    # Configure for mask output
    config = SegmentationConfig(
        mask_type="greenscreen",  # Green screen mask for easy processing
        output_video=True,
        video_fps=int(fps),
        annotation_type="mask"
    )
    
    try:
        # Run SAM2 video tracking
        result = segmenter.segment_video_advanced(
            video_path,
            [ClickPoint(x=x, y=y, frame=f, label=1, object_id="head") 
             for x, y, f in click_points],
            config,
            output_mask_path
        )
        
        # Download result if it's a URL
        if isinstance(result, str) and result.startswith('http'):
            print(f"  Downloading mask from Replicate...")
            response = requests.get(result)
            with open(output_mask_path, 'wb') as f:
                f.write(response.content)
        
        print(f"  ✅ SAM2 mask saved to: {output_mask_path}")
        return True
        
    except Exception as e:
        print(f"❌ SAM2 tracking failed: {e}")
        return False


def find_head_bounds(head_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the bounding box of the head in the mask.
    
    Args:
        head_mask: Binary mask where 255 = head, 0 = background
        
    Returns:
        (x_min, y_min, x_max, y_max) of head bounds, or None if no head found
    """
    # Find all non-zero pixels
    head_pixels = np.where(head_mask > 0)
    
    if len(head_pixels[0]) == 0:
        return None
    
    # Get bounding box
    y_min = np.min(head_pixels[0])
    y_max = np.max(head_pixels[0])
    x_min = np.min(head_pixels[1])
    x_max = np.max(head_pixels[1])
    
    return (x_min, y_min, x_max, y_max)


def extract_head_mask_from_sam2(sam2_mask_frame: np.ndarray) -> np.ndarray:
    """
    Extract head mask from SAM2 green screen output.
    SAM2 outputs green where the object is NOT present.
    
    Returns:
        Binary mask where 255 = head, 0 = background
    """
    # SAM2 uses green for background
    green_screen_color = np.array([0, 255, 0], dtype=np.uint8)
    tolerance = 50
    
    # Convert to RGB if needed
    if len(sam2_mask_frame.shape) == 2:
        sam2_mask_frame = cv2.cvtColor(sam2_mask_frame, cv2.COLOR_GRAY2BGR)
    
    # Detect green screen (background)
    diff = np.abs(sam2_mask_frame.astype(np.int16) - green_screen_color.astype(np.int16))
    is_green = np.all(diff <= tolerance, axis=2)
    
    # Head mask is NOT green
    head_mask = (~is_green).astype(np.uint8) * 255
    
    return head_mask


def check_text_head_overlap(text_bbox: Tuple[int, int, int, int], 
                           head_mask: np.ndarray) -> bool:
    """
    Check if text overlaps with head AT ALL (even 1 pixel).
    
    Args:
        text_bbox: (x1, y1, x2, y2) bounding box of text
        head_mask: Binary mask where 255 = head, 0 = background
        
    Returns:
        True if ANY pixel of text overlaps with head
    """
    x1, y1, x2, y2 = text_bbox
    
    # Ensure bbox is within frame bounds
    h, w = head_mask.shape
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    # Check if ANY head pixels in text region
    text_region = head_mask[y1:y2, x1:x2]
    return np.any(text_region > 0)  # Any non-zero pixel = overlap


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


def calculate_optimal_text_size(text: str, base_font_size: int, frame_width: int, 
                                max_multiplier: float = 1.5, min_multiplier: float = 0.3,
                                margin_pixels: int = 30) -> float:
    """
    Calculate optimal size multiplier for text that goes behind.
    Returns the LARGEST multiplier that keeps text within frame margins.
    
    Args:
        text: Text to measure
        base_font_size: Base font size
        frame_width: Width of video frame
        max_multiplier: Maximum size multiplier (for enlarging)
        min_multiplier: Minimum size multiplier (for shrinking if needed)
        margin_pixels: Margin to leave on each side (default 30px)
    """
    # Create temporary font to measure text
    temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp_img)
    
    # Available width is frame width minus margins on both sides
    available_width = frame_width - (2 * margin_pixels)
    
    # Binary search for the LARGEST multiplier that fits
    left, right = min_multiplier, max_multiplier
    optimal = 1.0
    
    for _ in range(15):  # More iterations for precision
        mid = (left + right) / 2
        test_size = int(base_font_size * mid)
        
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', test_size)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= available_width:
            # Text fits, try to make it larger
            optimal = mid
            left = mid
        else:
            # Text too wide, make it smaller
            right = mid
    
    return optimal


def find_optimal_y_position_for_behind_text(merged_text: str, font_size: int, 
                                           phrase_start: float, phrase_end: float, 
                                           fps: float, cap_mask, position: str, 
                                           frame_width: int) -> int:
    """
    Find Y position with least accumulated foreground occlusion for text that goes behind.
    Tests positions from default up to 100px higher in 5px increments.
    
    Args:
        merged_text: Text to display
        font_size: Font size in pixels
        phrase_start: Start time in seconds
        phrase_end: End time in seconds
        fps: Video FPS
        cap_mask: Person mask video capture
        position: "top" or "bottom"
        frame_width: Frame width
        
    Returns:
        Optimal Y position
    """
    # Calculate text dimensions
    temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp_img)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), merged_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1] + 10  # Add some padding
    
    # Calculate centered X position
    x_pos = (frame_width - text_width) // 2
    
    # Define search range
    if position == "top":
        default_y = 180
        # Test from 100px above default to default position
        min_y = max(text_height + 20, default_y - 100)  # Don't go too high
        max_y = default_y
    else:
        default_y = 540
        # For bottom, test from default to 100px above
        min_y = default_y - 100
        max_y = default_y
    
    start_frame = int(phrase_start * fps)
    end_frame = int(phrase_end * fps)
    
    best_y = default_y
    min_occlusion = float('inf')
    
    # Save current position
    orig_mask_pos = cap_mask.get(cv2.CAP_PROP_POS_FRAMES)
    
    # Test every 5 pixels
    for test_y in range(min_y, max_y + 1, 5):
        total_occlusion = 0
        
        # Sample frames throughout phrase duration (every 10 frames)
        sample_frames = range(start_frame, min(end_frame + 1, int(cap_mask.get(cv2.CAP_PROP_FRAME_COUNT))), 10)
        
        for frame_idx in sample_frames:
            # Read mask frame
            cap_mask.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, mask_frame = cap_mask.read()
            if not ret:
                continue
            
            # Extract person mask
            green_screen_color = np.array([154, 254, 119], dtype=np.uint8)
            tolerance = 25
            diff = np.abs(mask_frame.astype(np.int16) - green_screen_color.astype(np.int16))
            is_green_screen = np.all(diff <= tolerance, axis=2)
            person_mask = (~is_green_screen).astype(np.uint8)
            
            # Calculate occlusion in text bounding box
            # Note: test_y is the TOP of the text (FFmpeg convention)
            y1 = max(0, test_y)
            y2 = min(mask_frame.shape[0], test_y + text_height)
            x1 = max(0, x_pos)
            x2 = min(mask_frame.shape[1], x_pos + text_width)
            
            if y2 > y1 and x2 > x1:
                text_region = person_mask[y1:y2, x1:x2]
                occlusion = np.sum(text_region)
                total_occlusion += occlusion
        
        # Update best position if less occluded
        if total_occlusion < min_occlusion:
            min_occlusion = total_occlusion
            best_y = test_y
    
    # Restore original position
    cap_mask.set(cv2.CAP_PROP_POS_FRAMES, orig_mask_pos)
    
    # Log the optimization result
    if best_y != default_y:
        shift = default_y - best_y
        print(f"      Y position optimized: shifted up {shift}px for less occlusion")
    
    return best_y


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


def apply_sam2_head_aware_sandwich(
    original_video: str,
    mask_video: str,
    transcript_path: str,
    output_path: str,
    visibility_threshold: float = 0.9
):
    """
    Apply SAM2 head-aware sandwich compositing.
    Text ALWAYS goes behind if it touches the head (even 1 pixel).
    Otherwise uses visibility threshold for body occlusion.
    
    Args:
        original_video: Path to original video
        mask_video: Path to green screen mask video (person mask)
        transcript_path: Path to enriched transcript JSON
        output_path: Output video path
        visibility_threshold: Min visibility for non-head areas (default 0.9 = 90%)
    """
    # Determine video name and folder
    video_name = Path(original_video).stem
    video_dir = Path(original_video).parent
    
    # Check for cached SAM2 head mask
    sam2_mask_path = video_dir / ".." / ".." / "uploads" / "assets" / "videos" / video_name.replace('_6sec', '') / f"{video_name.replace('_6sec', '')}_sam2_head_mask.mp4"
    sam2_mask_path = sam2_mask_path.resolve()
    
    print(f"\n🔍 Checking for cached SAM2 head mask at: {sam2_mask_path}")
    
    if not sam2_mask_path.exists():
        print("  No cached mask found, generating with SAM2...")
        if not generate_sam2_head_mask(original_video, str(sam2_mask_path)):
            print("  ⚠️ Failed to generate SAM2 mask, proceeding without head detection")
            sam2_mask_path = None
    else:
        print(f"  ✅ Using cached SAM2 head mask")
    
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
    cap_head = cv2.VideoCapture(str(sam2_mask_path)) if sam2_mask_path else None
    
    # Get video properties
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties: {width}x{height}, {fps} fps, {total_frames} frames")
    print(f"Visibility threshold: {visibility_threshold:.0%} for non-head areas")
    print(f"Head overlap rule: ANY overlap = text goes behind")
    
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
    
    # Pre-analyze phrases to determine which go behind and optimize their placement
    print("\n🔍 Pre-analyzing phrases for optimal placement...")
    phrase_optimizations = {}  # Store size multiplier and Y position for each phrase
    
    # First pass: determine which phrases go behind
    for phrase in transcript_data.get("phrases", []):
        phrase_key = f"{phrase['start_time']:.2f}_{phrase['text'][:20]}"
        scene_end = scene_end_times.get(phrase_key, phrase["end_time"])
        
        # Quick check if phrase will go behind (sample middle of phrase duration)
        mid_time = (phrase["start_time"] + scene_end) / 2
        mid_frame = int(mid_time * fps)
        
        # Check for head overlap
        will_go_behind = False
        if cap_head and mid_frame < total_frames:
            cap_head.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret_head, head_frame = cap_head.read()
            if ret_head:
                head_mask = extract_head_mask_from_sam2(head_frame)
                
                # Estimate text position
                font_size = int(48 * phrase["visual_style"]["font_size_multiplier"])
                text_width = len(phrase["text"]) * int(font_size * 0.6)
                x = (width - text_width) // 2
                y = 180 if phrase.get("position") == "top" else 540
                text_bbox = (x, y - font_size, x + text_width, y + int(font_size * 0.5))
                
                if check_text_head_overlap(text_bbox, head_mask):
                    will_go_behind = True
        
        # If not behind due to head, check visibility
        if not will_go_behind and mid_frame < total_frames:
            cap_mask.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, mask_frame = cap_mask.read()
            if ret:
                green_screen_color = np.array([154, 254, 119], dtype=np.uint8)
                tolerance = 25
                diff = np.abs(mask_frame.astype(np.int16) - green_screen_color.astype(np.int16))
                is_green_screen = np.all(diff <= tolerance, axis=2)
                person_mask = (~is_green_screen).astype(np.uint8)
                
                font_size = int(48 * phrase["visual_style"]["font_size_multiplier"])
                text_width = len(phrase["text"]) * int(font_size * 0.6)
                x = (width - text_width) // 2
                y = 180 if phrase.get("position") == "top" else 540
                text_bbox = (x, y - font_size, x + text_width, y + int(font_size * 0.5))
                
                visibility = calculate_text_visibility(text_bbox, person_mask)
                if visibility > visibility_threshold:
                    will_go_behind = True
                
                # Debug visibility for key phrases
                if "AI created" in phrase["text"]:
                    print(f"  '{phrase['text']}': visibility={visibility:.1%} (threshold={visibility_threshold:.1%}) -> behind={visibility > visibility_threshold}")
        
        # Store initial decision
        phrase_optimizations[phrase_key] = {
            "goes_behind": will_go_behind,
            "size_multiplier": 1.0,  # Will be updated
            "merged_group": None,  # Will be set if part of merged group
            "is_primary": False  # True for the main phrase in merged group
        }
        
        # Debug key phrases
        if "AI created" in phrase["text"]:
            print(f"  Pre-analysis: '{phrase['text']}' -> goes_behind={will_go_behind}")
    
    # Second pass: group consecutive behind-face phrases by position
    # We need to find consecutive phrases that:
    # 1. Have the same position (top/bottom)
    # 2. Both go behind the face
    # 3. Are temporally adjacent (one ends when the other starts, or overlapping)
    
    # Build a list of all phrases with their optimization info
    all_phrases_with_opt = []
    for phrase in transcript_data.get("phrases", []):
        phrase_key = f"{phrase['start_time']:.2f}_{phrase['text'][:20]}"
        all_phrases_with_opt.append((phrase, phrase_key, phrase_optimizations[phrase_key]))
    
    # Sort by start time to ensure we process in temporal order
    all_phrases_with_opt.sort(key=lambda x: x[0]["start_time"])
    
    # Group consecutive behind-face phrases
    merged_groups = []
    current_group = []
    current_position = None
    
    for phrase, phrase_key, opt in all_phrases_with_opt:
        if opt["goes_behind"]:
            position = phrase.get("position", "bottom")
            
            # Check if this phrase can be added to current group
            if current_group and current_position == position:
                # Check temporal adjacency (within 0.5 seconds gap)
                last_phrase = current_group[-1][0]
                last_end = scene_end_times.get(current_group[-1][1], last_phrase["end_time"])
                time_gap = phrase["start_time"] - last_end
                
                if time_gap <= 0.5:  # Allow up to 0.5 second gap
                    current_group.append((phrase, phrase_key))
                else:
                    # Gap too large, finalize current group and start new one
                    if len(current_group) >= 2:
                        merged_groups.append((current_position, current_group))
                    current_group = [(phrase, phrase_key)]
                    current_position = position
            else:
                # Start new group
                if current_group and len(current_group) >= 2:
                    merged_groups.append((current_position, current_group))
                current_group = [(phrase, phrase_key)]
                current_position = position
        else:
            # Not a behind phrase, finalize any current group
            if current_group and len(current_group) >= 2:
                merged_groups.append((current_position, current_group))
            current_group = []
            current_position = None
    
    # Don't forget the last group
    if current_group and len(current_group) >= 2:
        merged_groups.append((current_position, current_group))
    
    # Process single behind-face phrases (not in any group)
    single_behind_phrases = []
    merged_phrase_keys = set()
    for position, group in merged_groups:
        for phrase, phrase_key in group:
            merged_phrase_keys.add(phrase_key)
    
    for phrase, phrase_key, opt in all_phrases_with_opt:
        if opt["goes_behind"] and phrase_key not in merged_phrase_keys:
            single_behind_phrases.append((phrase, phrase_key))
    
    # Create merged groups where 2+ consecutive phrases go behind
    for position, behind_phrases in merged_groups:
        if len(behind_phrases) >= 2:
            # Merge these phrases
            merged_texts = []
            phrase_keys = []
            merged_word_timings = []  # Collect all word timings
            earliest_start = float('inf')
            latest_end = 0
            
            for phrase, phrase_key in behind_phrases:
                merged_texts.append(phrase["text"])
                phrase_keys.append(phrase_key)
                scene_end = scene_end_times.get(phrase_key, phrase["end_time"])
                earliest_start = min(earliest_start, phrase["start_time"])
                latest_end = max(latest_end, scene_end)
                
                # Collect word timings if available
                if "word_timings" in phrase and phrase["word_timings"]:
                    merged_word_timings.extend(phrase["word_timings"])
            
            merged_text = " ".join(merged_texts)
            
            # Calculate optimal size for merged text - ENLARGE as much as possible!
            base_font_size = 48
            size_multiplier = calculate_optimal_text_size(
                merged_text, base_font_size, width,
                max_multiplier=2.5,  # Allow enlarging up to 2.5x for maximum visibility
                min_multiplier=0.3,  # Can shrink if absolutely necessary
                margin_pixels=30     # 30px margin on each side
            )
            
            # Create merged group ID
            merged_group_id = f"merged_{position}_{earliest_start:.2f}"
            
            # Calculate optimal Y position for merged text
            actual_font_size = int(48 * size_multiplier)
            optimal_y = find_optimal_y_position_for_behind_text(
                merged_text, actual_font_size, 
                earliest_start, latest_end,
                fps, cap_mask, position, width
            )
            
            # Update all phrases in the group
            for i, phrase_key in enumerate(phrase_keys):
                phrase_optimizations[phrase_key]["merged_group"] = merged_group_id
                phrase_optimizations[phrase_key]["is_primary"] = (i == 0)  # First phrase is primary
                phrase_optimizations[phrase_key]["size_multiplier"] = size_multiplier
                phrase_optimizations[phrase_key]["merged_text"] = merged_text
                phrase_optimizations[phrase_key]["merged_start"] = earliest_start
                phrase_optimizations[phrase_key]["merged_end"] = latest_end
                phrase_optimizations[phrase_key]["merged_word_timings"] = merged_word_timings
                phrase_optimizations[phrase_key]["optimal_y"] = optimal_y
            
            print(f"  Merging {len(phrase_keys)} {position} phrases: font scaled to {size_multiplier:.2f}x")
            print(f"    Merged text: '{merged_text[:50]}...'")
    
    # Process single behind-face phrases
    for phrase, phrase_key in single_behind_phrases:
        # Check if phrase is very short (less than 7 characters)
        if len(phrase["text"]) < 7:
            # Short phrase - will track to the left of head instead of going behind
            phrase_optimizations[phrase_key]["track_head"] = True
            phrase_optimizations[phrase_key]["goes_behind"] = False  # Override - don't go behind
            phrase_optimizations[phrase_key]["size_multiplier"] = 1.2  # Slightly larger but not huge
            # Head-tracking phrases use slide-from-behind animation (special case)
            phrase_optimizations[phrase_key]["entrance_effect"] = EntranceEffect.FADE_SLIDE_FROM_RIGHT
            print(f"  '{phrase['text']}': will track head with slide-from-behind animation")
        else:
            # Normal behind-face processing for longer phrases
            base_font_size = 48
            size_multiplier = calculate_optimal_text_size(
                phrase["text"], base_font_size, width, 
                max_multiplier=2.5,  # Allow enlarging up to 2.5x
                min_multiplier=0.3,  # Can shrink if needed
                margin_pixels=30     # 30px margin on each side
            )
            
            # Calculate optimal Y position for single phrase
            actual_font_size = int(48 * size_multiplier)
            scene_end = scene_end_times.get(phrase_key, phrase["end_time"])
            optimal_y = find_optimal_y_position_for_behind_text(
                phrase["text"], actual_font_size,
                phrase["start_time"], scene_end,
                fps, cap_mask, phrase.get("position", "bottom"), width
            )
            
            phrase_optimizations[phrase_key]["size_multiplier"] = size_multiplier
            phrase_optimizations[phrase_key]["optimal_y"] = optimal_y
            print(f"  '{phrase['text'][:30]}...': enlarged {size_multiplier:.2f}x")
    
    # Third pass: Assign random entrance effects to all non-head-tracking phrases
    print("\n🎨 Assigning random entrance effects to phrases...")
    for phrase in transcript_data.get("phrases", []):
        phrase_key = f"{phrase['start_time']:.2f}_{phrase['text'][:20]}"
        opt = phrase_optimizations[phrase_key]
        
        # Skip if already has an effect (e.g., head-tracking phrases)
        if opt.get("entrance_effect") is not None:
            continue
        
        # Assign random effect (excluding side slides which are for head-tracking)
        opt["entrance_effect"] = EntranceEffect.get_random_effect(exclude_side_effects=True)
        
        # Log effect assignment for first few phrases
        if phrase_key in [f"{p['start_time']:.2f}_{p['text'][:20]}" for p in transcript_data.get("phrases", [])[:5]]:
            effect_name = opt["entrance_effect"].value
            print(f"  '{phrase['text'][:20]}...' → {effect_name}")
    
    # Fourth pass: Assign disappearance effects per scene (all phrases in a scene get same effect)
    print("\n💨 Assigning disappearance effects per scene...")
    scene_disappearance_effects = {}  # Map scene_idx to disappearance effect
    
    for scene_idx in scenes.keys():
        # Assign a random disappearance effect for this scene
        scene_effect = DisappearanceEffect.get_random_effect()
        scene_disappearance_effects[scene_idx] = scene_effect
        
        # Apply to all phrases in this scene
        for phrase in scenes[scene_idx]:
            phrase_key = f"{phrase['start_time']:.2f}_{phrase['text'][:20]}"
            phrase_optimizations[phrase_key]["disappearance_effect"] = scene_effect
        
        # Log for first few scenes
        if scene_idx < 3:
            effect_name = scene_effect.value
            phrase_count = len(scenes[scene_idx])
            print(f"  Scene {scene_idx}: {phrase_count} phrases → {effect_name}")
    
    # Reset video positions
    cap_mask.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if cap_head:
        cap_head.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\nProcessing {total_frames} frames with SAM2 head-aware compositing...")
    print(f"Found {len(scenes)} scenes with grouped timing")
    print(f"\nDisappearance effects by scene:")
    for scene_idx in sorted(scenes.keys())[:5]:  # Show first 5 scenes
        effect = scene_disappearance_effects.get(scene_idx, DisappearanceEffect.FADE_OUT)
        print(f"  Scene {scene_idx}: {effect.value}")
    
    # Track statistics
    behind_head = 0
    behind_visibility = 0
    front_count = 0
    
    # Process each frame
    for frame_idx in range(total_frames):
        ret_orig, frame_original = cap_orig.read()
        ret_mask, mask_frame = cap_mask.read()
        
        if not ret_orig or not ret_mask:
            break
        
        # Read head mask if available
        head_mask = None
        if cap_head:
            ret_head, head_frame = cap_head.read()
            if ret_head:
                head_mask = extract_head_mask_from_sam2(head_frame)
        
        current_time = frame_idx / fps
        
        # Track vertical positions for stacking
        top_phrases = []
        bottom_phrases = []
        
        # First pass: collect visible phrases and organize by position
        disappear_duration = 0.5  # Must match render_phrase duration
        for phrase in transcript_data.get("phrases", []):
            phrase_key = f"{phrase['start_time']:.2f}_{phrase['text'][:20]}"
            logical_end = scene_end_times.get(phrase_key, phrase["end_time"])
            draw_end = logical_end + disappear_duration
            
            # Check if phrase is visible at current time (match render_phrase gate)
            if phrase["start_time"] <= current_time <= draw_end:
                if phrase.get("position") == "top":
                    top_phrases.append((phrase, phrase_key, logical_end))
                else:
                    bottom_phrases.append((phrase, phrase_key, logical_end))
        
        # Start with original frame
        composite = frame_original.copy()
        
        # Extract foreground mask (person body)
        green_screen_color = np.array([154, 254, 119], dtype=np.uint8)
        tolerance = 25
        
        diff = np.abs(mask_frame.astype(np.int16) - green_screen_color.astype(np.int16))
        is_green_screen = np.all(diff <= tolerance, axis=2)
        
        # Person mask: NOT green screen
        person_mask = (~is_green_screen).astype(np.uint8)
        
        # Light erosion
        kernel = np.ones((2,2), np.uint8)
        person_mask = cv2.erode(person_mask, kernel, iterations=1)
        
        # Calculate Y positions for stacking with pre-computed merging decisions
        top_y_positions = []
        bottom_y_positions = []
        processed_groups = set()  # Track which merged groups we've already handled
        
        # Stack top phrases (starting at y=180, going down)
        current_y = 180
        for i, (phrase, phrase_key, _) in enumerate(top_phrases):
            optimization = phrase_optimizations.get(phrase_key, {})
            merged_group = optimization.get("merged_group")
            
            if merged_group and merged_group not in processed_groups:
                # This is the first phrase in a merged group
                processed_groups.add(merged_group)
                size_mult = optimization.get("size_multiplier", 1.0)
                font_size = int(48 * size_mult)
                top_y_positions.append(current_y)
                current_y += int(font_size * 1.5)
            elif merged_group:
                # This phrase is part of an already-processed merged group
                top_y_positions.append(None)  # Skip it
            else:
                # Normal phrase (not merged)
                size_mult = optimization.get("size_multiplier", 1.0)
                if optimization.get("goes_behind", False) and size_mult > 1.0:
                    font_size = int(48 * size_mult)
                else:
                    font_size = int(48 * phrase["visual_style"]["font_size_multiplier"])
                
                top_y_positions.append(current_y)
                current_y += int(font_size * 1.5)
        
        # Stack bottom phrases (starting at y=540, going down)
        current_y = 540
        for i, (phrase, phrase_key, _) in enumerate(bottom_phrases):
            optimization = phrase_optimizations.get(phrase_key, {})
            merged_group = optimization.get("merged_group")
            
            if merged_group and merged_group not in processed_groups:
                # This is the first phrase in a merged group
                processed_groups.add(merged_group)
                size_mult = optimization.get("size_multiplier", 1.0)
                font_size = int(48 * size_mult)
                bottom_y_positions.append(current_y)
                current_y += int(font_size * 1.5)
            elif merged_group:
                # This phrase is part of an already-processed merged group
                bottom_y_positions.append(None)  # Skip it
            else:
                # Normal phrase (not merged)
                size_mult = optimization.get("size_multiplier", 1.0)
                if optimization.get("goes_behind", False) and size_mult > 1.0:
                    font_size = int(48 * size_mult)
                else:
                    font_size = int(48 * phrase["visual_style"]["font_size_multiplier"])
                
                bottom_y_positions.append(current_y)
                current_y += int(font_size * 1.5)
        
        # Check if we need to shift stacks due to heavy occlusion
        occlusion_threshold = 0.1  # 10% visibility = 90% occluded
        
        # First pass: render top phrases and check occlusion
        top_needs_shift = False
        shift_amount = 0
        temp_top_renders = []
        
        for i, (phrase, phrase_key, scene_end) in enumerate(top_phrases):
            optimization = phrase_optimizations.get(phrase_key, {})
            size_mult = optimization.get("size_multiplier", 1.0)
            y_pos = top_y_positions[i]
            
            phrase_img = phrase_renderer.render_phrase(
                phrase, current_time, (height, width), scene_end, y_pos, size_mult,
                entrance_effect=optimization.get("entrance_effect"),
                disappearance_effect=optimization.get("disappearance_effect")
            )
            
            if phrase_img is not None and "_render_bbox" in phrase:
                # Check visibility
                visibility = calculate_text_visibility(phrase["_render_bbox"], person_mask)
                
                # If any phrase is heavily occluded, we need to shift the entire stack
                if visibility < occlusion_threshold:
                    if not top_needs_shift:
                        top_needs_shift = True
                        # IMPORTANT: Mark ALL heavily occluded phrases to go behind
                        # This ensures they stay behind during disappearance
                        for _, pk, _ in top_phrases:
                            if pk in phrase_optimizations:
                                phrase_optimizations[pk]["runtime_goes_behind"] = True
                                if "AI created" in phrase_optimizations[pk].get("text", ""):
                                    print(f"    Runtime: Marking '{phrase['text']}' as goes_behind due to occlusion ({visibility:.1%})")
                    # Calculate how much to shift up
                    # Try shifting up by 20 pixels at a time until we find good visibility
                    test_y = y_pos
                    for shift_try in range(1, 10):  # Max 9 attempts
                        test_y = y_pos - (shift_try * 20)
                        if test_y < 50:  # Don't go too high
                            break
                        # Test visibility at new position
                        x1, _, x2, _ = phrase["_render_bbox"]
                        text_height = phrase["_render_bbox"][3] - phrase["_render_bbox"][1]
                        test_bbox = (x1, test_y - text_height, x2, test_y)
                        test_visibility = calculate_text_visibility(test_bbox, person_mask)
                        if test_visibility > 0.3:  # At least 30% visible
                            shift_amount = shift_try * 20
                            if frame_idx % 30 == 0:  # Only print occasionally
                                print(f"    Shifting top stack UP by {shift_amount}px (visibility {visibility:.1%} -> {test_visibility:.1%})")
                            break
                
                temp_top_renders.append((phrase_img, phrase, phrase_key, scene_end, i))
        
        # Re-render phrases with shift if needed
        phrases_to_render = []
        
        # Render top phrases
        for i, (phrase, phrase_key, scene_end) in enumerate(top_phrases):
            optimization = phrase_optimizations.get(phrase_key, {})
            
            # Skip if Y position is None (non-primary merged phrase)
            if top_y_positions[i] is None:
                continue
            
            # Check if this phrase tracks the head
            if optimization.get("track_head", False):
                # Check if phrase is visible at current time (including disappearance)
                disappear_duration = 0.5
                draw_end = scene_end + disappear_duration
                if not (phrase["start_time"] <= current_time <= draw_end):
                    continue
                    
                # Find head position in current frame
                if head_mask is not None:
                    head_bounds = find_head_bounds(head_mask)
                    if head_bounds:
                        x_min, y_min, x_max, y_max = head_bounds
                        
                        # Render with custom position
                        size_mult = optimization.get("size_multiplier", 1.2)
                        
                        # Create custom render with absolute positioning
                        temp_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                        draw = ImageDraw.Draw(temp_img)
                        font_size = int(48 * size_mult)
                        
                        try:
                            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
                        except:
                            font = ImageFont.load_default()
                        
                        # Get text dimensions
                        text = phrase["text"]
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height_pixels = bbox[3] - bbox[1]
                        
                        # Calculate animation progress (entrance and disappearance)
                        time_since_start = current_time - phrase["start_time"]
                        time_until_end = scene_end - current_time
                        entrance_duration = 0.5  # 500ms for slide + fade in
                        disappear_duration = 0.5  # 500ms for fade out
                        
                        # Get disappearance effect for this phrase's scene
                        disappearance_effect = optimization.get("disappearance_effect", DisappearanceEffect.FADE_OUT)
                        
                        if time_since_start < entrance_duration:
                            # ENTRANCE ANIMATION
                            progress = time_since_start / entrance_duration
                            # Ease-out cubic for smooth deceleration
                            eased_progress = 1 - pow(1 - progress, 3)
                            
                            # Opacity: fade in from 0 to 1
                            opacity = int(255 * progress)
                            
                            # Position: slide from behind face to final position
                            gap_from_face = 35  # Final gap
                            final_x = x_min - gap_from_face - text_width
                            
                            # Start position: behind the face (right edge at face center)
                            start_x = (x_min + x_max) // 2 - text_width
                            
                            # Interpolate X position
                            text_x = start_x + (final_x - start_x) * eased_progress
                            text_x = max(10, text_x)  # Keep at least 10px from left edge
                        elif current_time < scene_end:
                            # STEADY STATE - full opacity, final position
                            opacity = 255
                            gap_from_face = 35
                            text_x = x_min - gap_from_face - text_width
                            text_x = max(10, text_x)
                        else:
                            # DISAPPEARANCE ANIMATION (current_time >= scene_end)
                            disappear_progress = min((current_time - scene_end) / disappear_duration, 1.0)
                            
                            # Apply disappearance effect (simplified for head-tracking)
                            if disappearance_effect in [DisappearanceEffect.SLIDE_OUT_LEFT, DisappearanceEffect.SLIDE_OUT_RIGHT]:
                                # Slide out horizontally
                                gap_from_face = 35
                                base_x = x_min - gap_from_face - text_width
                                if disappearance_effect == DisappearanceEffect.SLIDE_OUT_LEFT:
                                    text_x = base_x - int(disappear_progress * text_width * 1.5)
                                else:
                                    text_x = base_x + int(disappear_progress * text_width * 1.5)
                                opacity = int(255 * (1 - disappear_progress))
                            else:
                                # Default: fade out in place
                                gap_from_face = 35
                                text_x = x_min - gap_from_face - text_width
                                text_x = max(10, text_x)
                                opacity = int(255 * (1 - disappear_progress))
                            
                            # Debug disappearance
                            if disappear_progress < 0.02 and "Yes" in text:
                                print(f"    🎬 HEAD-TRACK DISAPPEAR: '{text}' starting at t={current_time:.2f}s")
                        
                        # Y: vertically centered with head (no animation)
                        text_y = (y_min + y_max) // 2 - text_height_pixels // 2
                        
                        # Black outline with opacity
                        outline_alpha = opacity
                        for dx in [-2, -1, 0, 1, 2]:
                            for dy in [-2, -1, 0, 1, 2]:
                                if dx != 0 or dy != 0:
                                    draw.text((text_x + dx, text_y + dy), text, 
                                             font=font, fill=(0, 0, 0, outline_alpha))
                        
                        # White text with opacity
                        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, opacity))
                        
                        phrase_img = np.array(temp_img)
                        
                        if phrase_img is not None:
                            # During entrance animation, check if text should be behind face
                            if time_since_start < entrance_duration:
                                # Check if text overlaps with face during slide
                                text_bbox = (int(text_x), int(text_y), 
                                           int(text_x + text_width), int(text_y + text_height_pixels))
                                
                                # Text goes behind if it overlaps with head during animation
                                text_overlaps_head = check_text_head_overlap(text_bbox, head_mask)
                                should_be_behind = text_overlaps_head
                            else:
                                # After animation, text is always in front
                                should_be_behind = False
                            
                            phrases_to_render.append((phrase_img, should_be_behind, phrase["text"], "head-track"))
                
                # Skip normal rendering for head-tracking phrases
                continue
            
            # Apply shift if needed (for non-head-tracking phrases)
            y_pos = top_y_positions[i] - shift_amount if top_needs_shift else top_y_positions[i]
            
            # Check if this is a merged phrase (primary only)
            if optimization.get("is_primary", False) and optimization.get("merged_group"):
                # Render merged text
                merged_text = optimization.get("merged_text", phrase["text"])
                merged_start = optimization.get("merged_start", phrase["start_time"])
                merged_end = optimization.get("merged_end", scene_end)
                merged_word_timings = optimization.get("merged_word_timings", [])
                
                # Use optimal Y if available (for behind-face text)
                if "optimal_y" in optimization:
                    y_pos = optimization["optimal_y"] - shift_amount if top_needs_shift else optimization["optimal_y"]
                
                # Check if merged phrase is visible (including disappearance)
                disappear_duration = 0.5
                if merged_start <= current_time <= merged_end + disappear_duration:
                    # Create a modified phrase with merged word timings
                    merged_phrase = phrase.copy()
                    if merged_word_timings:
                        merged_phrase["word_timings"] = merged_word_timings
                    
                    phrase_img = phrase_renderer.render_phrase(
                        merged_phrase, 
                        current_time, 
                        (height, width), 
                        merged_end, 
                        y_pos, 
                        optimization["size_multiplier"],
                        merged_text=merged_text,
                        entrance_effect=optimization.get("entrance_effect"),
                        disappearance_effect=optimization.get("disappearance_effect")
                    )
                    
                    if phrase_img is not None:
                        # Merged phrases always go behind (that's why we merged them)
                        phrases_to_render.append((phrase_img, True, f"[MERGED]", "merged"))
            else:
                # Normal phrase rendering
                size_mult = optimization.get("size_multiplier", 1.0)
                
                # Use optimal Y if available (for single behind-face phrases)
                if optimization.get("goes_behind", False) and "optimal_y" in optimization:
                    y_pos = optimization["optimal_y"] - shift_amount if top_needs_shift else optimization["optimal_y"]
                
                phrase_img = phrase_renderer.render_phrase(
                    phrase, current_time, (height, width), scene_end, y_pos, size_mult,
                    entrance_effect=optimization.get("entrance_effect"),
                    disappearance_effect=optimization.get("disappearance_effect")
                )
                
                if phrase_img is not None:
                    # Use decision about behind/front - runtime occlusion overrides pre-computed
                    should_be_behind = optimization.get("runtime_goes_behind", optimization.get("goes_behind", False))
                    reason = "runtime-occluded" if optimization.get("runtime_goes_behind", False) else "pre-computed"
                    phrases_to_render.append((phrase_img, should_be_behind, phrase["text"][:20], reason))
        
        # Check bottom phrases for occlusion
        bottom_needs_shift = False
        bottom_shift_amount = 0
        
        for i, (phrase, phrase_key, scene_end) in enumerate(bottom_phrases):
            optimization = phrase_optimizations.get(phrase_key, {})
            size_mult = optimization.get("size_multiplier", 1.0)
            y_pos = bottom_y_positions[i]
            
            temp_img = phrase_renderer.render_phrase(
                phrase, current_time, (height, width), scene_end, y_pos, size_mult,
                entrance_effect=optimization.get("entrance_effect"),
                disappearance_effect=optimization.get("disappearance_effect")
            )
            
            if temp_img is not None and "_render_bbox" in phrase:
                visibility = calculate_text_visibility(phrase["_render_bbox"], person_mask)
                
                if visibility < occlusion_threshold:
                    if not bottom_needs_shift:
                        bottom_needs_shift = True
                        # IMPORTANT: Mark ALL heavily occluded bottom phrases to go behind
                        # This ensures they stay behind during disappearance
                        for _, pk, _ in bottom_phrases:
                            if pk in phrase_optimizations:
                                phrase_optimizations[pk]["runtime_goes_behind"] = True
                    # For bottom, shift down
                    for shift_try in range(1, 10):
                        test_y = y_pos + (shift_try * 20)
                        if test_y > height - 50:  # Don't go too low
                            break
                        x1, _, x2, _ = phrase["_render_bbox"]
                        text_height = phrase["_render_bbox"][3] - phrase["_render_bbox"][1]
                        test_bbox = (x1, test_y, x2, test_y + text_height)
                        test_visibility = calculate_text_visibility(test_bbox, person_mask)
                        if test_visibility > 0.3:
                            bottom_shift_amount = shift_try * 20
                            break
        
        # Render bottom phrases
        for i, (phrase, phrase_key, scene_end) in enumerate(bottom_phrases):
            optimization = phrase_optimizations.get(phrase_key, {})
            
            # Skip if Y position is None (non-primary merged phrase)
            if bottom_y_positions[i] is None:
                continue
            
            # Check if this phrase tracks the head
            if optimization.get("track_head", False):
                # Check if phrase is visible at current time (including disappearance)
                disappear_duration = 0.5
                draw_end = scene_end + disappear_duration
                if not (phrase["start_time"] <= current_time <= draw_end):
                    continue
                    
                # Find head position in current frame
                if head_mask is not None:
                    head_bounds = find_head_bounds(head_mask)
                    if head_bounds:
                        x_min, y_min, x_max, y_max = head_bounds
                        
                        # Render with custom position
                        size_mult = optimization.get("size_multiplier", 1.2)
                        
                        # Create custom render with absolute positioning
                        temp_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                        draw = ImageDraw.Draw(temp_img)
                        font_size = int(48 * size_mult)
                        
                        try:
                            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
                        except:
                            font = ImageFont.load_default()
                        
                        # Get text dimensions
                        text = phrase["text"]
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height_pixels = bbox[3] - bbox[1]
                        
                        # Calculate animation progress (entrance and disappearance)
                        time_since_start = current_time - phrase["start_time"]
                        time_until_end = scene_end - current_time
                        entrance_duration = 0.5  # 500ms for slide + fade in
                        disappear_duration = 0.5  # 500ms for fade out
                        
                        # Get disappearance effect for this phrase's scene
                        disappearance_effect = optimization.get("disappearance_effect", DisappearanceEffect.FADE_OUT)
                        
                        if time_since_start < entrance_duration:
                            # ENTRANCE ANIMATION
                            progress = time_since_start / entrance_duration
                            # Ease-out cubic for smooth deceleration
                            eased_progress = 1 - pow(1 - progress, 3)
                            
                            # Opacity: fade in from 0 to 1
                            opacity = int(255 * progress)
                            
                            # Position: slide from behind face to final position
                            gap_from_face = 35  # Final gap
                            final_x = x_min - gap_from_face - text_width
                            
                            # Start position: behind the face (right edge at face center)
                            start_x = (x_min + x_max) // 2 - text_width
                            
                            # Interpolate X position
                            text_x = start_x + (final_x - start_x) * eased_progress
                            text_x = max(10, text_x)  # Keep at least 10px from left edge
                        elif current_time < scene_end:
                            # STEADY STATE - full opacity, final position
                            opacity = 255
                            gap_from_face = 35
                            text_x = x_min - gap_from_face - text_width
                            text_x = max(10, text_x)
                        else:
                            # DISAPPEARANCE ANIMATION (current_time >= scene_end)
                            disappear_progress = min((current_time - scene_end) / disappear_duration, 1.0)
                            
                            # Apply disappearance effect (simplified for head-tracking)
                            if disappearance_effect in [DisappearanceEffect.SLIDE_OUT_LEFT, DisappearanceEffect.SLIDE_OUT_RIGHT]:
                                # Slide out horizontally
                                gap_from_face = 35
                                base_x = x_min - gap_from_face - text_width
                                if disappearance_effect == DisappearanceEffect.SLIDE_OUT_LEFT:
                                    text_x = base_x - int(disappear_progress * text_width * 1.5)
                                else:
                                    text_x = base_x + int(disappear_progress * text_width * 1.5)
                                opacity = int(255 * (1 - disappear_progress))
                            else:
                                # Default: fade out in place
                                gap_from_face = 35
                                text_x = x_min - gap_from_face - text_width
                                text_x = max(10, text_x)
                                opacity = int(255 * (1 - disappear_progress))
                            
                            # Debug disappearance
                            if disappear_progress < 0.02 and "Yes" in text:
                                print(f"    🎬 HEAD-TRACK DISAPPEAR: '{text}' starting at t={current_time:.2f}s")
                        
                        # Y: vertically centered with head (no animation)
                        text_y = (y_min + y_max) // 2 - text_height_pixels // 2
                        
                        # Black outline with opacity
                        outline_alpha = opacity
                        for dx in [-2, -1, 0, 1, 2]:
                            for dy in [-2, -1, 0, 1, 2]:
                                if dx != 0 or dy != 0:
                                    draw.text((text_x + dx, text_y + dy), text, 
                                             font=font, fill=(0, 0, 0, outline_alpha))
                        
                        # White text with opacity
                        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, opacity))
                        
                        phrase_img = np.array(temp_img)
                        
                        if phrase_img is not None:
                            # During entrance animation, check if text should be behind face
                            if time_since_start < entrance_duration:
                                # Check if text overlaps with face during slide
                                text_bbox = (int(text_x), int(text_y), 
                                           int(text_x + text_width), int(text_y + text_height_pixels))
                                
                                # Text goes behind if it overlaps with head during animation
                                text_overlaps_head = check_text_head_overlap(text_bbox, head_mask)
                                should_be_behind = text_overlaps_head
                            else:
                                # After animation, text is always in front
                                should_be_behind = False
                            
                            phrases_to_render.append((phrase_img, should_be_behind, phrase["text"], "head-track"))
                
                # Skip normal rendering for head-tracking phrases
                continue
            
            # Apply shift if needed (for non-head-tracking phrases)
            y_pos = bottom_y_positions[i] + bottom_shift_amount if bottom_needs_shift else bottom_y_positions[i]
            
            # Check if this is a merged phrase (primary only)
            if optimization.get("is_primary", False) and optimization.get("merged_group"):
                # Render merged text
                merged_text = optimization.get("merged_text", phrase["text"])
                merged_start = optimization.get("merged_start", phrase["start_time"])
                merged_end = optimization.get("merged_end", scene_end)
                merged_word_timings = optimization.get("merged_word_timings", [])
                
                # Use optimal Y if available (for behind-face text)
                if "optimal_y" in optimization:
                    y_pos = optimization["optimal_y"] + bottom_shift_amount if bottom_needs_shift else optimization["optimal_y"]
                
                # Check if merged phrase is visible (including disappearance)
                disappear_duration = 0.5
                if merged_start <= current_time <= merged_end + disappear_duration:
                    # Create a modified phrase with merged word timings
                    merged_phrase = phrase.copy()
                    if merged_word_timings:
                        merged_phrase["word_timings"] = merged_word_timings
                    
                    phrase_img = phrase_renderer.render_phrase(
                        merged_phrase, 
                        current_time, 
                        (height, width), 
                        merged_end, 
                        y_pos, 
                        optimization["size_multiplier"],
                        merged_text=merged_text,
                        entrance_effect=optimization.get("entrance_effect"),
                        disappearance_effect=optimization.get("disappearance_effect")
                    )
                    
                    if phrase_img is not None:
                        # Merged phrases always go behind (that's why we merged them)
                        phrases_to_render.append((phrase_img, True, f"[MERGED]", "merged"))
            else:
                # Normal phrase rendering
                size_mult = optimization.get("size_multiplier", 1.0)
                
                # Use optimal Y if available (for single behind-face phrases)
                if optimization.get("goes_behind", False) and "optimal_y" in optimization:
                    y_pos = optimization["optimal_y"] + bottom_shift_amount if bottom_needs_shift else optimization["optimal_y"]
                
                phrase_img = phrase_renderer.render_phrase(
                    phrase, current_time, (height, width), scene_end, y_pos, size_mult,
                    entrance_effect=optimization.get("entrance_effect"),
                    disappearance_effect=optimization.get("disappearance_effect")
                )
                
                if phrase_img is not None:
                    # Use decision about behind/front - runtime occlusion overrides pre-computed
                    should_be_behind = optimization.get("runtime_goes_behind", optimization.get("goes_behind", False))
                    reason = "runtime-occluded" if optimization.get("runtime_goes_behind", False) else "pre-computed"
                    phrases_to_render.append((phrase_img, should_be_behind, phrase["text"][:20], reason))
        
        # Process phrases in two passes: behind first, then in front
        
        # Pass 1: Render phrases that go behind
        for phrase_img, should_be_behind, phrase_text, reason in phrases_to_render:
            if should_be_behind:
                alpha = phrase_img[:, :, 3:4] / 255.0
                composite = composite * (1 - alpha) + phrase_img[:, :, :3] * alpha
                if reason == "head":
                    behind_head += 1
                else:
                    behind_visibility += 1
        
        # Apply foreground (person) on top of background+behind-phrases
        person_mask_3ch = np.stack([person_mask, person_mask, person_mask], axis=2)
        composite = np.where(person_mask_3ch == 1, frame_original, composite)
        
        # Pass 2: Render phrases that go in front
        for phrase_img, should_be_behind, phrase_text, reason in phrases_to_render:
            if not should_be_behind:
                alpha = phrase_img[:, :, 3:4] / 255.0
                composite = composite * (1 - alpha) + phrase_img[:, :, :3] * alpha
                front_count += 1
        
        composite = composite.astype(np.uint8)
        out.write(composite)
        
        # More frequent logging around scene transitions
        should_log = frame_idx % 30 == 0
        
        # Log around scene 1 end (2.92s = frame 73)
        if 70 <= frame_idx <= 85:
            should_log = True
        
        if should_log:
            time_str = f"t={frame_idx/fps:.2f}s"
            print(f"Processed {frame_idx}/{total_frames} frames ({time_str})")
            if len(phrases_to_render) > 0:
                for _, should_be_behind, phrase_text, reason in phrases_to_render:
                    position = f"behind ({reason})" if should_be_behind else "front"
                    print(f"  '{phrase_text}': {position}")
            
            # Special logging around scene 1 end (2.92s)
            if 2.90 <= frame_idx/fps <= 2.94:
                print(f"  ⚠️ Scene 1 END at {time_str} - disappearance should START")
            elif 3.40 <= frame_idx/fps <= 3.44:
                print(f"  ⚠️ Scene 1 disappearance should be COMPLETE by {time_str}")
    
    # Clean up
    cap_orig.release()
    cap_mask.release()
    if cap_head:
        cap_head.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nCompositing statistics:")
    print(f"  Phrases placed behind (head overlap): {behind_head}")
    print(f"  Phrases placed behind (visibility): {behind_visibility}")
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
    # Always save output to outputs folder
    output_video = "../../outputs/ai_math1_sam2_head_aware.mp4"
    
    print("\n" + "="*60)
    print("SAM2 HEAD-AWARE SANDWICH COMPOSITING")
    print("="*60)
    print("\nFeatures:")
    print("  • Uses SAM2 video tracking for consistent head detection")
    print("  • Caches head mask to avoid recomputation")
    print("  • Text ALWAYS goes behind if it touches head (even 1 pixel)")
    print("  • Otherwise uses visibility threshold for body")
    print("  • Per-phrase independent decisions")
    print("  • Vertical stacking for same-position phrases")
    print("  • Auto-merges 2+ behind-face phrases into single line with scaled font")
    
    final_video = apply_sam2_head_aware_sandwich(
        input_video,
        mask_video,
        transcript_path,
        output_video,
        visibility_threshold=0.9  # For non-head areas
    )
    
    print(f"\n✅ SAM2 head-aware video created: {final_video}")
    print("\nKey rules:")
    print("  • Head overlap → ALWAYS behind (no exceptions)")
    print("  • Body overlap → behind if >90% visible")
    print("  • Otherwise → text stays in front")


if __name__ == "__main__":
    main()