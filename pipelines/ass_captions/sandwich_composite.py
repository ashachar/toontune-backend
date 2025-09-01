#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sandwich compositing for text-behind-head effect.
Separates foreground/background, renders text between layers.
"""

import cv2
import numpy as np
import json
from typing import List, Dict, Tuple
import subprocess
import os
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

@dataclass
class TextEvent:
    """Represents a text event to render."""
    text: str
    start_time: float
    end_time: float
    x: int
    y: int
    font_size: int
    color: Tuple[int, int, int]
    bold: bool
    animation_duration: float  # milliseconds

def render_text_on_frame(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_size: int,
    color: Tuple[int, int, int],
    bold: bool,
    opacity: float = 1.0
) -> np.ndarray:
    """Render text on frame with black border using PIL for better quality.
    Note: y is the desired TOP of the text."""
    frame_copy = frame.copy()
    
    # Convert to PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Load font
    try:
        font_name = "/System/Library/Fonts/Helvetica.ttc"
        if bold:
            # Try to use bold variant
            try:
                font_name = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            except:
                pass
        pil_font = ImageFont.truetype(font_name, font_size)
    except:
        # Fallback to default
        pil_font = ImageFont.load_default()
    
    # Draw black border (stroke)
    border_width = 3
    for dx in range(-border_width, border_width + 1):
        for dy in range(-border_width, border_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=pil_font, fill=(0, 0, 0))
    
    # Apply opacity to main text color
    if opacity < 1.0:
        r, g, b = color
        # Create semi-transparent text by blending with background
        # This is a simple approximation
        alpha = int(255 * opacity)
        draw.text((x, y), text, font=pil_font, fill=(r, g, b, alpha))
    else:
        draw.text((x, y), text, font=pil_font, fill=color)
    
    # Convert back to OpenCV format
    frame_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return frame_with_text

def composite_sandwich(
    original_frame: np.ndarray,
    text_layer: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Composite the sandwich: original frame -> text -> foreground pixels.
    
    Args:
        original_frame: Original video frame
        text_layer: Frame with text rendered on black background
        mask: Mask where black/dark=person, white/bright=background
    """
    # Ensure mask is single channel
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Render text onto the original frame
    # Only where text_layer has non-zero pixels
    text_mask = np.any(text_layer > 0, axis=2).astype(float)
    text_mask_3ch = np.stack([text_mask, text_mask, text_mask], axis=2)
    
    # Composite text onto original frame
    frame_with_text = original_frame * (1 - text_mask_3ch) + text_layer * text_mask_3ch
    frame_with_text = frame_with_text.astype(np.uint8)
    
    # Step 2: Use mask as alpha for smooth compositing
    # Normalize mask to 0-1 range (inverted: dark=1=person, bright=0=background)
    alpha = 1.0 - (mask.astype(float) / 255.0)
    
    # Apply slight adjustment to ensure clean edges
    # Push values towards 0 or 1 to reduce gray areas
    alpha = np.where(alpha > 0.5, np.minimum(alpha * 1.2, 1.0), alpha * 0.8)
    
    # Convert to 3-channel
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
    
    # Final composite with smooth alpha blending
    # Where alpha=1 (person), use original. Where alpha=0 (background), use text.
    result = original_frame * alpha_3ch + frame_with_text * (1 - alpha_3ch)
    result = result.astype(np.uint8)
    
    return result

def process_video_sandwich(
    video_path: str,
    mask_video_path: str,
    transcript_path: str,
    output_path: str
):
    """
    Process video with sandwich compositing for text-behind effect.
    """
    # Load transcript
    with open(transcript_path, 'r') as f:
        data = json.load(f)
    
    # Open videos
    cap_video = cv2.VideoCapture(video_path)
    cap_mask = cv2.VideoCapture(mask_video_path)
    
    # Get video properties
    fps = cap_video.get(cv2.CAP_PROP_FPS)
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames...")
    
    # Process each frame
    for frame_idx in range(total_frames):
        ret_video, frame = cap_video.read()
        ret_mask, mask_frame = cap_mask.read()
        
        if not ret_video or not ret_mask:
            break
        
        current_time = frame_idx / fps
        
        # Create text layer (transparent initially)
        text_layer = np.zeros_like(frame)
        
        # Find active phrases at current time
        for phrase in data["phrases"]:
            if phrase["start_time"] <= current_time <= phrase["end_time"]:
                # Calculate word-by-word timing
                words = phrase["words"]
                total_duration = phrase["end_time"] - phrase["start_time"]
                time_per_word = total_duration / len(words)
                
                # Get text properties
                font_size = int(48 * phrase["visual_style"]["font_size_multiplier"])
                color = (255, 255, 255)  # White default
                if phrase["visual_style"]["color_tint"]:
                    color = tuple(phrase["visual_style"]["color_tint"])
                
                # Calculate base position
                if phrase["position"] == "top":
                    base_y = int(height * 0.25)  # 25% from top
                else:
                    base_y = int(height * 0.75)  # 25% from bottom
                
                # Calculate total phrase width for centering using PIL
                try:
                    font_name = "/System/Library/Fonts/Helvetica.ttc"
                    if phrase["visual_style"]["bold"]:
                        try:
                            font_name = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
                        except:
                            pass
                    pil_font = ImageFont.truetype(font_name, font_size)
                except:
                    pil_font = ImageFont.load_default()
                
                # Measure each word and calculate total width
                word_widths = []
                total_width = 0
                for word in words:
                    bbox = pil_font.getbbox(word)
                    w = bbox[2] - bbox[0]
                    word_widths.append(w)
                    total_width += w
                # Add spacing between words
                spacing = int(font_size * 0.3)
                total_width += spacing * (len(words) - 1)
                
                # Starting X for centered text
                start_x = (width - total_width) // 2
                current_x = start_x
                
                # Determine which words should be visible
                for word_idx, word in enumerate(words):
                    word_start = phrase["start_time"] + (word_idx * time_per_word)
                    
                    if word_start <= current_time <= phrase["end_time"]:
                        # Calculate animation progress
                        word_progress = min(1.0, (current_time - word_start) / 0.3)  # 300ms animation
                        
                        # Calculate position with slide animation
                        slide_distance = 40
                        y_offset = int(slide_distance * (1 - word_progress))
                        y = base_y - y_offset
                        
                        # Render text with fade
                        opacity = word_progress
                        text_layer = render_text_on_frame(
                            text_layer,
                            word,
                            current_x,
                            y,
                            font_size,
                            color,
                            phrase["visual_style"]["bold"],
                            opacity
                        )
                        
                    # Move to next word position (always update even if word not visible yet)
                    if word_idx < len(word_widths):
                        current_x += word_widths[word_idx] + spacing
        
        # Composite sandwich: original frame -> text -> foreground pixels on top
        final_frame = composite_sandwich(frame, text_layer, mask_frame)
        
        # Write frame
        out.write(final_frame)
        
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    # Clean up
    cap_video.release()
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

if __name__ == "__main__":
    process_video_sandwich(
        "ai_math1_6sec.mp4",
        "../../uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4",
        "../../uploads/assets/videos/ai_math1/transcript_enriched_partial.json",
        "ai_math1_sandwich_composite.mp4"
    )