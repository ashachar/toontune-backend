#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate word-by-word ASS animations with proper two-line layout.
Words appear sequentially in their correct positions within sentences.
"""

import os
import subprocess
from PIL import ImageFont
from typing import List, Dict, Tuple

def ass_time(ms: int) -> str:
    """Convert milliseconds to ASS time format."""
    s = ms / 1000.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    cs = int(round((s - int(s)) * 100))
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"

def measure_text(text: str, font: ImageFont.FreeTypeFont) -> int:
    """Measure text width."""
    try:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]
    except:
        # Fallback estimation: ~30 pixels per character for 48pt font
        return len(text) * 30

def split_into_lines(segments: List[Dict], max_words_per_line: int = 7) -> List[List[Dict]]:
    """Split segments into lines for better readability."""
    lines = []
    current_line = []
    
    # First line: "Yes, AI created new math."
    lines.append(segments[0:5])  # First 5 words
    
    # Second line: "Would you be surprised if AI invented a new"
    lines.append(segments[5:14])  # Remaining words
    
    return lines

def calculate_line_positions(line_segments: List[Dict], font, W: int, spacing: int = 15) -> List[Tuple[int, int]]:
    """Calculate x,y positions for each word in a line."""
    positions = []
    
    # Calculate total line width
    total_width = 0
    for seg in line_segments:
        total_width += measure_text(seg["text"], font)
        if seg != line_segments[-1]:  # Add spacing except for last word
            total_width += spacing
    
    # Center the line
    start_x = (W - total_width) // 2
    current_x = start_x
    
    # Calculate position for each word
    for seg in line_segments:
        positions.append(current_x)
        word_width = measure_text(seg["text"], font)
        current_x += word_width + spacing
    
    return positions

def create_ass_file():
    """Create ASS file with proper two-line word-by-word animations."""
    
    # Video parameters
    W, H = 1280, 720
    font_size = 48
    font_name = "Arial"
    spacing = 15  # Space between words
    line_height = 70  # Vertical space between lines
    y_line1 = 550  # First line Y position
    y_line2 = y_line1 + line_height  # Second line Y position
    
    # Words and timings from transcript (first 6 seconds)
    segments = [
        {"text": "Yes,", "start_ms": 0, "end_ms": 500},
        {"text": "AI", "start_ms": 500, "end_ms": 1000},
        {"text": "created", "start_ms": 1000, "end_ms": 1500},
        {"text": "new", "start_ms": 1500, "end_ms": 2000},
        {"text": "math.", "start_ms": 2000, "end_ms": 2800},
        {"text": "Would", "start_ms": 2800, "end_ms": 3200},
        {"text": "you", "start_ms": 3200, "end_ms": 3400},
        {"text": "be", "start_ms": 3400, "end_ms": 3600},
        {"text": "surprised", "start_ms": 3600, "end_ms": 4200},
        {"text": "if", "start_ms": 4200, "end_ms": 4400},
        {"text": "AI", "start_ms": 4400, "end_ms": 4800},
        {"text": "invented", "start_ms": 4800, "end_ms": 5400},
        {"text": "a", "start_ms": 5400, "end_ms": 5500},
        {"text": "new", "start_ms": 5500, "end_ms": 5800}
    ]
    
    # ASS header
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Word,{font_name},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H7F000000,-1,0,0,0,100,100,0,0,1,3,2,7,0,0,0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    events = []
    
    # Load font for measurements
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
        except:
            # Fallback font object for estimation
            class FallbackFont:
                def getbbox(self, text):
                    width = len(text) * 30
                    return (0, 0, width, font_size)
            font = FallbackFont()
    
    # Split into lines
    lines = split_into_lines(segments)
    
    # Process each line
    for line_idx, line_segments in enumerate(lines):
        # Determine Y position for this line
        y_pos = y_line1 if line_idx == 0 else y_line2
        
        # Calculate X positions for words in this line
        x_positions = calculate_line_positions(line_segments, font, W, spacing)
        
        # Create dialogue entries for each word
        for word_idx, seg in enumerate(line_segments):
            start = seg["start_ms"]
            # Words stay visible until end of video
            end = 6000
            word = seg["text"]
            x = x_positions[word_idx]
            
            # Animation duration should match the word's duration in transcript
            word_duration = seg["end_ms"] - seg["start_ms"]
            # Cap the animation duration to be reasonable (max 300ms)
            animation_duration = min(word_duration, 300)
            
            # Animation effects
            slide_px = 40
            
            # Build the dialogue line with fade and slide from above effects
            # Use \move WITHOUT \pos to avoid conflicts - move handles positioning
            dialogue = (
                f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Word,,0,0,0,,"
                f"{{\\an7"
                f"\\fad({animation_duration},0)"
                f"\\move({x},{y_pos - slide_px},{x},{y_pos},0,{animation_duration})}}"
                f"{word}"
            )
            
            events.append(dialogue)
    
    # Write ASS file
    ass_content = header + "\n".join(events) + "\n"
    
    ass_file = "ai_math1_6sec_captions.ass"
    with open(ass_file, "w", encoding="utf-8") as f:
        f.write(ass_content)
    
    print(f"Created ASS file: {ass_file}")
    print(f"Line 1: {' '.join([s['text'] for s in lines[0]])}")
    print(f"Line 2: {' '.join([s['text'] for s in lines[1]])}")
    return ass_file

def burn_subtitles(input_video, ass_file, output_video):
    """Burn ASS subtitles into video using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", f"subtitles={ass_file}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "copy",
        output_video
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Created final video: {output_video}")

if __name__ == "__main__":
    # Create ASS file
    ass_file = create_ass_file()
    
    # Burn subtitles into video
    input_video = "ai_math1_6sec.mp4"
    output_video = "ai_math1_6sec_with_captions.mp4"
    
    burn_subtitles(input_video, ass_file, output_video)