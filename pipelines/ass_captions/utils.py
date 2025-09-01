#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for ASS caption generation.
"""

from PIL import ImageFont
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SubPhrase:
    """Represents a phrase with all its properties."""
    text: str
    words: List[str]
    start_time: float
    end_time: float
    importance: float
    emphasis_type: str
    font_size_multiplier: float
    bold: bool
    color_tint: Optional[List[int]]
    position: str  # top/bottom
    appearance_index: int
    opacity_boost: float

def ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def rgb_to_ass_color(r: int, g: int, b: int) -> str:
    """Convert RGB to ASS color format (&HBBGGRR&)."""
    return f"&H{b:02X}{g:02X}{r:02X}&"

def measure_text(text: str, font_size: int) -> Tuple[int, int]:
    """Estimate text dimensions."""
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        bbox = font.getbbox(text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width, height
    except:
        # Fallback estimation
        width = len(text) * int(font_size * 0.6)
        height = int(font_size * 1.2)
        return width, height

def split_text_into_lines(words: List[str], max_words: int = 6) -> List[str]:
    """Split words into lines if > max_words."""
    if len(words) <= max_words:
        return [" ".join(words)]
    
    # Split roughly in half
    mid = len(words) // 2
    line1 = " ".join(words[:mid])
    line2 = " ".join(words[mid:])
    return [line1, line2]

def get_ass_header(W: int, H: int) -> str:
    """Generate ASS header with multiple styles for different importance levels."""
    return f"""[Script Info]
Title: Advanced Captions
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,2,5,30,30,30,1
Style: Important,Arial,55,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,5,30,30,30,1
Style: Critical,Arial,62,&H009999FF,&H009999FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,3,5,30,30,30,1
Style: MegaTitle,Arial,86,&H0000D7FF,&H0000D7FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,4,5,30,30,30,1
Style: Masked,Arial,72,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,5,30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""