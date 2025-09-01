#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layout optimization and head detection for ASS captions.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from utils import SubPhrase, measure_text, split_text_into_lines

def extract_mask_frame(mask_video_path: str, time_ms: int) -> Optional[np.ndarray]:
    """Extract mask frame at specific time."""
    if not os.path.exists(mask_video_path):
        return None
    
    cap = cv2.VideoCapture(mask_video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert to grayscale mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray
    return None

def detect_text_head_intersection(
    text_x: int, text_y: int, text_w: int, text_h: int,
    mask: Optional[np.ndarray]
) -> bool:
    """Check if text bounding box intersects with head (white areas in mask)."""
    if mask is None:
        return False
    
    # Define text bounding box
    y1 = max(0, text_y)
    y2 = min(mask.shape[0], text_y + text_h)
    x1 = max(0, text_x)
    x2 = min(mask.shape[1], text_x + text_w)
    
    if y2 <= y1 or x2 <= x1:
        return False
    
    # Check if there are significant white pixels (head) in text area
    text_region = mask[y1:y2, x1:x2]
    white_pixels = np.sum(text_region > 128)  # Threshold for white
    total_pixels = text_region.size
    
    # If more than 10% of text area overlaps with head
    if total_pixels > 0:
        overlap_ratio = white_pixels / total_pixels
        return overlap_ratio > 0.1
    
    return False

def optimize_layout(
    phrases: List[SubPhrase],
    current_time: float,
    W: int, H: int,
    mask: Optional[np.ndarray]
) -> Dict[int, Tuple[int, int, int, bool]]:
    """
    Optimize layout for all phrases in the scene.
    Returns dict: phrase_idx -> (x, y, final_font_size, use_mask_effect)
    """
    # Process ALL phrases in the scene, not just those visible at current_time
    # This ensures all phrases get positioned correctly
    visible_phrases = [(i, p) for i, p in enumerate(phrases)]
    
    if not visible_phrases:
        return {}
    
    # Define vertical zones (25% from top and bottom)
    top_zone_y = int(H * 0.25)  # 180 for 720p
    bottom_zone_y = int(H * 0.75)  # 540 for 720p
    
    # Group by position preference
    top_phrases = [(i, p) for i, p in visible_phrases if p.position == "top"]
    bottom_phrases = [(i, p) for i, p in visible_phrases if p.position == "bottom"]
    
    layout = {}
    base_font_size = 48
    
    # Layout top phrases
    current_y = top_zone_y
    for idx, phrase in top_phrases:
        font_size = int(base_font_size * phrase.font_size_multiplier)
        
        # Check for multi-line
        lines = split_text_into_lines(phrase.words)
        total_height = len(lines) * int(font_size * 1.3)
        
        # Center horizontally
        max_width = max(measure_text(line, font_size)[0] for line in lines)
        x = (W - max_width) // 2
        
        # Check head intersection
        intersects = detect_text_head_intersection(
            x, current_y, max_width, total_height, mask
        )
        
        # If intersects, enlarge and mark for mask effect
        if intersects:
            font_size = int(font_size * 1.5)  # Enlarge by 50%
            
        layout[idx] = (x, current_y, font_size, intersects)
        current_y += total_height + 20  # Add spacing
    
    # Layout bottom phrases
    current_y = bottom_zone_y
    for idx, phrase in bottom_phrases:
        font_size = int(base_font_size * phrase.font_size_multiplier)
        
        # Check for multi-line
        lines = split_text_into_lines(phrase.words)
        total_height = len(lines) * int(font_size * 1.3)
        
        # Center horizontally
        max_width = max(measure_text(line, font_size)[0] for line in lines)
        x = (W - max_width) // 2
        
        # Check head intersection
        intersects = detect_text_head_intersection(
            x, current_y, max_width, total_height, mask
        )
        
        # If intersects, enlarge and mark for mask effect
        if intersects:
            font_size = int(font_size * 1.5)  # Enlarge by 50%
            
        layout[idx] = (x, current_y, font_size, intersects)
        current_y += total_height + 20  # Add spacing
    
    return layout