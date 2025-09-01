#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word-by-word animation logic for ASS captions.
"""

from utils import ass_time, rgb_to_ass_color, measure_text

def animate_words_in_phrase(
    phrase, x_base, y_base, font_size, use_mask, 
    scene_end, W, style, lines
):
    """Create word-by-word animations for a phrase."""
    events = []
    
    # Apply color tint if specified
    color_override = ""
    if phrase.color_tint:
        r, g, b = phrase.color_tint
        color_override = f"\\c{rgb_to_ass_color(r, g, b)}"
    
    # Apply opacity boost - DISABLED to ensure 100% opacity
    alpha_override = ""
    # if phrase.opacity_boost != 0:
    #     alpha = max(0, min(255, int(128 - phrase.opacity_boost * 128)))
    #     alpha_override = f"\\alpha&H{alpha:02X}&"
    
    # Calculate timing for each word
    total_duration = phrase.end_time - phrase.start_time
    time_per_word = total_duration / len(phrase.words)
    
    # Animation settings
    slide_distance = 40
    
    # Process each line
    word_index = 0
    for line_idx, line_text in enumerate(lines):
        line_words = line_text.split()
        line_y = y_base + line_idx * int(font_size * 1.3)
        
        # Calculate x position for each word in line
        line_width = measure_text(line_text, font_size)[0]
        start_x = (W - line_width) // 2
        current_x = start_x
        
        # Animate each word individually
        for word in line_words:
            # Calculate word timing
            word_start = phrase.start_time + (word_index * time_per_word)
            word_appear_duration = min(300, int(time_per_word * 500))
            
            # Word width
            word_width = measure_text(word, font_size)[0]
            
            # Build effects string
            effects = []
            effects.append(f"\\an7")  # Left-top alignment
            effects.append(f"\\fs{font_size}")
            effects.append(f"\\bord3")  # Add black border explicitly
            if phrase.bold:
                effects.append("\\b1")
            if color_override:
                effects.append(color_override)
            if alpha_override:
                effects.append(alpha_override)
            
            # Fade in effect
            effects.append(f"\\fad({word_appear_duration},0)")
            
            # Slide from above effect
            effects.append(f"\\move({current_x},{line_y - slide_distance},{current_x},{line_y},0,{word_appear_duration})")
            
            # Add mask effect if text is behind head
            # Removed transparency - text should be 100% opaque even behind head
            # if use_mask:
            #     effects.append("\\alpha&H80&")
            
            # Create dialogue entry for this word
            # Word appears at its time but disappears at scene end
            dialogue = (
                f"Dialogue: {2 if use_mask else 1},"
                f"{ass_time(word_start)},{ass_time(scene_end)},"
                f"{style},,0,0,0,,"
                f"{{{' '.join(effects)}}}{word}"
            )
            
            events.append(dialogue)
            
            # Move to next word position
            current_x += word_width + int(font_size * 0.3)  # Add spacing
            word_index += 1
    
    return events