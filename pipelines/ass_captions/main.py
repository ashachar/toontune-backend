#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate sophisticated ASS animations from enriched transcripts.
Word-by-word animation with slide-from-above, scene-based timing,
dynamic positioning, importance-based styling, and head detection.
"""

import json
import subprocess
from typing import Optional
from collections import defaultdict
from utils import SubPhrase, get_ass_header, split_text_into_lines
from layout import extract_mask_frame, optimize_layout
from word_animator import animate_words_in_phrase

def create_ass_file(
    transcript_path: str,
    video_path: str,
    mask_video_path: Optional[str] = None,
    output_ass_path: str = "output_captions.ass"
):
    """Create ASS file with word-by-word animation and advanced features."""
    
    # Load enriched transcript
    with open(transcript_path, 'r') as f:
        data = json.load(f)
    
    phrases = [
        SubPhrase(
            text=p["text"],
            words=p["words"],
            start_time=p["start_time"],
            end_time=p["end_time"],
            importance=p["importance"],
            emphasis_type=p["emphasis_type"],
            font_size_multiplier=p["visual_style"]["font_size_multiplier"],
            bold=p["visual_style"]["bold"],
            color_tint=p["visual_style"]["color_tint"],
            position=p["position"],
            appearance_index=p["appearance_index"],
            opacity_boost=p["visual_style"]["opacity_boost"]
        )
        for p in data["phrases"]
    ]
    
    # Group phrases by appearance_index (scenes)
    scenes = defaultdict(list)
    for i, phrase in enumerate(phrases):
        scenes[phrase.appearance_index].append((i, phrase))
    
    # Video parameters
    W, H = 1280, 720
    
    # Get ASS header
    header = get_ass_header(W, H)
    events = []
    
    # Process each scene
    for scene_idx, scene_phrases in scenes.items():
        # Find scene timing (all words disappear together at scene end)
        scene_start = min(p.start_time for _, p in scene_phrases)
        scene_end = max(p.end_time for _, p in scene_phrases)
        
        # Get mask for mid-point of scene
        mask = None
        if mask_video_path:
            mid_time = (scene_start + scene_end) / 2
            mask = extract_mask_frame(mask_video_path, int(mid_time * 1000))
        
        # Optimize layout for this scene
        # Create a list of just phrases for layout optimization
        scene_phrase_list = [p for _, p in scene_phrases]
        layout = optimize_layout(
            scene_phrase_list,
            (scene_start + scene_end) / 2,
            W, H, mask
        )
        
        # Process each phrase word-by-word
        for scene_local_idx, (phrase_idx, phrase) in enumerate(scene_phrases):
            if scene_local_idx not in layout:
                continue
                
            x_base, y_base, font_size, use_mask = layout[scene_local_idx]
            
            # Choose style based on emphasis
            style_map = {
                "minor": "Default",
                "normal": "Default", 
                "important": "Important",
                "critical": "Critical",
                "mega_title": "MegaTitle"
            }
            style = style_map.get(phrase.emphasis_type, "Default")
            
            # Use masked style if needed
            if use_mask:
                style = "Masked"
            
            # Check if we need to split into multiple lines
            lines = split_text_into_lines(phrase.words)
            
            # Animate words in this phrase
            phrase_events = animate_words_in_phrase(
                phrase, x_base, y_base, font_size, use_mask,
                scene_end, W, style, lines
            )
            events.extend(phrase_events)
    
    # Write ASS file
    ass_content = header + "\n".join(events) + "\n"
    
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
    
    print(f"Created ASS file: {output_ass_path}")
    return output_ass_path

def burn_subtitles(input_video: str, ass_file: str, output_video: str):
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
    # Paths
    transcript_path = "../../uploads/assets/videos/ai_math1/transcript_enriched_partial.json"
    video_path = "ai_math1_6sec.mp4"
    mask_video_path = "../../uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4"
    ass_file = "ai_math1_wordbyword_captions.ass"
    output_video = "ai_math1_wordbyword_with_captions.mp4"
    
    # Create ASS file
    create_ass_file(transcript_path, video_path, mask_video_path, ass_file)
    
    # Burn subtitles into video
    burn_subtitles(video_path, ass_file, output_video)