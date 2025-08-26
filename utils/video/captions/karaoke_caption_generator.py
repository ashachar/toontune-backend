#!/usr/bin/env python3
"""
Karaoke-style Caption Generator
Creates captions with word-by-word highlighting, center-bottom positioned.
Max 6 words per line, highlights current word in yellow.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
import re

class KaraokeCaptionGenerator:
    def __init__(self, video_width: int = 1168, video_height: int = 526):
        """Initialize with video dimensions for positioning."""
        self.video_width = video_width
        self.video_height = video_height
        
        # Caption positioning (center-bottom)
        self.caption_y = video_height - 100  # 100px from bottom
        self.line_height = 60  # Space between lines
        self.word_spacing = 15  # Space between words
        
        # Font settings
        self.font_size = 48
        
        # Colors
        self.default_color = "white"
        self.highlight_color = "yellow" 
        self.outline_color = "black"
        self.outline_width = 3
    
    def group_words_into_lines(self, words: List[Dict], max_words_per_line: int = 6) -> List[List[Dict]]:
        """
        Group words into lines with max words per line.
        Try to break at natural phrase boundaries.
        """
        lines = []
        current_line = []
        
        # Natural break points (punctuation or conjunctions)
        break_after = {'.', ',', '!', '?', ';', ':'}
        prefer_break_before = {'and', 'but', 'or', 'when', 'while', 'if', 'because', 'since', 'although'}
        
        for i, word in enumerate(words):
            current_line.append(word)
            word_text = word['word'].lower()
            
            # Check if we should break the line
            should_break = False
            
            # Break if we hit max words
            if len(current_line) >= max_words_per_line:
                should_break = True
            
            # Break after punctuation
            elif any(p in word['word'] for p in break_after):
                should_break = True
            
            # Consider breaking before conjunctions if line has 3+ words
            elif (i + 1 < len(words) and len(current_line) >= 3 and 
                  words[i + 1]['word'].lower() in prefer_break_before):
                should_break = True
            
            if should_break and current_line:
                lines.append(current_line)
                current_line = []
        
        # Add remaining words
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def calculate_line_width(self, words: List[Dict]) -> float:
        """Estimate the width of a line of text."""
        # Approximate character width (adjust based on font)
        char_width = self.font_size * 0.6
        
        total_width = 0
        for i, word in enumerate(words):
            # Convert to uppercase for display
            word_text = word['word'].upper()
            total_width += len(word_text) * char_width
            if i < len(words) - 1:
                total_width += self.word_spacing
        
        return total_width
    
    def calculate_word_positions(self, line_words: List[Dict]) -> List[Tuple[float, float]]:
        """Calculate x positions for each word in a centered line."""
        positions = []
        
        # Calculate total line width
        line_width = self.calculate_line_width(line_words)
        
        # Start position (centered)
        start_x = (self.video_width - line_width) / 2
        
        current_x = start_x
        char_width = self.font_size * 0.6
        
        for i, word in enumerate(line_words):
            positions.append((current_x, self.caption_y))
            
            # Move x position for next word
            word_text = word['word'].upper()
            current_x += len(word_text) * char_width + self.word_spacing
        
        return positions
    
    def generate_ffmpeg_filters(self, lines: List[List[Dict]], scene_start: float = 0.0) -> str:
        """
        Generate FFmpeg filter complex for karaoke-style captions.
        Creates two drawtext filters per word (normal and highlighted) with proper timing.
        """
        filters = []
        
        for line_idx, line_words in enumerate(lines):
            if not line_words:
                continue
            
            # Calculate when this line should be visible
            line_start = line_words[0]['start'] - scene_start
            line_end = line_words[-1]['end'] - scene_start
            
            # Skip if line is outside scene
            if line_end < 0:
                continue
            
            # Calculate word positions for this line
            word_positions = self.calculate_word_positions(line_words)
            
            # Create drawtext filters for each word
            for word_idx, (word, (x_pos, y_pos)) in enumerate(zip(line_words, word_positions)):
                word_text = word['word'].upper()
                word_start = word['start'] - scene_start
                word_end = word['end'] - scene_start
                
                # Escape special characters for FFmpeg
                word_text = word_text.replace("'", "'\\''")
                word_text = word_text.replace(":", "\\:")
                word_text = word_text.replace(",", "\\,")
                word_text = word_text.replace("[", "\\[")
                word_text = word_text.replace("]", "\\]")
                
                # Create TWO filters per word:
                
                # 1. White version (shown before and after highlight)
                white_filter = (
                    f"drawtext="
                    f"text='{word_text}':"
                    f"x={x_pos:.0f}:"
                    f"y={y_pos:.0f}:"
                    f"fontsize={self.font_size}:"
                    f"fontcolor={self.default_color}:"
                    f"bordercolor={self.outline_color}:"
                    f"borderw={self.outline_width}:"
                    # Show when line is active BUT word is not being spoken
                    f"enable='between(t,{line_start:.3f},{line_end:.3f})*"
                    f"not(between(t,{word_start:.3f},{word_end:.3f}))'"
                )
                filters.append(white_filter)
                
                # 2. Yellow version (shown during word timing)
                yellow_filter = (
                    f"drawtext="
                    f"text='{word_text}':"
                    f"x={x_pos:.0f}:"
                    f"y={y_pos:.0f}:"
                    f"fontsize={self.font_size}:"
                    f"fontcolor={self.highlight_color}:"
                    f"bordercolor={self.outline_color}:"
                    f"borderw={self.outline_width}:"
                    # Show only when word is being spoken
                    f"enable='between(t,{word_start:.3f},{word_end:.3f})'"
                )
                filters.append(yellow_filter)
        
        return ",".join(filters)
    
    def generate_video_with_karaoke_captions(self, 
                                            input_video: str,
                                            output_video: str,
                                            words: List[Dict],
                                            scene_start: float = 0.0):
        """Generate video with karaoke-style captions."""
        
        print("Generating karaoke-style captions...")
        
        # Group words into lines
        lines = self.group_words_into_lines(words)
        print(f"Grouped {len(words)} words into {len(lines)} lines")
        
        # Show line composition
        for i, line in enumerate(lines[:5]):  # Show first 5 lines
            line_text = " ".join([w['word'].upper() for w in line])
            print(f"  Line {i+1}: {line_text}")
        
        # Generate FFmpeg filters
        filter_complex = self.generate_ffmpeg_filters(lines, scene_start)
        
        # FFmpeg command
        cmd = [
            "ffmpeg",
            "-i", input_video,
            "-vf", filter_complex,
            "-codec:a", "copy",
            "-y",
            output_video
        ]
        
        print(f"\nGenerating video: {output_video}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Video generated successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ FFmpeg error: {e.stderr[:500]}")
            return False


def main():
    """Test the karaoke caption generator."""
    
    # Load word timings
    transcript_file = Path("uploads/assets/videos/do_re_mi/transcripts/transcript_words.json")
    with open(transcript_file) as f:
        transcript = json.load(f)
    
    # Get words for scene 1
    scene1_words = [w for w in transcript["words"] if w["start"] <= 56.74]
    
    # Load scene metadata
    metadata_file = Path("uploads/assets/videos/do_re_mi/metadata/scenes.json")
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    scene1 = metadata["scenes"][0]
    
    # Generate video with karaoke captions
    generator = KaraokeCaptionGenerator()
    
    input_video = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    output_video = "uploads/assets/videos/do_re_mi/scenes/final/scene_001_karaoke_test.mp4"
    
    # Ensure output dir exists
    Path(output_video).parent.mkdir(exist_ok=True, parents=True)
    
    generator.generate_video_with_karaoke_captions(
        input_video=input_video,
        output_video=output_video,
        words=scene1_words,
        scene_start=scene1["start_seconds"]
    )
    
    print("\n📝 Caption Style:")
    print("  - Position: Center-bottom")
    print("  - Max words per line: 6")
    print("  - Current word: Yellow")
    print("  - Other words: White")
    print("  - Text: ALL CAPS with black outline")


if __name__ == "__main__":
    main()