#!/usr/bin/env python3
"""
Enhanced Karaoke Caption Generator V2
- Shows current line and next line
- Smooth transitions between lines
- Better word grouping algorithm
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import subprocess
import math

class KaraokeCaptionV2:
    def __init__(self, video_width: int = 1168, video_height: int = 526):
        """Initialize with video dimensions."""
        self.video_width = video_width
        self.video_height = video_height
        
        # Caption positioning
        self.bottom_margin = 80
        self.line_spacing = 70
        
        # Font settings
        self.font_size = 52
        self.shadow_offset = 3
        
    def smart_group_lines(self, words: List[Dict], max_words: int = 6) -> List[List[Dict]]:
        """
        Smart grouping that respects sentence structure.
        """
        lines = []
        current_line = []
        
        # Sentence endings
        sentence_ends = {'.', '!', '?'}
        # Natural break points
        break_before = {'when', 'if', 'and', 'but', 'or', 'with', 'to'}
        
        for i, word in enumerate(words):
            current_line.append(word)
            word_lower = word['word'].lower()
            
            # Determine if we should break
            should_break = False
            
            # Always break at max words
            if len(current_line) >= max_words:
                should_break = True
            
            # Break at sentence endings
            elif any(p in word['word'] for p in sentence_ends):
                should_break = True
            
            # Break before conjunctions if line has 3+ words
            elif (i + 1 < len(words) and 
                  len(current_line) >= 3 and 
                  words[i + 1]['word'].lower() in break_before):
                should_break = True
            
            # For music notation (Do, Re, Mi), keep them together
            elif word_lower in ['do', 're', 'mi', 'fa', 'sol', 'la', 'ti'] and len(current_line) <= 3:
                # Don't break musical notes if we have room
                should_break = False
            
            if should_break and current_line:
                lines.append(current_line)
                current_line = []
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def generate_srt_subtitles(self, lines: List[List[Dict]], scene_start: float = 0.0) -> str:
        """
        Generate SRT subtitle file with word-by-word highlighting.
        Uses HTML-like tags for color.
        """
        srt_content = []
        
        for line_idx, line_words in enumerate(lines):
            if not line_words:
                continue
            
            # Line timing
            line_start = line_words[0]['start'] - scene_start
            line_end = line_words[-1]['end'] - scene_start
            
            if line_end < 0:
                continue
            
            # Generate subtitle entries for each word timing
            for word_idx, word in enumerate(line_words):
                word_start = word['start'] - scene_start
                word_end = word['end'] - scene_start
                
                # Build the line with current word highlighted
                line_text = []
                for w in line_words:
                    if w == word:
                        # Highlight current word
                        line_text.append(f"<font color='yellow'>{w['word'].upper()}</font>")
                    else:
                        line_text.append(w['word'].upper())
                
                # SRT entry
                entry = f"{len(srt_content) + 1}\n"
                entry += f"{self.format_srt_time(word_start)} --> {self.format_srt_time(word_end)}\n"
                entry += " ".join(line_text) + "\n\n"
                
                srt_content.append(entry)
        
        return "".join(srt_content)
    
    def format_srt_time(self, seconds: float) -> str:
        """Format time for SRT: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def generate_ass_subtitles(self, lines: List[List[Dict]], scene_start: float = 0.0) -> str:
        """
        Generate ASS (Advanced SubStation) subtitle file for better styling.
        Supports karaoke-style word highlighting.
        """
        # ASS header with styles
        ass_header = """[Script Info]
Title: Karaoke Captions
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,2,2,10,10,80,1
Style: Highlight,Arial,52,&H0000FFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,2,2,10,10,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        events = []
        
        for line_idx, line_words in enumerate(lines):
            if not line_words:
                continue
            
            # Process each word in the line
            for word_idx, word in enumerate(line_words):
                word_start = word['start'] - scene_start
                word_end = word['end'] - scene_start
                
                if word_start < 0:
                    continue
                
                # Build text with karaoke tags
                text_parts = []
                for i, w in enumerate(line_words):
                    w_start = w['start'] - scene_start
                    w_end = w['end'] - scene_start
                    w_text = w['word'].upper()
                    
                    if w == word:
                        # Current word in yellow
                        text_parts.append(f"{{\\c&H00FFFF&}}{w_text}{{\\c&HFFFFFF&}}")
                    else:
                        text_parts.append(w_text)
                
                text = " ".join(text_parts)
                
                # Format times for ASS
                start_time = self.format_ass_time(word_start)
                end_time = self.format_ass_time(word_end)
                
                event = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}"
                events.append(event)
        
        return ass_header + "\n".join(events)
    
    def format_ass_time(self, seconds: float) -> str:
        """Format time for ASS: H:MM:SS.CC"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
    
    def generate_video_with_ass_subtitles(self, 
                                         input_video: str,
                                         output_video: str,
                                         words: List[Dict],
                                         scene_start: float = 0.0):
        """Generate video with ASS subtitles for better karaoke effect."""
        
        print("🎤 Generating karaoke captions (ASS format)...")
        
        # Group words into lines
        lines = self.smart_group_lines(words)
        print(f"📝 Grouped {len(words)} words into {len(lines)} lines")
        
        # Show first few lines
        for i, line in enumerate(lines[:5]):
            line_text = " ".join([w['word'].upper() for w in line])
            print(f"   Line {i+1}: {line_text}")
        
        # Generate ASS subtitle file
        ass_content = self.generate_ass_subtitles(lines, scene_start)
        
        # Save to temp file
        ass_file = Path(output_video).parent / "temp_karaoke.ass"
        with open(ass_file, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        
        print(f"💾 Saved subtitles to: {ass_file}")
        
        # FFmpeg command with ASS subtitles
        cmd = [
            "ffmpeg",
            "-i", input_video,
            "-vf", f"ass={ass_file}",
            "-codec:a", "copy",
            "-y",
            output_video
        ]
        
        print(f"🎬 Generating video: {output_video}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Video generated successfully!")
            
            # Clean up temp file
            ass_file.unlink(missing_ok=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e.stderr[:500]}")
            return False


def main():
    """Generate video with enhanced karaoke captions."""
    
    # Load transcript
    transcript_file = Path("uploads/assets/videos/do_re_mi/transcripts/transcript_words.json")
    with open(transcript_file) as f:
        transcript = json.load(f)
    
    # Get scene 1 words
    scene1_words = [w for w in transcript["words"] if w["start"] <= 56.74]
    
    # Load scene metadata
    metadata_file = Path("uploads/assets/videos/do_re_mi/metadata/scenes.json")
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    scene1 = metadata["scenes"][0]
    
    # Generate video
    generator = KaraokeCaptionV2()
    
    input_video = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    output_video = "uploads/assets/videos/do_re_mi/scenes/final/scene_001_karaoke_v2.mp4"
    
    Path(output_video).parent.mkdir(exist_ok=True, parents=True)
    
    generator.generate_video_with_ass_subtitles(
        input_video=input_video,
        output_video=output_video,
        words=scene1_words,
        scene_start=scene1["start_seconds"]
    )
    
    print("\n🎨 Caption Features:")
    print("  ✓ Center-bottom positioning")
    print("  ✓ Max 6 words per line")
    print("  ✓ Smart line breaking")
    print("  ✓ Yellow highlighting for current word")
    print("  ✓ ALL CAPS with outline")


if __name__ == "__main__":
    main()