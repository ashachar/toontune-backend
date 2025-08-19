#!/usr/bin/env python3
"""
No-Flicker Karaoke Caption Generator
- Lines stay visible for their entire duration
- Last word remains highlighted during pauses
- Smooth, continuous display
"""

import json
from pathlib import Path
import subprocess
from typing import List, Dict, Tuple, Optional

class NoFlickerKaraokeGenerator:
    def __init__(self):
        """Initialize with optimal settings."""
        self.font_size = 56
        self.font_bold = True
        self.outline_width = 4
        self.margin_bottom = 80
        
        # Colors in ASS BGR format
        self.normal_color = "&HFFFFFF"     # White
        self.highlight_color = "&H00FFFF"  # Yellow
        self.outline_color = "&H000000"    # Black
    
    def group_words_into_lines(self, words: List[Dict], max_words: int = 6) -> List[List[Dict]]:
        """Group words into lines with smart breaking."""
        lines = []
        current_line = []
        
        for i, word in enumerate(words):
            current_line.append(word)
            
            should_break = False
            
            # Break at max words
            if len(current_line) >= max_words:
                should_break = True
            
            # Break at sentence endings
            elif word['word'].rstrip()[-1:] in '.!?,;:':
                should_break = True
            
            # Break before conjunctions
            elif (i + 1 < len(words) and len(current_line) >= 3):
                next_word = words[i + 1]['word'].lower()
                if next_word in ['when', 'and', 'but', 'or', 'if', 'with', 'to']:
                    should_break = True
            
            # Keep musical notes together
            elif word['word'].lower() in ['do', 're', 'mi', 'fa', 'sol', 'la', 'ti']:
                if i + 1 < len(words):
                    next_word = words[i + 1]['word'].lower()
                    if next_word not in ['do', 're', 'mi', 'fa', 'sol', 'la', 'ti']:
                        should_break = True
            
            if should_break:
                lines.append(current_line)
                current_line = []
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def create_line_timeline(self, line_words: List[Dict]) -> List[Tuple[float, float, Optional[int]]]:
        """
        Create timeline for a line showing which word is highlighted when.
        Returns list of (start_time, end_time, word_index) tuples.
        word_index is None for pauses between words.
        """
        timeline = []
        
        for i, word in enumerate(line_words):
            # Add this word's active period
            timeline.append((word['start'], word['end'], i))
            
            # If there's a gap before the next word, keep current word highlighted
            if i < len(line_words) - 1:
                next_word = line_words[i + 1]
                if word['end'] < next_word['start']:
                    # Gap between words - keep current word highlighted
                    timeline.append((word['end'], next_word['start'], i))
        
        return timeline
    
    def generate_ass_file(self, lines: List[List[Dict]], output_path: str, scene_start: float = 0.0):
        """Generate ASS subtitle file without flickering."""
        
        # ASS header
        header = f"""[Script Info]
Title: No-Flicker Karaoke Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601
PlayResX: 1168
PlayResY: 526

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{self.font_size},{self.normal_color},&H000000FF,{self.outline_color},&H80000000,{1 if self.font_bold else 0},0,0,0,100,100,0,0,1,{self.outline_width},0,2,10,10,{self.margin_bottom},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        events = []
        
        for line_idx, line_words in enumerate(lines):
            if not line_words:
                continue
            
            # Get line display period (from first word to last word)
            line_start = line_words[0]['start'] - scene_start
            line_end = line_words[-1]['end'] - scene_start
            
            # Skip if before scene
            if line_end < 0:
                continue
            
            # Create timeline for this line
            timeline = self.create_line_timeline(line_words)
            
            # Generate events for each timeline segment
            for segment_start, segment_end, highlighted_word_idx in timeline:
                seg_start = segment_start - scene_start
                seg_end = segment_end - scene_start
                
                if seg_start < 0:
                    seg_start = 0
                
                # Build line text with appropriate highlighting
                text_parts = []
                for word_idx, word in enumerate(line_words):
                    word_text = word['word'].upper()
                    
                    if highlighted_word_idx is not None and word_idx == highlighted_word_idx:
                        # This word is highlighted
                        text_parts.append(f"{{\\c{self.highlight_color}}}{word_text}{{\\c{self.normal_color}}}")
                    elif highlighted_word_idx is not None and word_idx < highlighted_word_idx:
                        # This word was already said (could optionally make it dimmer)
                        text_parts.append(word_text)
                    else:
                        # This word hasn't been said yet
                        text_parts.append(word_text)
                
                text = " ".join(text_parts)
                
                # Format times
                start_str = self.format_time(seg_start)
                end_str = self.format_time(seg_end)
                
                event = f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}"
                events.append(event)
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header + "\n".join(events))
    
    def format_time(self, seconds: float) -> str:
        """Format time for ASS: H:MM:SS.CC"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
    
    def generate_video(self, input_video: str, output_video: str, words: List[Dict], scene_start: float = 0.0):
        """Generate video with no-flicker karaoke captions."""
        
        print("\n🎤 NO-FLICKER KARAOKE GENERATOR")
        print("=" * 50)
        
        # Group words
        lines = self.group_words_into_lines(words)
        print(f"📝 Processing {len(words)} words into {len(lines)} lines")
        
        # Show preview
        print("\n📋 Line Preview (first 5):")
        for i, line in enumerate(lines[:5], 1):
            text = " ".join([w['word'].upper() for w in line])
            duration = line[-1]['end'] - line[0]['start']
            print(f"   {i}. {text} ({duration:.1f}s)")
        
        # Check for gaps in first line
        if lines:
            first_line = lines[0]
            gaps = []
            for i in range(len(first_line) - 1):
                gap = first_line[i + 1]['start'] - first_line[i]['end']
                if gap > 0.01:
                    gaps.append((first_line[i]['word'], gap))
            
            if gaps:
                print("\n⏸️  Pause handling in first line:")
                for word, gap in gaps[:3]:
                    print(f"   After '{word}': {gap:.2f}s pause (word stays highlighted)")
        
        # Generate ASS file
        ass_path = Path(output_video).parent / "no_flicker_karaoke.ass"
        self.generate_ass_file(lines, str(ass_path), scene_start)
        print(f"\n💾 Subtitle file created: {ass_path.name}")
        
        # Run FFmpeg
        cmd = [
            "ffmpeg",
            "-i", input_video,
            "-vf", f"ass={ass_path}",
            "-codec:a", "copy",
            "-y",
            output_video
        ]
        
        print(f"🎬 Generating video...")
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Success! Video saved to: {Path(output_video).name}")
            
            # Clean up
            ass_path.unlink(missing_ok=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e.stderr[:300]}")
            return False


def main():
    """Generate no-flicker karaoke video."""
    
    print("\n" + "🌟" * 25)
    print("  NO-FLICKER KARAOKE VIDEO")
    print("🌟" * 25)
    
    # Load data
    transcript_file = Path("uploads/assets/videos/do_re_mi/transcripts/transcript_words.json")
    with open(transcript_file) as f:
        transcript = json.load(f)
    
    scene1_words = [w for w in transcript["words"] if w["start"] <= 56.74]
    
    metadata_file = Path("uploads/assets/videos/do_re_mi/metadata/scenes.json")
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    scene1 = metadata["scenes"][0]
    
    # Generate
    generator = NoFlickerKaraokeGenerator()
    
    input_video = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    output_video = "uploads/assets/videos/do_re_mi/scenes/final/scene_001_karaoke_no_flicker.mp4"
    
    Path(output_video).parent.mkdir(exist_ok=True, parents=True)
    
    success = generator.generate_video(
        input_video=input_video,
        output_video=output_video,
        words=scene1_words,
        scene_start=scene1["start_seconds"]
    )
    
    if success:
        print("\n" + "✨" * 25)
        print("  NO-FLICKER COMPLETE!")
        print("✨" * 25)
        print("\n🎨 Improvements:")
        print("  ✓ Lines stay visible for entire duration")
        print("  ✓ No flickering between words")
        print("  ✓ Last word stays highlighted during pauses")
        print("  ✓ Smooth, continuous display")
        print(f"\n📹 Output: {output_video}")


if __name__ == "__main__":
    main()