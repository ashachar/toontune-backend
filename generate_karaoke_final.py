#!/usr/bin/env python3
"""
Final Karaoke Caption Generator
Production-ready script for generating videos with karaoke-style captions.
"""

import json
from pathlib import Path
import subprocess
from typing import List, Dict

class KaraokeCaptionFinal:
    def __init__(self):
        """Initialize caption generator with optimal settings."""
        # Font settings matching the reference image
        self.font_size = 56
        self.font_bold = True
        self.outline_width = 4
        
        # Colors
        self.normal_color = "&HFFFFFF"     # White
        self.highlight_color = "&H00FFFF"  # Yellow (in BGR format for ASS)
        self.outline_color = "&H000000"    # Black
        
        # Positioning
        self.margin_bottom = 80
        
    def group_words_intelligently(self, words: List[Dict], max_words: int = 6) -> List[List[Dict]]:
        """
        Group words into lines with intelligent breaking.
        Optimized for song lyrics and natural speech patterns.
        """
        lines = []
        current_line = []
        
        for i, word in enumerate(words):
            current_line.append(word)
            
            # Check for line break conditions
            should_break = False
            
            # 1. Max words reached
            if len(current_line) >= max_words:
                should_break = True
            
            # 2. Natural phrase endings
            elif word['word'].rstrip()[-1:] in '.!?,;:':
                should_break = True
            
            # 3. Before conjunctions if line has enough words
            elif (i + 1 < len(words) and len(current_line) >= 3):
                next_word = words[i + 1]['word'].lower()
                if next_word in ['when', 'and', 'but', 'or', 'if', 'with']:
                    should_break = True
            
            # 4. Keep musical notes together (Do Re Mi)
            elif word['word'].lower() in ['do', 're', 'mi', 'fa', 'sol', 'la', 'ti']:
                # Try to keep musical phrases together
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
    
    def generate_ass_file(self, lines: List[List[Dict]], output_path: str, scene_start: float = 0.0):
        """Generate ASS subtitle file with karaoke effect."""
        
        # ASS header with custom styles
        header = f"""[Script Info]
Title: Karaoke Subtitles
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
            
            # Generate events for each word timing in the line
            for word_idx, current_word in enumerate(line_words):
                word_start = current_word['start'] - scene_start
                word_end = current_word['end'] - scene_start
                
                # Skip if before scene start
                if word_end < 0:
                    continue
                
                # Build the line text with highlighting
                text_parts = []
                for w in line_words:
                    w_text = w['word'].upper()
                    
                    if w == current_word:
                        # Highlight current word in yellow
                        text_parts.append(f"{{\\c{self.highlight_color}}}{w_text}{{\\c{self.normal_color}}}")
                    else:
                        text_parts.append(w_text)
                
                text = " ".join(text_parts)
                
                # Format times (H:MM:SS.CC)
                start_str = self.format_time(word_start)
                end_str = self.format_time(word_end)
                
                event = f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}"
                events.append(event)
        
        # Write ASS file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header + "\n".join(events))
    
    def format_time(self, seconds: float) -> str:
        """Format time for ASS format: H:MM:SS.CC"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
    
    def generate_video(self, input_video: str, output_video: str, words: List[Dict], scene_start: float = 0.0):
        """Generate video with karaoke captions."""
        
        print("\n🎤 KARAOKE CAPTION GENERATOR")
        print("=" * 50)
        
        # Group words into lines
        lines = self.group_words_intelligently(words)
        print(f"📝 Processing {len(words)} words into {len(lines)} lines")
        
        # Display line preview
        print("\n📋 Line Preview (first 5):")
        for i, line in enumerate(lines[:5], 1):
            text = " ".join([w['word'].upper() for w in line])
            print(f"   {i}. {text}")
        
        # Generate ASS subtitle file
        ass_path = Path(output_video).parent / "karaoke_subtitles.ass"
        self.generate_ass_file(lines, str(ass_path), scene_start)
        print(f"\n💾 Subtitle file created: {ass_path.name}")
        
        # Run FFmpeg with ASS subtitles
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
            
            # Clean up subtitle file
            ass_path.unlink(missing_ok=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error generating video")
            print(f"   FFmpeg error: {e.stderr[:300]}")
            return False


def main():
    """Main function to generate karaoke video for scene 1."""
    
    print("\n" + "🎭" * 25)
    print("  FINAL KARAOKE VIDEO GENERATOR")
    print("🎭" * 25)
    
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
    
    # Initialize generator
    generator = KaraokeCaptionFinal()
    
    # Set paths
    input_video = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    output_video = "uploads/assets/videos/do_re_mi/scenes/final/scene_001_karaoke_final.mp4"
    
    # Ensure output directory exists
    Path(output_video).parent.mkdir(exist_ok=True, parents=True)
    
    # Generate video
    success = generator.generate_video(
        input_video=input_video,
        output_video=output_video,
        words=scene1_words,
        scene_start=scene1["start_seconds"]
    )
    
    if success:
        print("\n" + "✨" * 25)
        print("  KARAOKE VIDEO COMPLETE!")
        print("✨" * 25)
        print("\n🎨 Features:")
        print("  • Center-bottom captions")
        print("  • Max 6 words per line")
        print("  • Current word highlighted in YELLOW")
        print("  • Other words in WHITE")
        print("  • Bold text with black outline")
        print("  • Smart line breaking")
        print(f"\n📹 Output: {output_video}")


if __name__ == "__main__":
    main()