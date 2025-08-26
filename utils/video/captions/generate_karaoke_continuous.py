#!/usr/bin/env python3
"""
Continuous Karaoke Caption Generator
- Each line displays continuously from first word start to last word end
- Words highlight progressively as they're spoken
- Last spoken word remains highlighted during pauses
- Smooth transitions between lines
"""

import json
from pathlib import Path
import subprocess
from typing import List, Dict, Tuple, Optional

class ContinuousKaraokeGenerator:
    def __init__(self):
        """Initialize settings."""
        self.font_size = 56
        self.font_bold = True
        self.outline_width = 4
        self.margin_bottom = 80
        
        # Colors in ASS BGR format
        self.normal_color = "&HFFFFFF"     # White
        self.highlight_color = "&H00FFFF"  # Yellow
        self.dim_color = "&HCCCCCC"       # Light gray for already spoken words
        self.outline_color = "&H000000"    # Black
    
    def group_words_into_lines(self, words: List[Dict], max_words: int = 6) -> List[List[Dict]]:
        """Group words into lines."""
        lines = []
        current_line = []
        
        for i, word in enumerate(words):
            current_line.append(word)
            
            should_break = False
            
            # Max words reached
            if len(current_line) >= max_words:
                should_break = True
            
            # Sentence endings
            elif word['word'].rstrip()[-1:] in '.!?,;:':
                should_break = True
            
            # Before conjunctions
            elif (i + 1 < len(words) and len(current_line) >= 3):
                next_word = words[i + 1]['word'].lower()
                if next_word in ['when', 'and', 'but', 'or', 'if', 'with', 'to']:
                    should_break = True
            
            # Musical notes
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
    
    def find_active_word_at_time(self, line_words: List[Dict], time: float) -> Tuple[Optional[int], bool]:
        """
        Find which word should be highlighted at a given time.
        Returns (word_index, is_during_word).
        If between words, returns the last spoken word.
        """
        last_spoken = None
        
        for i, word in enumerate(line_words):
            if time < word['start']:
                # Before this word starts
                return (last_spoken, False) if last_spoken is not None else (None, False)
            elif word['start'] <= time <= word['end']:
                # During this word
                return (i, True)
            else:
                # After this word
                last_spoken = i
        
        # After all words - keep last word highlighted
        return (last_spoken, False) if last_spoken is not None else (None, False)
    
    def generate_continuous_ass_file(self, lines: List[List[Dict]], output_path: str, scene_start: float = 0.0):
        """Generate ASS file with continuous display."""
        
        # ASS header
        header = f"""[Script Info]
Title: Continuous Karaoke Subtitles
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
            
            # Line displays from first word start to last word end
            line_start = line_words[0]['start'] - scene_start
            line_end = line_words[-1]['end'] - scene_start
            
            if line_end < 0:
                continue
            
            # Create change points (when highlighting needs to update)
            change_points = []
            
            # Add start of each word and end of each word as change points
            for word in line_words:
                change_points.append(word['start'] - scene_start)
                change_points.append(word['end'] - scene_start)
            
            # Sort and remove duplicates
            change_points = sorted(set(change_points))
            
            # Add line start if not already there
            if line_start not in change_points:
                change_points.insert(0, line_start)
            
            # Add line end if not already there
            if line_end not in change_points:
                change_points.append(line_end)
            
            # Generate event for each segment between change points
            for i in range(len(change_points) - 1):
                segment_start = change_points[i]
                segment_end = change_points[i + 1]
                
                # Skip if before scene
                if segment_start < 0:
                    segment_start = 0
                if segment_end <= 0:
                    continue
                
                # Find what to highlight at midpoint of this segment
                midpoint = (segment_start + segment_end) / 2 + scene_start
                highlighted_idx, is_during = self.find_active_word_at_time(line_words, midpoint)
                
                # Build text with appropriate highlighting
                text_parts = []
                for word_idx, word in enumerate(line_words):
                    word_text = word['word'].upper()
                    
                    if highlighted_idx is not None and word_idx == highlighted_idx:
                        # Current/last spoken word - highlight in yellow
                        text_parts.append(f"{{\\c{self.highlight_color}}}{word_text}{{\\c{self.normal_color}}}")
                    elif highlighted_idx is not None and word_idx < highlighted_idx:
                        # Already spoken - optionally make dimmer
                        text_parts.append(f"{{\\c{self.dim_color}}}{word_text}{{\\c{self.normal_color}}}")
                    else:
                        # Not yet spoken - normal white
                        text_parts.append(word_text)
                
                text = " ".join(text_parts)
                
                # Format times
                start_str = self.format_time(segment_start)
                end_str = self.format_time(segment_end)
                
                event = f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}"
                events.append(event)
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header + "\n".join(events))
    
    def format_time(self, seconds: float) -> str:
        """Format time for ASS: H:MM:SS.CC"""
        if seconds < 0:
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
    
    def generate_video(self, input_video: str, output_video: str, words: List[Dict], scene_start: float = 0.0):
        """Generate video with continuous karaoke captions."""
        
        print("\n🎵 CONTINUOUS KARAOKE GENERATOR")
        print("=" * 50)
        
        # Group words
        lines = self.group_words_into_lines(words)
        print(f"📝 Processing {len(words)} words into {len(lines)} lines")
        
        # Analyze continuity
        print("\n🔄 Continuity Analysis:")
        total_gaps = 0
        for line in lines[:5]:  # Check first 5 lines
            for i in range(len(line) - 1):
                gap = line[i + 1]['start'] - line[i]['end']
                if gap > 0.01:
                    total_gaps += 1
        
        if total_gaps > 0:
            print(f"   Found {total_gaps} pauses between words")
            print("   ✓ Last word will stay highlighted during pauses")
        else:
            print("   ✓ No significant pauses detected")
        
        # Show preview
        print("\n📋 Line Preview (first 5):")
        for i, line in enumerate(lines[:5], 1):
            text = " ".join([w['word'].upper() for w in line])
            duration = line[-1]['end'] - line[0]['start']
            print(f"   {i}. {text} ({duration:.1f}s)")
        
        # Generate ASS file
        ass_path = Path(output_video).parent / "continuous_karaoke.ass"
        self.generate_continuous_ass_file(lines, str(ass_path), scene_start)
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
    """Generate continuous karaoke video."""
    
    print("\n" + "🎤" * 30)
    print("  CONTINUOUS KARAOKE - NO FLICKER")
    print("🎤" * 30)
    
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
    generator = ContinuousKaraokeGenerator()
    
    input_video = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    output_video = "uploads/assets/videos/do_re_mi/scenes/final/scene_001_karaoke_continuous.mp4"
    
    Path(output_video).parent.mkdir(exist_ok=True, parents=True)
    
    success = generator.generate_video(
        input_video=input_video,
        output_video=output_video,
        words=scene1_words,
        scene_start=scene1["start_seconds"]
    )
    
    if success:
        print("\n" + "✨" * 30)
        print("  PERFECT KARAOKE COMPLETE!")
        print("✨" * 30)
        print("\n🎨 Features:")
        print("  ✅ NO FLICKERING - Lines stay visible continuously")
        print("  ✅ Progressive highlighting as words are spoken")
        print("  ✅ Last word stays highlighted during pauses")
        print("  ✅ Already spoken words shown in light gray")
        print("  ✅ Smooth transitions between words")
        print(f"\n📹 Final Output: {output_video}")


if __name__ == "__main__":
    main()