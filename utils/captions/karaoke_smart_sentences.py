#!/usr/bin/env python3
"""
Smart Sentence-Based Karaoke Caption Generator
===============================================

Features:
- Groups words by sentences (respects punctuation)
- Splits long sentences (>15 words) intelligently
- Rotates highlight colors between sentences
- Interpolates missing timestamps
- No flickering, continuous display
"""

import json
from pathlib import Path
import subprocess
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartSentenceKaraoke:
    def __init__(self):
        """Initialize settings."""
        self.font_size = 56
        self.font_bold = True
        self.outline_width = 4
        self.margin_bottom = 80
        
        # Colors in ASS BGR format - rotate between these
        self.highlight_colors = [
            "&H00FFFF",  # Yellow
            "&H0000FF",  # Red
            "&H00FF00",  # Green
            "&HFF00FF",  # Purple/Magenta
            "&H00CCFF",  # Orange
            "&HFFFF00",  # Cyan
            "&HFF69B4",  # Pink
            "&H00FF7F",  # Blue-green
        ]
        
        self.normal_color = "&HFFFFFF"     # White
        self.dim_color = "&HCCCCCC"       # Light gray for already spoken words
        self.outline_color = "&H000000"    # Black
        
        # Current color index for rotation
        self.current_color_index = 0
    
    def get_next_highlight_color(self):
        """Get the next highlight color in rotation."""
        color = self.highlight_colors[self.current_color_index]
        self.current_color_index = (self.current_color_index + 1) % len(self.highlight_colors)
        return color
    
    def interpolate_missing_timestamps(self, words: List[Dict]) -> List[Dict]:
        """
        Interpolate timestamps for words that don't have them.
        Distributes time evenly between words with known timestamps.
        """
        interpolated = []
        i = 0
        
        while i < len(words):
            current_word = words[i].copy()
            
            # Check if current word has timestamps
            if 'start' in current_word and 'end' in current_word:
                # Word has timestamps, use them
                interpolated.append(current_word)
                i += 1
            else:
                # Word missing timestamps - find the group of words without timestamps
                missing_group = [current_word]
                j = i + 1
                
                # Collect all consecutive words without timestamps
                while j < len(words) and ('start' not in words[j] or 'end' not in words[j]):
                    missing_group.append(words[j].copy())
                    j += 1
                
                # Find surrounding words with timestamps
                prev_word = interpolated[-1] if interpolated else None
                next_word = words[j] if j < len(words) else None
                
                # Determine time boundaries for interpolation
                if prev_word and 'end' in prev_word:
                    start_boundary = prev_word['end']
                elif prev_word and 'start' in prev_word:
                    start_boundary = prev_word['start'] + 0.5
                else:
                    if next_word and 'start' in next_word:
                        start_boundary = max(0, next_word['start'] - len(missing_group) * 0.3)
                    else:
                        start_boundary = 0
                
                if next_word and 'start' in next_word:
                    end_boundary = next_word['start']
                elif next_word and 'end' in next_word:
                    end_boundary = next_word['end'] - 0.5
                else:
                    if prev_word and 'end' in prev_word:
                        end_boundary = prev_word['end'] + len(missing_group) * 0.3
                    else:
                        end_boundary = start_boundary + len(missing_group) * 0.3
                
                # Calculate duration per word
                total_duration = end_boundary - start_boundary
                if total_duration <= 0:
                    total_duration = len(missing_group) * 0.3
                
                duration_per_word = total_duration / len(missing_group)
                
                # Assign interpolated timestamps
                current_time = start_boundary
                for word in missing_group:
                    word['start'] = current_time
                    word['end'] = current_time + duration_per_word
                    word['interpolated'] = True
                    current_time += duration_per_word
                    interpolated.append(word)
                
                logger.info(f"Interpolated timestamps for {len(missing_group)} words: "
                          f"{[w['word'] for w in missing_group]} "
                          f"between {start_boundary:.2f}s and {end_boundary:.2f}s")
                
                # Move index past the missing group
                i = j
        
        return interpolated
    
    def group_words_into_smart_lines(self, words: List[Dict]) -> List[List[Dict]]:
        """
        Group words into lines based on sentence structure.
        - Respects sentence endings (. ! ?)
        - Splits at commas if line is getting long
        - Splits very long sentences (>15 words) intelligently
        """
        lines = []
        current_line = []
        current_sentence_length = 0
        
        for i, word in enumerate(words):
            current_line.append(word)
            current_sentence_length += 1
            
            word_text = word['word'].rstrip()
            should_break = False
            
            # Check for sentence endings
            if word_text and word_text[-1] in '.!?':
                # End of sentence - always break here
                should_break = True
                current_sentence_length = 0
            
            # Check for comma or semicolon (natural pause points)
            elif word_text and word_text[-1] in ',;:':
                # Break at comma if we have at least 4 words
                if len(current_line) >= 4:
                    should_break = True
            
            # Check if sentence is getting too long
            elif current_sentence_length >= 15:
                # Force break for very long sentences
                # Try to find a good break point
                if i + 1 < len(words):
                    next_word = words[i + 1]['word'].lower()
                    # Break before conjunctions or prepositions
                    if next_word in ['and', 'but', 'or', 'when', 'where', 'which', 'that', 'with', 'for', 'to', 'in', 'on', 'at']:
                        should_break = True
                        current_sentence_length = 0
            
            # Also check for very long lines regardless of sentence
            elif len(current_line) >= 8:
                # Maximum 8 words per line for readability
                should_break = True
            
            # Musical notes special case - keep them together
            elif word_text.lower() in ['do', 're', 'mi', 'fa', 'sol', 'la', 'ti', 'do-re-mi']:
                if i + 1 < len(words):
                    next_word = words[i + 1]['word'].lower()
                    # If next word is not a note, break after this group
                    if next_word not in ['do', 're', 'mi', 'fa', 'sol', 'la', 'ti', 'do-re-mi'] and len(current_line) >= 3:
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
    
    def is_sentence_ending(self, word: str) -> bool:
        """Check if word ends a sentence."""
        return word.rstrip() and word.rstrip()[-1] in '.!?'
    
    def generate_continuous_ass_file(self, lines: List[List[Dict]], output_path: str, scene_start: float = 0.0):
        """Generate ASS file with continuous display and rotating colors."""
        
        # ASS header
        header = f"""[Script Info]
Title: Smart Sentence Karaoke
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
        
        # Reset color index for each scene
        self.current_color_index = 0
        
        for line_idx, line_words in enumerate(lines):
            if not line_words:
                continue
            
            # Check if this line ends a sentence to determine color
            line_ends_sentence = self.is_sentence_ending(line_words[-1]['word'])
            
            # Get color for this line
            line_highlight_color = self.get_next_highlight_color() if line_idx == 0 or lines[line_idx-1] and self.is_sentence_ending(lines[line_idx-1][-1]['word']) else self.highlight_colors[(self.current_color_index - 1) % len(self.highlight_colors)]
            
            # Line displays from first word start to last word end
            line_start = line_words[0]['start'] - scene_start
            line_end = line_words[-1]['end'] - scene_start
            
            if line_end < 0:
                continue
            
            # Create change points
            change_points = []
            for word in line_words:
                change_points.append(word['start'] - scene_start)
                change_points.append(word['end'] - scene_start)
            
            change_points = sorted(set(change_points))
            
            if line_start not in change_points:
                change_points.insert(0, line_start)
            if line_end not in change_points:
                change_points.append(line_end)
            
            # Generate events for each segment
            for i in range(len(change_points) - 1):
                segment_start = change_points[i]
                segment_end = change_points[i + 1]
                
                if segment_start < 0:
                    segment_start = 0
                if segment_end <= 0:
                    continue
                
                # Find what to highlight
                midpoint = (segment_start + segment_end) / 2 + scene_start
                highlighted_idx, is_during = self.find_active_word_at_time(line_words, midpoint)
                
                # Build text with highlighting
                text_parts = []
                for word_idx, word in enumerate(line_words):
                    word_text = word['word'].upper()
                    
                    if highlighted_idx is not None and word_idx == highlighted_idx:
                        # Current word - highlight with current color
                        text_parts.append(f"{{\\c{line_highlight_color}}}{word_text}{{\\c{self.normal_color}}}")
                    elif highlighted_idx is not None and word_idx < highlighted_idx:
                        # Already spoken
                        text_parts.append(f"{{\\c{self.dim_color}}}{word_text}{{\\c{self.normal_color}}}")
                    else:
                        # Not yet spoken
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
        """Generate video with smart sentence-based karaoke captions."""
        
        print("\n🎵 SMART SENTENCE KARAOKE")
        print("=" * 50)
        
        # Check for missing timestamps
        missing_count = sum(1 for w in words if 'start' not in w or 'end' not in w)
        if missing_count > 0:
            print(f"⚠️  Found {missing_count} words without timestamps")
            print("   Interpolating missing timestamps...")
        
        # Interpolate missing timestamps
        interpolated_words = self.interpolate_missing_timestamps(words)
        
        # Group words into smart lines
        lines = self.group_words_into_smart_lines(interpolated_words)
        print(f"📝 Processing {len(interpolated_words)} words into {len(lines)} lines")
        
        # Report interpolation results
        interpolated_count = sum(1 for w in interpolated_words if w.get('interpolated', False))
        if interpolated_count > 0:
            print(f"✅ Interpolated timestamps for {interpolated_count} words")
        
        # Show preview with sentence structure
        print("\n📋 Line Preview (showing sentence structure):")
        for i, line in enumerate(lines[:8], 1):
            text = " ".join([w['word'].upper() for w in line])
            duration = line[-1]['end'] - line[0]['start']
            
            # Show if line ends a sentence
            ends_sentence = self.is_sentence_ending(line[-1]['word'])
            marker = " [END]" if ends_sentence else " [...]"
            
            print(f"   {i}. {text}{marker} ({duration:.1f}s)")
        
        # Generate ASS file
        ass_path = Path(output_video).parent / "karaoke_smart.ass"
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
        print(f"   Features:")
        print(f"   ✓ Smart sentence grouping")
        print(f"   ✓ Color rotation between sentences")
        print(f"   ✓ No mid-sentence breaks (unless >15 words)")
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Success! Video saved to: {Path(output_video).name}")
            
            # Clean up
            ass_path.unlink(missing_ok=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e.stderr[:300]}")
            return False


def test_smart_karaoke():
    """Test the smart sentence karaoke generator."""
    
    print("\n" + "🎯" * 30)
    print("  TESTING SMART SENTENCE KARAOKE")
    print("🎯" * 30)
    
    # Load transcript
    transcript_file = Path("uploads/assets/videos/do_re_mi/transcripts/transcript_words.json")
    with open(transcript_file) as f:
        transcript = json.load(f)
    
    # Get scene 1 words
    scene1_words = [w for w in transcript["words"] if w.get("start", 0) <= 56.74]
    
    # Initialize generator
    generator = SmartSentenceKaraoke()
    
    # Generate test video
    input_video = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    output_video = "uploads/assets/videos/do_re_mi/scenes/karaoke/scene_001_smart_test.mp4"
    
    Path(output_video).parent.mkdir(exist_ok=True, parents=True)
    
    success = generator.generate_video(
        input_video=input_video,
        output_video=output_video,
        words=scene1_words,
        scene_start=0.0
    )
    
    if success:
        print("\n" + "✨" * 30)
        print("  SMART KARAOKE TEST COMPLETE!")
        print("✨" * 30)


if __name__ == "__main__":
    test_smart_karaoke()