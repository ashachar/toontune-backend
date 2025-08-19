#!/usr/bin/env python3
"""
Enhanced Karaoke Caption Generator with Timestamp Interpolation
- Interpolates missing timestamps for words
- Ensures every word gets highlighted
- No flickering, continuous display
"""

import json
from pathlib import Path
import subprocess
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaraokeWithInterpolation:
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
                    # Fallback if prev word only has start
                    start_boundary = prev_word['start'] + 0.5
                else:
                    # No previous word, use next word's start minus some time
                    if next_word and 'start' in next_word:
                        start_boundary = max(0, next_word['start'] - len(missing_group) * 0.3)
                    else:
                        # Fallback to 0
                        start_boundary = 0
                
                if next_word and 'start' in next_word:
                    end_boundary = next_word['start']
                elif next_word and 'end' in next_word:
                    # Fallback if next word only has end
                    end_boundary = next_word['end'] - 0.5
                else:
                    # No next word, use previous word's end plus some time
                    if prev_word and 'end' in prev_word:
                        end_boundary = prev_word['end'] + len(missing_group) * 0.3
                    else:
                        # Fallback
                        end_boundary = start_boundary + len(missing_group) * 0.3
                
                # Calculate duration per word
                total_duration = end_boundary - start_boundary
                if total_duration <= 0:
                    # Safety fallback
                    total_duration = len(missing_group) * 0.3
                
                duration_per_word = total_duration / len(missing_group)
                
                # Assign interpolated timestamps
                current_time = start_boundary
                for word in missing_group:
                    word['start'] = current_time
                    word['end'] = current_time + duration_per_word
                    word['interpolated'] = True  # Mark as interpolated
                    current_time += duration_per_word
                    interpolated.append(word)
                
                logger.info(f"Interpolated timestamps for {len(missing_group)} words: "
                          f"{[w['word'] for w in missing_group]} "
                          f"between {start_boundary:.2f}s and {end_boundary:.2f}s")
                
                # Move index past the missing group
                i = j
        
        return interpolated
    
    def validate_and_fix_timestamps(self, words: List[Dict]) -> List[Dict]:
        """
        Validate and fix any timestamp issues after interpolation.
        Ensures no overlaps and proper ordering.
        """
        if not words:
            return words
        
        fixed = []
        for i, word in enumerate(words):
            word_copy = word.copy()
            
            # Ensure end > start
            if word_copy.get('end', 0) <= word_copy.get('start', 0):
                word_copy['end'] = word_copy['start'] + 0.2
            
            # Fix overlaps with previous word
            if i > 0 and fixed:
                prev_word = fixed[-1]
                if word_copy.get('start', 0) < prev_word.get('end', 0):
                    # Adjust current word's start to previous word's end
                    gap = 0.01  # Small gap between words
                    word_copy['start'] = prev_word['end'] + gap
                    
                    # Ensure minimum duration
                    if word_copy['end'] <= word_copy['start']:
                        word_copy['end'] = word_copy['start'] + 0.2
            
            fixed.append(word_copy)
        
        return fixed
    
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
Title: Karaoke with Interpolation
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
                    
                    # Add indicator for interpolated words (optional)
                    if word.get('interpolated') and word_idx == highlighted_idx:
                        # Could add a subtle marker for interpolated words
                        pass
                    
                    if highlighted_idx is not None and word_idx == highlighted_idx:
                        # Current word - highlight
                        text_parts.append(f"{{\\c{self.highlight_color}}}{word_text}{{\\c{self.normal_color}}}")
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
        """Generate video with karaoke captions, interpolating missing timestamps."""
        
        print("\n🎵 KARAOKE WITH INTERPOLATION")
        print("=" * 50)
        
        # Check for missing timestamps
        missing_count = sum(1 for w in words if 'start' not in w or 'end' not in w)
        if missing_count > 0:
            print(f"⚠️  Found {missing_count} words without timestamps")
            print("   Interpolating missing timestamps...")
        
        # Interpolate missing timestamps
        interpolated_words = self.interpolate_missing_timestamps(words)
        
        # Validate and fix any issues
        fixed_words = self.validate_and_fix_timestamps(interpolated_words)
        
        # Report interpolation results
        interpolated_count = sum(1 for w in fixed_words if w.get('interpolated', False))
        if interpolated_count > 0:
            print(f"✅ Interpolated timestamps for {interpolated_count} words")
            
            # Show examples
            examples = [w for w in fixed_words if w.get('interpolated', False)][:3]
            for w in examples:
                print(f"   • '{w['word']}': {w['start']:.2f}s - {w['end']:.2f}s")
        
        # Group words
        lines = self.group_words_into_lines(fixed_words)
        print(f"\n📝 Processing {len(fixed_words)} words into {len(lines)} lines")
        
        # Show preview
        print("\n📋 Line Preview (first 5):")
        for i, line in enumerate(lines[:5], 1):
            text = " ".join([w['word'].upper() for w in line])
            duration = line[-1]['end'] - line[0]['start']
            interpolated_in_line = sum(1 for w in line if w.get('interpolated', False))
            if interpolated_in_line > 0:
                print(f"   {i}. {text} ({duration:.1f}s) [{interpolated_in_line} interpolated]")
            else:
                print(f"   {i}. {text} ({duration:.1f}s)")
        
        # Generate ASS file
        ass_path = Path(output_video).parent / "karaoke_interpolated.ass"
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


def test_with_missing_timestamps():
    """Test with words that have missing timestamps."""
    
    print("\n" + "🧪" * 30)
    print("  TESTING INTERPOLATION")
    print("🧪" * 30)
    
    # Create test data with some missing timestamps
    test_words = [
        {"word": "Let's", "start": 7.92, "end": 8.56},
        {"word": "start"},  # Missing timestamp
        {"word": "at"},     # Missing timestamp
        {"word": "the", "start": 9.34, "end": 9.579},
        {"word": "very"},   # Missing timestamp
        {"word": "beginning", "start": 10.1, "end": 11.479},
    ]
    
    generator = KaraokeWithInterpolation()
    
    print("\nOriginal words:")
    for w in test_words:
        if 'start' in w:
            print(f"  '{w['word']}': {w.get('start', '?'):.2f}s - {w.get('end', '?'):.2f}s")
        else:
            print(f"  '{w['word']}': NO TIMESTAMP")
    
    # Interpolate
    interpolated = generator.interpolate_missing_timestamps(test_words)
    
    print("\nAfter interpolation:")
    for w in interpolated:
        marker = " [INTERPOLATED]" if w.get('interpolated') else ""
        print(f"  '{w['word']}': {w['start']:.2f}s - {w['end']:.2f}s{marker}")


if __name__ == "__main__":
    test_with_missing_timestamps()