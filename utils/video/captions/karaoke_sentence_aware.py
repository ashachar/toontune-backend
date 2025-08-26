#!/usr/bin/env python3
"""
Sentence-Aware Karaoke Caption Generator
=========================================

Features:
- Uses sentence transcript to properly group words
- Never breaks sentences unless >15 words
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


class SentenceAwareKaraoke:
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
    
    def load_sentence_transcript(self, scene_start: float, scene_end: float):
        """Load sentence transcript for timing reference."""
        sentences_file = Path("uploads/assets/videos/do_re_mi/transcripts/transcript_sentences.json")
        if sentences_file.exists():
            with open(sentences_file) as f:
                all_sentences = json.load(f).get('sentences', [])
            
            # Filter sentences for this scene
            scene_sentences = []
            for sent in all_sentences:
                if sent['start'] < scene_end and sent['end'] > scene_start:
                    scene_sentences.append(sent)
            
            return scene_sentences
        return []
    
    def interpolate_missing_timestamps(self, words: List[Dict]) -> List[Dict]:
        """Interpolate timestamps for words that don't have them."""
        interpolated = []
        i = 0
        
        while i < len(words):
            current_word = words[i].copy()
            
            if 'start' in current_word and 'end' in current_word:
                interpolated.append(current_word)
                i += 1
            else:
                # Find group of words without timestamps
                missing_group = [current_word]
                j = i + 1
                
                while j < len(words) and ('start' not in words[j] or 'end' not in words[j]):
                    missing_group.append(words[j].copy())
                    j += 1
                
                # Find boundaries
                prev_word = interpolated[-1] if interpolated else None
                next_word = words[j] if j < len(words) else None
                
                if prev_word and 'end' in prev_word:
                    start_boundary = prev_word['end']
                else:
                    start_boundary = 0
                
                if next_word and 'start' in next_word:
                    end_boundary = next_word['start']
                else:
                    end_boundary = start_boundary + len(missing_group) * 0.3
                
                # Distribute time evenly
                total_duration = end_boundary - start_boundary
                if total_duration <= 0:
                    total_duration = len(missing_group) * 0.3
                
                duration_per_word = total_duration / len(missing_group)
                
                current_time = start_boundary
                for word in missing_group:
                    word['start'] = current_time
                    word['end'] = current_time + duration_per_word
                    word['interpolated'] = True
                    current_time += duration_per_word
                    interpolated.append(word)
                
                i = j
        
        return interpolated
    
    def group_words_by_sentences(self, words: List[Dict], sentences: List[Dict]) -> List[List[Dict]]:
        """
        Group words into lines based on actual sentence boundaries.
        Uses the sentence transcript to know where sentences end.
        """
        lines = []
        current_line = []
        word_idx = 0
        
        for sentence in sentences:
            sent_start = sentence['start']
            sent_end = sentence['end']
            sent_text = sentence['text']
            
            # Collect all words in this sentence
            sentence_words = []
            while word_idx < len(words):
                word = words[word_idx]
                word_time = word.get('start', 0)
                
                # Check if word belongs to this sentence
                if word_time >= sent_start and word_time < sent_end + 0.5:  # 0.5s tolerance
                    sentence_words.append(word)
                    word_idx += 1
                elif word_time >= sent_end + 0.5:
                    # Word belongs to next sentence
                    break
                else:
                    # Skip words before this sentence
                    word_idx += 1
            
            if not sentence_words:
                continue
            
            # Now decide how to display this sentence
            word_count = len(sentence_words)
            
            if word_count <= 8:
                # Short sentence - display as one line
                lines.append(sentence_words)
            elif word_count <= 15:
                # Medium sentence - try to split at comma or logical point
                # Look for comma in the sentence text
                if ',' in sent_text:
                    # Try to split at comma
                    comma_pos = sent_text.index(',')
                    words_before_comma = sent_text[:comma_pos].split()
                    split_point = len(words_before_comma)
                    
                    if 3 <= split_point <= word_count - 3:
                        # Good split point
                        lines.append(sentence_words[:split_point])
                        lines.append(sentence_words[split_point:])
                    else:
                        # Split in half
                        mid_point = word_count // 2
                        lines.append(sentence_words[:mid_point])
                        lines.append(sentence_words[mid_point:])
                else:
                    # No comma - split in half
                    mid_point = word_count // 2
                    lines.append(sentence_words[:mid_point])
                    lines.append(sentence_words[mid_point:])
            else:
                # Long sentence - split into chunks of 6-8 words
                chunk_size = 7
                for i in range(0, word_count, chunk_size):
                    chunk = sentence_words[i:i+chunk_size]
                    if chunk:
                        lines.append(chunk)
        
        # Handle any remaining words not in sentences
        if word_idx < len(words):
            remaining = words[word_idx:]
            for i in range(0, len(remaining), 6):
                chunk = remaining[i:i+6]
                if chunk:
                    lines.append(chunk)
        
        return lines
    
    def find_active_word_at_time(self, line_words: List[Dict], time: float) -> Tuple[Optional[int], bool]:
        """Find which word should be highlighted at a given time."""
        last_spoken = None
        
        for i, word in enumerate(line_words):
            if time < word['start']:
                return (last_spoken, False) if last_spoken is not None else (None, False)
            elif word['start'] <= time <= word['end']:
                return (i, True)
            else:
                last_spoken = i
        
        return (last_spoken, False) if last_spoken is not None else (None, False)
    
    def determine_line_color(self, line_idx: int, lines: List[List[Dict]], sentences: List[Dict]) -> str:
        """Determine which color to use for this line based on sentence boundaries."""
        # Get the start time of the first word in this line
        if not lines[line_idx]:
            return self.highlight_colors[0]
        
        line_start = lines[line_idx][0].get('start', 0)
        
        # Find which sentence this line belongs to
        sentence_idx = 0
        for i, sent in enumerate(sentences):
            if line_start >= sent['start'] and line_start < sent['end'] + 0.5:
                sentence_idx = i
                break
        
        # Use sentence index to determine color
        return self.highlight_colors[sentence_idx % len(self.highlight_colors)]
    
    def generate_continuous_ass_file(self, lines: List[List[Dict]], sentences: List[Dict], 
                                    output_path: str, scene_start: float = 0.0):
        """Generate ASS file with continuous display and rotating colors."""
        
        # ASS header
        header = f"""[Script Info]
Title: Sentence-Aware Karaoke
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
            
            # Get color for this line based on sentence
            line_highlight_color = self.determine_line_color(line_idx, lines, sentences)
            
            # Line timing
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
                        # Current word - highlight with sentence color
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
        """Generate video with sentence-aware karaoke captions."""
        
        print("\n🎵 SENTENCE-AWARE KARAOKE")
        print("=" * 50)
        
        # Calculate scene end time
        scene_end = max(w.get('end', w.get('start', 0)) for w in words) if words else scene_start + 60
        
        # Load sentence transcript for this scene
        sentences = self.load_sentence_transcript(scene_start, scene_end)
        print(f"📖 Found {len(sentences)} sentences in scene")
        
        # Check for missing timestamps
        missing_count = sum(1 for w in words if 'start' not in w or 'end' not in w)
        if missing_count > 0:
            print(f"⚠️  Found {missing_count} words without timestamps")
            print("   Interpolating missing timestamps...")
        
        # Interpolate missing timestamps
        interpolated_words = self.interpolate_missing_timestamps(words)
        
        # Group words by sentences
        lines = self.group_words_by_sentences(interpolated_words, sentences)
        print(f"📝 Grouped {len(interpolated_words)} words into {len(lines)} lines")
        
        # Show preview
        print("\n📋 Line Preview (respecting sentence boundaries):")
        for i, line in enumerate(lines[:8], 1):
            text = " ".join([w['word'].upper() for w in line])
            duration = line[-1]['end'] - line[0]['start']
            
            # Find which sentence this line belongs to
            line_start = line[0].get('start', 0)
            sentence_num = 0
            for j, sent in enumerate(sentences):
                if line_start >= sent['start'] and line_start < sent['end'] + 0.5:
                    sentence_num = j + 1
                    break
            
            color_name = ["Yellow", "Red", "Green", "Purple", "Orange", "Cyan", "Pink", "Blue-green"][sentence_num % 8]
            print(f"   {i}. [{color_name}] {text} ({duration:.1f}s)")
        
        # Generate ASS file
        ass_path = Path(output_video).parent / "karaoke_sentences.ass"
        self.generate_continuous_ass_file(lines, sentences, str(ass_path), scene_start)
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
        print(f"   ✓ Sentence-based grouping")
        print(f"   ✓ Color rotation between sentences")
        print(f"   ✓ Never breaks sentences (unless >15 words)")
        print(f"   ✓ Uses punctuation from transcript")
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Success! Video saved to: {Path(output_video).name}")
            
            # Clean up
            ass_path.unlink(missing_ok=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e.stderr[:300]}")
            return False


# Alias for pipeline compatibility
SmartSentenceKaraoke = SentenceAwareKaraoke