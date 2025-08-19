#!/usr/bin/env python3
"""
Precise Multi-line Karaoke Caption Generator
=============================================

Features:
- PRECISE sentence boundaries from transcript_sentences.json
- Two-line display for 6+ word sentences
- Smart word counting (ignores <4 letter words)
- Includes punctuation
- Color rotation between sentences
"""

import json
import re
from pathlib import Path
import subprocess
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreciseKaraoke:
    def __init__(self):
        """Initialize settings."""
        self.font_size = 48  # Slightly smaller for two-line display
        self.font_bold = True
        self.outline_width = 3
        self.margin_bottom = 60
        
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
                # Include sentence if it overlaps with scene at all
                if sent['end'] > scene_start and sent['start'] < scene_end:
                    scene_sentences.append(sent)
            
            return scene_sentences
        return []
    
    def add_punctuation_to_words(self, words: List[Dict], sentence_text: str) -> List[Dict]:
        """Add punctuation to words based on the sentence text."""
        # Create a clean word list from sentence
        import re
        tokens = re.findall(r'\w+|[.,!?;:]', sentence_text)
        
        punctuated = []
        token_idx = 0
        
        for word in words:
            word_copy = word.copy()
            word_copy['display'] = word['word']
            
            # Match word with tokens
            while token_idx < len(tokens):
                if tokens[token_idx].lower() == word['word'].lower():
                    token_idx += 1
                    # Check if next token is punctuation
                    if token_idx < len(tokens) and tokens[token_idx] in '.,!?;:':
                        word_copy['display'] = word['word'] + tokens[token_idx]
                        token_idx += 1
                    break
                elif tokens[token_idx] in '.,!?;:':
                    # Skip standalone punctuation
                    token_idx += 1
                else:
                    # Token doesn't match, move on
                    break
            
            punctuated.append(word_copy)
        
        return punctuated
    
    def count_significant_words(self, words: List[Dict]) -> int:
        """Count significant words (≥4 letters, not numbers)."""
        count = 0
        for word in words:
            word_text = word.get('display', word['word'])
            # Remove punctuation for counting
            clean_word = re.sub(r'[.,!?;:]', '', word_text)
            # Count if ≥4 letters and not a number
            if len(clean_word) >= 4 and not clean_word.isdigit():
                count += 1
        return count
    
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
                missing_group = [current_word]
                j = i + 1
                
                while j < len(words) and ('start' not in words[j] or 'end' not in words[j]):
                    missing_group.append(words[j].copy())
                    j += 1
                
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
    
    def assign_words_to_sentences(self, words: List[Dict], sentences: List[Dict]) -> List[Tuple[List[Dict], Dict]]:
        """
        Assign words to sentences using PRECISE boundaries.
        Returns list of (sentence_words, sentence) tuples.
        """
        sentence_groups = []
        
        for sentence in sentences:
            sent_start = sentence['start']
            sent_end = sentence['end']
            sent_text = sentence['text']
            
            # Collect words that fall within this sentence's EXACT boundaries
            sentence_words = []
            for word in words:
                word_start = word.get('start', 0)
                
                # Use STRICT boundaries - word starts at or after sentence start
                # and before sentence end (not at sentence end)
                if sent_start <= word_start < sent_end:
                    sentence_words.append(word)
            
            if sentence_words:
                # Add punctuation to words
                sentence_words = self.add_punctuation_to_words(sentence_words, sent_text)
                sentence_groups.append((sentence_words, sentence))
                
                # Debug log
                logger.info(f"Sentence '{sent_text[:30]}...' [{sent_start:.3f}-{sent_end:.3f}s] has {len(sentence_words)} words")
                if sentence_words:
                    logger.info(f"  First word: {sentence_words[0]['word']} at {sentence_words[0].get('start', 0):.3f}s")
                    logger.info(f"  Last word: {sentence_words[-1]['word']} at {sentence_words[-1].get('start', 0):.3f}s")
        
        return sentence_groups
    
    def group_sentences_into_frames(self, sentence_groups: List[Tuple[List[Dict], Dict]]) -> List[Dict]:
        """
        Group sentences into display frames.
        Each frame can have 1-2 lines.
        """
        frames = []
        
        for sentence_words, sentence in sentence_groups:
            if not sentence_words:
                continue
            
            # Count significant words
            significant_count = self.count_significant_words(sentence_words)
            total_words = len(sentence_words)
            
            # Decide how to display this sentence
            if significant_count <= 6:
                # Short sentence - single line frame
                frames.append({
                    'lines': [sentence_words],
                    'sentence': sentence,
                    'type': 'single'
                })
            elif total_words <= 12:
                # Medium sentence - two-line frame
                split_point = self.find_best_split(sentence_words)
                
                frames.append({
                    'lines': [
                        sentence_words[:split_point],
                        sentence_words[split_point:]
                    ],
                    'sentence': sentence,
                    'type': 'double'
                })
            else:
                # Long sentence - multiple frames
                # Split into chunks
                chunks = []
                remaining = sentence_words[:]
                
                while remaining:
                    if len(remaining) <= 6:
                        chunks.append(remaining)
                        remaining = []
                    elif len(remaining) <= 12:
                        split = self.find_best_split(remaining)
                        chunks.append(remaining[:split])
                        chunks.append(remaining[split:])
                        remaining = []
                    else:
                        # Take first 6-7 words
                        split = min(7, len(remaining))
                        chunks.append(remaining[:split])
                        remaining = remaining[split:]
                
                # Create frames from chunks
                i = 0
                while i < len(chunks):
                    if i + 1 < len(chunks) and len(chunks[i]) + len(chunks[i+1]) <= 12:
                        # Two chunks fit in one frame
                        frames.append({
                            'lines': [chunks[i], chunks[i+1]],
                            'sentence': sentence,
                            'type': 'double'
                        })
                        i += 2
                    else:
                        # Single chunk
                        frames.append({
                            'lines': [chunks[i]],
                            'sentence': sentence,
                            'type': 'single'
                        })
                        i += 1
        
        return frames
    
    def find_best_split(self, words: List[Dict]) -> int:
        """Find the best split point for two-line display."""
        total = len(words)
        ideal_split = total // 2
        
        # Look for comma
        for i, word in enumerate(words):
            if ',' in word.get('display', word['word']):
                # Split after comma if reasonable
                if 0.3 * total <= i+1 <= 0.7 * total:
                    return i + 1
        
        # Look for conjunctions  
        for i, word in enumerate(words):
            if word['word'].lower() in ['and', 'but', 'or', 'when', 'where', 'which', 'that', 'with']:
                if 0.3 * total <= i <= 0.7 * total:
                    return i
        
        # Default to middle
        return ideal_split
    
    def find_active_word_at_time(self, words: List[Dict], time: float) -> Tuple[Optional[int], bool]:
        """Find which word should be highlighted at a given time."""
        last_spoken = None
        
        for i, word in enumerate(words):
            if time < word['start']:
                return (last_spoken, False) if last_spoken is not None else (None, False)
            elif word['start'] <= time <= word['end']:
                return (i, True)
            else:
                last_spoken = i
        
        return (last_spoken, False) if last_spoken is not None else (None, False)
    
    def generate_continuous_ass_file(self, frames: List[Dict], output_path: str, scene_start: float = 0.0):
        """Generate ASS file with multi-line frames."""
        
        # ASS header
        header = f"""[Script Info]
Title: Precise Multi-line Karaoke
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
        current_sentence_idx = -1
        current_color = self.highlight_colors[0]
        last_sentence_text = None
        
        for frame_idx, frame in enumerate(frames):
            # Check if we're in a new sentence
            sentence_text = frame['sentence'].get('text', '')
            if sentence_text != last_sentence_text:
                # New sentence - rotate color
                current_sentence_idx += 1
                current_color = self.highlight_colors[current_sentence_idx % len(self.highlight_colors)]
                last_sentence_text = sentence_text
            
            # Get all words in this frame
            all_frame_words = []
            for line in frame['lines']:
                all_frame_words.extend(line)
            
            if not all_frame_words:
                continue
            
            # Frame timing
            frame_start = all_frame_words[0]['start'] - scene_start
            frame_end = all_frame_words[-1]['end'] - scene_start
            
            if frame_end < 0:
                continue
            
            # Create change points
            change_points = []
            for word in all_frame_words:
                change_points.append(word['start'] - scene_start)
                change_points.append(word['end'] - scene_start)
            
            change_points = sorted(set(change_points))
            
            if frame_start not in change_points:
                change_points.insert(0, frame_start)
            if frame_end not in change_points:
                change_points.append(frame_end)
            
            # Generate events
            for i in range(len(change_points) - 1):
                segment_start = change_points[i]
                segment_end = change_points[i + 1]
                
                if segment_start < 0:
                    segment_start = 0
                if segment_end <= 0:
                    continue
                
                # Find highlighted word
                midpoint = (segment_start + segment_end) / 2 + scene_start
                highlighted_idx, is_during = self.find_active_word_at_time(all_frame_words, midpoint)
                
                # Build text for all lines
                text_lines = []
                word_counter = 0
                
                for line_words in frame['lines']:
                    text_parts = []
                    for word in line_words:
                        word_text = word.get('display', word['word']).upper()
                        
                        if highlighted_idx is not None and word_counter == highlighted_idx:
                            text_parts.append(f"{{\\c{current_color}}}{word_text}{{\\c{self.normal_color}}}")
                        elif highlighted_idx is not None and word_counter < highlighted_idx:
                            text_parts.append(f"{{\\c{self.dim_color}}}{word_text}{{\\c{self.normal_color}}}")
                        else:
                            text_parts.append(word_text)
                        
                        word_counter += 1
                    
                    text_lines.append(" ".join(text_parts))
                
                # Join lines with \N for multi-line
                text = "\\N".join(text_lines)
                
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
        """Generate video with precise multi-line karaoke captions."""
        
        print("\n🎵 PRECISE MULTI-LINE KARAOKE")
        print("=" * 50)
        
        # Calculate scene end
        scene_end = max(w.get('end', w.get('start', 0)) for w in words) if words else scene_start + 60
        
        # Load sentence transcript
        sentences = self.load_sentence_transcript(scene_start, scene_end)
        print(f"📖 Found {len(sentences)} sentences in scene")
        
        # Show sentence boundaries for debugging
        print("\n📍 Sentence boundaries:")
        for i, sent in enumerate(sentences[:6], 1):
            print(f"   {i}. [{sent['start']:.3f}s - {sent['end']:.3f}s] \"{sent['text'][:40]}...\"")
        
        # Interpolate missing timestamps
        missing_count = sum(1 for w in words if 'start' not in w or 'end' not in w)
        if missing_count > 0:
            print(f"⚠️  Found {missing_count} words without timestamps")
            words = self.interpolate_missing_timestamps(words)
        
        # Assign words to sentences with PRECISE boundaries
        sentence_groups = self.assign_words_to_sentences(words, sentences)
        print(f"\n✅ Assigned words to {len(sentence_groups)} sentences using precise boundaries")
        
        # Group into frames
        frames = self.group_sentences_into_frames(sentence_groups)
        print(f"📝 Organized into {len(frames)} frames")
        
        # Show preview
        print("\n📋 Frame Preview:")
        current_sentence = None
        for i, frame in enumerate(frames[:10], 1):
            if frame['sentence']['text'] != current_sentence:
                current_sentence = frame['sentence']['text']
                color_idx = [s for s in sentences if s['text'] == current_sentence][0] if current_sentence else 0
                idx = sentences.index(color_idx) if color_idx in sentences else 0
                color_name = ["Yellow", "Red", "Green", "Purple", "Orange", "Cyan", "Pink", "Blue-green"][idx % 8]
            
            print(f"\n   Frame {i} [{color_name}] ({frame['type']}):")
            for j, line in enumerate(frame['lines'], 1):
                text = " ".join([w.get('display', w['word']).upper() for w in line])
                print(f"     Line {j}: {text}")
        
        # Generate ASS file
        ass_path = Path(output_video).parent / "karaoke_precise.ass"
        self.generate_continuous_ass_file(frames, str(ass_path), scene_start)
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
        print(f"   ✓ PRECISE sentence boundaries")
        print(f"   ✓ Multi-line display (2 lines for 6+ word sentences)")
        print(f"   ✓ Punctuation included")
        print(f"   ✓ Color rotation between sentences")
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Success! Video saved to: {Path(output_video).name}")
            
            # Clean up
            ass_path.unlink(missing_ok=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e.stderr[:300]}")
            return False


# Aliases for compatibility
MultilineKaraoke = PreciseKaraoke
PunctuatedKaraoke = PreciseKaraoke
SentenceAwareKaraoke = PreciseKaraoke