"""
Step 7: Karaoke Caption Generation (FIXED)
===========================================

Generates karaoke-style captions for edited videos using simple subtitle overlay.
"""

import json
import subprocess
from pathlib import Path


class KaraokeStep:
    """Handles karaoke caption generation operations."""
    
    def __init__(self, pipeline_state, dirs, config):
        """Initialize with pipeline references."""
        self.pipeline_state = pipeline_state
        self.dirs = dirs
        self.config = config
    
    def run(self, use_final_with_cartoons=False):
        """Generate karaoke-style captions for edited videos."""
        print("\n" + "-"*60)
        print("STEP 7: GENERATING KARAOKE CAPTIONS (SIMPLIFIED)")
        print("-"*60)
        
        # Load word-level transcript
        words_file = self.dirs['transcripts'] / 'transcript_words.json'
        if not words_file.exists():
            print("  ⚠ No word-level transcript found. Skipping karaoke generation.")
            return
        
        with open(words_file, 'r') as f:
            transcript_data = json.load(f)
            words_transcript = transcript_data.get('words', [])
        
        # Load scene metadata
        scene_metadata_file = self.dirs['metadata'] / 'scenes.json'
        if not scene_metadata_file.exists():
            print("  ⚠ No scene metadata found. Skipping karaoke generation.")
            return
        
        with open(scene_metadata_file, 'r') as f:
            scenes_data = json.load(f)
        
        # Output to edited directory
        output_dir = self.dirs['scenes_edited']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Style: Simple subtitle overlay")
        print(f"  Features:")
        print("    ✓ Bottom positioning")
        print("    ✓ White text with black outline")
        print("    ✓ Sentence-based display")
        
        # Process each scene
        for scene in scenes_data.get('scenes', [])[:1]:  # Just scene 1 for now
            scene_num = scene['scene_number']
            scene_start = scene['start_seconds']
            scene_end = scene['end_seconds']
            
            # Get words for this scene
            scene_words = []
            for word in words_transcript:
                word_start = word.get('start', 0)
                if scene_start <= word_start < scene_end:
                    scene_words.append(word)
            
            if not scene_words:
                print(f"  ⚠ No words found for scene {scene_num}")
                continue
            
            # Input/output paths
            input_video = self.dirs['scenes_edited'] / f"scene_{scene_num:03d}.mp4"
            output_video = self.dirs['scenes_edited'] / f"scene_{scene_num:03d}_temp.mp4"
            
            if not input_video.exists():
                print(f"  ⚠ No video found for scene {scene_num}")
                continue
            
            print(f"\n  Processing Scene {scene_num}:")
            print(f"    - Words to caption: {len(scene_words)}")
            print(f"    - Duration: {scene_end - scene_start:.1f}s")
            
            # Create simple ASS subtitle file
            ass_path = output_dir / f"scene_{scene_num:03d}_karaoke.ass"
            self._create_simple_ass(ass_path, scene_words, scene_start)
            
            # Apply subtitles with FFmpeg
            cmd = [
                'ffmpeg', '-i', str(input_video),
                '-vf', f"subtitles={ass_path}:force_style='Fontsize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1,Alignment=2,MarginV=30'",
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-c:a', 'copy',
                '-y', str(output_video)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace original with karaoke version
                subprocess.run(['mv', str(output_video), str(input_video)], check=True)
                print(f"  ✓ Scene {scene_num} karaoke generated")
            else:
                print(f"  ⚠ Scene {scene_num} karaoke generation failed")
                print(f"    Error: {result.stderr[:200]}")
        
        print(f"\n  ✓ Karaoke videos saved to: {output_dir}")
    
    def _create_simple_ass(self, output_path, words, scene_start):
        """Create a simple ASS subtitle file."""
        # Group words into sentences
        sentences = []
        current_sentence = []
        
        for word in words:
            current_sentence.append(word)
            # End sentence on punctuation
            if word['word'].rstrip()[-1:] in '.!?,;:':
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
        
        if current_sentence:
            sentences.append(current_sentence)
        
        # Create ASS content
        ass_content = """[Script Info]
Title: Karaoke
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        for sentence in sentences[:10]:  # Limit to first 10 sentences for testing
            if not sentence:
                continue
            
            # Get sentence timing
            start_time = sentence[0]['start'] - scene_start
            end_time = sentence[-1].get('end', sentence[-1]['start'] + 1) - scene_start
            
            # Format text
            text = ' '.join(w['word'] for w in sentence).upper()
            
            # Convert to ASS timestamp format
            def to_ass_time(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = seconds % 60
                return f"{h}:{m:02d}:{s:05.2f}"
            
            start_str = to_ass_time(start_time)
            end_str = to_ass_time(end_time)
            
            ass_content += f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}\n"
        
        with open(output_path, 'w') as f:
            f.write(ass_content)
        
        print(f"    💾 Created subtitle file: {output_path.name}")