"""
Step 7: Karaoke Caption Generation
===================================

Generates karaoke-style captions for edited videos.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "utils" / "captions"))
try:
    from karaoke_precise import PreciseKaraoke as KaraokeWithInterpolation
except ImportError:
    try:
        from karaoke_sentence_punctuated import PunctuatedKaraoke as KaraokeWithInterpolation
    except ImportError:
        try:
            from karaoke_sentence_aware import SentenceAwareKaraoke as KaraokeWithInterpolation
        except ImportError:
            try:
                from karaoke_smart_sentences import SmartSentenceKaraoke as KaraokeWithInterpolation
            except ImportError:
                from karaoke_with_interpolation import KaraokeWithInterpolation


class KaraokeStep:
    """Handles karaoke caption generation operations."""
    
    def __init__(self, pipeline_state, dirs, config):
        """Initialize with pipeline references."""
        self.pipeline_state = pipeline_state
        self.dirs = dirs
        self.config = config
    
    def run(self, use_final_with_cartoons=False):
        """Generate karaoke-style captions for edited videos.
        
        Args:
            use_final_with_cartoons: If True, use final_with_cartoons as input, otherwise use edited
        """
        print("\n" + "-"*60)
        print("STEP 7: GENERATING KARAOKE CAPTIONS")
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
        
        # Output to edited directory (final output location for all processing)
        output_dir = self.dirs['scenes_edited']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize karaoke generator with interpolation support
        generator = KaraokeWithInterpolation()
        
        print(f"  Style: {self.config.karaoke_style}")
        print(f"  Features:")
        print("    ✓ Center-bottom positioning")
        print("    ✓ Max 6 words per line")
        print("    ✓ Word-by-word highlighting")
        print("    ✓ No flickering (continuous display)")
        print("    ✓ Timestamp interpolation for missing words")
        
        # Process each scene
        for scene in scenes_data.get('scenes', []):
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
            
            # Determine input video based on pipeline state
            # Priority: final_with_cartoons > embedded_phrases > edited > original
            final_video = self.dirs['scenes'] / 'final_with_cartoons' / f"scene_{scene_num:03d}.mp4"
            phrases_video = self.dirs['scenes'] / 'embedded_phrases' / f"scene_{scene_num:03d}.mp4"
            edited_video = self.dirs['scenes'] / 'edited' / f"scene_{scene_num:03d}.mp4"
            original_video = self.dirs['scenes'] / 'original' / f"scene_{scene_num:03d}.mp4"
            
            # Use the most processed version available
            if use_final_with_cartoons and final_video.exists():
                input_video = str(final_video)
                print(f"\n  Processing Scene {scene_num} (with cartoons):")
            elif phrases_video.exists():
                input_video = str(phrases_video)
                print(f"\n  Processing Scene {scene_num} (with phrases):")
            elif edited_video.exists():
                input_video = str(edited_video)
                print(f"\n  Processing Scene {scene_num} (edited version):")
            elif original_video.exists():
                input_video = str(original_video)
                print(f"\n  Processing Scene {scene_num} (original version):")
            else:
                print(f"  ⚠ No video found for scene {scene_num}")
                continue
            
            # Output video to edited folder (final location)
            output_video = str(output_dir / f"scene_{scene_num:03d}.mp4")
            
            print(f"    - Words to caption: {len(scene_words)}")
            print(f"    - Duration: {scene_end - scene_start:.1f}s")
            
            # Generate karaoke video
            success = generator.generate_video(
                input_video=input_video,
                output_video=output_video,
                words=scene_words,
                scene_start=scene_start
            )
            
            if success:
                print(f"  ✓ Scene {scene_num} karaoke generated: {Path(output_video).name}")
            else:
                print(f"  ⚠ Scene {scene_num} karaoke generation failed")
        
        print(f"\n  ✓ Karaoke videos saved to: {output_dir}")