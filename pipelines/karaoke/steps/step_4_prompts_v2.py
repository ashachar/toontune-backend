"""
Step 4 V2: Prompt Generation with Key Phrases and Cartoon Characters
=====================================================================

Generates prompts focusing on key phrases and cartoon characters instead of individual words.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.video_description_generator_v2 import VideoDescriptionGeneratorV2


class PromptsStepV2:
    """Handles prompt generation with new key phrases and cartoon character approach."""
    
    def __init__(self, pipeline_state, dirs, config):
        """Initialize with pipeline references."""
        self.pipeline_state = pipeline_state
        self.dirs = dirs
        self.config = config
    
    def run(self):
        """Generate prompts with key phrases and cartoon characters."""
        print("\n" + "-"*60)
        print("STEP 4 V2: GENERATING PROMPTS (KEY PHRASES & CHARACTERS)")
        print("-"*60)
        
        # Initialize the V2 generator
        generator = VideoDescriptionGeneratorV2()
        
        # Load word-level transcript for timing reference
        words_transcript = []
        words_file = self.dirs['transcripts'] / 'transcript_words.json'
        if words_file.exists():
            with open(words_file, 'r') as f:
                words_data = json.load(f)
                words_transcript = words_data.get('words', [])
        
        # Load sentence-level transcript
        transcript_sentences = []
        sentences_file = self.dirs['transcripts'] / 'transcript_sentences.json'
        if sentences_file.exists():
            with open(sentences_file, 'r') as f:
                transcript_data = json.load(f)
                transcript_sentences = transcript_data.get('sentences', [])
        
        # Generate prompts for each scene
        scenes_data = self.pipeline_state.get('scenes', {})
        
        # If no scene data in pipeline state, try loading from metadata
        if not scenes_data or 'scenes' not in scenes_data:
            scene_metadata_file = self.dirs['metadata'] / 'scenes.json'
            if scene_metadata_file.exists():
                with open(scene_metadata_file, 'r') as f:
                    scenes_data = json.load(f)
                print(f"  Loaded scene data from: {scene_metadata_file}")
        
        if 'scenes' in scenes_data and scenes_data['scenes']:
            scenes = scenes_data['scenes']
            
            for scene in scenes:
                scene_num = scene['scene_number']
                start_time = scene['start_seconds']
                end_time = scene['end_seconds']
                duration = scene['duration']
                description = scene['description']
                
                print(f"\n  Processing Scene {scene_num}:")
                print(f"    Duration: {duration:.1f}s")
                
                # Calculate constraints for this scene
                max_key_phrases = max(1, int(duration / 20))  # One phrase per 20 seconds
                max_cartoon_chars = max(1, int(duration / 20))  # One character per 20 seconds (same as phrases)
                
                print(f"    Max key phrases: {max_key_phrases}")
                print(f"    Max cartoon characters: {max_cartoon_chars}")
                print(f"    ⚠️  Key phrases and cartoons must NOT overlap (3+ seconds separation)")
                
                # Extract relevant transcript for this scene
                scene_words = []
                scene_text_parts = []
                for word in words_transcript:
                    # Check if word is within scene time range
                    if 'start' in word and 'end' in word:
                        word_start = word['start']
                        word_end = word['end']
                        if start_time <= word_start < end_time:
                            # Adjust timing relative to scene start
                            scene_words.append({
                                'word': word['word'],
                                'start_seconds': word_start - start_time,
                                'end_seconds': word_end - start_time,
                            })
                            scene_text_parts.append(word['word'])
                
                # Create full text for scene
                scene_full_text = ' '.join(scene_text_parts)
                
                # Create transcript data structure for the prompt
                transcript_data = {
                    'scene_number': scene_num,
                    'total_words': len(scene_words),
                    'duration_seconds': duration,
                    'full_text': scene_full_text,
                    'words_with_timing': scene_words
                } if scene_words else None
                
                # Set transcript data in generator
                generator.transcript_data = transcript_data
                
                # Extract effects documentation if not already done
                if not generator.effects_documentation:
                    print(f"  Extracting effects documentation...")
                    generator.extract_effects_documentation()
                
                # Build the full prompt
                prompt = generator.build_prompt()
                
                # Add scene-specific context and constraints
                scene_context = f"""

SCENE-SPECIFIC CONTEXT AND CONSTRAINTS:
========================================
This is scene {scene_num} of {len(scenes)} total scenes.
Scene duration: {duration:.1f} seconds
Time range in full video: {start_time:.1f}s - {end_time:.1f}s
Scene description: {description}

CRITICAL CONSTRAINTS FOR THIS SCENE:
- Maximum {max_key_phrases} key phrase(s) total (one every 20 seconds)
- Maximum {max_cartoon_chars} cartoon character(s) total (one every 20 seconds)
- IMPORTANT: Key phrases and cartoon characters must NEVER appear simultaneously
  * They must be separated by at least 3 seconds
  * Example: If key phrase at 10s-14s, cartoon must be before 7s or after 17s
- Key phrases should be the MOST important/memorable parts only
- Cartoon characters must directly relate to the content

Scene text preview: "{scene_full_text[:100]}..." ({len(scene_words)} words total)
"""
                
                # Insert scene context before the effects documentation
                prompt = prompt.replace("Use the following comprehensive library", 
                                       scene_context + "\nUse the following comprehensive library")
                
                # Save V2 prompt
                prompt_file = self.dirs['prompts'] / f"scene_{scene_num:03d}_prompt_v2.txt"
                with open(prompt_file, 'w') as f:
                    f.write(prompt)
                
                # Also save a backup of the old prompt if it exists
                old_prompt_file = self.dirs['prompts'] / f"scene_{scene_num:03d}_prompt.txt"
                if old_prompt_file.exists():
                    backup_file = self.dirs['prompts'] / f"scene_{scene_num:03d}_prompt_old.txt"
                    old_prompt_file.rename(backup_file)
                
                # Rename V2 to standard name
                prompt_file.rename(old_prompt_file)
                
                # Calculate prompt stats
                prompt_lines = prompt.count('\n')
                effects_count = prompt.count('def apply_')
                
                print(f"  ✓ Scene {scene_num} V2 prompt saved")
                print(f"    - Prompt size: {len(prompt):,} characters, {prompt_lines:,} lines")
                print(f"    - Effects documented: {effects_count} functions")
                print(f"    - Scene text: {len(scene_words)} words → {max_key_phrases} key phrases max")
                print(f"    - Cartoon characters: {max_cartoon_chars} maximum")
            
            self.pipeline_state['prompts_generated'] = len(scenes)
            self.pipeline_state['prompt_version'] = 'v2_key_phrases_and_characters'
        else:
            print("  ⚠ No scene data found. Please run scene splitting first.")
            self.pipeline_state['prompts_generated'] = 0
        
        self.pipeline_state['steps_completed'].append('generate_prompts_v2')