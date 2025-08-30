"""
Step 4: Prompt Generation
==========================

Generates comprehensive prompts for each scene with effects documentation.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.video_description_generator import VideoDescriptionGenerator


class PromptsStep:
    """Handles prompt generation operations."""
    
    def __init__(self, pipeline_state, dirs, config):
        """Initialize with pipeline references."""
        self.pipeline_state = pipeline_state
        self.dirs = dirs
        self.config = config
    
    def run(self):
        """Generate comprehensive prompts for each scene with effects documentation."""
        print("\n" + "-"*60)
        print("STEP 4: GENERATING COMPREHENSIVE PROMPTS")
        print("-"*60)
        
        # Initialize the comprehensive video description generator
        generator = VideoDescriptionGenerator()
        
        # Load word-level transcript for more precise timing
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
        
        # Generate prompts for each scene using real scene data
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
                
                # Extract relevant word-level transcript for this scene
                scene_words = []
                for word in words_transcript:
                    # Check if word is within scene time range
                    if 'start' in word and 'end' in word:
                        word_start = word['start']
                        word_end = word['end']
                        if start_time <= word_start < end_time:
                            # Adjust timing relative to scene start
                            scene_words.append({
                                'word': word['word'],
                                'start_seconds': f"{word_start - start_time:.3f}",
                                'end_seconds': f"{word_end - start_time:.3f}",
                                'duration_ms': int((word_end - word_start) * 1000)
                            })
                
                # Create transcript data structure for the prompt
                transcript_data = {
                    'scene_number': scene_num,
                    'total_words': len(scene_words),
                    'duration_seconds': duration,
                    'words': scene_words
                } if scene_words else None
                
                # Build comprehensive prompt with effects documentation
                # Temporarily set transcript data in generator
                generator.transcript_data = transcript_data
                
                # Extract effects documentation if not already done
                if not generator.effects_documentation:
                    print(f"  Extracting effects documentation...")
                    generator.extract_effects_documentation()
                
                # Build the full prompt
                prompt = generator.build_prompt()
                
                # Add scene-specific context to the prompt
                scene_context = f"\n\nSCENE-SPECIFIC CONTEXT:\n"
                scene_context += f"This is scene {scene_num} of {len(scenes)} total scenes.\n"
                scene_context += f"Scene duration: {duration:.1f} seconds\n"
                scene_context += f"Time range in full video: {start_time:.1f}s - {end_time:.1f}s\n"
                scene_context += f"Scene description: {description}\n"
                
                # Insert scene context before the effects documentation
                prompt = prompt.replace("Use the following comprehensive library", 
                                       scene_context + "\nUse the following comprehensive library")
                
                # Save comprehensive prompt
                prompt_file = self.dirs['prompts'] / f"scene_{scene_num:03d}_prompt.txt"
                with open(prompt_file, 'w') as f:
                    f.write(prompt)
                
                # Calculate prompt stats
                prompt_lines = prompt.count('\n')
                effects_count = prompt.count('def apply_')
                word_count = len(scene_words)
                
                print(f"  ✓ Scene {scene_num} prompt saved: {prompt_file.name}")
                print(f"    - Prompt size: {len(prompt):,} characters, {prompt_lines:,} lines")
                print(f"    - Effects documented: {effects_count} functions")
                print(f"    - Transcript words: {word_count} words with precise timing")
            
            self.pipeline_state['prompts_generated'] = len(scenes)
        else:
            print("  ⚠ No scene data found. Please run scene splitting first.")
            self.pipeline_state['prompts_generated'] = 0
        
        self.pipeline_state['steps_completed'].append('generate_prompts')