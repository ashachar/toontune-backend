"""
Step 6: Video Editing
======================

Edits original videos based on inference results.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from apply_effects_to_original_scene import OriginalSceneProcessor
from utils.text_placement.intelligent_text_placer import IntelligentTextPlacer


class VideoEditStep:
    """Handles video editing operations."""
    
    def __init__(self, pipeline_state, dirs, config):
        """Initialize with pipeline references."""
        self.pipeline_state = pipeline_state
        self.dirs = dirs
        self.config = config
        self.video_name = None  # Will be set by pipeline
    
    def run(self):
        """Edit original videos based on inference results."""
        print("\n" + "-"*60)
        print("STEP 6: EDITING VIDEOS")
        print("-"*60)
        
        if self.config.dry_run:
            print("  [DRY RUN MODE - No actual video editing]")
            print("  Would apply effects from inference results to original scenes")
            
            scenes_data = self.pipeline_state.get('scenes', {})
            if 'scenes' in scenes_data:
                scene_count = len(scenes_data.get('scenes', []))
            else:
                scene_count = scenes_data.get('count', 4)
            
            for scene_num in range(1, scene_count + 1):
                original_scene = self.dirs['scenes_original'] / f"scene_{scene_num:03d}.mp4"
                edited_scene = self.dirs['scenes_edited'] / f"scene_{scene_num:03d}.mp4"
                inference_file = self.dirs['inferences'] / f"scene_{scene_num:03d}_inference.json"
                
                print(f"  Would edit scene {scene_num}:")
                print(f"    Input: {original_scene}")
                print(f"    Inference: {inference_file}")
                print(f"    Output: {edited_scene}")
        else:
            print("  [FULL MODE - Applying effects to videos]")
            
            # Load word-level transcript for ALL words
            words_transcript = []
            words_file = self.dirs['transcripts'] / 'transcript_words.json'
            if words_file.exists():
                with open(words_file, 'r') as f:
                    words_data = json.load(f)
                    words_transcript = words_data.get('words', [])
                print(f"  Loaded {len(words_transcript)} words from transcript")
            
            # Load scene metadata to get time ranges
            scenes_metadata = self.pipeline_state.get('scenes', {})
            if not scenes_metadata or 'scenes' not in scenes_metadata:
                scene_metadata_file = self.dirs['metadata'] / 'scenes.json'
                if scene_metadata_file.exists():
                    with open(scene_metadata_file, 'r') as f:
                        scenes_metadata = json.load(f)
            
            # Use the OriginalSceneProcessor
            base_dir = self.dirs['base'].parent
            processor = OriginalSceneProcessor(base_dir=str(base_dir))
            
            scenes_data = self.pipeline_state.get('scenes', {})
            if 'scenes' in scenes_data:
                scene_count = len(scenes_data.get('scenes', []))
            else:
                scene_count = scenes_data.get('count', 4)
            
            for scene_num in range(1, scene_count + 1):
                original_scene = self.dirs['scenes_original'] / f"scene_{scene_num:03d}.mp4"
                
                if not original_scene.exists():
                    print(f"  ⚠ Scene file not found: {original_scene}")
                    continue
                
                # Get scene time range
                scene_info = None
                if 'scenes' in scenes_metadata:
                    for s in scenes_metadata['scenes']:
                        if s['scene_number'] == scene_num:
                            scene_info = s
                            break
                
                if not scene_info:
                    print(f"  ⚠ No scene info found for scene {scene_num}")
                    continue
                
                scene_start = scene_info['start_seconds']
                scene_end = scene_info['end_seconds']
                
                # Extract words for this scene
                scene_words = []
                for word in words_transcript:
                    word_start = word.get('start', 0)
                    word_end = word.get('end', word_start + 0.5)
                    
                    # Check if word is within this scene
                    if scene_start <= word_start < scene_end:
                        # Adjust timing relative to scene start
                        scene_words.append({
                            'word': word.get('word', ''),
                            'start': word_start - scene_start,
                            'end': word_end - scene_start
                        })
                
                print(f"\n  Processing Scene {scene_num}:")
                print(f"    - Scene duration: {scene_end - scene_start:.1f}s")
                print(f"    - Words to place: {len(scene_words)}")
                
                # Use IntelligentTextPlacer to find optimal positions for each word
                backgrounds_dir = self.dirs['scenes'] / 'backgrounds' / f"scene_{scene_num:03d}"
                placer = IntelligentTextPlacer(str(original_scene), str(backgrounds_dir))
                
                print(f"  Extracting backgrounds and finding optimal text positions...")
                positioned_words = placer.generate_word_positions(scene_words)
                
                # Load inference file for sound effects and visual effects
                inference_file = self.dirs['inferences'] / f"scene_{scene_num:03d}_inference.json"
                sound_effects = []
                suggested_effects = []
                
                if inference_file.exists():
                    with open(inference_file, 'r') as f:
                        inference = json.load(f)
                    
                    # Extract sound effects
                    for effect in inference.get('sound_effects', []):
                        sound_effects.append({
                            'sound': effect.get('sound', ''),
                            'timestamp': float(effect.get('timestamp', 0)),
                            'volume': 0.5  # Default volume at 50%
                        })
                    
                    # Extract visual effects if in test mode
                    if self.config.test_mode and 'scenes' in inference:
                        for scene in inference['scenes']:
                            if 'scene_description' in scene and 'suggested_effects' in scene['scene_description']:
                                suggested_effects.extend(scene['scene_description'].get('suggested_effects', []))
                
                print(f"    - Sound effects: {len(sound_effects)}")
                print(f"    - Visual effects: {len(suggested_effects)}")
                
                # Process the scene with ALL words intelligently positioned
                success = processor.process_scene(
                    video_name=self.video_name,
                    scene_number=scene_num,
                    text_overlays=positioned_words,  # Use ALL words with intelligent positions
                    sound_effects=sound_effects,
                    test_mode=self.config.test_mode,
                    suggested_effects=suggested_effects if self.config.test_mode else None
                )
                
                if success:
                    print(f"  ✓ Scene {scene_num} edited successfully with {len(positioned_words)} words")
                else:
                    print(f"  ⚠ Scene {scene_num} editing had issues")
        
        self.pipeline_state['steps_completed'].append('edit_videos')