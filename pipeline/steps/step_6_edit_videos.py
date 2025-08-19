"""
Step 6: Video Editing (Effects Only - No Text Overlays)
========================================================

Edits original videos based on inference results.
Only applies visual and sound effects - NO text overlays.
Text overlays are handled by step_8_embed_phrases.py
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from apply_effects_to_original_scene import OriginalSceneProcessor


class VideoEditStep:
    """Handles video editing operations - effects only, no text."""
    
    def __init__(self, pipeline_state, dirs, config):
        """Initialize with pipeline references."""
        self.pipeline_state = pipeline_state
        self.dirs = dirs
        self.config = config
        self.video_name = None  # Will be set by pipeline
    
    def run(self):
        """Edit original videos based on inference results - NO TEXT OVERLAYS."""
        print("\n" + "-"*60)
        print("STEP 6: EDITING VIDEOS (Effects Only)")
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
            print("  NOTE: Text overlays disabled - handled by embedding steps")
            
            # Use the OriginalSceneProcessor
            base_dir = self.dirs['base'].parent
            processor = OriginalSceneProcessor(base_dir=str(base_dir))
            
            scenes_data = self.pipeline_state.get('scenes', {})
            if 'scenes' in scenes_data:
                scene_count = len(scenes_data.get('scenes', []))
            else:
                scene_count = scenes_data.get('count', 4)
            
            for scene_num in range(1, scene_count + 1):
                # Use downsampled scenes as input to avoid any old text overlays
                downsampled_scene = self.dirs['scenes_downsampled'] / f"scene_{scene_num:03d}.mp4"
                original_scene = self.dirs['scenes_original'] / f"scene_{scene_num:03d}.mp4"
                
                # Prefer downsampled if it exists, otherwise use original
                if downsampled_scene.exists():
                    input_scene = downsampled_scene
                    print(f"  Using downsampled scene for editing: {downsampled_scene.name}")
                elif original_scene.exists():
                    input_scene = original_scene
                    print(f"  Using original scene for editing: {original_scene.name}")
                else:
                    print(f"  ⚠ No scene file found for scene {scene_num}")
                    continue
                
                # Load inference file for sound effects and visual effects ONLY
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
                    
                    # Extract visual effects from suggested_effects
                    if 'scenes' in inference:
                        for scene in inference['scenes']:
                            suggested_effects.extend(scene.get('suggested_effects', []))
                
                print(f"\n  Processing Scene {scene_num}:")
                print(f"    - Sound effects: {len(sound_effects)}")
                print(f"    - Visual effects: {len(suggested_effects)}")
                print(f"    - Text overlays: DISABLED (handled by embedding steps)")
                
                # Process the scene with NO text overlays
                success = processor.process_scene(
                    video_name=self.video_name,
                    scene_number=scene_num,
                    text_overlays=[],  # EMPTY - No text overlays in editing step
                    sound_effects=sound_effects,
                    test_mode=self.config.test_mode,
                    suggested_effects=suggested_effects if len(suggested_effects) > 0 else None
                )
                
                if success:
                    print(f"  ✓ Scene {scene_num} edited successfully (effects only)")
                else:
                    print(f"  ⚠ Scene {scene_num} editing had issues")
        
        self.pipeline_state['steps_completed'].append('edit_videos')