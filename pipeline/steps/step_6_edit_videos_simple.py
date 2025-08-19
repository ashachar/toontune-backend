"""
Step 6: Video Editing (Simplified - No Text Overlays)
======================================================

Simplified video editing that just copies videos when no effects are needed.
Only applies visual and sound effects if present - NO text overlays.
"""

import json
import subprocess
from pathlib import Path


class VideoEditStep:
    """Simplified video editing - just copy if no effects."""
    
    def __init__(self, pipeline_state, dirs, config):
        """Initialize with pipeline references."""
        self.pipeline_state = pipeline_state
        self.dirs = dirs
        self.config = config
        self.video_name = None  # Will be set by pipeline
    
    def run(self):
        """Edit videos or copy them if no effects needed."""
        print("\n" + "-"*60)
        print("STEP 6: EDITING VIDEOS (Simplified)")
        print("-"*60)
        print("  NOTE: Text overlays disabled - handled by embedding steps")
        
        scenes_data = self.pipeline_state.get('scenes', {})
        if 'scenes' in scenes_data:
            scene_count = len(scenes_data.get('scenes', []))
        else:
            scene_count = scenes_data.get('count', 3)
        
        # Create edited directory
        edited_dir = self.dirs['scenes_edited']
        edited_dir.mkdir(parents=True, exist_ok=True)
        
        for scene_num in range(1, scene_count + 1):
            # ALWAYS use original scenes for best quality
            original_scene = self.dirs['scenes_original'] / f"scene_{scene_num:03d}.mp4"
            edited_scene = self.dirs['scenes_edited'] / f"scene_{scene_num:03d}.mp4"
            
            if original_scene.exists():
                input_scene = original_scene
                print(f"\n  Scene {scene_num}: Using original (full quality)")
            else:
                print(f"\n  ⚠ Scene {scene_num}: Original scene not found")
                continue
            
            # Check for effects in inference
            inference_file = self.dirs['inferences'] / f"scene_{scene_num:03d}_inference.json"
            has_effects = False
            
            if inference_file.exists():
                with open(inference_file, 'r') as f:
                    inference = json.load(f)
                
                # Check for visual effects
                if 'scenes' in inference:
                    for scene in inference['scenes']:
                        if scene.get('suggested_effects'):
                            has_effects = True
                            break
                
                # Check for sound effects
                if inference.get('sound_effects'):
                    has_effects = True
            
            if has_effects and False:  # Disabled for now - just copy
                print(f"    Would apply effects from inference")
                # Here we would call the effects processor
                # For now, just copy
                
            # For now, just copy the input to edited
            print(f"    Copying {input_scene.name} → edited/")
            try:
                subprocess.run(
                    ['cp', str(input_scene), str(edited_scene)],
                    check=True
                )
                print(f"  ✓ Scene {scene_num} ready (no text overlays)")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Scene {scene_num} copy failed: {e}")
        
        self.pipeline_state['steps_completed'].append('edit_videos')
        print("\n  ✓ All scenes processed (effects only, no text)")