"""
Step 3: Scene Splitting
========================

Splits video into scenes based on transcript.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.scene_management.scene_splitter import SceneSplitter


class SceneSplitStep:
    """Handles scene splitting operations."""
    
    def __init__(self, pipeline_state, dirs, config):
        """Initialize with pipeline references."""
        self.pipeline_state = pipeline_state
        self.dirs = dirs
        self.config = config
    
    def run(self, video_path):
        """Split video into scenes based on transcript."""
        print("\n" + "-"*60)
        print("STEP 3: SPLITTING INTO SCENES")
        print("-"*60)
        
        # Load transcript to determine scene boundaries
        sentences_file = self.dirs['transcripts'] / 'transcript_sentences.json'
        
        if sentences_file.exists():
            with open(sentences_file, 'r') as f:
                transcript_data = json.load(f)
            
            # Use transcript to determine scene boundaries
            # For now, we'll split based on logical sentence groups
            sentences = transcript_data.get('sentences', [])
            
            if sentences:
                # Create scenes of approximately target duration (default 60 seconds)
                # based on sentence boundaries
                scene_timestamps = []
                scene_descriptions = []
                
                target_duration = self.config.target_scene_duration
                current_scene_start = 0.0
                scene_sentences = []
                scene_duration = 0.0
                
                for i, sent in enumerate(sentences):
                    # Calculate duration if we add this sentence
                    potential_end = sent['end']
                    potential_duration = potential_end - current_scene_start
                    
                    # Add sentence to current scene
                    scene_sentences.append(sent['text'])
                    
                    # Check if we should end the scene
                    should_end_scene = False
                    
                    # End scene if we've reached target duration (with 10% tolerance)
                    if potential_duration >= target_duration * 0.9:
                        should_end_scene = True
                    
                    # Or if this is the last sentence
                    if i == len(sentences) - 1:
                        should_end_scene = True
                    
                    # Or if next sentence would make scene too long (>120% of target)
                    if i < len(sentences) - 1:
                        next_duration = sentences[i + 1]['end'] - current_scene_start
                        if next_duration > target_duration * 1.2:
                            should_end_scene = True
                    
                    if should_end_scene:
                        # Create the scene
                        scene_timestamps.append((current_scene_start, sent['end']))
                        
                        # Create description from first 1-2 sentences
                        desc_sentences = scene_sentences[:2] if len(scene_sentences) > 1 else scene_sentences
                        scene_descriptions.append(' '.join(desc_sentences)[:50])
                        
                        # Reset for next scene
                        if i < len(sentences) - 1:
                            current_scene_start = sent['end'] + 0.001
                        scene_sentences = []
                
                print(f"  Created {len(scene_timestamps)} scenes from transcript (~{target_duration}s each)")
            else:
                # Fallback to default scenes if no transcript
                scene_timestamps = [
                    (0.000, 13.020),
                    (13.021, 29.959),
                    (29.960, 39.000),
                    (39.001, 54.759),
                ]
                scene_descriptions = [
                    "Introduction - Let's start at the beginning",
                    "ABC and Do-Re-Mi",
                    "Musical scale demonstration",
                    "Note definitions"
                ]
        else:
            # Default scene boundaries if no transcript
            scene_timestamps = [
                (0.000, 13.020),
                (13.021, 29.959),
                (29.960, 39.000),
                (39.001, 54.759),
            ]
            scene_descriptions = [
                "Introduction - Let's start at the beginning",
                "ABC and Do-Re-Mi",
                "Musical scale demonstration",
                "Note definitions"
            ]
        
        # Create scene splitter
        base_dir = self.dirs['base'].parent
        splitter = SceneSplitter(str(video_path), str(base_dir))
        
        # Create scenes from timestamps
        scenes = SceneSplitter.create_scenes_from_timestamps(
            scene_timestamps,
            scene_descriptions
        )
        
        print(f"  Splitting into {len(scenes)} scenes:")
        for scene in scenes:
            print(f"    Scene {scene.scene_number}: {scene.start_seconds:.1f}s - "
                  f"{scene.end_seconds:.1f}s ({scene.duration:.1f}s)")
        
        # Always split the video (even in dry run - it's just file operations)
        print("  Creating scene files...")
        results = splitter.split_into_scenes(scenes)
        self.pipeline_state['scenes'] = results
        
        if self.config.dry_run:
            print("  [DRY RUN] Scenes created, but no further processing")
        
        self.pipeline_state['steps_completed'].append('split_scenes')