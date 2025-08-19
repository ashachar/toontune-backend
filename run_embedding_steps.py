#!/usr/bin/env python3
"""
Run just the embedding steps in correct order on existing edited videos
"""

import sys
from pathlib import Path

# Add to path for imports
sys.path.append(str(Path(__file__).parent))

from pipeline.steps.step_7_karaoke import KaraokeStep
from pipeline.steps.step_8_embed_phrases import EmbedPhrasesStep
from pipeline.steps.step_9_embed_cartoons import EmbedCartoonsStep

def main():
    print("="*70)
    print("RUNNING EMBEDDING STEPS IN CORRECTED ORDER")
    print("="*70)
    
    # Set up paths
    video_dir = Path("uploads/assets/videos/do_re_mi")
    
    # Create config object
    class Config:
        def __init__(self):
            self.video_dir = video_dir
            self.video_name = "do_re_mi"
            self.karaoke_style = "default"
    
    config = Config()
    
    # Create pipeline state and dirs for karaoke
    pipeline_state = {'steps_completed': []}
    dirs = {
        'transcripts': video_dir / 'transcripts',
        'scenes': video_dir / 'scenes',
        'scenes_edited': video_dir / 'scenes' / 'edited',
        'metadata': video_dir / 'metadata'
    }
    
    print("\nOrder of execution:")
    print("1. Karaoke (base layer)")
    print("2. Key phrases (on top of karaoke)")
    print("3. Cartoon characters (on top of everything)")
    print()
    
    # Step 1: Karaoke FIRST (base layer)
    print("\n[STEP 7] Generating Karaoke Captions")
    print("-"*40)
    karaoke_step = KaraokeStep(pipeline_state, dirs, config)
    karaoke_step.run(use_final_with_cartoons=False)
    
    # Step 2: Embed phrases on top of karaoke
    print("\n[STEP 8] Embedding Key Phrases")
    print("-"*40)
    phrases_step = EmbedPhrasesStep(config)
    phrases_step.run()
    
    # Step 3: Embed cartoons on top of everything
    print("\n[STEP 9] Embedding Cartoon Characters")
    print("-"*40)
    cartoons_step = EmbedCartoonsStep(config)
    cartoons_step.run()
    
    print("\n" + "="*70)
    print("✅ EMBEDDING COMPLETE")
    print("="*70)
    print(f"Final videos with ALL features: {video_dir / 'scenes/edited'}")
    print("\nFeatures layered in order:")
    print("  1. Karaoke captions (bottom layer)")
    print("  2. Key phrases")
    print("  3. Cartoon characters (top layer)")

if __name__ == "__main__":
    main()