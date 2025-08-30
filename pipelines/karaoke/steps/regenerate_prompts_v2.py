#!/usr/bin/env python3
"""
Regenerate Prompts with V2 (Key Phrases and Cartoon Characters)
================================================================

This script regenerates all prompts for existing scenes using the new approach:
- Key phrases (max 4 words) instead of individual words
- Cartoon character suggestions related to content
"""

import sys
import json
import argparse
from pathlib import Path

# Import pipeline components
from karaoke import PipelineConfig
from karaoke.steps.step_4_prompts_v2 import PromptsStepV2


def regenerate_prompts_for_video(video_name: str, base_dir: str = "uploads/assets/videos"):
    """Regenerate prompts for a specific video using V2 approach"""
    
    print("=" * 70)
    print("PROMPT REGENERATION V2 - KEY PHRASES & CARTOON CHARACTERS")
    print("=" * 70)
    print(f"Video: {video_name}")
    print(f"Base directory: {base_dir}")
    print("-" * 70)
    
    # Setup directories
    video_dir = Path(base_dir) / video_name
    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return False
    
    dirs = {
        'base': video_dir,
        'transcripts': video_dir / 'transcripts',
        'scenes': video_dir / 'scenes',
        'prompts': video_dir / 'prompts',
        'metadata': video_dir / 'metadata'
    }
    
    # Check required directories exist
    for name, path in dirs.items():
        if not path.exists():
            if name == 'prompts':
                path.mkdir(parents=True, exist_ok=True)
                print(f"  Created prompts directory: {path}")
            else:
                print(f"❌ Required directory not found: {path}")
                return False
    
    # Load existing pipeline state
    pipeline_state = {
        'steps_completed': [],
        'scenes': {}
    }
    
    # Load scenes metadata
    scenes_file = dirs['metadata'] / 'scenes.json'
    if scenes_file.exists():
        with open(scenes_file, 'r') as f:
            pipeline_state['scenes'] = json.load(f)
        print(f"✅ Loaded scenes metadata: {len(pipeline_state['scenes'].get('scenes', []))} scenes")
    else:
        print(f"❌ Scenes metadata not found: {scenes_file}")
        return False
    
    # Check for transcript
    words_file = dirs['transcripts'] / 'transcript_words.json'
    if words_file.exists():
        print(f"✅ Found word-level transcript: {words_file}")
    else:
        print(f"⚠️ No word-level transcript found - prompts will be generated without lyrics")
    
    # Create config - use a dummy video path since we're only regenerating prompts
    dummy_video = dirs['base'] / 'video.mp4'
    config = PipelineConfig(
        video_path=str(dummy_video),
        downsample_width=256,
        target_scene_duration=60,
        dry_run=False
    )
    
    # Initialize and run the V2 prompts step
    print("\n" + "=" * 70)
    print("GENERATING V2 PROMPTS")
    print("=" * 70)
    
    prompts_step = PromptsStepV2(pipeline_state, dirs, config)
    prompts_step.run()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    num_prompts = pipeline_state.get('prompts_generated', 0)
    print(f"✅ Generated {num_prompts} V2 prompts")
    
    # List generated prompts
    prompt_files = sorted(dirs['prompts'].glob('scene_*_prompt.txt'))
    if prompt_files:
        print("\nGenerated prompt files:")
        for pf in prompt_files[:5]:  # Show first 5
            size_kb = pf.stat().st_size / 1024
            print(f"  • {pf.name} ({size_kb:.1f} KB)")
        if len(prompt_files) > 5:
            print(f"  ... and {len(prompt_files) - 5} more")
    
    # Show key differences
    print("\n📝 KEY CHANGES IN V2 PROMPTS:")
    print("  ✓ Removed word-by-word overlay instructions")
    print("  ✓ Added key_phrases field (max 4 words, once per 20 seconds)")
    print("  ✓ Added cartoon_characters field (related to content)")
    print("  ✓ Focus on enhancing, not cluttering, the video")
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Regenerate prompts with V2 approach (key phrases and cartoon characters)"
    )
    parser.add_argument(
        "--video",
        default="do_re_mi_with_music",
        help="Video name/directory (default: do_re_mi_with_music)"
    )
    parser.add_argument(
        "--base-dir",
        default="uploads/assets/videos",
        help="Base directory for videos (default: uploads/assets/videos)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerate prompts for all videos in base directory"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Process all videos
        base_path = Path(args.base_dir)
        if not base_path.exists():
            print(f"❌ Base directory not found: {base_path}")
            sys.exit(1)
        
        video_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        print(f"Found {len(video_dirs)} video directories")
        
        success_count = 0
        for video_dir in video_dirs:
            video_name = video_dir.name
            print(f"\n{'=' * 70}")
            print(f"Processing: {video_name}")
            
            if regenerate_prompts_for_video(video_name, args.base_dir):
                success_count += 1
            else:
                print(f"⚠️ Skipped {video_name}")
        
        print(f"\n{'=' * 70}")
        print(f"FINAL SUMMARY: {success_count}/{len(video_dirs)} videos processed")
    else:
        # Process single video
        success = regenerate_prompts_for_video(args.video, args.base_dir)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()