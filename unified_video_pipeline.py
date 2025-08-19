#!/usr/bin/env python3
"""
Unified Video Processing Pipeline
==================================

This is the main entry point for the unified video processing pipeline.
All functionality has been organized into modular components in the pipeline/ directory.

Pipeline Structure:
- pipeline/core/config.py       - Configuration dataclass
- pipeline/core/pipeline.py     - Main pipeline orchestrator  
- pipeline/steps/               - Individual step modules
  - step_1_downsample.py       - Video downsampling
  - step_2_transcripts.py      - Transcript generation
  - step_3_scenes.py           - Scene splitting
  - step_4_prompts.py          - Prompt generation (V2: key phrases & cartoons)
  - step_5_inference.py        - LLM inference
  - step_6_edit_videos.py      - Video editing with effects
  - step_7_karaoke.py          - Karaoke caption generation

Directory Structure Created:
uploads/assets/videos/{video_name}/
├── video.mp4                    # Original video
├── video_downsampled.mp4        # Downsampled full video
├── transcripts/
│   ├── transcript_sentences.json
│   └── transcript_words.json
├── scenes/
│   ├── original/               # Full resolution scenes
│   │   ├── scene_001.mp4
│   │   └── ...
│   ├── downsampled/            # Downsampled scenes (parallel to original/)
│   │   ├── scene_001.mp4
│   │   └── ...
│   ├── edited/                 # Edited scenes with effects
│   │   ├── scene_001.mp4
│   │   └── ...
│   └── karaoke/                # Karaoke versions
│       ├── scene_001.mp4
│       └── ...
├── prompts/
│   ├── scene_001_prompt.txt    # V2: key phrases & cartoon characters
│   └── ...
├── inferences/
│   ├── scene_001_inference.json
│   └── ...
└── metadata/
    └── pipeline_state.json

Usage:
    # Full pipeline
    python unified_video_pipeline.py video.mp4
    
    # Skip specific steps
    python unified_video_pipeline.py video.mp4 --no-downsample --no-transcript
    
    # Dry run (skip LLM inference)
    python unified_video_pipeline.py video.mp4 --dry-run
    
    # Generate karaoke
    python unified_video_pipeline.py video.mp4 --karaoke
"""

import sys
import json
import argparse
import traceback

# Import the pipeline components
from pipeline import PipelineConfig, UnifiedVideoPipeline


def main():
    """Main entry point for the unified pipeline."""
    parser = argparse.ArgumentParser(
        description="Unified Video Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a new video completely
  python unified_video_pipeline.py my_video.mp4
  
  # Regenerate prompts and run inference only
  python unified_video_pipeline.py my_video.mp4 --no-downsample --no-transcript --no-scenes
  
  # Test mode with effect labels
  python unified_video_pipeline.py my_video.mp4 --test-mode
  
  # Generate karaoke version
  python unified_video_pipeline.py my_video.mp4 --karaoke
        """
    )
    
    parser.add_argument(
        "video",
        help="Path to the input video file"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-base",
        default="uploads/assets/videos",
        help="Base directory for output (default: uploads/assets/videos)"
    )
    
    # Processing options
    parser.add_argument(
        "--downsample-width",
        type=int,
        default=256,
        help="Width for downsampled videos (default: 256)"
    )
    parser.add_argument(
        "--target-scene-duration",
        type=int,
        default=60,
        help="Target duration for each scene in seconds (default: 60)"
    )
    
    # Skip flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM inference only (still does transcript, scenes, etc.)"
    )
    parser.add_argument(
        "--no-downsample",
        action="store_true",
        help="Skip video downsampling"
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Skip transcript generation (use existing or mock)"
    )
    parser.add_argument(
        "--no-scenes",
        action="store_true",
        help="Skip scene splitting"
    )
    parser.add_argument(
        "--no-prompts",
        action="store_true",
        help="Skip prompt generation"
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip LLM inference (same as --dry-run)"
    )
    parser.add_argument(
        "--no-editing",
        action="store_true",
        help="Skip video editing"
    )
    parser.add_argument(
        "--no-embed-phrases",
        action="store_true",
        help="Skip key phrase embedding"
    )
    parser.add_argument(
        "--no-embed-cartoons",
        action="store_true",
        help="Skip cartoon character embedding"
    )
    
    # Feature flags
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Enable test mode to show effect labels on video"
    )
    parser.add_argument(
        "--karaoke",
        action="store_true",
        help="Generate karaoke-style captions for videos"
    )
    parser.add_argument(
        "--karaoke-style",
        choices=["continuous", "simple"],
        default="continuous",
        help="Karaoke caption style (default: continuous for no flickering)"
    )
    parser.add_argument(
        "--no-karaoke",
        action="store_true",
        help="Skip karaoke caption generation (overrides --karaoke)"
    )
    
    # V2 Prompt options
    parser.add_argument(
        "--use-v2-prompts",
        action="store_true",
        help="Use V2 prompts with key phrases and cartoon characters instead of word-by-word"
    )
    
    args = parser.parse_args()
    
    # Create pipeline configuration
    config = PipelineConfig(
        video_path=args.video,
        output_base_dir=args.output_base,
        downsample_width=args.downsample_width,
        dry_run=args.dry_run,
        downsample_video=not args.no_downsample,
        generate_transcripts=not args.no_transcript,
        split_scenes=not args.no_scenes,
        generate_prompts=not args.no_prompts,
        run_inference=not (args.dry_run or args.no_inference),
        edit_videos=not args.no_editing,
        target_scene_duration=args.target_scene_duration,
        test_mode=args.test_mode,
        generate_karaoke=args.karaoke and not args.no_karaoke,
        karaoke_style=args.karaoke_style,
        embed_phrases=not args.no_embed_phrases,
        embed_cartoons=not args.no_embed_cartoons
    )
    
    # Create and run pipeline
    pipeline = UnifiedVideoPipeline(config)
    
    try:
        pipeline.run()
        
        # Print summary
        summary = pipeline.get_summary()
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        print(json.dumps(summary, indent=2))
        
        # Print next steps hint
        if config.dry_run:
            print("\n💡 Dry run complete. To run inference:")
            print(f"   python unified_video_pipeline.py {args.video} --no-downsample --no-transcript --no-scenes --no-prompts")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()