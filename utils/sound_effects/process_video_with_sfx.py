#!/usr/bin/env python3
"""
Main script to process videos with sound effects.
Reads video metadata JSON and applies sound effects at specified timestamps.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.sound_effects.sound_effects_manager import SoundEffectsManager
from utils.sound_effects.video_sound_overlay import VideoSoundOverlay


def load_video_metadata(json_path: str) -> Dict:
    """Load video metadata from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def process_video_with_sound_effects(
    video_path: str,
    metadata: Dict,
    output_dir: str = "output/videos_with_sfx",
    download_sounds: bool = True
) -> Optional[str]:
    """
    Process a video with sound effects based on metadata.
    
    Args:
        video_path: Path to input video
        metadata: Video metadata dictionary containing sound_effects list
        output_dir: Directory for output video
        download_sounds: Whether to download missing sound effects
    
    Returns:
        Path to output video or None if failed
    """
    # Extract sound effects from metadata
    sound_effects = metadata.get("sound_effects", [])
    
    if not sound_effects:
        print("No sound effects found in metadata")
        return None
    
    print(f"Processing video: {video_path}")
    print(f"Found {len(sound_effects)} sound effects to add")
    
    # Initialize managers
    sfx_manager = SoundEffectsManager()
    video_overlay = VideoSoundOverlay()
    
    # Process sound effects (download if needed)
    if download_sounds:
        print("\n1. Obtaining sound effects...")
        sound_files = sfx_manager.process_sound_effects_list(sound_effects)
    else:
        # Just look for existing files
        sound_files = {}
        for effect in sound_effects:
            sound_name = effect.get("sound", "").lower()
            existing = sfx_manager.find_existing_sound(sound_name)
            if existing:
                sound_files[sound_name] = existing
    
    if not sound_files:
        print("No sound files available")
        return None
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    video_name = Path(video_path).stem
    output_path = str(output_dir / f"{video_name}_with_sfx.mp4")
    
    # Overlay sound effects
    print(f"\n2. Overlaying {len(sound_files)} sound effects onto video...")
    success = video_overlay.overlay_sound_effects(
        video_path,
        sound_effects,
        sound_files,
        output_path,
        preserve_original_audio=True
    )
    
    if success:
        print(f"\n✓ Success! Video saved to: {output_path}")
        
        # Save attributions
        attributions_file = output_dir / f"{video_name}_attributions.txt"
        with open(attributions_file, 'w') as f:
            f.write("Sound Effect Attributions\n")
            f.write("=" * 50 + "\n\n")
            for attr in sfx_manager.get_attributions():
                f.write(attr + "\n")
        print(f"✓ Attributions saved to: {attributions_file}")
        
        return output_path
    else:
        print("\n✗ Failed to create video with sound effects")
        return None


def process_scene_videos(
    scenes_dir: str,
    metadata: Dict,
    output_dir: str = "output/scenes_with_sfx"
) -> List[str]:
    """
    Process all scene videos with sound effects.
    
    Args:
        scenes_dir: Directory containing scene video files
        metadata: Video metadata dictionary
        output_dir: Directory for output videos
    
    Returns:
        List of output video paths
    """
    scenes_path = Path(scenes_dir)
    if not scenes_path.exists():
        print(f"Scenes directory not found: {scenes_dir}")
        return []
    
    # Get all scene videos
    scene_videos = sorted(scenes_path.glob("scene_*.mp4"))
    if not scene_videos:
        print(f"No scene videos found in: {scenes_dir}")
        return []
    
    print(f"Found {len(scene_videos)} scene videos")
    
    results = []
    for video_path in scene_videos:
        print(f"\n{'='*60}")
        print(f"Processing scene: {video_path.name}")
        print('='*60)
        
        output_path = process_video_with_sound_effects(
            str(video_path),
            metadata,
            output_dir
        )
        
        if output_path:
            results.append(output_path)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process videos with sound effects based on metadata JSON"
    )
    parser.add_argument(
        "input",
        help="Input video file or scenes directory"
    )
    parser.add_argument(
        "--metadata",
        help="Path to metadata JSON file (default: uses provided do_re_mi metadata)",
        default=None
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for processed videos",
        default="output/videos_with_sfx"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download missing sound effects"
    )
    parser.add_argument(
        "--process-scenes",
        action="store_true",
        help="Process all scene videos in directory"
    )
    
    args = parser.parse_args()
    
    # Load metadata
    if args.metadata:
        metadata = load_video_metadata(args.metadata)
    else:
        # Use the provided do_re_mi metadata
        metadata = {
            "sound_effects": [
                {"sound": "ding", "timestamp": "3.579"},
                {"sound": "swoosh", "timestamp": "13.050"},
                {"sound": "chime", "timestamp": "17.700"},
                {"sound": "chime", "timestamp": "26.180"},
                {"sound": "sparkle", "timestamp": "40.119"},
                {"sound": "pop", "timestamp": "44.840"},
                {"sound": "pop", "timestamp": "48.520"},
                {"sound": "pop", "timestamp": "52.479"}
            ]
        }
    
    # Process video(s)
    if args.process_scenes:
        # Process all scenes in directory
        results = process_scene_videos(
            args.input,
            metadata,
            args.output_dir
        )
        
        if results:
            print(f"\n{'='*60}")
            print(f"✓ Successfully processed {len(results)} videos")
            print("Output files:")
            for path in results:
                print(f"  - {path}")
        else:
            print("\n✗ No videos were successfully processed")
    else:
        # Process single video
        result = process_video_with_sound_effects(
            args.input,
            metadata,
            args.output_dir,
            download_sounds=not args.no_download
        )
        
        if result:
            print(f"\n✓ Processing complete!")
        else:
            print("\n✗ Processing failed")
            sys.exit(1)


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("FREESOUND_API_KEY"):
        print("Warning: FREESOUND_API_KEY not set in environment")
        print("Please set it to download sound effects from Freesound")
        print("Example: export FREESOUND_API_KEY=your_api_key_here")
        print()
    
    main()