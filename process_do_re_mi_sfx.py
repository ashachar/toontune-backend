#!/usr/bin/env python3
"""
Script to process the do_re_mi video with sound effects.
This is the main entry point for adding sound effects to the video.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from utils.sound_effects.sound_effects_manager import SoundEffectsManager
from utils.sound_effects.video_sound_overlay import VideoSoundOverlay


# The metadata provided by the user
DO_RE_MI_METADATA = {
    "characters_in_video": [
        {
            "name": "Adult Woman",
            "description": "A woman with short, blonde hair, wearing a light-colored blouse and a brown, rustic-style jumper. She is playing an acoustic guitar."
        },
        {
            "name": "Children Group",
            "description": "A group of several children of varying ages, dressed in matching light green and white patterned outfits, sitting on the grass."
        }
    ],
    "video_description": "The video features a woman joyfully singing and playing a guitar for a group of children in a scenic, sunlit meadow against a backdrop of green mountains.",
    "sound_effects": [
        {"sound": "ding", "timestamp": 3.579},
        {"sound": "swoosh", "timestamp": 13.050},
        {"sound": "chime", "timestamp": 17.700},
        {"sound": "chime", "timestamp": 26.180},
        {"sound": "sparkle", "timestamp": 40.119},
        {"sound": "pop", "timestamp": 44.840},
        {"sound": "pop", "timestamp": 48.520},
        {"sound": "pop", "timestamp": 52.479}
    ]
}


def main():
    """Process the do_re_mi video with sound effects."""
    
    # Check for API key
    if not os.environ.get("FREESOUND_API_KEY"):
        print("=" * 60)
        print("SETUP REQUIRED")
        print("=" * 60)
        print("\nTo download sound effects, I need a Freesound API key.")
        print("\nPlease:")
        print("1. Sign up at https://freesound.org/")
        print("2. Get your API key from https://freesound.org/apiv2/apply/")
        print("3. Set the environment variable:")
        print("   export FREESOUND_API_KEY=your_api_key_here")
        print("\nAlternatively, you can manually download sound effects")
        print("and place them in the sound_effects/downloaded/ directory")
        print("=" * 60)
        response = input("\nDo you want to continue without downloading? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Video path
    video_path = "uploads/assets/videos/do_re_mi_with_music/scenes/scene_001.mp4"
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        print("Please ensure the video file exists at the specified path")
        return
    
    print("=" * 60)
    print("PROCESSING DO RE MI VIDEO WITH SOUND EFFECTS")
    print("=" * 60)
    
    # Initialize managers
    print("\n1. Initializing sound effects system...")
    sfx_manager = SoundEffectsManager()
    video_overlay = VideoSoundOverlay()
    
    # Process sound effects (download if API key available)
    print("\n2. Obtaining sound effects...")
    
    # Better search queries for each sound
    sound_queries = {
        "ding": "bell ding bright musical",
        "swoosh": "whoosh swoosh transition smooth",
        "chime": "chime bell magical sparkle",
        "sparkle": "sparkle magic shimmer twinkle",
        "pop": "pop bubble cartoon playful"
    }
    
    sound_files = {}
    for effect in DO_RE_MI_METADATA["sound_effects"]:
        sound_name = effect["sound"].lower()
        timestamp = effect["timestamp"]
        
        print(f"\n   Processing: {sound_name} (at {timestamp}s)")
        
        # Check for existing or download new
        query = sound_queries.get(sound_name, sound_name)
        filepath = sfx_manager.get_or_download_sound(sound_name, query)
        
        if filepath:
            sound_files[sound_name] = filepath
            print(f"   ✓ Ready: {filepath}")
        else:
            print(f"   ⚠ Could not obtain: {sound_name}")
    
    if not sound_files:
        print("\nError: No sound files available. Cannot proceed.")
        return
    
    print(f"\n3. Successfully obtained {len(sound_files)} sound effects")
    
    # Create output directory
    output_dir = Path("output/do_re_mi_with_sfx")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output path
    output_path = str(output_dir / "scene_001_with_sfx.mp4")
    
    # Overlay sound effects
    print(f"\n4. Overlaying sound effects onto video...")
    print(f"   Input:  {video_path}")
    print(f"   Output: {output_path}")
    
    success = video_overlay.overlay_sound_effects(
        video_path,
        DO_RE_MI_METADATA["sound_effects"],
        sound_files,
        output_path,
        preserve_original_audio=True
    )
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"\n✓ Video with sound effects saved to:")
        print(f"  {output_path}")
        
        # Save metadata and attributions
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(DO_RE_MI_METADATA, f, indent=2)
        print(f"\n✓ Metadata saved to:")
        print(f"  {metadata_file}")
        
        # Save attributions
        attributions_file = output_dir / "sound_attributions.txt"
        with open(attributions_file, 'w') as f:
            f.write("SOUND EFFECT ATTRIBUTIONS\n")
            f.write("=" * 50 + "\n")
            f.write("These sound effects were used under Creative Commons licenses.\n")
            f.write("Please include these attributions if you distribute the video.\n\n")
            
            for attr in sfx_manager.get_attributions():
                f.write(attr + "\n\n")
        
        print(f"\n✓ Attributions saved to:")
        print(f"  {attributions_file}")
        
        # Export sound mapping for reference
        mapping_file = output_dir / "sound_mapping.json"
        mapping = sfx_manager.export_sound_mapping(str(mapping_file))
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Added {len(sound_files)} unique sound effects:")
        for name, path in sound_files.items():
            print(f"  • {name:10} -> {Path(path).name}")
        
        print(f"\nTotal sound effect events: {len(DO_RE_MI_METADATA['sound_effects'])}")
        print("\nYou can now play the video with sound effects!")
        
    else:
        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)
        print("Failed to create video with sound effects.")
        print("Please check the error messages above.")


if __name__ == "__main__":
    main()