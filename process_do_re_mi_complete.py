#!/usr/bin/env python3
"""
Complete processor for do_re_mi video with ALL effects:
- Sound effects (with 50% volume and < 0.5s duration)
- Text overlays with animations
- Visual effects (bloom, zoom, light sweep, shake)
- Test mode with debug indicators
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

from utils.video_effects.comprehensive_video_processor import ComprehensiveVideoProcessor


# Complete metadata from user
DO_RE_MI_COMPLETE_METADATA = {
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
    ],
    "scenes": [
        {
            "start_seconds": 0.000,
            "end_seconds": 13.020,
            "scene_description": {
                "characters": ["Adult Woman"],
                "most_prominent_figure": "Adult Woman",
                "character_actions": {
                    "Adult Woman": "She looks down at her guitar, then looks up and begins to sing with a smile."
                },
                "background": "A lush, green meadow with a blurry view of a forest and a castle on a hill in the distance.",
                "camera_angle": "Medium Close-Up",
                "suggested_effects": [
                    {
                        "effect_fn_name": "apply_bloom_effect",
                        "effect_timestamp": 2.000,
                        "effect_fn_params": {
                            "threshold": 180,
                            "bloom_intensity": 1.2,
                            "blur_radius": 15
                        },
                        "top_left_pixels": {"x": 0, "y": 0},
                        "bottom_right_pixels": {"x": 256, "y": 144}
                    }
                ],
                "text_overlays": [
                    {
                        "word": "Let's",
                        "start_seconds": 2.779,
                        "end_seconds": 3.579,
                        "top_left_pixels": {"x": 10, "y": 90},
                        "bottom_right_pixels": {"x": 60, "y": 110},
                        "text_effect": "WordBuildup",
                        "text_effect_params": {"buildup_mode": "fade", "word_delay": 5},
                        "interaction_style": "anchored_to_background"
                    },
                    {
                        "word": "start",
                        "start_seconds": 3.579,
                        "end_seconds": 4.079,
                        "top_left_pixels": {"x": 65, "y": 90},
                        "bottom_right_pixels": {"x": 115, "y": 110},
                        "text_effect": "WordBuildup",
                        "text_effect_params": {"buildup_mode": "fade", "word_delay": 5},
                        "interaction_style": "anchored_to_background"
                    },
                    {
                        "word": "beginning",
                        "start_seconds": 5.280,
                        "end_seconds": 6.420,
                        "top_left_pixels": {"x": 10, "y": 90},
                        "bottom_right_pixels": {"x": 100, "y": 110},
                        "text_effect": "WordBuildup",
                        "text_effect_params": {"buildup_mode": "fade", "word_delay": 5},
                        "interaction_style": "anchored_to_background"
                    }
                ]
            }
        },
        {
            "start_seconds": 13.021,
            "end_seconds": 29.959,
            "scene_description": {
                "characters": ["Adult Woman", "Children Group"],
                "most_prominent_figure": "Adult Woman",
                "character_actions": {
                    "Adult Woman": "Sits centrally, playing her guitar and singing to the children.",
                    "Children Group": "The children are sitting on the grass around her, listening."
                },
                "background": "A vast green hillside overlooking a valley, with mountains in the background.",
                "camera_angle": "Long Shot",
                "suggested_effects": [
                    {
                        "effect_fn_name": "apply_smooth_zoom",
                        "effect_timestamp": 14.000,
                        "effect_fn_params": {
                            "zoom_factor": 1.1,
                            "zoom_type": "in",
                            "easing": "ease_in_out"
                        },
                        "top_left_pixels": {"x": 0, "y": 0},
                        "bottom_right_pixels": {"x": 256, "y": 144}
                    }
                ],
                "text_overlays": [
                    {
                        "word": "A",
                        "start_seconds": 13.020,
                        "end_seconds": 13.720,
                        "top_left_pixels": {"x": 180, "y": 10},
                        "bottom_right_pixels": {"x": 200, "y": 30},
                        "text_effect": "Typewriter",
                        "text_effect_params": {"typing_speed": 2},
                        "interaction_style": "anchored_to_background"
                    },
                    {
                        "word": "B",
                        "start_seconds": 13.720,
                        "end_seconds": 14.119,
                        "top_left_pixels": {"x": 205, "y": 10},
                        "bottom_right_pixels": {"x": 225, "y": 30},
                        "text_effect": "Typewriter",
                        "text_effect_params": {"typing_speed": 2},
                        "interaction_style": "anchored_to_background"
                    },
                    {
                        "word": "C",
                        "start_seconds": 14.439,
                        "end_seconds": 14.460,
                        "top_left_pixels": {"x": 230, "y": 10},
                        "bottom_right_pixels": {"x": 250, "y": 30},
                        "text_effect": "Typewriter",
                        "text_effect_params": {"typing_speed": 2},
                        "interaction_style": "anchored_to_background"
                    },
                    {
                        "word": "Do-Re-Mi",
                        "start_seconds": 17.700,
                        "end_seconds": 18.440,
                        "top_left_pixels": {"x": 10, "y": 10},
                        "bottom_right_pixels": {"x": 100, "y": 30},
                        "text_effect": "WordBuildup",
                        "text_effect_params": {"buildup_mode": "fade", "word_delay": 3},
                        "interaction_style": "anchored_to_background"
                    }
                ]
            }
        },
        {
            "start_seconds": 29.960,
            "end_seconds": 39.000,
            "scene_description": {
                "characters": ["Adult Woman"],
                "most_prominent_figure": "Adult Woman",
                "character_actions": {
                    "Adult Woman": "She sings the musical scale, then looks up thoughtfully."
                },
                "background": "The green meadow with the blurry castle in the background.",
                "camera_angle": "Medium Close-Up",
                "suggested_effects": [
                    {
                        "effect_fn_name": "apply_light_sweep",
                        "effect_timestamp": 37.500,
                        "effect_fn_params": {
                            "sweep_duration": 1.5,
                            "sweep_width": 80,
                            "sweep_angle": 30,
                            "sweep_intensity": 0.4
                        },
                        "top_left_pixels": {"x": 0, "y": 0},
                        "bottom_right_pixels": {"x": 256, "y": 144}
                    }
                ],
                "text_overlays": [
                    {
                        "word": "Do",
                        "start_seconds": 30.360,
                        "end_seconds": 30.760,
                        "top_left_pixels": {"x": 10, "y": 10},
                        "bottom_right_pixels": {"x": 40, "y": 30},
                        "text_effect": "WordBuildup",
                        "text_effect_params": {"buildup_mode": "fade", "word_delay": 1},
                        "interaction_style": "floating_with_character"
                    },
                    {
                        "word": "Re",
                        "start_seconds": 31.059,
                        "end_seconds": 31.459,
                        "top_left_pixels": {"x": 45, "y": 10},
                        "bottom_right_pixels": {"x": 75, "y": 30},
                        "text_effect": "WordBuildup",
                        "text_effect_params": {"buildup_mode": "fade", "word_delay": 1},
                        "interaction_style": "floating_with_character"
                    },
                    {
                        "word": "Mi",
                        "start_seconds": 31.159,
                        "end_seconds": 31.379,
                        "top_left_pixels": {"x": 80, "y": 10},
                        "bottom_right_pixels": {"x": 110, "y": 30},
                        "text_effect": "WordBuildup",
                        "text_effect_params": {"buildup_mode": "fade", "word_delay": 1},
                        "interaction_style": "floating_with_character"
                    },
                    {
                        "word": "easier",
                        "start_seconds": 36.659,
                        "end_seconds": 37.659,
                        "top_left_pixels": {"x": 170, "y": 90},
                        "bottom_right_pixels": {"x": 246, "y": 110},
                        "text_effect": "WordBuildup",
                        "text_effect_params": {"buildup_mode": "fade", "word_delay": 5},
                        "interaction_style": "anchored_to_background"
                    }
                ]
            }
        },
        {
            "start_seconds": 39.001,
            "end_seconds": 54.759,
            "scene_description": {
                "characters": ["Adult Woman"],
                "most_prominent_figure": "Adult Woman",
                "character_actions": {
                    "Adult Woman": "She sings with great expression, defining each musical note."
                },
                "background": "The green meadow with the blurry castle in the background.",
                "camera_angle": "Medium Close-Up",
                "suggested_effects": [
                    {
                        "effect_fn_name": "apply_handheld_shake",
                        "effect_timestamp": 40.000,
                        "effect_fn_params": {
                            "shake_intensity": 1.0,
                            "shake_frequency": 1.5,
                            "rotation_amount": 0.5,
                            "smooth_motion": True
                        },
                        "top_left_pixels": {"x": 0, "y": 0},
                        "bottom_right_pixels": {"x": 256, "y": 144}
                    }
                ],
                "text_overlays": [
                    {
                        "word": "Doe, a deer",
                        "start_seconds": 40.119,
                        "end_seconds": 41.939,
                        "top_left_pixels": {"x": 10, "y": 90},
                        "bottom_right_pixels": {"x": 120, "y": 110},
                        "text_effect": "SplitText",
                        "text_effect_params": {
                            "split_mode": "word",
                            "split_direction": "horizontal",
                            "split_distance": 50
                        },
                        "interaction_style": "anchored_to_background"
                    },
                    {
                        "word": "Ray, a drop of golden sun",
                        "start_seconds": 44.840,
                        "end_seconds": 47.200,
                        "top_left_pixels": {"x": 100, "y": 10},
                        "bottom_right_pixels": {"x": 250, "y": 30},
                        "text_effect": "SplitText",
                        "text_effect_params": {
                            "split_mode": "word",
                            "split_direction": "vertical",
                            "split_distance": 30
                        },
                        "interaction_style": "anchored_to_background"
                    },
                    {
                        "word": "Me, a name I call myself",
                        "start_seconds": 48.520,
                        "end_seconds": 51.680,
                        "top_left_pixels": {"x": 10, "y": 90},
                        "bottom_right_pixels": {"x": 220, "y": 110},
                        "text_effect": "SplitText",
                        "text_effect_params": {
                            "split_mode": "word",
                            "split_direction": "horizontal",
                            "split_distance": 50
                        },
                        "interaction_style": "anchored_to_background"
                    },
                    {
                        "word": "Far, a long, long way to run",
                        "start_seconds": 52.479,
                        "end_seconds": 54.759,
                        "top_left_pixels": {"x": 80, "y": 10},
                        "bottom_right_pixels": {"x": 250, "y": 30},
                        "text_effect": "SplitText",
                        "text_effect_params": {
                            "split_mode": "word",
                            "split_direction": "vertical",
                            "split_distance": 30
                        },
                        "interaction_style": "anchored_to_background"
                    }
                ]
            }
        }
    ]
}


def main():
    """Process do_re_mi video with ALL effects."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process do_re_mi video with complete effects"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=True,
        help="Enable test mode with debug overlays (default: True)"
    )
    parser.add_argument(
        "--sound-volume",
        type=float,
        default=0.5,
        help="Sound effects volume (0.0 to 1.0, default 0.5)"
    )
    parser.add_argument(
        "--output-dir",
        default="output/do_re_mi_complete",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("FREESOUND_API_KEY"):
        print("Warning: FREESOUND_API_KEY not found")
        print("Sound effects may not download properly")
    
    # Video path
    video_path = "uploads/assets/videos/do_re_mi_with_music/scenes/scene_001.mp4"
    
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output path
    output_path = str(output_dir / "scene_001_complete.mp4")
    
    print("="*70)
    print("PROCESSING DO RE MI VIDEO WITH COMPLETE EFFECTS")
    print("="*70)
    print(f"Test Mode: {'ENABLED' if args.test else 'DISABLED'}")
    print(f"Sound Volume: {args.sound_volume * 100}%")
    print(f"Output: {output_path}")
    print()
    
    # Save complete metadata
    metadata_file = output_dir / "complete_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(DO_RE_MI_COMPLETE_METADATA, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_file}")
    
    # Create processor with test mode
    processor = ComprehensiveVideoProcessor(test_mode=args.test)
    
    # Process video
    success = processor.process_video(
        video_path,
        DO_RE_MI_COMPLETE_METADATA,
        output_path,
        sound_volume=args.sound_volume
    )
    
    if success:
        print("\n" + "="*70)
        print("SUCCESS - ALL EFFECTS APPLIED!")
        print("="*70)
        print("\n✓ Video with complete effects saved to:")
        print(f"  {output_path}")
        
        # Summary
        total_text_overlays = sum(
            len(scene["scene_description"].get("text_overlays", []))
            for scene in DO_RE_MI_COMPLETE_METADATA["scenes"]
        )
        total_visual_effects = sum(
            len(scene["scene_description"].get("suggested_effects", []))
            for scene in DO_RE_MI_COMPLETE_METADATA["scenes"]
        )
        total_sound_effects = len(DO_RE_MI_COMPLETE_METADATA["sound_effects"])
        
        print("\nEffects Applied:")
        print(f"  • {total_text_overlays} Text Overlays")
        print(f"  • {total_visual_effects} Visual Effects (bloom, zoom, etc.)")
        print(f"  • {total_sound_effects} Sound Effects (at {args.sound_volume*100}% volume)")
        
        if args.test:
            print("\n✓ Test Mode: Debug indicators added to show applied effects")
    else:
        print("\n✗ Processing failed")


if __name__ == "__main__":
    main()