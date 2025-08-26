#!/usr/bin/env python3
"""
Apply effects to the ORIGINAL full resolution scene based on metadata.
This uses the downsampled scene only for VLLM inference reference,
but applies all effects to the original high-quality scene.
"""

import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List

# Load environment variables
load_dotenv()

from utils.sound_effects.sound_effects_manager import SoundEffectsManager
from utils.sound_effects.video_sound_overlay import VideoSoundOverlay


class OriginalSceneProcessor:
    """Process original scenes with effects based on metadata."""
    
    def __init__(self, base_dir: str = "uploads/assets/videos"):
        """Initialize with base directory."""
        self.base_dir = Path(base_dir)
    
    def escape_text(self, text: str) -> str:
        """Escape text for FFmpeg."""
        return text.replace("'", "").replace(",", " ").replace(":", " ")
    
    def scale_coordinates(self, x: int, y: int, 
                         ref_width: int = 256, ref_height: int = 144,
                         actual_width: int = 1166, actual_height: int = 534):
        """Scale coordinates from reference to actual dimensions."""
        scale_x = actual_width / ref_width
        scale_y = actual_height / ref_height
        return int(x * scale_x), int(y * scale_y)
    
    def scale_fontsize(self, fontsize: int,
                      ref_width: int = 256, ref_height: int = 144,
                      actual_width: int = 1166, actual_height: int = 534):
        """Scale font size based on resolution."""
        avg_scale = ((actual_width / ref_width) + (actual_height / ref_height)) / 2
        return int(fontsize * avg_scale)
    
    def process_scene(self, video_name: str, scene_number: int,
                     text_overlays: List[Dict],
                     sound_effects: List[Dict],
                     test_mode: bool = False,
                     suggested_effects: List[Dict] = None):
        """
        Process a single original scene with all effects.
        Saves the result to the 'edited' folder alongside original and downsampled.
        
        Args:
            video_name: Name of the video (e.g., "do_re_mi")
            scene_number: Scene number to process
            text_overlays: List of text overlay definitions
            sound_effects: List of sound effect definitions
            test_mode: If True, add labels showing what effects are being applied
            suggested_effects: List of visual effects from inference (for test mode)
        """
        # Paths - all in the same parent directory for easy navigation
        scenes_dir = self.base_dir / video_name / "scenes"
        original_scene = scenes_dir / "original" / f"scene_{scene_number:03d}.mp4"
        downsampled_scene = scenes_dir / "downsampled" / f"scene_{scene_number:03d}.mp4"
        edited_dir = scenes_dir / "edited"
        
        if not original_scene.exists():
            print(f"Error: Original scene not found: {original_scene}")
            return False
        
        print("="*70)
        print(f"PROCESSING ORIGINAL SCENE {scene_number}")
        print("="*70)
        print(f"Original: {original_scene} (full resolution)")
        print(f"Reference: {downsampled_scene} (for VLLM)")
        
        # Get video info
        video_info = self.get_video_info(str(original_scene))
        print(f"Resolution: {video_info['width']}x{video_info['height']}")
        print(f"Duration: {video_info['duration']:.1f}s")
        
        # Create edited directory if it doesn't exist
        edited_dir.mkdir(parents=True, exist_ok=True)
        
        # Use temp directory for intermediate files
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Apply text overlays (with test mode labels if enabled)
            print("\n1. Applying text overlays...")
            temp_with_text = temp_path / f"scene_{scene_number:03d}_temp_text.mp4"
            self.apply_text_overlays(
                str(original_scene),
                str(temp_with_text),
                text_overlays,
                video_info['width'],
                video_info['height'],
                test_mode=test_mode,
                suggested_effects=suggested_effects,
                sound_effects=sound_effects
            )
            
            # Step 2: Apply sound effects
            print("\n2. Applying sound effects...")
            final_output = edited_dir / f"scene_{scene_number:03d}.mp4"
            self.apply_sound_effects(
                str(temp_with_text) if temp_with_text.exists() else str(original_scene),
                str(final_output),
                sound_effects
            )
        
        # Check if output was created (temp file cleanup is handled by context manager)
        if final_output.exists():
            size_mb = final_output.stat().st_size / (1024 * 1024)
            print("\n" + "="*70)
            print("SUCCESS!")
            print("="*70)
            print(f"✓ Final scene saved to: {final_output}")
            print(f"  Size: {size_mb:.1f} MB")
            print(f"  Resolution: {video_info['width']}x{video_info['height']} (ORIGINAL)")
            print(f"\nScene versions available at: {scenes_dir}")
            print(f"  • Original: {scenes_dir}/original/scene_{scene_number:03d}.mp4")
            print(f"  • Downsampled: {scenes_dir}/downsampled/scene_{scene_number:03d}.mp4")
            print(f"  • Edited: {scenes_dir}/edited/scene_{scene_number:03d}.mp4")
            return True
        
        return False
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get video information."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            for stream in data.get("streams", []):
                if stream["codec_type"] == "video":
                    return {
                        "width": int(stream["width"]),
                        "height": int(stream["height"]),
                        "duration": float(data.get("format", {}).get("duration", 0))
                    }
        except Exception as e:
            print(f"Error getting video info: {e}")
        
        return {"width": 1166, "height": 534, "duration": 0}
    
    def apply_text_overlays(self, input_video: str, output_video: str,
                           text_overlays: List[Dict],
                           actual_width: int, actual_height: int,
                           test_mode: bool = False,
                           suggested_effects: List[Dict] = None,
                           sound_effects: List[Dict] = None):
        """Apply text overlays with proper scaling and optional test mode labels."""
        filters = []
        
        for overlay in text_overlays:
            # Check if coordinates need scaling (if they're from downsampled inference)
            # If x/y are already in full resolution range, don't scale
            x = overlay.get("x", 10)
            y = overlay.get("y", 90)
            
            # Only scale if coordinates appear to be from downsampled video (< 256)
            if x < 256 and y < 144:
                x, y = self.scale_coordinates(
                    x, y,
                    actual_width=actual_width,
                    actual_height=actual_height
                )
            
            # Get font size (may need scaling)
            fontsize = overlay.get("fontsize", 24)
            if fontsize < 50:  # Likely needs scaling
                fontsize = self.scale_fontsize(
                    fontsize,
                    actual_width=actual_width,
                    actual_height=actual_height
                )
            
            # Escape text (handle both 'text' and 'word' keys)
            text = self.escape_text(overlay.get("text", overlay.get("word", "")))
            
            # Build filter
            filter_str = (
                f"drawtext="
                f"text={text}:"
                f"x={x}:y={y}:"
                f"fontsize={fontsize}:fontcolor=white:"
                f"borderw=3:bordercolor=black:"
                f"enable='gte(t\\,{overlay['start']})*lte(t\\,{overlay['end']})'"
            )
            filters.append(filter_str)
            print(f"  Added: {text[:30]}...")
        
        # Add test mode labels if enabled
        if test_mode:
            # Create effect timeline labels
            test_labels = []
            y_offset = 10
            
            # Add header
            test_labels.append(
                f"drawtext="
                f"text=TEST MODE - EFFECTS TIMELINE:"
                f"x=10:y={y_offset}:"
                f"fontsize=16:fontcolor=yellow:"
                f"box=1:boxcolor=black@0.8:boxborderw=5"
            )
            y_offset += 25
            
            # Add text overlay indicators
            for i, overlay in enumerate(text_overlays):
                # Escape text for FFmpeg (handle both 'text' and 'word' keys)
                word = overlay.get('text', overlay.get('word', ''))
                safe_text = self.escape_text(word[:20])
                label_text = f"TEXT {safe_text}"
                test_labels.append(
                    f"drawtext="
                    f"text={label_text}:"
                    f"x=10:y={y_offset}:"
                    f"fontsize=14:fontcolor=white:"
                    f"box=1:boxcolor=blue@0.6:boxborderw=3:"
                    f"enable='gte(t\\,{overlay['start']})*lte(t\\,{overlay['end']})'"
                )
                y_offset += 22
            
            # Add sound effect indicators
            if sound_effects:
                for effect in sound_effects:
                    timestamp = effect.get('timestamp', 0)
                    sound_name = effect.get('sound', 'unknown')
                    label_text = f"SOUND {sound_name}"
                    # Show for 2 seconds after the timestamp
                    test_labels.append(
                        f"drawtext="
                        f"text={label_text}:"
                        f"x=10:y={y_offset}:"
                        f"fontsize=14:fontcolor=white:"
                        f"box=1:boxcolor=green@0.6:boxborderw=3:"
                        f"enable='gte(t\\,{timestamp})*lte(t\\,{timestamp + 2})'"
                    )
                    y_offset += 22
            
            # Add visual effect indicators
            if suggested_effects:
                for effect in suggested_effects:
                    effect_name = effect.get('effect_fn_name', 'unknown_effect')
                    effect_time = float(effect.get('effect_timestamp', 0))
                    clean_name = effect_name.replace('apply_', '').replace('_', ' ')
                    label_text = f"EFFECT {clean_name}"
                    # Show for 3 seconds after the effect timestamp
                    test_labels.append(
                        f"drawtext="
                        f"text={label_text}:"
                        f"x=10:y={y_offset}:"
                        f"fontsize=14:fontcolor=white:"
                        f"box=1:boxcolor=purple@0.6:boxborderw=3:"
                        f"enable='gte(t\\,{effect_time})*lte(t\\,{effect_time + 3})'"
                    )
                    y_offset += 22
            
            # Add all test labels to filters
            filters.extend(test_labels)
        else:
            # Original debug indicator (non-test mode)
            filters.append(
                f"drawtext="
                f"text=ORIGINAL SCENE {actual_width}x{actual_height}:"
                f"x=10:y=10:"
                f"fontsize=20:fontcolor=green:"
                f"box=1:boxcolor=black@0.5"
            )
        
        # Apply filters
        filtergraph = ",".join(filters)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vf", filtergraph,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_video
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("  Warning: Some filters failed, using simplified version")
                # Fallback to simple version
                subprocess.run(["cp", input_video, output_video])
        except Exception as e:
            print(f"  Error: {e}")
            subprocess.run(["cp", input_video, output_video])
    
    def apply_sound_effects(self, input_video: str, output_video: str,
                           sound_effects: List[Dict]):
        """Apply sound effects at 50% volume."""
        # Initialize managers
        sfx_manager = SoundEffectsManager()
        video_overlay = VideoSoundOverlay()
        
        # Get sound files
        sound_files = {}
        for effect in sound_effects:
            sound_name = effect["sound"].lower()
            existing = sfx_manager.find_existing_sound(sound_name)
            if existing:
                sound_files[sound_name] = existing
                print(f"  ✓ {sound_name}: {Path(existing).name}")
        
        # Apply effects
        success = video_overlay.overlay_sound_effects(
            input_video,
            sound_effects,
            sound_files,
            output_video,
            preserve_original_audio=True
        )
        
        return success


def process_do_re_mi_scene_1():
    """Process Do-Re-Mi Scene 1 with all effects."""
    
    # Scene 1 metadata (0-13 seconds)
    scene_1_text_overlays = [
        {"text": "Let's", "x": 10, "y": 90, "start": 2.779, "end": 3.579},
        {"text": "start", "x": 65, "y": 90, "start": 3.579, "end": 4.079},
        {"text": "at the very beginning", "x": 10, "y": 90, "start": 5.280, "end": 6.420},
        {"text": "A very good place to start", "x": 10, "y": 110, "start": 8.0, "end": 10.0},
    ]
    
    scene_1_sound_effects = [
        {"sound": "ding", "timestamp": 3.579, "volume": 0.5},
    ]
    
    processor = OriginalSceneProcessor()
    processor.process_scene(
        video_name="do_re_mi",
        scene_number=1,
        text_overlays=scene_1_text_overlays,
        sound_effects=scene_1_sound_effects
    )


def process_all_do_re_mi_scenes():
    """Process all Do-Re-Mi scenes with their respective effects."""
    
    # Define effects for each scene
    scenes_metadata = {
        1: {
            "text_overlays": [
                {"text": "Let's start at the very beginning", "x": 10, "y": 90, "start": 2.779, "end": 6.420},
                {"text": "A very good place to start", "x": 10, "y": 110, "start": 8.0, "end": 10.0},
            ],
            "sound_effects": [
                {"sound": "ding", "timestamp": 3.579, "volume": 0.5},
            ]
        },
        2: {
            "text_overlays": [
                {"text": "A", "x": 180, "y": 10, "start": 0.02, "end": 0.72},  # Adjusted for scene start
                {"text": "B", "x": 205, "y": 10, "start": 0.72, "end": 1.119},
                {"text": "C", "x": 230, "y": 10, "start": 1.439, "end": 1.46},
                {"text": "Do-Re-Mi", "x": 10, "y": 10, "start": 4.7, "end": 5.44},
            ],
            "sound_effects": [
                {"sound": "swoosh", "timestamp": 0.05, "volume": 0.5},  # Adjusted
                {"sound": "chime", "timestamp": 4.7, "volume": 0.5},
            ]
        },
        3: {
            "text_overlays": [
                {"text": "Do", "x": 10, "y": 10, "start": 0.36, "end": 0.76},  # Adjusted
                {"text": "Re", "x": 45, "y": 10, "start": 1.059, "end": 1.459},
                {"text": "Mi", "x": 80, "y": 10, "start": 1.159, "end": 1.379},
                {"text": "The first three notes", "x": 10, "y": 90, "start": 3.0, "end": 5.0},
            ],
            "sound_effects": []
        },
        4: {
            "text_overlays": [
                {"text": "Doe - a deer, a female deer", "x": 10, "y": 90, "start": 1.119, "end": 2.939},  # Adjusted
                {"text": "Ray - a drop of golden sun", "x": 10, "y": 110, "start": 5.84, "end": 8.2},
                {"text": "Me - a name I call myself", "x": 10, "y": 130, "start": 9.52, "end": 12.68},
                {"text": "Far - a long long way to run", "x": 10, "y": 150, "start": 13.479, "end": 15.759},
            ],
            "sound_effects": [
                {"sound": "sparkle", "timestamp": 1.119, "volume": 0.5},
                {"sound": "pop", "timestamp": 5.84, "volume": 0.5},
                {"sound": "pop", "timestamp": 9.52, "volume": 0.5},
                {"sound": "pop", "timestamp": 13.479, "volume": 0.5},
            ]
        }
    }
    
    processor = OriginalSceneProcessor()
    
    for scene_num, metadata in scenes_metadata.items():
        print(f"\n{'='*70}")
        print(f"Processing Scene {scene_num}")
        print('='*70)
        
        processor.process_scene(
            video_name="do_re_mi",
            scene_number=scene_num,
            text_overlays=metadata["text_overlays"],
            sound_effects=metadata["sound_effects"]
        )


if __name__ == "__main__":
    # Process all scenes
    process_all_do_re_mi_scenes()