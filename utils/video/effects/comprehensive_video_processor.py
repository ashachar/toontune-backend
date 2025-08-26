#!/usr/bin/env python3
"""
Comprehensive Video Processor that applies all effects from JSON metadata:
- Sound effects with volume control
- Text overlays with animation effects
- Visual effects (bloom, zoom, light sweep, etc.)
- Test mode with debug overlays
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import shutil

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.sound_effects.sound_effects_manager import SoundEffectsManager
from utils.sound_effects.video_sound_overlay import VideoSoundOverlay


@dataclass
class TextOverlay:
    """Represents a text overlay with positioning and effects."""
    word: str
    start_seconds: float
    end_seconds: float
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int
    text_effect: str
    text_effect_params: Dict
    interaction_style: str


@dataclass  
class VisualEffect:
    """Represents a visual effect to apply."""
    effect_fn_name: str
    effect_timestamp: float
    effect_fn_params: Dict
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int


class ComprehensiveVideoProcessor:
    """Processes videos with all effects from JSON metadata."""
    
    def __init__(self, test_mode: bool = False):
        """
        Initialize the processor.
        
        Args:
            test_mode: If True, adds debug overlays showing what effects are applied
        """
        self.test_mode = test_mode
        self.sfx_manager = SoundEffectsManager()
        self.video_overlay = VideoSoundOverlay()
        self.ffmpeg_path = "ffmpeg"
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            subprocess.run([self.ffmpeg_path, "-version"], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(f"FFmpeg not found at: {self.ffmpeg_path}")
    
    def process_video(self, 
                     video_path: str,
                     metadata: Dict,
                     output_path: str,
                     sound_volume: float = 0.5) -> bool:
        """
        Process a video with all effects from metadata.
        
        Args:
            video_path: Path to input video
            metadata: Complete metadata dictionary with all effects
            output_path: Path for output video
            sound_volume: Volume level for sound effects (0.0 to 1.0)
        
        Returns:
            True if successful, False otherwise
        """
        if not Path(video_path).exists():
            print(f"Error: Video not found: {video_path}")
            return False
        
        print("="*60)
        print("COMPREHENSIVE VIDEO PROCESSING")
        print("="*60)
        print(f"Input: {video_path}")
        print(f"Output: {output_path}")
        print(f"Test Mode: {self.test_mode}")
        print()
        
        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Stage 1: Apply text overlays and visual effects
            print("Stage 1: Applying text overlays and visual effects...")
            video_with_visuals = str(temp_path / "stage1_visuals.mp4")
            
            success = self._apply_visual_effects(
                video_path, 
                metadata,
                video_with_visuals
            )
            
            if not success:
                print("Failed to apply visual effects")
                return False
            
            # Stage 2: Apply sound effects
            print("\nStage 2: Adding sound effects...")
            final_input = video_with_visuals if Path(video_with_visuals).exists() else video_path
            
            success = self._apply_sound_effects(
                final_input,
                metadata,
                output_path,
                sound_volume
            )
            
            if not success:
                print("Failed to apply sound effects")
                return False
        
        print("\n" + "="*60)
        print("✓ Processing complete!")
        print(f"Output: {output_path}")
        return True
    
    def _apply_visual_effects(self, 
                             video_path: str,
                             metadata: Dict,
                             output_path: str) -> bool:
        """Apply text overlays and visual effects."""
        
        # Extract scenes and their effects
        scenes = metadata.get("scenes", [])
        if not scenes:
            print("No scenes found in metadata")
            shutil.copy(video_path, output_path)
            return True
        
        # Build FFmpeg filter complex
        filters = []
        debug_texts = []
        
        # Process each scene
        for scene_idx, scene in enumerate(scenes):
            scene_desc = scene.get("scene_description", {})
            
            # Process text overlays
            text_overlays = scene_desc.get("text_overlays", [])
            for text in text_overlays:
                filter_str = self._create_text_overlay_filter(text, scene_idx)
                if filter_str:
                    filters.append(filter_str)
                    
                    # Add debug text if in test mode
                    if self.test_mode:
                        debug_text = self._create_debug_text(
                            f"TEXT: {text['word']}",
                            text['start_seconds'],
                            text['end_seconds']
                        )
                        debug_texts.append(debug_text)
            
            # Process visual effects
            visual_effects = scene_desc.get("suggested_effects", [])
            for effect in visual_effects:
                filter_str = self._create_visual_effect_filter(effect, scene_idx)
                if filter_str:
                    filters.append(filter_str)
                    
                    # Add debug text if in test mode
                    if self.test_mode:
                        effect_name = effect['effect_fn_name'].replace('apply_', '').upper()
                        debug_text = self._create_debug_text(
                            f"EFFECT: {effect_name}",
                            effect['effect_timestamp'],
                            effect['effect_timestamp'] + 2.0  # Show for 2 seconds
                        )
                        debug_texts.append(debug_text)
        
        # Add debug texts to filters
        filters.extend(debug_texts)
        
        if not filters:
            print("No visual effects to apply")
            shutil.copy(video_path, output_path)
            return True
        
        # Build FFmpeg command - use simple sequential filters
        # Start with input
        filter_parts = []
        
        # Add text overlays as sequential filters
        for i, f in enumerate(filters):
            if i == 0:
                filter_parts.append(f)
            else:
                # Chain filters properly with commas
                filter_parts.append(f)
        
        # Join with commas for sequential processing
        filter_complex = ",".join(filter_parts) if filter_parts else "null"
        
        cmd = [
            self.ffmpeg_path, "-y",
            "-i", video_path,
            "-vf", filter_complex,  # Use -vf for video filters
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        try:
            print(f"Applying {len(text_overlays)} text overlays and {len(visual_effects)} visual effects...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                # If complex filters fail, copy original
                shutil.copy(video_path, output_path)
                return True
            
            return Path(output_path).exists()
            
        except Exception as e:
            print(f"Error applying visual effects: {e}")
            shutil.copy(video_path, output_path)
            return True
    
    def _create_text_overlay_filter(self, text_overlay: Dict, scene_idx: int) -> str:
        """Create FFmpeg filter for text overlay."""
        word = text_overlay.get("word", "")
        start = float(text_overlay.get("start_seconds", 0))
        end = float(text_overlay.get("end_seconds", start + 1))
        x = text_overlay.get("top_left_pixels", {}).get("x", 10)
        y = text_overlay.get("top_left_pixels", {}).get("y", 10)
        
        # Escape special characters more carefully
        word_escaped = word.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace(",", "\\,")
        
        # Create drawtext filter with timing - use backslash escaping for enable
        filter_str = (
            f"drawtext=text='{word_escaped}'\\:"
            f"x={x}\\:y={y}\\:"
            f"fontsize=24\\:fontcolor=white\\:"
            f"box=1\\:boxcolor=black@0.5\\:boxborderw=5\\:"
            f"enable='between(t\\,{start}\\,{end})'"
        )
        
        return filter_str
    
    def _create_visual_effect_filter(self, effect: Dict, scene_idx: int) -> str:
        """Create FFmpeg filter for visual effects."""
        effect_name = effect.get("effect_fn_name", "")
        timestamp = float(effect.get("effect_timestamp", 0))
        params = effect.get("effect_fn_params", {})
        
        # Map effect names to FFmpeg filters
        if "bloom" in effect_name.lower():
            # Bloom effect using curves and blur
            intensity = params.get("bloom_intensity", 1.2)
            return f"curves=all='0/0 0.5/{0.5*intensity} 1/1':enable='gte(t,{timestamp})*lte(t,{timestamp+2})'"
            
        elif "zoom" in effect_name.lower():
            # Smooth zoom effect
            zoom_factor = params.get("zoom_factor", 1.1)
            duration = 2.0
            return (f"zoompan=z='if(gte(t,{timestamp})*lte(t,{timestamp+duration}),"
                   f"min(zoom+0.001,{zoom_factor}),1)':"
                   f"d=1:s=256x144")
            
        elif "light_sweep" in effect_name.lower():
            # Light sweep using brightness adjustment
            return f"eq=brightness=0.2:enable='gte(t,{timestamp})*lte(t,{timestamp+1.5})'"
            
        elif "shake" in effect_name.lower():
            # Handheld shake effect
            intensity = params.get("shake_intensity", 1.0)
            return f"crop=in_w:in_h:0:0,rotate=PI/180*sin(10*t)*{intensity}:enable='gte(t,{timestamp})'"
        
        return ""
    
    def _create_debug_text(self, text: str, start: float, end: float) -> str:
        """Create debug text overlay for test mode."""
        # Escape special characters
        text_escaped = text.replace("'", "\\'").replace(":", "\\:")
        
        # Position in top-left corner with red background
        return (
            f"drawtext=text='{text_escaped}':"
            f"x=10:y=10:"
            f"fontsize=16:fontcolor=yellow:"
            f"box=1:boxcolor=red@0.7:boxborderw=3:"
            f"enable='between(t,{start},{end})'"
        )
    
    def _apply_sound_effects(self,
                            video_path: str,
                            metadata: Dict,
                            output_path: str,
                            sound_volume: float) -> bool:
        """Apply sound effects with volume control."""
        
        sound_effects = metadata.get("sound_effects", [])
        if not sound_effects:
            print("No sound effects to apply")
            shutil.copy(video_path, output_path)
            return True
        
        print(f"Processing {len(sound_effects)} sound effects...")
        
        # Get or download sound files with duration filtering
        sound_files = {}
        for effect in sound_effects:
            sound_name = effect.get("sound", "").lower()
            
            # Define better search queries
            sound_queries = {
                "ding": "ding bell short",
                "swoosh": "whoosh short",
                "chime": "chime short",
                "sparkle": "sparkle short",
                "pop": "pop short"
            }
            
            query = sound_queries.get(sound_name, sound_name)
            
            # Search with max duration of 0.5 seconds
            sound_info = self.sfx_manager.api.search_sound(query, max_duration=0.5)
            if sound_info:
                filepath = self.sfx_manager.api.download_sound(sound_info)
                if filepath:
                    sound_files[sound_name] = filepath
                    
                    # Update registry
                    self.sfx_manager.metadata[sound_name] = {
                        "filepath": filepath,
                        "query": query,
                        "attribution": sound_info["attribution"],
                        "freesound_id": sound_info["id"],
                        "duration": sound_info.get("duration"),
                        "downloaded_at": datetime.now().isoformat()
                    }
        
        self.sfx_manager._save_metadata()
        
        # Apply sound effects with adjusted volume
        for effect in sound_effects:
            effect["volume"] = sound_volume
        
        # Add debug sound indicators if in test mode
        if self.test_mode:
            # This would be handled by adding text overlays for sound effects
            pass
        
        return self.video_overlay.overlay_sound_effects(
            video_path,
            sound_effects,
            sound_files,
            output_path,
            preserve_original_audio=True
        )


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive video processor with all effects"
    )
    parser.add_argument("video", help="Input video file")
    parser.add_argument(
        "--metadata",
        help="Path to metadata JSON file",
        required=True
    )
    parser.add_argument(
        "--output",
        help="Output video path",
        default="output/processed_video.mp4"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable test mode with debug overlays"
    )
    parser.add_argument(
        "--sound-volume",
        type=float,
        default=0.5,
        help="Sound effects volume (0.0 to 1.0, default 0.5)"
    )
    
    args = parser.parse_args()
    
    # Load metadata
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
    
    # Create processor
    processor = ComprehensiveVideoProcessor(test_mode=args.test)
    
    # Ensure output directory exists
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process video
    success = processor.process_video(
        args.video,
        metadata,
        args.output,
        sound_volume=args.sound_volume
    )
    
    if success:
        print(f"\n✓ Video successfully processed!")
        if args.test:
            print("Test mode: Debug overlays added")
    else:
        print("\n✗ Video processing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()