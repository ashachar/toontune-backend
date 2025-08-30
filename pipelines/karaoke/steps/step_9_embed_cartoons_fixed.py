#!/usr/bin/env python3
"""
Step 9: Embed Cartoon Characters (FIXED - Proper Layering)
===========================================================

This step embeds cartoon characters as overlays that properly layer
on top of existing video content instead of replacing it.
"""

import json
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EmbedCartoonsStep:
    """Step to embed cartoon characters with proper layering."""
    
    def __init__(self, config):
        """Initialize the embed cartoons step."""
        self.config = config
        self.video_dir = Path(config.video_dir)
        self.scenes_dir = self.video_dir / "scenes"
        self.inferences_dir = self.video_dir / "inferences"
        self.output_dir = self.scenes_dir / "edited"
        
        # Assets directories for cartoon images
        self.assets_dirs = [
            Path("cartoon-test"),
            Path("uploads/assets/batch_images_transparent_bg"),
            self.video_dir / "cartoon_assets"
        ]
        
    def run(self):
        """Run the embed cartoons step."""
        logger.info("Starting cartoon characters embedding step (FIXED)")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process videos in edited directory (in-place modification)
        input_videos = sorted(self.output_dir.glob("scene_*.mp4"))
        
        for video_path in input_videos:
            scene_name = video_path.stem
            inference_path = self.inferences_dir / f"{scene_name}_inference.json"
            
            if not inference_path.exists():
                logger.warning(f"No inference found for {scene_name}. Skipping.")
                continue
                
            # Load inference
            with open(inference_path, 'r') as f:
                inference_data = json.load(f)
            
            # Process each scene
            for scene in inference_data.get('scenes', []):
                cartoon_characters = scene.get('cartoon_characters', [])
                
                if not cartoon_characters:
                    logger.info(f"No cartoon characters for {scene_name}. Keeping video as-is.")
                    continue
                
                # Get cartoon assets
                cartoon_assets = self._prepare_cartoon_assets(cartoon_characters)
                
                # Create temp file then move to avoid corruption
                temp_path = self.output_dir / f"{video_path.stem}_temp.mp4"
                success = self._embed_cartoons_properly(video_path, temp_path, cartoon_characters, cartoon_assets)
                
                if success:
                    # Replace original with embedded version
                    subprocess.run(["mv", str(temp_path), str(video_path)], check=True)
                    logger.info(f"✓ Embedded {len(cartoon_characters)} cartoons in {scene_name}")
                
        logger.info(f"✓ Cartoon characters embedded")
        
    def _prepare_cartoon_assets(self, cartoon_characters):
        """Prepare cartoon character assets."""
        assets = {}
        
        for char in cartoon_characters:
            char_type = char.get('character_type', '')
            
            # Try to find matching asset
            asset_path = self._find_matching_asset(char_type)
            
            if asset_path:
                assets[char_type] = asset_path
                logger.info(f"Using asset {asset_path.name} for {char_type}")
            else:
                logger.warning(f"No asset found for {char_type}")
                
        return assets
        
    def _find_matching_asset(self, char_type):
        """Find a matching asset image for the character type."""
        # Use spring.png as default for all characters for now
        for assets_dir in self.assets_dirs:
            if not assets_dir.exists():
                continue
            
            # Try spring.png first
            spring_path = assets_dir / "spring.png"
            if spring_path.exists():
                return spring_path
            
            # Try other PNG files
            for png in assets_dir.glob("*.png"):
                return png
                
        return None
        
    def _embed_cartoons_properly(self, input_path, output_path, cartoon_characters, cartoon_assets):
        """Embed cartoon characters with PROPER LAYERING on existing video."""
        
        if not cartoon_assets:
            logger.warning("No cartoon assets available")
            return False
        
        # Build FFmpeg command with proper layering
        input_files = ['-i', str(input_path)]
        
        # Add cartoon images as inputs
        cartoon_inputs = []
        for char in cartoon_characters:
            char_type = char.get('character_type', '')
            asset_path = cartoon_assets.get(char_type)
            if asset_path and asset_path.exists():
                input_files.extend(['-i', str(asset_path)])
                cartoon_inputs.append(char)
        
        if not cartoon_inputs:
            logger.warning("No valid cartoon inputs")
            return False
        
        # Build filter complex that properly layers
        # CRITICAL: Each overlay must build on the previous result, not start fresh
        filter_parts = []
        
        # First, name the video stream
        current_stream = "0:v"
        
        for i, char in enumerate(cartoon_inputs):
            input_idx = i + 1  # 0 is the main video
            
            # Get timing and position
            start_seconds = float(char.get('start_seconds', 0))
            duration_seconds = float(char.get('duration_seconds', 3))
            end_seconds = start_seconds + duration_seconds
            
            # Position (simplified for now)
            x = 200 + (i * 200)  # Offset each cartoon
            y = 200
            
            # Size
            width = 100
            height = 120
            
            # Scale the cartoon
            filter_parts.append(f"[{input_idx}:v]scale={width}:{height}[cartoon{i}]")
            
            # Overlay on current stream
            next_stream = f"v{i}" if i < len(cartoon_inputs) - 1 else "vout"
            filter_parts.append(
                f"[{current_stream}][cartoon{i}]overlay="
                f"x={x}:y={y}:"
                f"enable='between(t,{start_seconds},{end_seconds})'"
                f"[{next_stream}]"
            )
            current_stream = next_stream
        
        # Join all filter parts
        filter_complex = ";".join(filter_parts)
        
        # Build FFmpeg command
        cmd = ['ffmpeg'] + input_files + [
            '-filter_complex', filter_complex,
            '-map', '[vout]',      # Use the final composited video
            '-map', '0:a?',         # Copy audio from original
            '-c:v', 'libx264',      # H.264 codec
            '-preset', 'fast',      # Fast encoding
            '-crf', '18',           # High quality
            '-c:a', 'copy',         # Copy audio
            '-movflags', '+faststart',
            '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return True
            else:
                logger.error(f"FFmpeg failed: {result.stderr[:500]}")
                return False
        except Exception as e:
            logger.error(f"Failed to embed cartoons: {e}")
            return False