#!/usr/bin/env python3
"""
Step 8: Embed Key Phrases
==========================

This step embeds key phrases from the inference as text overlays on the video.
It reads the key_phrases from each scene's inference and applies them to the edited videos.
"""

import json
import subprocess
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EmbedPhrasesStep:
    """Step to embed key phrases as text overlays on videos."""
    
    def __init__(self, config):
        """Initialize the embed phrases step."""
        self.config = config
        self.video_dir = Path(config.video_dir)
        self.scenes_dir = self.video_dir / "scenes"
        self.inferences_dir = self.video_dir / "inferences"
        # Always output to edited directory - this is our final output location
        self.output_dir = self.scenes_dir / "edited"
        
    def run(self):
        """Run the embed phrases step."""
        logger.info("Starting key phrases embedding step")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process videos in edited directory (in-place modification)
        if not self.output_dir.exists():
            logger.warning("No edited directory found. Creating it.")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        edited_videos = sorted(self.output_dir.glob("scene_*.mp4"))
        
        for video_path in edited_videos:
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
                key_phrases = scene.get('key_phrases', [])
                
                if not key_phrases:
                    logger.info(f"No key phrases for {scene_name}. Keeping video as-is.")
                    continue
                
                # Build FFmpeg command with all key phrases
                # Create temp file then move to avoid corruption
                temp_path = self.output_dir / f"{video_path.stem}_temp.mp4"
                self._embed_phrases_ffmpeg(video_path, temp_path, key_phrases)
                
                # Replace original with embedded version
                if temp_path.exists():
                    subprocess.run(["mv", str(temp_path), str(video_path)], check=True)
                    logger.info(f"✓ Embedded {len(key_phrases)} phrases in {scene_name}")
                
        logger.info(f"✓ Key phrases embedding complete")
        
    def _embed_phrases_ffmpeg(self, input_path, output_path, key_phrases):
        """Embed key phrases using FFmpeg drawtext filter with HIGH QUALITY."""
        
        # Build complex filter for all phrases
        filters = []
        
        for i, phrase_info in enumerate(key_phrases):
            phrase = phrase_info.get('phrase', '')
            start_seconds = float(phrase_info.get('start_seconds', 0))
            duration_seconds = float(phrase_info.get('duration_seconds', 3))
            
            # Get position (using top-left as reference for FFmpeg)
            x = phrase_info.get('top_left_pixels', {}).get('x', 50)
            y = phrase_info.get('top_left_pixels', {}).get('y', 50)
            
            # Style mapping
            style = phrase_info.get('style', 'default')
            importance = phrase_info.get('importance', 'normal')
            
            # Set font size based on importance
            fontsize = 24 if importance == 'critical' else 20 if importance == 'high' else 16
            
            # Escape special characters in text
            phrase_escaped = phrase.replace("'", "\\'").replace(":", "\\:")
            
            # Set color and effects based on style
            if style == 'elegant_fade':
                fontcolor = 'white'
                # Add fade in/out effect
                fade_duration = 0.5
                drawtext = (
                    f"drawtext=text='{phrase_escaped}':fontsize={fontsize}:fontcolor={fontcolor}:"
                    f"x={x}:y={y}:fontfile=/System/Library/Fonts/Helvetica.ttc:"
                    f"enable='between(t,{start_seconds},{start_seconds + duration_seconds})':"
                    f"alpha='if(lt(t,{start_seconds + fade_duration}),"
                    f"(t-{start_seconds})/{fade_duration},"
                    f"if(gt(t,{start_seconds + duration_seconds - fade_duration}),"
                    f"({start_seconds + duration_seconds}-t)/{fade_duration},1))'"
                )
            elif style == 'playful_bounce':
                fontcolor = 'yellow'
                # Add slight vertical movement
                drawtext = (
                    f"drawtext=text='{phrase_escaped}':fontsize={fontsize}:fontcolor={fontcolor}:"
                    f"x={x}:y='{y}+10*sin(2*PI*t)':"
                    f"fontfile=/System/Library/Fonts/Helvetica.ttc:"
                    f"enable='between(t,{start_seconds},{start_seconds + duration_seconds})'"
                )
            else:
                fontcolor = 'white'
                drawtext = (
                    f"drawtext=text='{phrase_escaped}':fontsize={fontsize}:fontcolor={fontcolor}:"
                    f"x={x}:y={y}:fontfile=/System/Library/Fonts/Helvetica.ttc:"
                    f"enable='between(t,{start_seconds},{start_seconds + duration_seconds})'"
                )
            
            filters.append(drawtext)
        
        # Combine all filters
        filter_complex = ','.join(filters)
        
        # Run FFmpeg command with HIGH QUALITY settings
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-vf', filter_complex,
            '-c:v', 'libx264',  # Use H.264 codec
            '-preset', 'slow',   # Better quality encoding
            '-crf', '18',        # High quality (lower = better, 18 is visually lossless)
            '-c:a', 'copy',      # Copy audio without re-encoding
            '-movflags', '+faststart',  # Optimize for streaming
            '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg failed: {result.stderr}")
                # Try without alpha channel as fallback
                filter_complex_simple = filter_complex.replace(":alpha='", "#:alpha='").replace("')", "'#)")
                cmd[2] = filter_complex_simple
                subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to embed phrases: {e.stderr if hasattr(e, 'stderr') else str(e)}")