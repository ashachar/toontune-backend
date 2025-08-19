#!/usr/bin/env python3
"""
Step 9: Embed Cartoon Characters with Horizon Tracking
======================================================

This step embeds cartoon characters from the inference as animated overlays on the video.
It uses horizon detection to properly place ground-based characters and tracks them.
"""

import json
import subprocess
import os
import sys
from pathlib import Path
import logging
import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.lines_detector import HorizonDetector
from utils.tracking.cotracker3 import CoTracker3

logger = logging.getLogger(__name__)


class EmbedCartoonsStep:
    """Step to embed cartoon characters with smart horizon-aware placement."""
    
    def __init__(self, config):
        """Initialize the embed cartoons step."""
        self.config = config
        self.video_dir = Path(config.video_dir)
        self.scenes_dir = self.video_dir / "scenes"
        self.inferences_dir = self.video_dir / "inferences"
        # Always work with edited directory - our single output location
        self.output_dir = self.scenes_dir / "edited"
        
        # Assets directories for cartoon images
        self.assets_dirs = [
            Path("uploads/assets/batch_images_transparent_bg"),
            Path("cartoon-test"),
            self.video_dir / "cartoon_assets"
        ]
        
        # Initialize horizon detector (lazy loading)
        self.horizon_detector = None
        self.tracker = None
        
    def _init_tracking_systems(self):
        """Initialize tracking systems if needed."""
        if self.horizon_detector is None:
            try:
                # Enable horizon tracking for smart placement
                # Comment out these lines to disable horizon tracking for faster processing
                from utils.lines_detector import HorizonDetector
                from utils.tracking.cotracker3 import CoTracker3
                self.horizon_detector = HorizonDetector(use_fast_model=True)
                self.tracker = CoTracker3(model_type="cotracker3_online")
                logger.info("Initialized horizon detection and tracking systems")
            except Exception as e:
                logger.warning(f"Could not initialize tracking: {e}. Using fallback placement.")
                
    def run(self):
        """Run the embed cartoons step."""
        logger.info("Starting cartoon characters embedding step with horizon tracking")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process videos in edited directory (in-place modification)
        if not self.output_dir.exists():
            logger.warning("No edited directory found. Skipping embed cartoons step.")
            return
            
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
                
                # Detect horizon for ground-based characters
                horizon_info = self._detect_horizon(video_path, cartoon_characters)
                
                # Get or create cartoon assets
                cartoon_assets = self._prepare_cartoon_assets(cartoon_characters)
                
                # Build FFmpeg command with smart placement
                # Create temp file then move to avoid corruption
                temp_path = self.output_dir / f"{video_path.stem}_temp.mp4"
                self._embed_cartoons_ffmpeg(video_path, temp_path, cartoon_characters, cartoon_assets, horizon_info)
                
                # Replace original with embedded version
                if temp_path.exists():
                    subprocess.run(["mv", str(temp_path), str(video_path)], check=True)
                
        logger.info(f"✓ Cartoon characters embedded in {len(input_videos)} videos")
        
    def _detect_horizon(self, video_path, cartoon_characters):
        """Detect horizon line for proper character placement."""
        # Check if any character needs horizon placement
        ground_chars = [c for c in cartoon_characters if 'deer' in c.get('character_type', '').lower() 
                       or 'animal' in c.get('character_type', '').lower()]
        
        if not ground_chars:
            return None
            
        try:
            # Initialize tracking systems if needed
            self._init_tracking_systems()
            
            if self.horizon_detector is None:
                return None
                
            # Load first frame
            cap = cv2.VideoCapture(str(video_path))
            ret, first_frame = cap.read()
            cap.release()
            
            if not ret:
                logger.warning("Could not read first frame for horizon detection")
                return None
                
            # Detect horizon
            horizon_result = self.horizon_detector.detect(first_frame)
            
            if horizon_result['horizon_points'] is None:
                logger.warning("No horizon detected, using fallback placement")
                return None
                
            logger.info(f"Horizon detected at angle {horizon_result['angle']:.1f}° with {horizon_result['confidence']:.1%} confidence")
            
            # Get multiple points along horizon for character placement
            positions = {}
            for pos in ['left', 'center', 'right']:
                x, y = self.horizon_detector.select_tracking_point(
                    horizon_result['horizon_points'],
                    position=pos
                )
                positions[pos] = (x, y)
                
            return {
                'horizon_points': horizon_result['horizon_points'],
                'angle': horizon_result['angle'],
                'positions': positions,
                'frame_height': first_frame.shape[0],
                'frame_width': first_frame.shape[1]
            }
            
        except Exception as e:
            logger.warning(f"Horizon detection failed: {e}")
            return None
            
    def _prepare_cartoon_assets(self, cartoon_characters):
        """Prepare cartoon character assets from existing images."""
        assets = {}
        
        for char in cartoon_characters:
            char_type = char.get('character_type', '')
            
            # Try to find matching asset
            asset_path = self._find_matching_asset(char_type)
            
            if asset_path:
                assets[char_type] = asset_path
                logger.info(f"Using asset {asset_path.name} for {char_type}")
            else:
                # Create placeholder
                placeholder_path = self.video_dir / "cartoon_assets" / f"{char_type}.png"
                placeholder_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not placeholder_path.exists():
                    self._create_simple_placeholder(placeholder_path, char_type)
                    
                assets[char_type] = placeholder_path
                
        return assets
        
    def _find_matching_asset(self, char_type):
        """Find a matching asset image for the character type."""
        # Map character types to potential asset names
        asset_mappings = {
            'happy_female_deer': ['deer', 'animal', 'spring', 'baby'],  # Use spring or baby as fallback
            'smiling_drop_of_sun': ['sun', 'star', 'balloon', 'spring'],
            'musical_note': ['note', 'music'],
            'singing_bird': ['bird', 'spring'],
            'dancing_flower': ['flower', 'spring']
        }
        
        search_terms = asset_mappings.get(char_type, [char_type])
        
        # Search in all asset directories
        for assets_dir in self.assets_dirs:
            if not assets_dir.exists():
                continue
                
            for asset_file in assets_dir.glob("*.png"):
                asset_name_lower = asset_file.stem.lower()
                
                for term in search_terms:
                    if term.lower() in asset_name_lower:
                        return asset_file
                        
        # Special fallback for specific types
        if 'deer' in char_type.lower():
            # Use baby or spring cartoon as deer substitute
            for assets_dir in self.assets_dirs:
                for name in ['baby.png', 'spring.png']:
                    path = assets_dir / name
                    if path.exists():
                        return path
                        
        if 'sun' in char_type.lower():
            # Use spring as sun substitute
            for assets_dir in self.assets_dirs:
                path = assets_dir / 'spring.png'
                if path.exists():
                    return path
                    
        return None
        
    def _create_simple_placeholder(self, output_path, char_type):
        """Create a simple colored circle as placeholder."""
        # Using ffmpeg to create a colored circle
        color = 'yellow' if 'sun' in char_type else 'brown' if 'deer' in char_type else 'blue'
        
        cmd = [
            'ffmpeg', '-f', 'lavfi', 
            '-i', f'color=c={color}:s=100x100:d=1',
            '-vf', 'format=rgba,geq=r=\'r(X,Y)\':g=\'g(X,Y)\':b=\'b(X,Y)\':a=\'if(pow(X-50,2)+pow(Y-50,2)<=2500,255,0)\'',
            '-frames:v', '1', '-y', str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Created placeholder for {char_type}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create placeholder: {e.stderr}")
            
    def _embed_cartoons_ffmpeg(self, input_path, output_path, cartoon_characters, cartoon_assets, horizon_info):
        """Embed cartoon characters with smart placement using FFmpeg."""
        
        # Build complex filter for all cartoon overlays
        filter_parts = []
        input_files = ['-i', str(input_path)]
        
        for i, char in enumerate(cartoon_characters):
            char_type = char.get('character_type', '')
            asset_path = cartoon_assets.get(char_type)
            
            if not asset_path or not asset_path.exists():
                continue
                
            # Add cartoon image as input
            input_files.extend(['-i', str(asset_path)])
            input_index = i + 1  # 0 is the main video
            
            # Get timing
            start_seconds = float(char.get('start_seconds', 0))
            duration_seconds = float(char.get('duration_seconds', 3))
            
            # Calculate position based on character type and horizon
            x, y = self._calculate_character_position(char, horizon_info)
            
            # Get size
            width = char.get('size_pixels', {}).get('width', 100)
            height = char.get('size_pixels', {}).get('height', 120)
            
            # Animation style
            animation_style = char.get('animation_style', 'static')
            
            # Build overlay filter based on animation style
            if animation_style == 'hop_across' and horizon_info:
                # For deer - hop along the horizon line
                # Calculate position along horizon
                horizon_y = horizon_info['positions']['center'][1] - height  # Place feet on horizon
                
                overlay_filter = (
                    f"[{input_index}:v]scale={width}:{height}[cartoon{i}];"
                    f"[{'0:v' if i == 0 else f'out{i-1}'}][cartoon{i}]overlay="
                    f"x='{x}+200*(t-{start_seconds})/{duration_seconds}':"
                    f"y='{horizon_y}+20*abs(sin(4*PI*(t-{start_seconds})))':"
                    f"enable='between(t,{start_seconds},{start_seconds + duration_seconds})'"
                    f"[out{i}]"
                )
            elif animation_style == 'float_across':
                # For sun - drift down from sky
                overlay_filter = (
                    f"[{input_index}:v]scale={width}:{height}[cartoon{i}];"
                    f"[{'0:v' if i == 0 else f'out{i-1}'}][cartoon{i}]overlay="
                    f"x='{x}+100*(t-{start_seconds})/{duration_seconds}':"
                    f"y='{y}+150*(t-{start_seconds})/{duration_seconds}':"
                    f"enable='between(t,{start_seconds},{start_seconds + duration_seconds})'"
                    f"[out{i}]"
                )
            else:
                # Static position
                overlay_filter = (
                    f"[{input_index}:v]scale={width}:{height}[cartoon{i}];"
                    f"[{'0:v' if i == 0 else f'out{i-1}'}][cartoon{i}]overlay="
                    f"x={x}:y={y}:"
                    f"enable='between(t,{start_seconds},{start_seconds + duration_seconds})'"
                    f"[out{i}]"
                )
            
            filter_parts.append(overlay_filter)
        
        if not filter_parts:
            # No cartoons to embed, just copy
            subprocess.run(["cp", str(input_path), str(output_path)], check=True)
            return
            
        # Combine all filters
        filter_complex = ';'.join(filter_parts)
        
        # Map the final output
        final_output = f"out{len(cartoon_characters)-1}"
        
        # Build FFmpeg command with HIGH QUALITY settings
        cmd = ['ffmpeg'] + input_files + [
            '-filter_complex', filter_complex,
            '-map', f'[{final_output}]',
            '-map', '0:a?',  # Copy audio if present
            '-c:v', 'libx264',  # Use H.264 codec
            '-preset', 'slow',   # Better quality encoding
            '-crf', '18',        # High quality (lower = better, 18 is visually lossless)
            '-c:a', 'copy',      # Copy audio without re-encoding
            '-movflags', '+faststart',  # Optimize for streaming
            '-y', str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✓ Embedded {len(cartoon_characters)} cartoons in {output_path.name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to embed cartoons: {e.stderr}")
            # Fallback: copy original
            subprocess.run(["cp", str(input_path), str(output_path)], check=True)
            
    def _calculate_character_position(self, char, horizon_info):
        """Calculate smart position for character based on type and horizon."""
        # Get original position from inference
        orig_x = char.get('position_pixels', {}).get('x', 100)
        orig_y = char.get('position_pixels', {}).get('y', 100)
        
        char_type = char.get('character_type', '').lower()
        
        # If no horizon info, use original positions
        if not horizon_info:
            return orig_x, orig_y
            
        # For ground-based characters (deer, animals)
        if 'deer' in char_type or 'animal' in char_type:
            # Place on horizon line
            # Choose position based on original x coordinate
            if orig_x < horizon_info['frame_width'] / 3:
                pos = 'left'
            elif orig_x > 2 * horizon_info['frame_width'] / 3:
                pos = 'right'
            else:
                pos = 'center'
                
            horizon_x, horizon_y = horizon_info['positions'][pos]
            
            # Adjust y to place character feet on horizon
            char_height = char.get('size_pixels', {}).get('height', 120)
            y = horizon_y - char_height + 20  # Small offset to make it look grounded
            
            return orig_x, int(y)
            
        # For sky-based characters (sun, clouds, birds)
        elif 'sun' in char_type or 'cloud' in char_type or 'bird' in char_type:
            # Keep in upper portion of frame but respect original position
            return orig_x, orig_y
            
        # Default: use original position
        return orig_x, orig_y