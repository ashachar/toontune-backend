#!/usr/bin/env python3
"""
Scene Splitter - Splits videos into scenes based on timestamps.
Creates both original (full resolution) and downsampled versions.

Directory structure:
uploads/assets/videos/{video_name}/
├── scenes/
│   ├── original/        # Full resolution scenes
│   │   ├── scene_001.mp4
│   │   ├── scene_002.mp4
│   │   └── ...
│   ├── downsampled/     # Downsampled for VLLM inference
│   │   ├── scene_001.mp4
│   │   ├── scene_002.mp4
│   │   └── ...
│   └── edited/          # Final scenes with effects applied
│       ├── scene_001.mp4
│       ├── scene_002.mp4
│       └── ...
├── metadata/
│   └── scenes.json      # Scene metadata with timestamps
└── transcripts/
    └── transcript.json  # Original transcript with timings
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class Scene:
    """Represents a scene with timing information."""
    scene_number: int
    start_seconds: float
    end_seconds: float
    duration: float
    description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class SceneSplitter:
    """Split videos into scenes with original and downsampled versions."""
    
    def __init__(self, video_path: str, output_base_dir: str = "uploads/assets/videos"):
        """
        Initialize scene splitter.
        
        Args:
            video_path: Path to input video
            output_base_dir: Base directory for output
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Create output directory structure
        video_name = self.video_path.stem
        self.output_dir = Path(output_base_dir) / video_name
        self.scenes_dir = self.output_dir / "scenes"
        self.original_dir = self.scenes_dir / "original"
        self.downsampled_dir = self.scenes_dir / "downsampled"
        self.edited_dir = self.scenes_dir / "edited"
        self.metadata_dir = self.output_dir / "metadata"
        self.transcripts_dir = self.output_dir / "transcripts"
        
        # Create all directories
        for dir_path in [self.original_dir, self.downsampled_dir, self.edited_dir,
                         self.metadata_dir, self.transcripts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.video_info = self._get_video_info()
    
    def _get_video_info(self) -> Dict:
        """Get video information using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            str(self.video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Extract video stream info
            for stream in data.get("streams", []):
                if stream["codec_type"] == "video":
                    return {
                        "width": int(stream["width"]),
                        "height": int(stream["height"]),
                        "duration": float(data.get("format", {}).get("duration", 0)),
                        "fps": eval(stream.get("r_frame_rate", "30/1")),
                        "codec": stream.get("codec_name", "unknown")
                    }
        except Exception as e:
            print(f"Error getting video info: {e}")
            return {"width": 1920, "height": 1080, "duration": 0, "fps": 30}
    
    def extract_scene(self, scene: Scene, output_path: str, 
                     downsample: bool = False,
                     target_width: int = 256) -> bool:
        """
        Extract a single scene from the video.
        
        Args:
            scene: Scene object with timing info
            output_path: Output file path
            downsample: Whether to downsample the video
            target_width: Target width for downsampling
        
        Returns:
            True if successful
        """
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.video_path),
            "-ss", str(scene.start_seconds),
            "-t", str(scene.duration)
        ]
        
        if downsample:
            # Calculate height maintaining aspect ratio
            aspect_ratio = self.video_info["height"] / self.video_info["width"]
            target_height = int(target_width * aspect_ratio)
            # Ensure even dimensions
            target_height = target_height + (target_height % 2)
            
            cmd.extend([
                "-vf", f"scale={target_width}:{target_height}",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23"
            ])
        else:
            # Copy video codec for original quality
            cmd.extend(["-c:v", "copy"])
        
        # Audio settings
        cmd.extend(["-c:a", "copy"])
        
        # Output
        cmd.append(output_path)
        
        try:
            print(f"Extracting scene {scene.scene_number}: "
                  f"{scene.start_seconds:.1f}s - {scene.end_seconds:.1f}s "
                  f"({'downsampled' if downsample else 'original'})")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: FFmpeg error: {result.stderr[:200]}")
                return False
            
            return Path(output_path).exists()
            
        except Exception as e:
            print(f"Error extracting scene: {e}")
            return False
    
    def split_into_scenes(self, scenes: List[Scene]) -> Dict:
        """
        Split video into multiple scenes.
        
        Args:
            scenes: List of Scene objects
        
        Returns:
            Dictionary with results
        """
        results = {
            "video_path": str(self.video_path),
            "video_info": self.video_info,
            "scenes": [],
            "original_scenes": [],
            "downsampled_scenes": []
        }
        
        for scene in scenes:
            # Original/full resolution scene
            original_path = self.original_dir / f"scene_{scene.scene_number:03d}.mp4"
            success_orig = self.extract_scene(scene, str(original_path), downsample=False)
            
            if success_orig:
                results["original_scenes"].append(str(original_path))
            
            # Downsampled scene for VLLM
            downsampled_path = self.downsampled_dir / f"scene_{scene.scene_number:03d}.mp4"
            success_down = self.extract_scene(scene, str(downsampled_path), downsample=True)
            
            if success_down:
                results["downsampled_scenes"].append(str(downsampled_path))
            
            # Add to results
            scene_info = scene.to_dict()
            scene_info["original_path"] = str(original_path) if success_orig else None
            scene_info["downsampled_path"] = str(downsampled_path) if success_down else None
            results["scenes"].append(scene_info)
        
        # Save metadata
        metadata_file = self.metadata_dir / "scenes.json"
        with open(metadata_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved scene metadata to: {metadata_file}")
        
        return results
    
    @staticmethod
    def create_scenes_from_timestamps(timestamps: List[Tuple[float, float]], 
                                     descriptions: Optional[List[str]] = None) -> List[Scene]:
        """
        Create Scene objects from timestamps.
        
        Args:
            timestamps: List of (start, end) tuples in seconds
            descriptions: Optional list of scene descriptions
        
        Returns:
            List of Scene objects
        """
        scenes = []
        for i, (start, end) in enumerate(timestamps, 1):
            duration = end - start
            description = descriptions[i-1] if descriptions and i-1 < len(descriptions) else None
            
            scene = Scene(
                scene_number=i,
                start_seconds=start,
                end_seconds=end,
                duration=duration,
                description=description
            )
            scenes.append(scene)
        
        return scenes


def split_do_re_mi_scenes():
    """Split the Do-Re-Mi with music video into scenes based on the transcript."""
    
    # Scene timestamps from the JSON metadata
    scene_timestamps = [
        (0.000, 13.020),     # Scene 1: Introduction
        (13.021, 29.959),    # Scene 2: ABC/Do-Re-Mi
        (29.960, 39.000),    # Scene 3: Musical scale
        (39.001, 54.759),    # Scene 4: Note definitions
    ]
    
    scene_descriptions = [
        "Woman begins to sing with guitar",
        "Wide shot with children, ABC and Do-Re-Mi",
        "Woman sings the musical scale",
        "Defining each note (Doe, Ray, Me, Far)"
    ]
    
    # Use the correct video with music
    video_path = "uploads/assets/videos/do_re_mi_with_music.mov"
    
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        # Try alternative without .mov extension
        video_path = "uploads/assets/videos/do_re_mi_with_music"
        if not Path(video_path).exists():
            print("Error: do_re_mi_with_music video not found")
            return None
    
    print(f"Using video: {video_path}")
    
    # Create scene splitter
    splitter = SceneSplitter(video_path)
    
    # Create scenes
    scenes = SceneSplitter.create_scenes_from_timestamps(
        scene_timestamps, 
        scene_descriptions
    )
    
    print(f"\nSplitting into {len(scenes)} scenes:")
    for scene in scenes:
        print(f"  Scene {scene.scene_number}: {scene.start_seconds:.1f}s - "
              f"{scene.end_seconds:.1f}s ({scene.duration:.1f}s) - {scene.description}")
    
    # Split the video
    print("\n" + "="*60)
    print("SPLITTING VIDEO INTO SCENES")
    print("="*60)
    
    results = splitter.split_into_scenes(scenes)
    
    print("\n" + "="*60)
    print("SCENE SPLITTING COMPLETE")
    print("="*60)
    print(f"\nOriginal scenes: {splitter.original_dir}")
    print(f"Downsampled scenes: {splitter.downsampled_dir}")
    print(f"Metadata: {splitter.metadata_dir}/scenes.json")
    
    return results


if __name__ == "__main__":
    results = split_do_re_mi_scenes()
    
    if results:
        print("\n✓ Scene splitting successful!")
        print(f"  • {len(results['original_scenes'])} original scenes")
        print(f"  • {len(results['downsampled_scenes'])} downsampled scenes")