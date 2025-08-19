#!/usr/bin/env python3
"""
Simple Video Scene Splitter

Splits video into fixed-duration scenes (default: 10 seconds each).
No complex analysis - just simple, predictable time-based splitting.
"""

import argparse
import subprocess
import json
import sys
from pathlib import Path
import time


def get_video_info(video_path):
    """Get video duration and metadata using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames',
            '-show_entries', 'format=duration',
            '-of', 'json',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            
            # Get duration from format section (more reliable)
            duration = float(info.get('format', {}).get('duration', 0))
            
            # Get stream info
            if info.get('streams'):
                stream = info['streams'][0]
                
                # Parse frame rate
                fps_str = stream.get('r_frame_rate', '25/1')
                fps_parts = fps_str.split('/')
                fps = float(fps_parts[0]) / float(fps_parts[1])
                
                return {
                    'width': int(stream.get('width', 0)),
                    'height': int(stream.get('height', 0)),
                    'fps': fps,
                    'duration': duration,
                    'total_frames': int(stream.get('nb_frames', fps * duration))
                }
            
            return {'duration': duration}
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None


def split_video_simple(video_path, scene_duration=10.0, output_dir=None):
    """
    Split video into fixed-duration scenes
    
    Args:
        video_path: Path to input video
        scene_duration: Duration of each scene in seconds (default: 10.0)
        output_dir: Output directory for scene clips
    
    Returns:
        List of scene dictionaries
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return None
    
    # Get video info
    print(f"[{time.strftime('%H:%M:%S')}] Analyzing video...")
    video_info = get_video_info(video_path)
    if not video_info:
        print("❌ Error: Could not get video information")
        return None
    
    duration = video_info['duration']
    fps = video_info.get('fps', 25)
    
    print(f"📹 Video duration: {duration:.2f}s")
    print(f"🔪 Splitting into {scene_duration}s scenes...")
    
    # Calculate scenes
    scenes = []
    current_time = 0.0
    scene_num = 1
    
    while current_time < duration:
        end_time = min(current_time + scene_duration, duration)
        
        scene = {
            'scene_num': scene_num,
            'start': current_time,
            'end': end_time,
            'duration': end_time - current_time,
            'frame_start': int(current_time * fps),
            'frame_end': int(end_time * fps)
        }
        
        scenes.append(scene)
        print(f"  Scene {scene_num}: {current_time:.1f}s - {end_time:.1f}s ({scene['duration']:.1f}s)")
        
        current_time = end_time
        scene_num += 1
    
    print(f"✅ Created {len(scenes)} scenes")
    
    # Create output directory with new structure
    video_name = video_path.stem
    base_dir = video_path.parent / video_name
    
    if output_dir is None:
        output_dir = base_dir / "scenes"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract scene clips
    print(f"\n[{time.strftime('%H:%M:%S')}] Extracting scene clips to: {output_dir}")
    
    for scene in scenes:
        output_path = output_dir / f"scene_{scene['scene_num']:03d}.mp4"
        
        # Use ffmpeg to extract scene
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-ss', str(scene['start']),
            '-t', str(scene['duration']),
            '-c', 'copy',  # Copy codec for speed
            '-avoid_negative_ts', 'make_zero',
            '-y', str(output_path)
        ]
        
        print(f"  Extracting scene {scene['scene_num']}/{len(scenes)}...")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # Try with re-encoding if copy fails
                cmd_alt = [
                    'ffmpeg', '-i', str(video_path),
                    '-ss', str(scene['start']),
                    '-t', str(scene['duration']),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-c:a', 'aac',
                    '-y', str(output_path)
                ]
                subprocess.run(cmd_alt, capture_output=True, text=True)
            
            scene['clip_path'] = str(output_path)
            
        except Exception as e:
            print(f"    Error extracting scene {scene['scene_num']}: {e}")
    
    # Save scene index
    index_path = output_dir / "scene_index.txt"
    with open(index_path, 'w') as f:
        f.write(f"Simple Scene Index for {video_path.name}\n")
        f.write(f"Scene duration: {scene_duration}s\n")
        f.write("=" * 50 + "\n\n")
        
        for scene in scenes:
            f.write(f"Scene {scene['scene_num']:03d}:\n")
            f.write(f"  Start: {scene['start']:.2f}s (frame {scene['frame_start']})\n")
            f.write(f"  End: {scene['end']:.2f}s (frame {scene['frame_end']})\n")
            f.write(f"  Duration: {scene['duration']:.2f}s\n")
            f.write(f"  File: scene_{scene['scene_num']:03d}.mp4\n\n")
    
    # Save JSON
    json_path = output_dir / "scenes_simple.json"
    with open(json_path, 'w') as f:
        json.dump(scenes, f, indent=2)
    
    print(f"\n✅ Scene splitting complete!")
    print(f"   {len(scenes)} scenes created")
    print(f"   Output directory: {output_dir}")
    print(f"   Scene data: {json_path}")
    
    return scenes


def main():
    parser = argparse.ArgumentParser(
        description='Simple video scene splitter - splits video into fixed-duration scenes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split into 10-second scenes (default)
  python simple_scene_splitter.py video.mp4
  
  # Split into 5-second scenes
  python simple_scene_splitter.py video.mp4 --duration 5
  
  # Split into 30-second scenes
  python simple_scene_splitter.py video.mp4 --duration 30
  
  # Custom output directory
  python simple_scene_splitter.py video.mp4 --output my_scenes
        """
    )
    
    parser.add_argument('input', help='Input video file')
    parser.add_argument('-d', '--duration', type=float, default=10.0,
                       help='Duration of each scene in seconds (default: 10.0)')
    parser.add_argument('-o', '--output', help='Output directory for scene clips')
    
    args = parser.parse_args()
    
    # Validate duration
    if args.duration <= 0:
        print(f"❌ Error: Duration must be positive")
        sys.exit(1)
    
    # Split scenes
    scenes = split_video_simple(args.input, args.duration, args.output)
    
    if scenes:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())