#!/usr/bin/env python3
"""
Transcript-Based Scene Splitter

Splits video into scenes based on transcript timestamps and speech patterns.
Creates natural scene boundaries at speech pauses while respecting maximum scene duration.
"""

import argparse
import subprocess
import json
import sys
from pathlib import Path
import time


def load_transcript(transcript_path):
    """Load transcript from JSON file"""
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading transcript: {e}")
        return None


def get_video_duration(video_path):
    """Get video duration in milliseconds"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            duration_s = float(info.get('format', {}).get('duration', 0))
            return duration_s * 1000  # Convert to milliseconds
    except Exception as e:
        print(f"Error getting video duration: {e}")
    return 0


def find_scene_boundaries(transcript_data, max_scene_duration_ms=60000, min_gap_ms=500):
    """
    Find optimal scene boundaries based on transcript
    
    Args:
        transcript_data: Transcript dictionary with segments
        max_scene_duration_ms: Maximum scene duration in milliseconds (default: 60000)
        min_gap_ms: Minimum gap between speech to consider as scene boundary (default: 500)
    
    Returns:
        List of scene boundary timestamps in milliseconds
    """
    if not transcript_data or 'transcript' not in transcript_data:
        return []
    
    segments = transcript_data['transcript']
    if not segments:
        return []
    
    # Get total duration
    total_duration = transcript_data.get('metadata', {}).get('total_duration_ms', 0)
    if not total_duration and segments:
        total_duration = max(seg['end_ms'] for seg in segments)
    
    boundaries = [0]  # Start with beginning
    current_scene_start = 0
    
    print(f"📊 Analyzing {len(segments)} transcript segments...")
    
    # Target duration is slightly less than max to allow flexibility
    target_duration_ms = max_scene_duration_ms * 0.9  # 54 seconds for 60 second max
    
    for i, segment in enumerate(segments):
        current_duration = segment['end_ms'] - current_scene_start
        
        # Check if we're approaching or exceeding target duration
        if current_duration >= target_duration_ms:
            # Look for the best natural break point
            best_boundary = None
            best_boundary_score = -1
            
            # Look ahead for natural breaks within the next few segments
            for j in range(i, min(i + 5, len(segments))):
                if j < len(segments) - 1:
                    current_seg = segments[j]
                    next_seg = segments[j + 1]
                    gap_duration = next_seg['start_ms'] - current_seg['end_ms']
                    scene_duration = current_seg['end_ms'] - current_scene_start
                    
                    # Score based on gap size and proximity to target duration
                    if gap_duration >= min_gap_ms and scene_duration <= max_scene_duration_ms:
                        # Higher score for larger gaps and closer to target duration
                        gap_score = min(gap_duration / 1000, 5)  # Cap at 5 points for gap
                        duration_score = 10 - abs(scene_duration - target_duration_ms) / 1000  # Closer to target is better
                        total_score = gap_score + duration_score
                        
                        if total_score > best_boundary_score:
                            best_boundary = current_seg['end_ms']
                            best_boundary_score = total_score
                
                # Stop if we're getting too long
                if segments[j]['end_ms'] - current_scene_start > max_scene_duration_ms:
                    break
            
            # Use the best boundary found, or force a split at max duration
            if best_boundary:
                boundaries.append(best_boundary)
                current_scene_start = best_boundary
                print(f"  Scene boundary at {best_boundary/1000:.1f}s (optimal break near {current_duration/1000:.1f}s)")
            elif current_duration > max_scene_duration_ms:
                # Force a split at segment boundary
                boundaries.append(segment['start_ms'])
                current_scene_start = segment['start_ms']
                print(f"  Scene boundary at {segment['start_ms']/1000:.1f}s (max duration exceeded)")
    
    # Add end boundary
    if total_duration > 0:
        boundaries.append(total_duration)
    elif segments:
        boundaries.append(segments[-1]['end_ms'])
    
    # Remove duplicates and sort
    boundaries = sorted(list(set(boundaries)))
    
    return boundaries


def split_video_by_transcript(video_path, transcript_path=None, max_scene_duration=60.0):
    """
    Split video into scenes based on transcript
    
    Args:
        video_path: Path to input video
        transcript_path: Path to transcript JSON (auto-detected if None)
        max_scene_duration: Maximum scene duration in seconds
    
    Returns:
        List of scene dictionaries
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return None
    
    # Find transcript if not provided
    if transcript_path is None:
        # Look in organized folder structure
        video_name = video_path.stem
        base_dir = video_path.parent / video_name
        soundtrack_dir = base_dir / "soundtrack"
        
        # Try sentence transcript first (optimized for scene splitting)
        transcript_path = soundtrack_dir / f"{video_name}_transcript_sentences.json"
        if not transcript_path.exists():
            # Try Whisper combined transcript
            transcript_path = soundtrack_dir / f"{video_name}_transcript_whisper.json"
        if not transcript_path.exists():
            # Try regular transcript
            transcript_path = soundtrack_dir / f"{video_name}_transcript.json"
        
        if not transcript_path.exists():
            print(f"❌ Error: Transcript not found at {transcript_path}")
            print(f"   Run extract_transcript_whisper.py or extract_transcript.py first to generate transcript")
            return None
    
    transcript_path = Path(transcript_path)
    
    print(f"[{time.strftime('%H:%M:%S')}] Loading transcript...")
    transcript_data = load_transcript(transcript_path)
    if not transcript_data:
        return None
    
    # Check if there's speech
    has_speech = transcript_data.get('metadata', {}).get('has_speech', True)
    if not has_speech or not transcript_data.get('transcript'):
        print("⚠️ No speech detected in transcript, falling back to time-based splitting")
        # Fall back to simple time-based splitting
        from simple_scene_splitter import split_video_simple
        return split_video_simple(video_path, max_scene_duration)
    
    # Get video duration for validation
    video_duration_ms = get_video_duration(video_path)
    
    # Find scene boundaries
    print(f"[{time.strftime('%H:%M:%S')}] Finding scene boundaries...")
    boundaries = find_scene_boundaries(
        transcript_data, 
        max_scene_duration_ms=max_scene_duration * 1000
    )
    
    if len(boundaries) < 2:
        print("⚠️ Could not find scene boundaries, falling back to time-based splitting")
        from simple_scene_splitter import split_video_simple
        return split_video_simple(video_path, max_scene_duration)
    
    # Create scenes from boundaries
    scenes = []
    for i in range(len(boundaries) - 1):
        start_ms = boundaries[i]
        end_ms = boundaries[i + 1]
        
        # Skip if scene is too short (less than 0.5 seconds)
        if end_ms - start_ms < 500:
            continue
        
        scene = {
            'scene_num': len(scenes) + 1,
            'start': start_ms / 1000,  # Convert to seconds
            'end': end_ms / 1000,
            'duration': (end_ms - start_ms) / 1000,
            'start_ms': start_ms,
            'end_ms': end_ms,
            'transcript_segments': []
        }
        
        # Find transcript segments in this scene
        for segment in transcript_data['transcript']:
            if segment['start_ms'] >= start_ms and segment['end_ms'] <= end_ms:
                scene['transcript_segments'].append(segment['text'])
        
        scenes.append(scene)
        print(f"  Scene {scene['scene_num']}: {scene['start']:.1f}s - {scene['end']:.1f}s ({scene['duration']:.1f}s)")
    
    print(f"✅ Created {len(scenes)} scenes based on transcript")
    
    # Create output directory in organized structure
    video_name = video_path.stem
    base_dir = video_path.parent / video_name
    scenes_dir = base_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract scene clips
    print(f"\n[{time.strftime('%H:%M:%S')}] Extracting scene clips to: {scenes_dir}")
    
    for scene in scenes:
        output_path = scenes_dir / f"scene_{scene['scene_num']:03d}.mp4"
        
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
    
    # Save scene index with transcript
    index_path = scenes_dir / "scene_index_transcript.txt"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcript-Based Scene Index for {video_path.name}\n")
        f.write("=" * 50 + "\n\n")
        
        for scene in scenes:
            f.write(f"Scene {scene['scene_num']:03d}:\n")
            f.write(f"  Time: {scene['start']:.2f}s - {scene['end']:.2f}s\n")
            f.write(f"  Duration: {scene['duration']:.2f}s\n")
            f.write(f"  File: scene_{scene['scene_num']:03d}.mp4\n")
            if scene['transcript_segments']:
                f.write(f"  Dialogue: {' '.join(scene['transcript_segments'][:100])}\n")
            f.write("\n")
    
    # Save JSON
    json_path = scenes_dir / "scenes_transcript.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(scenes, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Scene splitting complete!")
    print(f"   {len(scenes)} scenes created")
    print(f"   Output directory: {scenes_dir}")
    print(f"   Scene data: {json_path}")
    
    return scenes


def main():
    parser = argparse.ArgumentParser(
        description='Split video into scenes based on transcript and speech patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Requirements:
  - Transcript must be generated first using extract_transcript.py
  - ffmpeg must be installed

Examples:
  # Split using auto-detected transcript
  python scene_splitter_by_transcript.py video.mp4
  
  # Use specific transcript file
  python scene_splitter_by_transcript.py video.mp4 --transcript my_transcript.json
  
  # Set maximum scene duration to 5 seconds
  python scene_splitter_by_transcript.py video.mp4 --max-duration 5
        """
    )
    
    parser.add_argument('video', help='Input video file')
    parser.add_argument('--transcript', help='Path to transcript JSON file (auto-detected if not provided)')
    parser.add_argument('--max-duration', type=float, default=60.0,
                       help='Maximum scene duration in seconds (default: 60.0)')
    
    args = parser.parse_args()
    
    # Split scenes
    scenes = split_video_by_transcript(
        args.video,
        args.transcript,
        args.max_duration
    )
    
    if scenes:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())