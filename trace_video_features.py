#!/usr/bin/env python3
"""
Trace all features in the final video with timestamps
"""

import json
import subprocess
from pathlib import Path
from datetime import timedelta

def format_time(seconds):
    """Format seconds to MM:SS.f format."""
    td = timedelta(seconds=seconds)
    minutes = int(td.total_seconds() // 60)
    secs = td.total_seconds() % 60
    return f"{minutes:02d}:{secs:05.2f}"

def trace_features():
    base_dir = Path("uploads/assets/videos/do_re_mi")
    
    print("="*70)
    print("🎬 VIDEO FEATURE TIMELINE")
    print("="*70)
    
    # Load inference to get expected features
    inference_file = base_dir / "inferences/scene_001_inference.json"
    with open(inference_file) as f:
        inference = json.load(f)
    
    scene = inference.get('scenes', [{}])[0]
    
    # Create timeline
    features = []
    
    # Add phrases
    for p in scene.get('key_phrases', []):
        start = float(p['start_seconds'])
        end = start + float(p['duration_seconds'])
        features.append({
            'type': 'PHRASE',
            'name': p['phrase'],
            'start': start,
            'end': end,
            'details': f"Style: {p.get('style')}, Pos: ({p.get('top_left_pixels', {}).get('x')}, {p.get('top_left_pixels', {}).get('y')})"
        })
    
    # Add cartoons  
    for c in scene.get('cartoon_characters', []):
        start = float(c['start_seconds'])
        end = start + float(c.get('duration_seconds', 3))
        features.append({
            'type': 'CARTOON',
            'name': c['character_type'],
            'start': start,
            'end': end,
            'details': f"Animation: {c.get('animation_style')}"
        })
    
    # Sort by start time
    features.sort(key=lambda x: x['start'])
    
    print("\n📋 EXPECTED FEATURES TIMELINE:")
    print("-"*70)
    print(f"{'Time Range':<20} {'Type':<10} {'Content':<25} {'Details':<15}")
    print("-"*70)
    
    for f in features:
        time_range = f"{format_time(f['start'])} - {format_time(f['end'])}"
        print(f"{time_range:<20} {f['type']:<10} {f['name']:<25} {f['details']:<15}")
    
    # Check video duration
    edited_video = base_dir / "scenes/edited/scene_001.mp4"
    if edited_video.exists():
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(edited_video)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        
        print(f"\n📹 Video Duration: {format_time(duration)}")
        
        # Check if features are within video duration
        print("\n✅ Feature Visibility Check:")
        for f in features:
            if f['end'] <= duration:
                print(f"  ✓ {f['name']}: Fully within video duration")
            elif f['start'] < duration:
                print(f"  ⚠ {f['name']}: Partially cut off (ends at {format_time(f['end'])}, video ends at {format_time(duration)})")
            else:
                print(f"  ✗ {f['name']}: Outside video duration (starts at {format_time(f['start'])})")
    
    # Extract frames at multiple points to verify
    print("\n🎞️ Extracting frames at key moments:")
    print("-"*50)
    
    debug_dir = base_dir / "debug_timeline"
    debug_dir.mkdir(exist_ok=True)
    
    # Sample points throughout the video
    sample_times = [0, 5, 10, 11.5, 15, 20, 23, 30, 40, 47.5, 51.5, 55]
    
    for t in sample_times:
        if t > duration:
            continue
            
        frame_path = debug_dir / f"frame_at_{t:05.1f}s.png"
        cmd = [
            'ffmpeg', '-ss', str(t),
            '-i', str(edited_video),
            '-frames:v', '1',
            '-y', str(frame_path)
        ]
        subprocess.run(cmd, capture_output=True)
        
        # Check what features should be visible at this time
        visible = []
        for f in features:
            if f['start'] <= t <= f['end']:
                visible.append(f"{f['type']}:{f['name']}")
        
        if visible:
            print(f"  {format_time(t)}: {', '.join(visible)}")
        else:
            print(f"  {format_time(t)}: [no overlays expected]")
    
    print("\n" + "="*70)
    print("💡 TO CHECK VISUALLY:")
    print(f"1. Open the video: {edited_video}")
    print(f"2. Jump to these timestamps to see each feature:")
    for f in features:
        print(f"   - {format_time(f['start'])}: {f['name']} ({f['type']})")
    print(f"3. Check extracted frames in: {debug_dir}")
    print("="*70)

if __name__ == "__main__":
    trace_features()