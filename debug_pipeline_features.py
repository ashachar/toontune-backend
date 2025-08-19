#!/usr/bin/env python3
"""
Debug Pipeline Features
=======================

Thoroughly checks what features are actually present in the pipeline output.
"""

import json
import subprocess
import cv2
from pathlib import Path
from datetime import datetime

def check_video_properties(video_path):
    """Get detailed video properties."""
    if not video_path.exists():
        return None
    
    # Get file size
    size_mb = video_path.stat().st_size / (1024*1024)
    
    # Get video info with ffprobe
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        
        video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), {})
        
        return {
            'size_mb': size_mb,
            'width': video_stream.get('width', 0),
            'height': video_stream.get('height', 0),
            'codec': video_stream.get('codec_name', 'unknown'),
            'duration': float(video_stream.get('duration', 0))
        }
    except:
        return {'size_mb': size_mb}

def extract_frame(video_path, time_seconds, output_path):
    """Extract a single frame at specified time."""
    cmd = [
        'ffmpeg', '-ss', str(time_seconds),
        '-i', str(video_path),
        '-frames:v', '1',
        '-y', str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)
    return output_path.exists()

def debug_pipeline():
    base_dir = Path("uploads/assets/videos/do_re_mi")
    
    print("="*70)
    print("🔍 DEEP PIPELINE DEBUGGING")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Check inference data
    print("📄 INFERENCE DATA:")
    print("-"*40)
    inference_file = base_dir / "inferences/scene_001_inference.json"
    if inference_file.exists():
        with open(inference_file) as f:
            inference = json.load(f)
        
        scene = inference.get('scenes', [{}])[0]
        
        # Key phrases
        key_phrases = scene.get('key_phrases', [])
        print(f"Key Phrases: {len(key_phrases)}")
        for i, p in enumerate(key_phrases, 1):
            print(f"  {i}. '{p['phrase']}' at {p['start_seconds']}s for {p['duration_seconds']}s")
            print(f"     Position: x={p.get('top_left_pixels', {}).get('x', '?')}, y={p.get('top_left_pixels', {}).get('y', '?')}")
            print(f"     Style: {p.get('style', 'default')}, Importance: {p.get('importance', 'normal')}")
        
        # Cartoons
        cartoons = scene.get('cartoon_characters', [])
        print(f"\nCartoon Characters: {len(cartoons)}")
        for i, c in enumerate(cartoons, 1):
            print(f"  {i}. {c['character_type']} at {c['start_seconds']}s")
            print(f"     Animation: {c.get('animation_style', 'static')}")
            print(f"     Duration: {c.get('duration_seconds', 3)}s")
    else:
        print("  ❌ No inference file found!")
    
    # 2. Check video files and sizes
    print("\n📹 VIDEO FILES:")
    print("-"*40)
    
    videos = {
        "Original": base_dir / "scenes/original/scene_001.mp4",
        "Downsampled": base_dir / "scenes/downsampled/scene_001.mp4",
        "Edited": base_dir / "scenes/edited/scene_001.mp4"
    }
    
    for name, path in videos.items():
        if path.exists():
            props = check_video_properties(path)
            if props:
                print(f"{name:12} {props['size_mb']:.1f} MB, {props.get('width', '?')}x{props.get('height', '?')}, {props.get('codec', '?')}")
        else:
            print(f"{name:12} NOT FOUND")
    
    # 3. Extract frames at key times to check visually
    print("\n🎬 EXTRACTING FRAMES FOR VISUAL CHECK:")
    print("-"*40)
    
    edited_video = base_dir / "scenes/edited/scene_001.mp4"
    if edited_video.exists() and key_phrases:
        debug_dir = base_dir / "debug_frames"
        debug_dir.mkdir(exist_ok=True)
        
        # Extract frame at each phrase time
        for i, phrase in enumerate(key_phrases, 1):
            time = float(phrase['start_seconds']) + 1.0  # 1 second after start
            frame_path = debug_dir / f"phrase_{i}_at_{time:.1f}s.png"
            
            if extract_frame(edited_video, time, frame_path):
                print(f"  ✓ Extracted frame for phrase '{phrase['phrase']}' at {time:.1f}s")
                print(f"    Saved to: {frame_path}")
            else:
                print(f"  ❌ Failed to extract frame for phrase '{phrase['phrase']}'")
        
        # Extract frame at cartoon times
        for i, cartoon in enumerate(cartoons, 1):
            time = float(cartoon['start_seconds']) + 1.0
            frame_path = debug_dir / f"cartoon_{i}_at_{time:.1f}s.png"
            
            if extract_frame(edited_video, time, frame_path):
                print(f"  ✓ Extracted frame for {cartoon['character_type']} at {time:.1f}s")
            else:
                print(f"  ❌ Failed to extract frame for {cartoon['character_type']}")
    
    # 4. Check cartoon assets
    print("\n🎨 CARTOON ASSETS:")
    print("-"*40)
    
    asset_dirs = [
        Path("cartoon-test"),
        Path("uploads/assets/batch_images_transparent_bg"),
        base_dir / "cartoon_assets"
    ]
    
    for asset_dir in asset_dirs:
        if asset_dir.exists():
            pngs = list(asset_dir.glob("*.png"))
            if pngs:
                print(f"  {asset_dir}: {len(pngs)} PNG files")
                for png in pngs[:3]:  # Show first 3
                    print(f"    - {png.name}")
        else:
            print(f"  {asset_dir}: NOT FOUND")
    
    # 5. Check what the embedding steps are looking for
    print("\n🔎 ASSET MAPPING CHECK:")
    print("-"*40)
    
    # Simulate what step_9 looks for
    for cartoon in cartoons:
        char_type = cartoon['character_type']
        print(f"\nLooking for '{char_type}':")
        
        # Check mappings from step_9
        if 'deer' in char_type.lower():
            search = ['deer', 'animal', 'spring', 'baby']
        elif 'sun' in char_type.lower():
            search = ['sun', 'star', 'balloon', 'spring']
        else:
            search = [char_type]
        
        print(f"  Search terms: {search}")
        
        found = False
        for asset_dir in asset_dirs:
            if not asset_dir.exists():
                continue
            for term in search:
                matches = list(asset_dir.glob(f"*{term}*.png"))
                if matches:
                    print(f"  ✓ Found in {asset_dir.name}: {matches[0].name}")
                    found = True
                    break
            if found:
                break
        
        if not found:
            print(f"  ❌ No matching asset found!")
    
    # 6. Check pipeline state
    print("\n📊 PIPELINE STATE:")
    print("-"*40)
    
    state_file = base_dir / "metadata/pipeline_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        
        print("Steps completed:")
        for step in state.get('steps_completed', []):
            print(f"  ✓ {step}")
    
    # 7. Test FFmpeg commands directly
    print("\n🔧 TESTING FFMPEG COMMANDS:")
    print("-"*40)
    
    # Test if we can add a simple text overlay
    if edited_video.exists():
        test_output = base_dir / "debug_frames/test_overlay.mp4"
        test_output.parent.mkdir(exist_ok=True)
        
        cmd = [
            'ffmpeg', '-i', str(edited_video),
            '-vf', "drawtext=text='TEST':fontsize=40:fontcolor=red:x=100:y=100:enable='between(t,0,5)'",
            '-t', '10',  # Just first 10 seconds
            '-c:v', 'libx264', '-crf', '18',
            '-y', str(test_output)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ FFmpeg drawtext test successful")
            print(f"    Test video: {test_output}")
        else:
            print("  ❌ FFmpeg drawtext test failed:")
            print(f"    {result.stderr[:200]}")
    
    print("\n" + "="*70)
    print("DEBUGGING COMPLETE")
    print("Check the debug_frames folder for extracted frames!")
    print("="*70)

if __name__ == "__main__":
    debug_pipeline()