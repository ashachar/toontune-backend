#!/usr/bin/env python3
"""
Verify that pipeline steps are chaining correctly
"""

import subprocess
import hashlib
from pathlib import Path
from datetime import datetime

def get_file_hash(filepath):
    """Get MD5 hash of a file."""
    if not filepath.exists():
        return None
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]

def get_video_info(filepath):
    """Get video codec and bitrate info."""
    if not filepath.exists():
        return {}
    
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', str(filepath)
    ]
    
    import json
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
        video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), {})
        return {
            'codec': video_stream.get('codec_name', '?'),
            'bit_rate': int(video_stream.get('bit_rate', 0)) // 1000,  # kbps
            'profile': video_stream.get('profile', '?')
        }
    except:
        return {}

def main():
    base_dir = Path("uploads/assets/videos/do_re_mi/scenes/edited")
    
    print("="*70)
    print("🔗 PIPELINE STEP CHAINING VERIFICATION")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")
    
    video_path = base_dir / "scene_001.mp4"
    
    if not video_path.exists():
        print("❌ No edited video found!")
        return
    
    # Get file info
    stat = video_path.stat()
    size_mb = stat.st_size / (1024 * 1024)
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    file_hash = get_file_hash(video_path)
    video_info = get_video_info(video_path)
    
    print(f"📹 Current Video State:")
    print(f"   Path: {video_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Hash: {file_hash}")
    print(f"   Modified: {mod_time.strftime('%H:%M:%S')}")
    print(f"   Codec: {video_info.get('codec', '?')}")
    print(f"   Bitrate: {video_info.get('bit_rate', 0)} kbps")
    print(f"   Profile: {video_info.get('profile', '?')}")
    
    # Now let's simulate what each step should do
    print("\n🔄 Simulating Pipeline Steps:")
    print("-"*50)
    
    # Test adding just phrases
    print("\n1. Testing PHRASES ONLY:")
    test_phrases = base_dir.parent / "test_pipeline"
    test_phrases.mkdir(exist_ok=True)
    
    phrases_only = test_phrases / "phrases_only.mp4"
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', "drawtext=text='TEST PHRASE 1':fontsize=30:fontcolor=yellow:x=100:y=100:enable='between(t,10,14)',drawtext=text='TEST PHRASE 2':fontsize=30:fontcolor=white:x=500:y=300:enable='between(t,22,26)'",
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-t', '30',  # Just first 30 seconds
        '-y', str(phrases_only)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✓ Phrases overlay successful")
        print(f"     Output: {phrases_only.name} ({phrases_only.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f"   ✗ Phrases failed: {result.stderr[:100]}")
    
    # Test adding cartoon overlay
    print("\n2. Testing CARTOON OVERLAY:")
    # Use spring.png as test cartoon
    spring_path = Path("cartoon-test/spring.png")
    if spring_path.exists():
        cartoon_only = test_phrases / "cartoon_only.mp4"
        cmd = [
            'ffmpeg', '-i', str(video_path), '-i', str(spring_path),
            '-filter_complex', "[1:v]scale=100:100[cartoon];[0:v][cartoon]overlay=x=200:y=200:enable='between(t,5,10)'",
            '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
            '-t', '15',
            '-y', str(cartoon_only)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✓ Cartoon overlay successful")
            print(f"     Output: {cartoon_only.name} ({cartoon_only.stat().st_size / (1024*1024):.1f} MB)")
        else:
            print(f"   ✗ Cartoon failed: {result.stderr[:100]}")
    
    # Test combining both
    print("\n3. Testing COMBINED (Phrases + Cartoon):")
    if spring_path.exists():
        combined = test_phrases / "combined.mp4"
        cmd = [
            'ffmpeg', '-i', str(video_path), '-i', str(spring_path),
            '-filter_complex', (
                "[1:v]scale=100:100[cartoon];"
                "[0:v][cartoon]overlay=x=200:y=200:enable='between(t,5,10)'[with_cartoon];"
                "[with_cartoon]drawtext=text='PHRASE ON TOP':fontsize=30:fontcolor=red:x=300:y=50:enable='between(t,0,15)'"
            ),
            '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
            '-t', '15',
            '-y', str(combined)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✓ Combined overlay successful")
            print(f"     Output: {combined.name} ({combined.stat().st_size / (1024*1024):.1f} MB)")
        else:
            print(f"   ✗ Combined failed: {result.stderr[:100]}")
    
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS:")
    print("-"*50)
    
    if size_mb < 20:
        print("⚠️  Video size is low - possible quality loss during processing")
        print("   Consider using -crf 18 or lower for better quality")
    
    if video_info.get('bit_rate', 0) < 2000:
        print("⚠️  Bitrate is low - may affect visual quality")
        print("   Original was likely higher quality")
    
    print("\n✅ To verify all features are present:")
    print(f"1. Play the test videos in: {test_phrases}")
    print("2. Check if you can see:")
    print("   - Yellow 'TEST PHRASE 1' at 10-14 seconds")
    print("   - White 'TEST PHRASE 2' at 22-26 seconds")
    print("   - Cartoon overlay in the test files")
    print("="*70)

if __name__ == "__main__":
    main()