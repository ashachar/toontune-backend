#!/usr/bin/env python3
"""
Test script for Video Scene Splitter

Tests the scene detection functionality with various configurations.
"""

import sys
import json
import subprocess
from pathlib import Path
import time


def create_test_video(output_path="test_video.mp4", duration=30):
    """Create a simple test video with distinct scenes using ffmpeg"""
    print(f"Creating test video with {duration}s duration...")
    
    # Create simpler video with color bars that change
    cmd = [
        'ffmpeg', '-f', 'lavfi',
        '-i', f'color=c=red:duration=5:s=640x480:r=25',
        '-f', 'lavfi', '-i', f'color=c=green:duration=5:s=640x480:r=25',
        '-f', 'lavfi', '-i', f'color=c=blue:duration=5:s=640x480:r=25',
        '-f', 'lavfi', '-i', f'color=c=yellow:duration=5:s=640x480:r=25',
        '-f', 'lavfi', '-i', f'color=c=cyan:duration=5:s=640x480:r=25',
        '-f', 'lavfi', '-i', f'color=c=magenta:duration=5:s=640x480:r=25',
        '-filter_complex', '[0][1][2][3][4][5]concat=n=6:v=1[out]',
        '-map', '[out]',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y', str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error creating test video: {result.stderr}")
        # Try even simpler approach
        cmd_simple = [
            'ffmpeg', '-f', 'lavfi',
            '-i', f'testsrc=duration={duration}:size=640x480:rate=25',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y', str(output_path)
        ]
        result = subprocess.run(cmd_simple, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error with simple test video: {result.stderr}")
            return False
    
    print(f"✅ Test video created: {output_path}")
    return True


def test_scene_detection(video_path, test_configs):
    """Test scene detection with various configurations"""
    from scene_splitter import SceneSplitter
    
    print(f"\n{'='*60}")
    print("TESTING SCENE SPLITTER")
    print(f"{'='*60}\n")
    
    results = []
    
    for config_name, config in test_configs.items():
        print(f"\n📝 Test: {config_name}")
        print(f"   Config: {config}")
        
        start_time = time.time()
        
        # Initialize splitter
        splitter = SceneSplitter(**config)
        
        # Detect scenes
        try:
            scenes = splitter.split_scenes(video_path)
            elapsed = time.time() - start_time
            
            print(f"   ✅ Success! Found {len(scenes)} scenes in {elapsed:.2f}s")
            
            # Validate JSON structure
            for scene in scenes:
                assert 'start' in scene
                assert 'end' in scene
                assert 'duration' in scene
                assert 'frame_start' in scene
                assert 'frame_end' in scene
                assert scene['end'] > scene['start']
                assert scene['duration'] > 0
            
            results.append({
                'test': config_name,
                'status': 'passed',
                'scenes': len(scenes),
                'time': elapsed
            })
            
            # Print scene details
            for i, scene in enumerate(scenes, 1):
                print(f"      Scene {i}: {scene['start']:.2f}s - {scene['end']:.2f}s "
                     f"({scene['duration']:.2f}s)")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'test': config_name,
                'status': 'failed',
                'error': str(e)
            })
    
    return results


def test_cli_interface(video_path):
    """Test the CLI interface"""
    print(f"\n{'='*60}")
    print("TESTING CLI INTERFACE")
    print(f"{'='*60}\n")
    
    tests = [
        {
            'name': 'Basic usage',
            'cmd': ['python', 'scene_splitter.py', str(video_path)]
        },
        {
            'name': 'High sensitivity',
            'cmd': ['python', 'scene_splitter.py', str(video_path), '--sensitivity', '0.8']
        },
        {
            'name': 'Custom output',
            'cmd': ['python', 'scene_splitter.py', str(video_path), '--output', 'test_scenes.json']
        },
        {
            'name': 'No audio analysis',
            'cmd': ['python', 'scene_splitter.py', str(video_path), '--no-audio']
        },
        {
            'name': 'Quiet mode',
            'cmd': ['python', 'scene_splitter.py', str(video_path), '--quiet']
        }
    ]
    
    for test in tests:
        print(f"📝 Test: {test['name']}")
        print(f"   Command: {' '.join(test['cmd'])}")
        
        result = subprocess.run(test['cmd'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ Success!")
            
            # Check if output file was created
            if '--output' in test['cmd']:
                output_idx = test['cmd'].index('--output')
                output_file = Path(test['cmd'][output_idx + 1])
            else:
                output_file = Path(video_path).with_suffix('.scenes.json')
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    scenes = json.load(f)
                print(f"   Found {len(scenes)} scenes")
                
                # Clean up test output
                if output_file.name == 'test_scenes.json':
                    output_file.unlink()
        else:
            print(f"   ❌ Failed with return code {result.returncode}")


def main():
    """Run all tests"""
    print("🎬 Scene Splitter Test Suite")
    
    # Check if test video exists or create it
    test_video = Path("test_video.mp4")
    
    if not test_video.exists():
        print("\n📹 Creating test video...")
        if not create_test_video(test_video):
            print("Failed to create test video")
            sys.exit(1)
    else:
        print(f"Using existing test video: {test_video}")
    
    # Test configurations
    test_configs = {
        "Default settings": {
            'sensitivity': 0.5,
            'min_scene_duration': 1.0,
            'sample_rate': 5,
            'use_audio': True,
            'verbose': True
        },
        "High sensitivity": {
            'sensitivity': 0.8,
            'min_scene_duration': 0.5,
            'sample_rate': 5,
            'use_audio': True,
            'verbose': True
        },
        "Low sensitivity": {
            'sensitivity': 0.2,
            'min_scene_duration': 2.0,
            'sample_rate': 5,
            'use_audio': True,
            'verbose': True
        },
        "Fast processing (no audio)": {
            'sensitivity': 0.5,
            'min_scene_duration': 1.0,
            'sample_rate': 10,
            'use_audio': False,
            'verbose': True
        },
        "Fine-grained (every frame)": {
            'sensitivity': 0.6,
            'min_scene_duration': 0.5,
            'sample_rate': 1,
            'use_audio': False,
            'verbose': False
        }
    }
    
    # Run scene detection tests
    results = test_scene_detection(test_video, test_configs)
    
    # Test CLI interface
    test_cli_interface(test_video)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}\n")
    
    passed = sum(1 for r in results if r['status'] == 'passed')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Please check the output above.")
    
    # Test with real video if provided
    if len(sys.argv) > 1:
        real_video = Path(sys.argv[1])
        if real_video.exists():
            print(f"\n{'='*60}")
            print(f"TESTING WITH REAL VIDEO: {real_video}")
            print(f"{'='*60}\n")
            
            from scene_splitter import SceneSplitter
            splitter = SceneSplitter(sensitivity=0.5, verbose=True)
            scenes = splitter.split_scenes(real_video)
            
            output_path = real_video.with_suffix('.scenes.json')
            with open(output_path, 'w') as f:
                json.dump(scenes, f, indent=2)
            
            print(f"\n✅ Detected {len(scenes)} scenes")
            print(f"   Output saved to: {output_path}")
            
            for i, scene in enumerate(scenes, 1):
                print(f"   Scene {i}: {scene['start']:.2f}s - {scene['end']:.2f}s "
                     f"({scene['duration']:.2f}s)")


if __name__ == '__main__':
    main()