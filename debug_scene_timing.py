#!/usr/bin/env python3
"""
Debug the timing alignment issue - check when audio actually starts in scene_001.mp4
"""

import subprocess
import json
import numpy as np
from pathlib import Path
import cv2

def analyze_scene_timing():
    """Analyze the actual content timing in scene_001.mp4"""
    
    print("="*70)
    print("SCENE TIMING ANALYSIS")
    print("="*70)
    
    # Check video duration and properties
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    
    # Get video info using ffprobe
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    
    duration = float(info["format"]["duration"])
    print(f"\nVideo file: {video_path}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Check for audio stream
    audio_stream = None
    video_stream = None
    for stream in info["streams"]:
        if stream["codec_type"] == "audio":
            audio_stream = stream
        elif stream["codec_type"] == "video":
            video_stream = stream
    
    if audio_stream:
        print(f"Audio: {audio_stream['codec_name']}, {audio_stream['sample_rate']} Hz")
        audio_start = float(audio_stream.get("start_time", 0))
        print(f"Audio start time: {audio_start:.3f}s")
    
    if video_stream:
        print(f"Video: {video_stream['width']}x{video_stream['height']}, {video_stream['r_frame_rate']} fps")
        video_start = float(video_stream.get("start_time", 0))
        print(f"Video start time: {video_start:.3f}s")
    
    # Analyze audio levels to find when speech starts
    print("\n" + "-"*50)
    print("Detecting audio activity (when speech begins)...")
    
    # Extract audio levels using ffmpeg
    cmd = [
        "ffmpeg", "-i", video_path,
        "-af", "volumedetect",
        "-f", "null", "-"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("\nAudio volume analysis:")
    for line in result.stderr.split('\n'):
        if 'volumedetect' in line:
            print(f"  {line.strip()}")
    
    # Get audio waveform data for first 10 seconds
    print("\n" + "-"*50)
    print("Extracting audio waveform to find speech onset...")
    
    # Extract raw audio samples
    cmd = [
        "ffmpeg", "-i", video_path,
        "-t", "10",  # First 10 seconds
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1",  # Mono
        "-ar", "8000",  # 8kHz sample rate for analysis
        "-"
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0:
        # Convert bytes to numpy array
        audio_data = np.frombuffer(result.stdout, dtype=np.float32)
        
        # Find first significant audio (above threshold)
        threshold = 0.01  # Adjust based on noise floor
        window_size = 800  # 0.1 second windows at 8kHz
        
        for i in range(0, len(audio_data) - window_size, window_size):
            window = audio_data[i:i+window_size]
            rms = np.sqrt(np.mean(window**2))
            if rms > threshold:
                time_offset = i / 8000.0
                print(f"\nFirst significant audio detected at: {time_offset:.3f} seconds")
                break
    
    # Load word timings
    print("\n" + "-"*50)
    print("Checking word timings from JSON...")
    
    json_path = "uploads/assets/videos/do_re_mi/inferences/scene_001_v3_safe.json"
    with open(json_path) as f:
        data = json.load(f)
    
    words = data["text_overlays"]
    
    # First few words
    print(f"\nFirst 5 words (original timings):")
    for word in words[:5]:
        print(f"  '{word['word']}': {word['start']:.2f}s - {word['end']:.2f}s")
    
    # What we're currently doing
    print(f"\nCurrent offset calculation:")
    print(f"  Scene offset: 7.92s")
    print(f"  First word 'Let's' at 7.92s - 7.92s = 0.00s in scene video")
    
    # Check if there's a mismatch
    print("\n" + "-"*50)
    print("POTENTIAL ISSUES:")
    
    # Load transcript to see original timing
    transcript_path = "uploads/assets/videos/do_re_mi/transcripts/transcript_words.json"
    if Path(transcript_path).exists():
        with open(transcript_path) as f:
            transcript = json.load(f)
        
        # Find "Let's" in original transcript
        for word_data in transcript["words"]:
            if word_data["word"] == "Let's":
                print(f"\n1. Original transcript has 'Let's' at {word_data['start']:.3f}s")
                print(f"   But we're placing it at 0.00s in scene video")
                print(f"   If scene has silence at start, this would be wrong!")
                break
    
    # Sample a few frames to check for visual content
    print("\n" + "-"*50)
    print("Checking visual content at key timestamps...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Check frames at 0s, 1s, 2s
    for timestamp in [0.0, 1.0, 2.0]:
        frame_num = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Check if frame has significant content (not black)
            mean_brightness = np.mean(frame)
            print(f"  t={timestamp:.1f}s: Mean brightness = {mean_brightness:.1f}")
            if mean_brightness < 10:
                print(f"    ⚠️ Very dark frame - might be black/padding")
    
    cap.release()
    
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)
    print("\nThe issue is likely one of:")
    print("1. Scene video has padding/silence at the beginning")
    print("2. Scene was extracted with different boundaries than expected")
    print("3. The 7.92s offset assumption is incorrect")
    print("\nTo fix: Detect actual speech onset and adjust timing accordingly")

if __name__ == "__main__":
    analyze_scene_timing()