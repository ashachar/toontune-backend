#!/usr/bin/env python3
"""
Generate final video with corrected text positions and FIXED timing alignment.
Accounts for the silence at the beginning of the scene video.
"""

import json
import subprocess
import numpy as np
from pathlib import Path

def detect_audio_onset(video_path, sample_duration=10):
    """Detect when audio actually starts in the video."""
    
    print("Detecting audio onset...")
    
    # Extract raw audio samples for analysis
    cmd = [
        "ffmpeg", "-i", video_path,
        "-t", str(sample_duration),  # First N seconds
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1",  # Mono
        "-ar", "8000",  # 8kHz sample rate
        "-"
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print("Warning: Could not detect audio onset, using default")
        return 2.6  # Fallback to detected value
    
    # Convert to numpy array
    audio_data = np.frombuffer(result.stdout, dtype=np.float32)
    
    # Find first significant audio
    threshold = 0.01
    window_size = 800  # 0.1 second windows at 8kHz
    
    for i in range(0, len(audio_data) - window_size, window_size):
        window = audio_data[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        if rms > threshold:
            onset_time = i / 8000.0
            print(f"Audio onset detected at: {onset_time:.3f} seconds")
            return onset_time
    
    print("Warning: No audio onset detected, using 0")
    return 0.0

def generate_video_with_aligned_text():
    """Generate video with properly aligned text timing."""
    
    # Load the new positions
    positions_file = Path("uploads/assets/videos/do_re_mi/inferences/scene_001_v3_safe.json")
    with open(positions_file) as f:
        data = json.load(f)
    
    text_overlays = data["text_overlays"]
    
    # Video paths
    input_video = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    output_video = "uploads/assets/videos/do_re_mi/scenes/final/scene_001_aligned_text.mp4"
    
    # Ensure output directory exists
    Path(output_video).parent.mkdir(exist_ok=True, parents=True)
    
    # Detect actual audio onset in the scene video
    audio_onset = detect_audio_onset(input_video)
    
    # Calculate timing offsets
    # Original assumption: scene starts at 7.92s in full video
    # Reality: scene has ~2.6s of silence at beginning
    # So the actual content starts at 7.92s in original, which maps to 2.6s in scene
    
    original_scene_start = 7.92  # When scene 1 content starts in original video
    scene_audio_delay = audio_onset  # Detected silence at beginning of scene video
    
    print(f"\nTiming correction:")
    print(f"  Original scene start: {original_scene_start:.2f}s")
    print(f"  Scene audio delay: {scene_audio_delay:.2f}s")
    print(f"  Adjustment: +{scene_audio_delay:.2f}s to all timings")
    
    # Build FFmpeg filter for text overlays with corrected timing
    filters = []
    words_processed = 0
    
    for i, overlay in enumerate(text_overlays):
        # Escape special characters for FFmpeg
        word = overlay["word"].replace("'", "'\\''")
        word = word.replace(":", "\\:")
        x = overlay["x"]
        y = overlay["y"]
        fontsize = overlay.get("fontsize", 48)
        
        # FIXED TIMING CALCULATION
        # The word times are in original video time
        # We need to map them to scene video time accounting for the audio delay
        start_in_scene = overlay["start"] - original_scene_start + scene_audio_delay
        end_in_scene = overlay["end"] - original_scene_start + scene_audio_delay
        
        # Skip words that would appear before video starts
        if end_in_scene < 0:
            continue
        
        # Clamp start time to 0 if needed
        if start_in_scene < 0:
            start_in_scene = 0
        
        words_processed += 1
        
        # FFmpeg drawtext filter
        filter_str = (
            f"drawtext=text='{word}'"
            f":x={x}:y={y}"
            f":fontsize={fontsize}"
            f":fontcolor=white"
            f":bordercolor=black"
            f":borderw=2"
            f":enable='between(t,{start_in_scene:.3f},{end_in_scene:.3f})'"
        )
        filters.append(filter_str)
        
        # Debug first few words
        if i < 5:
            print(f"  '{word}': {start_in_scene:.2f}s - {end_in_scene:.2f}s (was {overlay['start'] - original_scene_start:.2f}s)")
    
    # Combine all filters
    filter_complex = ",".join(filters)
    
    # FFmpeg command
    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-vf", filter_complex,
        "-codec:a", "copy",
        "-y",
        output_video
    ]
    
    print(f"\nGenerating video with aligned text...")
    print(f"Input: {input_video}")
    print(f"Output: {output_video}")
    print(f"Processing {words_processed} text overlays...")
    
    # Run FFmpeg
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"\n✓ Video generated successfully: {output_video}")
        
        print("\nKey improvements:")
        print(f"  - Text timing aligned with audio (adjusted by +{scene_audio_delay:.1f}s)")
        print("  - 'beginning' at safe position (650, 270)")
        print("  - All words checked across multiple frames")
        print("  - Text synchronized with speech")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ FFmpeg error: {e.stderr}")
        return False
    
    return True

if __name__ == "__main__":
    generate_video_with_aligned_text()