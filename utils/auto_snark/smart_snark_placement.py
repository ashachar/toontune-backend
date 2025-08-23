#!/usr/bin/env python3
"""
Smart snark placement that:
1. Detects silent moments in the original audio
2. Places snarks only during silence or quiet background music
3. Uses slow motion as fallback when needed
"""

import os
import numpy as np
import subprocess
import json
from pydub import AudioSegment
from pydub.silence import detect_silence
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import librosa

def analyze_audio_silence(audio_path, min_silence_len=1500, silence_thresh=-40):
    """
    Analyze audio to find silent or quiet moments
    Returns list of (start_ms, end_ms) tuples for silent periods
    """
    print("🔍 Analyzing audio for silent moments...")
    
    # Load audio
    audio = AudioSegment.from_file(audio_path)
    
    # Detect silence (or very quiet moments)
    silent_ranges = detect_silence(
        audio, 
        min_silence_len=min_silence_len,  # Minimum 1.5 seconds of silence
        silence_thresh=silence_thresh  # dB threshold for silence
    )
    
    # Also detect quieter moments (background music only)
    quiet_ranges = detect_silence(
        audio,
        min_silence_len=1000,  # 1 second minimum
        silence_thresh=-35  # Less strict threshold
    )
    
    print(f"  Found {len(silent_ranges)} silent periods (< {silence_thresh}dB)")
    print(f"  Found {len(quiet_ranges)} quiet periods (< -35dB)")
    
    return silent_ranges, quiet_ranges

def get_speech_segments(audio_path):
    """
    Detect speech vs non-speech segments using energy analysis
    """
    print("🎤 Detecting speech segments...")
    
    # Load audio with librosa for more detailed analysis
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Calculate energy
    hop_length = 512
    frame_length = 2048
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert to time
    frames = range(len(energy))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    
    # Threshold for speech (adaptive)
    energy_threshold = np.percentile(energy, 30)  # Bottom 30% considered quiet
    
    # Find quiet segments
    quiet_segments = []
    in_quiet = False
    start_time = 0
    
    for i, (t, e) in enumerate(zip(times, energy)):
        if e < energy_threshold and not in_quiet:
            in_quiet = True
            start_time = t
        elif e >= energy_threshold and in_quiet:
            in_quiet = False
            if t - start_time > 1.0:  # At least 1 second
                quiet_segments.append((start_time * 1000, t * 1000))
    
    print(f"  Found {len(quiet_segments)} quiet segments suitable for snarks")
    return quiet_segments

def find_best_snark_positions(video_path, snark_durations):
    """
    Find the best positions to insert snarks without overlapping speech
    """
    print("\n📊 Finding optimal snark positions...")
    
    # Extract audio from video
    temp_audio = "temp_audio_analysis.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        temp_audio
    ], capture_output=True)
    
    # Analyze for silence
    silent_ranges, quiet_ranges = analyze_audio_silence(temp_audio)
    
    # Get speech segments for more accurate detection
    try:
        speech_quiet = get_speech_segments(temp_audio)
        quiet_ranges.extend(speech_quiet)
    except:
        print("  ⚠️ Advanced speech detection unavailable, using basic silence detection")
    
    # Sort and merge overlapping quiet periods
    all_quiet = sorted(set(silent_ranges + quiet_ranges))
    
    # Match snarks to quiet periods
    snark_positions = []
    used_ranges = []
    
    # Existing snark files with their durations
    snarks = [
        ("v3_snark_1_sarcastic.mp3", 3.8, 2400),  # Original time, duration in ms
        ("v3_snark_2_condescending.mp3", 16.8, 2300),
        ("v3_snark_3_mocking.mp3", 25.5, 2100),
        ("v3_snark_4_skeptical.mp3", 41.8, 2100),
        ("v3_snark_5_deadpan.mp3", 48.0, 2100),
        ("v3_snark_6_sarcastic.mp3", 52.5, 1800)
    ]
    
    print("\n🎯 Matching snarks to quiet moments:")
    
    for snark_file, original_time, duration_ms in snarks:
        best_position = None
        best_score = float('inf')
        
        # Try to find a quiet spot near the original time
        target_ms = original_time * 1000
        
        for start_ms, end_ms in all_quiet:
            if (start_ms, end_ms) in used_ranges:
                continue
                
            # Check if snark fits in this quiet period
            if end_ms - start_ms >= duration_ms:
                # Score based on distance from original position
                distance = abs(start_ms - target_ms)
                
                # Prefer positions close to original but in silence
                if distance < best_score and distance < 5000:  # Within 5 seconds
                    best_position = start_ms
                    best_score = distance
        
        if best_position is not None:
            snark_positions.append({
                "file": snark_file,
                "original_time": original_time,
                "new_time": best_position / 1000,
                "duration": duration_ms / 1000,
                "placement": "silent"
            })
            used_ranges.append((best_position, best_position + duration_ms))
            print(f"  ✅ {os.path.basename(snark_file)}: {original_time:.1f}s → {best_position/1000:.1f}s (silent)")
        else:
            # No silent spot found - use ducking at original position
            snark_positions.append({
                "file": snark_file,
                "original_time": original_time,
                "new_time": original_time,
                "duration": duration_ms / 1000,
                "placement": "ducked"
            })
            print(f"  ⚠️ {os.path.basename(snark_file)}: {original_time:.1f}s (with ducking)")
    
    # Cleanup
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    
    return snark_positions

def create_smart_video(video_path, snark_positions, output_path):
    """
    Create video with snarks placed intelligently
    """
    print("\n🎬 Creating smart placement video...")
    
    # Load video
    video = VideoFileClip(video_path)
    
    # Separate silent placements from ducked placements
    silent_snarks = [s for s in snark_positions if s["placement"] == "silent"]
    ducked_snarks = [s for s in snark_positions if s["placement"] == "ducked"]
    
    print(f"\n🎚️ Mixing audio with {len(silent_snarks)} silent placements and {len(ducked_snarks)} ducked placements...")
    
    # Extract original audio
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "original_audio.wav"
    ], capture_output=True)
    
    # Mix all snarks (both silent and ducked)
    all_snarks = silent_snarks + ducked_snarks
    
    if all_snarks:
        cmd = ["ffmpeg", "-y", "-i", "original_audio.wav"]
        
        for snark in all_snarks:
            cmd.extend(["-i", snark["file"]])
        
        # Build complex filter
        filter_parts = []
        
        # First, apply ducking to original audio for ducked snarks
        audio_stream = "0:a"
        for i, snark in enumerate(all_snarks):
            if snark["placement"] == "ducked":
                start_ms = int(snark["new_time"] * 1000)
                duration_ms = int(snark["duration"] * 1000)
                # Duck audio during this snark (reduce volume to 20%)
                filter_parts.append(f"[{audio_stream}]volume=enable='between(t,{snark['new_time']},{snark['new_time'] + snark['duration']})':volume=0.2[duck{i}]")
                audio_stream = f"duck{i}"
        
        # Rename final ducked stream
        if audio_stream != "0:a":
            filter_parts.append(f"[{audio_stream}]anull[ducked]")
            audio_stream = "ducked"
        
        # Add delays to snark files
        for i, snark in enumerate(all_snarks):
            delay_ms = int(snark["new_time"] * 1000)
            filter_parts.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[snark{i}]")
        
        # Mix everything together
        mix_inputs = [audio_stream] + [f"snark{i}" for i in range(len(all_snarks))]
        filter_parts.append(f"[{''.join(f'[{inp}]' for inp in mix_inputs)}]amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0")
        
        filter_str = ";".join(filter_parts)
        
        cmd.extend([
            "-filter_complex", filter_str,
            "-ac", "2",
            "mixed_audio_smart.wav"
        ])
        
        subprocess.run(cmd, capture_output=True)
    
    final_video = video
    
    # Apply the smart audio mix
    if os.path.exists("mixed_audio_smart.wav"):
        smart_audio = AudioFileClip("mixed_audio_smart.wav")
        final_video = final_video.set_audio(smart_audio)
    
    # Export final video
    print(f"\n💾 Exporting to {output_path}...")
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp_audio.m4a",
        remove_temp=True,
        verbose=False,
        logger=None
    )
    
    # Cleanup
    for f in ["original_audio.wav", "mixed_audio_smart.wav", "temp_audio.m4a"]:
        if os.path.exists(f):
            os.remove(f)
    
    print(f"✅ Smart placement video created: {output_path}")

def main():
    print("=" * 70)
    print("🎯 SMART SNARK PLACEMENT - NO OVERLAP WITH SPEECH")
    print("=" * 70)
    
    video_path = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi/scenes/downsampled/scene_001.mp4"
    output_path = "do_re_mi_smart_placement.mp4"
    
    # Check for existing snark audio files
    snark_files = [
        "v3_snark_1_sarcastic.mp3",
        "v3_snark_2_condescending.mp3",
        "v3_snark_3_mocking.mp3",
        "v3_snark_4_skeptical.mp3",
        "v3_snark_5_deadpan.mp3",
        "v3_snark_6_sarcastic.mp3"
    ]
    
    missing = [f for f in snark_files if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing snark files: {missing}")
        print("Please run the ElevenLabs demo first to generate audio files")
        return
    
    print("✅ Found all 6 pre-generated snark audio files")
    print("💰 No additional API costs - reusing existing audio\n")
    
    # Analyze and find best positions
    snark_positions = find_best_snark_positions(video_path, snark_files)
    
    # Create the smart video
    create_smart_video(video_path, snark_positions, output_path)
    
    # Save report
    report = {
        "strategy": "smart_placement",
        "total_snarks": len(snark_positions),
        "silent_placements": len([s for s in snark_positions if s["placement"] == "silent"]),
        "ducked_placements": len([s for s in snark_positions if s["placement"] == "ducked"]),
        "positions": snark_positions
    }
    
    with open("smart_placement_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Report saved: smart_placement_report.json")
    print(f"🎬 Play the result: open {output_path}")

if __name__ == "__main__":
    main()