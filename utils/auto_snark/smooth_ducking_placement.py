#!/usr/bin/env python3
"""
Smart snark placement with SMOOTH gradual audio ducking
Uses advanced FFmpeg filters for professional-grade transitions
"""

import os
import subprocess
import json
from pydub import AudioSegment
from pydub.silence import detect_silence

def find_quiet_moments(audio_path):
    """Find quiet moments in audio where we can insert snarks"""
    
    print("🔍 Analyzing audio for quiet moments...")
    
    # Load audio
    audio = AudioSegment.from_file(audio_path)
    
    # Find silent periods (very quiet)
    silent_ranges = detect_silence(
        audio,
        min_silence_len=1500,  # At least 1.5 seconds
        silence_thresh=-40,  # Very quiet threshold
        seek_step=100
    )
    
    # Find quieter periods (background music level)
    quiet_ranges = detect_silence(
        audio,
        min_silence_len=1000,  # At least 1 second
        silence_thresh=-30,  # Less strict
        seek_step=100
    )
    
    print(f"  Found {len(silent_ranges)} very quiet periods")
    print(f"  Found {len(quiet_ranges)} moderately quiet periods")
    
    # Combine and sort all quiet periods
    all_quiet = []
    for start, end in silent_ranges:
        all_quiet.append((start, end, "silent"))
    for start, end in quiet_ranges:
        if not any(start >= s and end <= e for s, e, _ in all_quiet):
            all_quiet.append((start, end, "quiet"))
    
    all_quiet.sort(key=lambda x: x[0])
    
    return all_quiet

def create_smooth_ducking_video(video_path, output_path):
    """Create video with smooth gradual audio ducking"""
    
    print("\n🎬 Creating smooth ducking version...")
    
    # Extract audio for analysis
    temp_audio = "temp_audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "22050",
        temp_audio
    ], capture_output=True)
    
    # Find quiet moments
    quiet_moments = find_quiet_moments(temp_audio)
    
    # Our existing snarks with approximate durations
    snarks = [
        {
            "file": "v3_snark_1_sarcastic.mp3",
            "ideal_time": 3.8,
            "duration": 2.4,
            "text": "Oh good, another musical..."
        },
        {
            "file": "v3_snark_2_condescending.mp3",
            "ideal_time": 16.8,
            "duration": 2.3,
            "text": "Yes, because ABC..."
        },
        {
            "file": "v3_snark_3_mocking.mp3",
            "ideal_time": 25.5,
            "duration": 2.1,
            "text": "We get it..."
        },
        {
            "file": "v3_snark_4_skeptical.mp3",
            "ideal_time": 41.8,
            "duration": 2.1,
            "text": "Easier?..."
        },
        {
            "file": "v3_snark_5_deadpan.mp3",
            "ideal_time": 48.0,
            "duration": 2.1,
            "text": "Revolutionary..."
        },
        {
            "file": "v3_snark_6_sarcastic.mp3",
            "ideal_time": 52.5,
            "duration": 1.8,
            "text": "Mi, the narcissism..."
        }
    ]
    
    # Match snarks to quiet moments
    placements = []
    used_moments = []
    
    print("\n📍 Placing snarks:")
    
    for snark in snarks:
        placed = False
        ideal_ms = snark["ideal_time"] * 1000
        duration_ms = snark["duration"] * 1000
        
        # Look for quiet moment near ideal time
        best_moment = None
        best_distance = float('inf')
        
        for start_ms, end_ms, quietness in quiet_moments:
            if (start_ms, end_ms) in used_moments:
                continue
            
            # Check if snark fits
            if end_ms - start_ms >= duration_ms:
                # Calculate distance from ideal position
                distance = abs(start_ms - ideal_ms)
                
                # Prefer closer positions, within 5 seconds
                if distance < best_distance and distance < 5000:
                    best_moment = (start_ms, end_ms, quietness)
                    best_distance = distance
        
        if best_moment:
            start_ms, end_ms, quietness = best_moment
            placements.append({
                "snark": snark,
                "time": start_ms / 1000,
                "type": quietness,
                "original_time": snark["ideal_time"]
            })
            used_moments.append((start_ms, end_ms))
            placed = True
            print(f"  ✅ {snark['text'][:20]}... at {start_ms/1000:.1f}s ({quietness})")
        else:
            # No quiet spot - place at original time with smooth ducking
            placements.append({
                "snark": snark,
                "time": snark["ideal_time"],
                "type": "smooth_duck",
                "original_time": snark["ideal_time"]
            })
            print(f"  🎚️ {snark['text'][:20]}... at {snark['ideal_time']:.1f}s (smooth ducking)")
    
    # Create FFmpeg command with smooth ducking
    print("\n🎚️ Applying smooth gradual ducking...")
    
    # Start building complex filter
    filter_parts = []
    
    # Extract original audio
    cmd = ["ffmpeg", "-y", "-i", video_path]
    
    # Add all snark files as inputs
    for p in placements:
        cmd.extend(["-i", p["snark"]["file"]])
    
    # Build the complex audio filter with smooth ducking
    audio_stream = "0:a"
    
    # Apply smooth ducking for each snark that needs it
    for i, p in enumerate(placements):
        if p["type"] in ["smooth_duck", "quiet"]:  # Apply ducking to non-silent placements
            start_time = p["time"]
            duration = p["snark"]["duration"]
            fade_duration = 0.5  # 500ms fade in/out
            
            # Create smooth volume envelope
            # Format: volume=enable='between(t,START,END)':volume='EXPRESSION'
            # Use smooth sine-based transition for gradual fade
            
            fade_in_start = max(0, start_time - fade_duration)
            fade_in_end = start_time
            fade_out_start = start_time + duration
            fade_out_end = start_time + duration + fade_duration
            
            # Build expression for smooth ducking
            # During fade-in: gradually reduce from 1.0 to 0.25
            # During snark: maintain at 0.25
            # During fade-out: gradually increase from 0.25 to 1.0
            
            volume_expr = (
                f"'if(between(t,{fade_in_start},{fade_in_end}),"
                f"1.0-0.75*(t-{fade_in_start})/{fade_duration},"
                f"if(between(t,{fade_in_end},{fade_out_start}),"
                f"0.25,"
                f"if(between(t,{fade_out_start},{fade_out_end}),"
                f"0.25+0.75*(t-{fade_out_start})/{fade_duration},"
                f"1.0)))'"
            )
            
            filter_parts.append(
                f"[{audio_stream}]volume=enable='between(t,{fade_in_start},{fade_out_end})':volume={volume_expr}[duck{i}]"
            )
            audio_stream = f"duck{i}"
    
    # If we applied ducking, use the ducked stream
    if audio_stream != "0:a":
        filter_parts.append(f"[{audio_stream}]anull[ducked]")
        audio_stream = "ducked"
    
    # Add delay to each snark audio
    for i, p in enumerate(placements):
        delay_ms = int(p["time"] * 1000)
        filter_parts.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[snark{i}]")
    
    # Mix all audio streams together
    mix_inputs = [audio_stream] + [f"snark{i}" for i in range(len(placements))]
    mix_filter = "[" + "][".join(mix_inputs) + f"]amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0[final]"
    filter_parts.append(mix_filter)
    
    # Join all filter parts
    filter_complex = ";".join(filter_parts)
    
    # Add filter to command
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "0:v",  # Keep original video
        "-map", "[final]",  # Use mixed audio
        "-c:v", "copy",  # Copy video stream
        "-c:a", "aac", "-b:a", "192k",  # High quality audio
        output_path
    ])
    
    # Execute command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ FFmpeg error: {result.stderr[:500]}")
        # Fallback to simpler approach
        print("⚠️ Falling back to simpler mixing approach...")
        create_simple_smooth_version(video_path, placements, output_path)
    else:
        print(f"✅ Smooth ducking video created: {output_path}")
    
    # Cleanup
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    
    # Save detailed report
    report = {
        "total_snarks": len(placements),
        "placements": [
            {
                "text": p["snark"]["text"][:30],
                "time": p["time"],
                "type": p["type"],
                "moved_from": p["original_time"],
                "fade_duration": "0.5s"
            }
            for p in placements
        ],
        "silent_placements": len([p for p in placements if p["type"] == "silent"]),
        "quiet_placements": len([p for p in placements if p["type"] == "quiet"]),
        "smooth_duck_placements": len([p for p in placements if p["type"] == "smooth_duck"]),
        "audio_settings": {
            "duck_level": "25% (-12dB)",
            "fade_duration": "500ms",
            "fade_type": "linear",
            "mixing": "gradual"
        }
    }
    
    with open("smooth_ducking_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Report saved: smooth_ducking_report.json")
    
    return report

def create_simple_smooth_version(video_path, placements, output_path):
    """Fallback: Create smooth version using pydub"""
    
    print("🎵 Creating smooth version with pydub...")
    
    # Extract audio
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "original_audio.wav"
    ], capture_output=True)
    
    # Load original audio
    original = AudioSegment.from_file("original_audio.wav")
    
    # Apply smooth ducking for each snark
    for p in placements:
        snark_audio = AudioSegment.from_mp3(p["snark"]["file"])
        position_ms = int(p["time"] * 1000)
        duration_ms = int(p["snark"]["duration"] * 1000)
        
        if p["type"] != "silent":
            # Apply smooth ducking
            fade_ms = 500  # 500ms fade
            
            # Calculate segments
            fade_in_start = max(0, position_ms - fade_ms)
            fade_out_end = min(len(original), position_ms + duration_ms + fade_ms)
            
            # Create ducked segment with crossfade
            before = original[:fade_in_start]
            fade_in = original[fade_in_start:position_ms].fade_out(fade_ms)
            during = original[position_ms:position_ms + duration_ms] - 12
            fade_out = original[position_ms + duration_ms:fade_out_end].fade_in(fade_ms) - 12
            after = original[fade_out_end:]
            
            # Reconstruct
            original = before + fade_in + during + fade_out + after
        
        # Overlay snark
        original = original.overlay(snark_audio, position=position_ms)
    
    # Export mixed audio
    original.export("mixed_smooth.wav", format="wav")
    
    # Combine with video
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", "mixed_smooth.wav",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v", "-map", "1:a",
        output_path
    ], capture_output=True)
    
    # Cleanup
    for f in ["original_audio.wav", "mixed_smooth.wav"]:
        if os.path.exists(f):
            os.remove(f)

def main():
    print("=" * 70)
    print("🎯 SMOOTH GRADUAL AUDIO DUCKING FOR SNARKS")
    print("=" * 70)
    
    video_path = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi/scenes/downsampled/scene_001.mp4"
    output_path = "do_re_mi_smooth_ducking.mp4"
    
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
        return
    
    print("✅ Found all snark audio files")
    print("💰 No additional API costs - reusing existing audio")
    print("🎚️ Fade duration: 0.5 seconds (smooth transition)")
    
    # Create smooth ducking version
    report = create_smooth_ducking_video(video_path, output_path)
    
    print("\n✨ SMOOTH DUCKING COMPLETE!")
    print(f"📹 Output: {output_path}")
    print("🎵 Features:")
    print("  • Gradual 500ms fade in/out transitions")
    print("  • Original audio ducks to 25% during snarks")
    print("  • No abrupt volume changes")
    print("  • Professional-grade audio mixing")

if __name__ == "__main__":
    main()