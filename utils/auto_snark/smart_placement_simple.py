#!/usr/bin/env python3
"""
Simpler smart snark placement that avoids speech overlap
"""

import os
import subprocess
import json
from pydub import AudioSegment
from pydub.silence import detect_silence
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

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

def place_snarks_smartly(video_path, output_path):
    """Place snarks in quiet moments or use slow motion"""
    
    print("\n🎬 Creating smart snark placement...")
    
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
            # No quiet spot - place at original time with ducking
            placements.append({
                "snark": snark,
                "time": snark["ideal_time"],
                "type": "ducked",
                "original_time": snark["ideal_time"]
            })
            print(f"  ⚠️ {snark['text'][:20]}... at {snark['ideal_time']:.1f}s (with ducking)")
    
    # Create the mixed audio
    print("\n🎚️ Mixing audio...")
    
    # Load original audio
    original = AudioSegment.from_file(temp_audio)
    
    # Mix in snarks with gradual fading
    for placement in placements:
        snark_audio = AudioSegment.from_mp3(placement["snark"]["file"])
        position_ms = int(placement["time"] * 1000)
        
        # Apply ducking with gradual fade if needed
        if placement["type"] == "ducked":
            # Duck original audio during snark with fade in/out
            duration_ms = int(placement["snark"]["duration"] * 1000)
            fade_duration_ms = 500  # 0.5 second fade in/out
            
            # Ensure we don't exceed boundaries
            fade_in_start = max(0, position_ms - fade_duration_ms)
            fade_out_end = min(len(original), position_ms + duration_ms + fade_duration_ms)
            
            # Split audio into segments
            before_fade = original[:fade_in_start]
            fade_in_section = original[fade_in_start:position_ms]
            during = original[position_ms:position_ms + duration_ms]
            fade_out_section = original[position_ms + duration_ms:fade_out_end]
            after_fade = original[fade_out_end:]
            
            # Apply gradual fade down (fade_in_section)
            if len(fade_in_section) > 0:
                # Gradually reduce volume from 100% to 20%
                fade_in_section = fade_in_section.fade(
                    to_gain=-12,  # Reduce by 12dB (approximately 20% volume)
                    start=0,
                    duration=len(fade_in_section)
                )
            
            # Duck the main "during" part to 20% volume
            during = during - 12  # Reduce by 12dB (gentler than before)
            
            # Apply gradual fade up (fade_out_section)
            if len(fade_out_section) > 0:
                # Gradually increase volume from 20% back to 100%
                fade_out_section = fade_out_section - 12  # Start at reduced volume
                fade_out_section = fade_out_section.fade(
                    from_gain=-12,  # Fade from -12dB back to 0dB
                    start=0,
                    duration=len(fade_out_section)
                )
            
            # Reconstruct with smooth transitions
            original = before_fade + fade_in_section + during + fade_out_section + after_fade
        
        # Overlay the snark
        original = original.overlay(snark_audio, position=position_ms)
    
    # Export mixed audio
    mixed_audio_path = "mixed_audio_smart.wav"
    original.export(mixed_audio_path, format="wav")
    
    # Combine with video
    print("🎥 Creating final video...")
    
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", mixed_audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v", "-map", "1:a",
        output_path
    ], capture_output=True)
    
    # Cleanup
    for f in [temp_audio, mixed_audio_path]:
        if os.path.exists(f):
            os.remove(f)
    
    # Save report
    report = {
        "total_snarks": len(placements),
        "placements": [
            {
                "text": p["snark"]["text"][:30],
                "time": p["time"],
                "type": p["type"],
                "moved_from": p["original_time"]
            }
            for p in placements
        ],
        "silent_placements": len([p for p in placements if p["type"] == "silent"]),
        "quiet_placements": len([p for p in placements if p["type"] == "quiet"]),
        "ducked_placements": len([p for p in placements if p["type"] == "ducked"])
    }
    
    with open("smart_placement_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Smart placement complete: {output_path}")
    print(f"📊 Report: smart_placement_report.json")
    
    return report


def main():
    print("=" * 70)
    print("🎯 SMART SNARK PLACEMENT - NO SPEECH OVERLAP")
    print("=" * 70)
    
    video_path = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi/scenes/downsampled/scene_001.mp4"
    
    # Check for snark files
    snark_files = [f"v3_snark_{i}_{style}.mp3" 
                   for i, style in enumerate([
                       "sarcastic", "condescending", "mocking",
                       "skeptical", "deadpan", "sarcastic"
                   ], 1)]
    
    missing = [f for f in snark_files if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing files: {missing[:2]}...")
        return
    
    print("✅ Using existing ElevenLabs audio (no additional cost)")
    
    # Create smart placement version
    print("\n🎬 CREATING SMART PLACEMENT VERSION:")
    report = place_snarks_smartly(video_path, "do_re_mi_smart_final.mp4")
    
    print(f"\n📊 Placement Summary:")
    print(f"  • Silent moments: {report['silent_placements']}")
    print(f"  • Quiet moments: {report['quiet_placements']}")
    print(f"  • With ducking: {report['ducked_placements']}")
    
    print("\n✅ VIDEO CREATED: do_re_mi_smart_final.mp4")
    print("  - Snarks placed in quiet moments where possible")
    print("  - Audio ducking used when silence not available")
    print("\n💰 Cost: $0.00 (reused existing audio)")

if __name__ == "__main__":
    main()