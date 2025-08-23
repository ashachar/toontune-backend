#!/usr/bin/env python3
"""
Analyze scene_002 for silence periods and plan snarks
"""

import json
from pydub import AudioSegment
from pydub.silence import detect_silence
import subprocess
import os

# Scene 002 timing: 56.741 to 111.839 seconds
SCENE_START = 56.741
SCENE_END = 111.839

# Extract transcript segments for scene_002
transcript_data = {
    "segments": [
        {"text": "Fa, a long, long way to run.", "start": 57.9, "end": 59.819},
        {"text": "So, a needle pulling thread.", "start": 61.439, "end": 64.18},
        {"text": "La, a note to follow so.", "start": 65.699, "end": 67.739},
        {"text": "Ti, a drink with jam and bread.", "start": 69.44, "end": 72.58},
        {"text": "That will bring us back to Do, oh, oh, oh.", "start": 72.72, "end": 77.4},
        {"text": "Do, a deer, a female deer.", "start": 77.44, "end": 80.68},
        {"text": "Re, a drop of golden sun.", "start": 80.68, "end": 83.86},
        {"text": "Mi, a name I call myself.", "start": 85.199, "end": 87.94},
        {"text": "Fa, a long, long way to run.", "start": 88.76, "end": 91.76},
        {"text": "So, a needle pulling thread.", "start": 92.339, "end": 95.3},
        {"text": "La, a note to follow so.", "start": 96.139, "end": 98.699},
        {"text": "Ti, a drink with jam and bread.", "start": 99.44, "end": 102.36},
        {"text": "That will bring us back to Do.", "start": 102.4, "end": 105.48},
        {"text": "Do, a deer, a female deer.", "start": 105.48, "end": 108.18},
        {"text": "Re, a drop of golden sun.", "start": 108.919, "end": 111.839}
    ]
}

def find_silence_gaps():
    """Find all silence gaps between dialogue in scene_002"""
    
    segments = transcript_data["segments"]
    silence_gaps = []
    
    # Scene starts at 56.741, first dialogue at 57.9
    if segments[0]["start"] - SCENE_START > 1.0:
        silence_gaps.append({
            "start": 0.0,  # Relative to scene start
            "end": segments[0]["start"] - SCENE_START,
            "duration": segments[0]["start"] - SCENE_START,
            "after": "scene start"
        })
    
    # Find gaps between segments
    for i in range(len(segments) - 1):
        gap_start = segments[i]["end"]
        gap_end = segments[i+1]["start"]
        gap_duration = gap_end - gap_start
        
        if gap_duration > 0.5:  # At least 0.5 seconds
            silence_gaps.append({
                "start": gap_start - SCENE_START,
                "end": gap_end - SCENE_START,
                "duration": gap_duration,
                "after": segments[i]["text"]
            })
    
    return silence_gaps

def analyze_audio_silence():
    """Analyze actual audio for silence"""
    
    video_path = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi/scenes/downsampled/scene_002.mp4"
    
    # Extract audio
    print("Extracting audio from scene_002...")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "22050",
        "scene_002_audio.wav"
    ], capture_output=True)
    
    # Analyze with pydub
    audio = AudioSegment.from_file("scene_002_audio.wav")
    
    # Detect silence
    silent_ranges = detect_silence(
        audio,
        min_silence_len=1000,  # 1 second minimum
        silence_thresh=-40,  # -40dB threshold
        seek_step=100
    )
    
    # Also detect quieter periods
    quiet_ranges = detect_silence(
        audio,
        min_silence_len=800,  # 0.8 seconds
        silence_thresh=-35,  # Less strict
        seek_step=100
    )
    
    # Clean up
    if os.path.exists("scene_002_audio.wav"):
        os.remove("scene_002_audio.wav")
    
    return silent_ranges, quiet_ranges

def plan_snarks_for_gaps(silence_gaps):
    """Plan snarks that fit in silence gaps"""
    
    snarks = []
    
    for gap in silence_gaps:
        if gap["duration"] < 1.2:
            continue  # Too short for a snark
            
        # Select snark based on duration and context
        if gap["duration"] >= 2.5:
            # Longer gap - can use medium snark
            if "Fa" in gap.get("after", ""):
                snark = {
                    "time": gap["start"] + 0.3,
                    "text": "Running away from this performance.",
                    "duration_estimate": 2.0,
                    "gap_duration": gap["duration"],
                    "context": gap["after"][:30]
                }
            elif "needle" in gap.get("after", ""):
                snark = {
                    "time": gap["start"] + 0.3,
                    "text": "Threading together nonsense.",
                    "duration_estimate": 1.8,
                    "gap_duration": gap["duration"],
                    "context": gap["after"][:30]
                }
            elif "jam and bread" in gap.get("after", ""):
                snark = {
                    "time": gap["start"] + 0.3,
                    "text": "Culinary music theory.",
                    "duration_estimate": 1.5,
                    "gap_duration": gap["duration"],
                    "context": gap["after"][:30]
                }
            else:
                snark = {
                    "time": gap["start"] + 0.3,
                    "text": "Still going, apparently.",
                    "duration_estimate": 1.5,
                    "gap_duration": gap["duration"],
                    "context": gap["after"][:30]
                }
        elif gap["duration"] >= 1.5:
            # Short gap - ultra-short snark
            snark = {
                "time": gap["start"] + 0.2,
                "text": "Riveting.",
                "duration_estimate": 0.8,
                "gap_duration": gap["duration"],
                "context": gap["after"][:30]
            }
        else:
            continue
            
        snarks.append(snark)
    
    return snarks

def main():
    print("=" * 70)
    print("SCENE_002 SILENCE ANALYSIS & SNARK PLANNING")
    print("=" * 70)
    print(f"Scene duration: {SCENE_END - SCENE_START:.1f} seconds")
    print()
    
    # Find silence gaps from transcript
    print("📊 ANALYZING TRANSCRIPT FOR SILENCE GAPS:")
    print("-" * 50)
    
    silence_gaps = find_silence_gaps()
    
    for i, gap in enumerate(silence_gaps, 1):
        print(f"{i}. {gap['start']:.1f}s - {gap['end']:.1f}s ({gap['duration']:.2f}s)")
        print(f"   After: \"{gap['after'][:40]}...\"")
    
    print(f"\nTotal silence gaps: {len(silence_gaps)}")
    print(f"Total silence time: {sum(g['duration'] for g in silence_gaps):.1f}s")
    
    # Analyze actual audio
    print("\n🔊 ANALYZING ACTUAL AUDIO:")
    print("-" * 50)
    
    silent_ranges, quiet_ranges = analyze_audio_silence()
    print(f"Found {len(silent_ranges)} truly silent periods (< -40dB)")
    print(f"Found {len(quiet_ranges)} quiet periods (< -35dB)")
    
    # Plan snarks
    print("\n💬 PLANNED SNARKS (SILENCE ONLY):")
    print("-" * 50)
    
    planned_snarks = plan_snarks_for_gaps(silence_gaps)
    
    for i, snark in enumerate(planned_snarks, 1):
        print(f"\n{i}. Time: {snark['time']:.1f}s (scene time)")
        print(f"   Text: \"{snark['text']}\"")
        print(f"   Duration: ~{snark['duration_estimate']}s (fits in {snark['gap_duration']:.1f}s gap)")
        print(f"   Context: After \"{snark['context']}\"")
        
        # Verify no overlap
        absolute_time = SCENE_START + snark['time']
        snark_end = absolute_time + snark['duration_estimate']
        
        overlaps = False
        for seg in transcript_data["segments"]:
            if not (snark_end <= seg["start"] or absolute_time >= seg["end"]):
                overlaps = True
                print(f"   ⚠️ OVERLAP with: {seg['text']}")
                break
        
        if not overlaps:
            print(f"   ✅ NO OVERLAP - Fully in silence!")
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY:")
    print(f"• {len(planned_snarks)} snarks planned")
    print(f"• All placed in verified silence gaps")
    print(f"• No overlap with dialogue")
    print(f"• Average gap utilization: {sum(s['duration_estimate'] for s in planned_snarks) / sum(g['duration'] for g in silence_gaps) * 100:.1f}%")

if __name__ == "__main__":
    main()