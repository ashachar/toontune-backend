#!/usr/bin/env python3
"""Debug script to analyze segment creation."""

import json
from pathlib import Path
from pydub import AudioSegment

# Load pipeline report
with open('uploads/assets/videos/ai_math/pipeline_report.json') as f:
    report = json.load(f)

# Load remarks
with open('uploads/assets/videos/ai_math/remarks.json') as f:
    remarks = json.load(f)

print("🔍 Analyzing segment creation...")
print("=" * 60)

# Get audio durations
audio_durations = {}
for r in remarks:
    if r['library_file']:
        audio_path = f'uploads/assets/sounds/snark_remarks/{r["library_file"]}'
    else:
        audio_path = f'uploads/assets/videos/ai_math/{r["audio_file"]}'
    
    try:
        audio = AudioSegment.from_mp3(audio_path)
        duration = len(audio) / 1000.0
        audio_durations[r['text']] = duration
    except:
        audio_durations[r['text']] = 0

# Simulate segment creation
current_pos = 0
segment_idx = 0
segments = []

for i, snark in enumerate(report['snarks']):
    gap_start = snark['time']
    gap_duration = snark['gap_duration']
    audio_duration = audio_durations.get(snark['text'], 0)
    
    # Calculate speed factor
    speed_factor = gap_duration / audio_duration if audio_duration > gap_duration else 1.0
    needs_slowdown = speed_factor < 1.0
    
    # Normal segment before gap
    if current_pos < gap_start:
        seg = {
            'type': 'normal',
            'idx': segment_idx,
            'start': current_pos,
            'end': gap_start,
            'duration': gap_start - current_pos
        }
        segments.append(seg)
        print(f"Segment {segment_idx:02d}: Normal [{current_pos:.2f}s → {gap_start:.2f}s] = {seg['duration']:.2f}s")
        segment_idx += 1
    
    # Gap segment
    if needs_slowdown:
        output_duration = audio_duration  # Slowed video will be this long
        seg = {
            'type': 'slowed',
            'idx': segment_idx,
            'start': gap_start,
            'source_duration': gap_duration,
            'output_duration': output_duration,
            'speed': f"{speed_factor*100:.1f}%"
        }
        print(f"Segment {segment_idx:02d}: Slowed [{gap_start:.2f}s for {gap_duration:.2f}s] → {output_duration:.2f}s @ {seg['speed']}")
    else:
        seg = {
            'type': 'normal_gap',
            'idx': segment_idx,
            'start': gap_start,
            'duration': gap_duration
        }
        print(f"Segment {segment_idx:02d}: Gap    [{gap_start:.2f}s for {gap_duration:.2f}s]")
    
    segments.append(seg)
    segment_idx += 1
    
    # Update position
    current_pos = gap_start + gap_duration
    print(f"            Remark: '{snark['text']}' - Audio: {audio_duration:.2f}s")
    print(f"            Current position: {current_pos:.2f}s")
    print("-" * 60)

# Final segment
if current_pos < 55.67:  # Approximate video duration
    seg = {
        'type': 'normal',
        'idx': segment_idx,
        'start': current_pos,
        'end': 55.67,
        'duration': 55.67 - current_pos
    }
    segments.append(seg)
    print(f"Segment {segment_idx:02d}: Normal [{current_pos:.2f}s → 55.67s] = {seg['duration']:.2f}s")

print("\n📊 Summary:")
print(f"Total segments: {len(segments)}")

# Calculate total output duration
total_output = 0
for seg in segments:
    if seg['type'] == 'slowed':
        total_output += seg['output_duration']
    else:
        total_output += seg.get('duration', 0)

print(f"Expected output duration: {total_output:.2f}s")