#!/usr/bin/env python3
"""
Analyze the exact segment boundaries being created by the pipeline.
"""

import json
from pathlib import Path

def simulate_pipeline_logic():
    """Simulate the exact logic from precise_gap_pipeline.py."""
    
    # Load gaps from Scribe
    gaps_path = Path("uploads/assets/videos/ai_math1/gaps_analysis_scribe.json")
    with open(gaps_path) as f:
        data = json.load(f)
        gaps_list = data["gaps"]
    
    # Load remarks
    remarks_path = Path("uploads/assets/videos/ai_math1/remarks.json")
    with open(remarks_path) as f:
        remarks = json.load(f)
    
    video_duration = 205.82
    
    print("="*70)
    print("SEGMENT BOUNDARY ANALYSIS")
    print("="*70)
    
    # Sort gaps by start time for chronological assignment
    gaps_list_sorted = sorted(gaps_list, key=lambda x: x["start"])
    
    split_points = []
    current_pos = 0
    
    # Use gaps in chronological order for remarks (first 10 gaps)
    for i, remark in enumerate(remarks):
        if i >= len(gaps_list_sorted):
            break
        
        gap = gaps_list_sorted[i]
        remark_time = gap["start"]
        
        # Add segment before gap
        if current_pos < remark_time:
            split_points.append({
                "type": "normal",
                "start": current_pos,
                "end": remark_time,
                "duration": remark_time - current_pos,
                "index": len(split_points)
            })
        
        # Add gap segment
        gap_segment_duration = gap["duration"]
        split_points.append({
            "type": "gap",
            "start": remark_time,
            "end": gap.get("end", remark_time + gap_segment_duration),
            "duration": gap_segment_duration,
            "remark_text": remark["text"],
            "index": len(split_points)
        })
        
        current_pos = gap.get("end", remark_time + gap_segment_duration)
    
    # Sort all gap segments by start time
    gap_segments = [sp for sp in split_points if sp["type"] == "gap"]
    gap_segments.sort(key=lambda x: x["start"])
    
    # Rebuild split_points with normal segments between gaps
    final_split_points = []
    current_pos = 0
    
    for gap in gap_segments:
        # Add normal segment before this gap
        if current_pos < gap["start"]:
            final_split_points.append({
                "type": "normal",
                "start": current_pos,
                "end": gap["start"],
                "duration": gap["start"] - current_pos,
                "index": len(final_split_points)
            })
        
        # Add the gap segment
        gap["index"] = len(final_split_points)
        final_split_points.append(gap)
        current_pos = gap["end"]
    
    # Add final normal segment
    if current_pos < video_duration:
        final_split_points.append({
            "type": "normal",
            "start": current_pos,
            "end": video_duration,
            "duration": video_duration - current_pos,
            "index": len(final_split_points)
        })
    
    # Print the boundaries
    print(f"\nTotal segments: {len(final_split_points)}")
    print("\nSegment boundaries:")
    
    for i, sp in enumerate(final_split_points):
        if sp["type"] == "gap":
            print(f"  [{i:2d}] GAP    {sp['start']:7.2f}s - {sp['end']:7.2f}s  '{sp.get('remark_text', 'unknown')}'")
        else:
            print(f"  [{i:2d}] NORMAL {sp['start']:7.2f}s - {sp['end']:7.2f}s")
        
        # Check for overlaps
        if i > 0:
            prev = final_split_points[i-1]
            if sp["start"] < prev["end"]:
                print(f"       ⚠️ OVERLAP: starts before previous ends!")
    
    # Focus on problem area (around segments 14-16)
    print("\n" + "="*70)
    print("PROBLEM AREA ANALYSIS (segments 13-17)")
    print("="*70)
    
    for i in range(13, min(18, len(final_split_points))):
        sp = final_split_points[i]
        print(f"\nSegment {i}:")
        print(f"  Type: {sp['type']}")
        print(f"  Range: {sp['start']:.2f}s - {sp['end']:.2f}s")
        print(f"  Duration: {sp['duration']:.3f}s")
        if sp["type"] == "gap":
            print(f"  Remark: '{sp.get('remark_text', 'unknown')}'")

if __name__ == "__main__":
    simulate_pipeline_logic()