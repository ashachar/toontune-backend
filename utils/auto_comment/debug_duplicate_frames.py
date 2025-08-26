#!/usr/bin/env python3
"""
Debug why duplicate frames appear in the final video.
"""

import cv2
import numpy as np
import hashlib
from pathlib import Path
import sys

def find_duplicate_source(video_path: str):
    """Find which segments contain duplicate frames."""
    
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    
    print(f"🎬 Analyzing: {video_path.name}")
    print("=" * 70)
    
    # Store frame hashes with their timestamps
    frame_hashes = {}
    frame_num = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    duplicates_found = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample every 10th frame
        if frame_num % 10 == 0:
            # Resize for faster hashing
            small = cv2.resize(frame, (160, 90))
            
            # Check if uniform (skip black/white frames)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray)
            
            if variance > 50:  # Non-uniform frame
                frame_hash = hashlib.md5(small.tobytes()).hexdigest()
                timestamp = frame_num / fps
                
                if frame_hash in frame_hashes:
                    # Found duplicate
                    original = frame_hashes[frame_hash]
                    time_diff = timestamp - original['time']
                    
                    if time_diff > 1.0:  # Significant duplicate
                        duplicates_found.append({
                            'frame': frame_num,
                            'time': timestamp,
                            'original_frame': original['frame'],
                            'original_time': original['time'],
                            'time_diff': time_diff
                        })
                        
                        if len(duplicates_found) <= 5:  # Show first 5
                            print(f"\n🔴 Duplicate found:")
                            print(f"   Frame {frame_num} at {timestamp:.2f}s")
                            print(f"   Duplicates frame {original['frame']} at {original['time']:.2f}s")
                            print(f"   Time difference: {time_diff:.2f}s")
                            
                            # Try to identify which segment this is in
                            print(f"   This is around segment boundary:")
                            if timestamp < 10:
                                print(f"      Segment 0-1 boundary (gap at 9.28s)")
                            elif 24 < timestamp < 27:
                                print(f"      Segment 4-6 boundary (gaps at 24.66s, 26.08s)")
                            elif 38 < timestamp < 43:
                                print(f"      Segment 9-14 boundary (gaps at 38.74s, 39.98s, 42.04s)")
                            elif 48 < timestamp < 50:
                                print(f"      Segment 15-16 boundary (gap at 48.98s)")
                            elif 56 < timestamp < 60:
                                print(f"      Segment 17-20 boundary (gaps at 56.22s, 58.92s)")
                else:
                    frame_hashes[frame_hash] = {'frame': frame_num, 'time': timestamp}
        
        frame_num += 1
    
    cap.release()
    
    print("\n" + "=" * 70)
    print(f"📊 SUMMARY:")
    print(f"   Total frames analyzed: {frame_num}")
    print(f"   Unique frames: {len(frame_hashes)}")
    print(f"   Duplicate frames (>1s apart): {len(duplicates_found)}")
    
    if duplicates_found:
        print(f"\n❌ PROBLEMATIC SEGMENTS:")
        # Group by time ranges
        ranges = {}
        for dup in duplicates_found:
            segment = int(dup['time'] / 10) * 10  # Group by 10s ranges
            if segment not in ranges:
                ranges[segment] = []
            ranges[segment].append(dup)
        
        for time_range, dups in sorted(ranges.items()):
            print(f"   {time_range}s-{time_range+10}s: {len(dups)} duplicates")
    
    return duplicates_found

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_duplicate_frames.py <video_path>")
        sys.exit(1)
    
    find_duplicate_source(sys.argv[1])