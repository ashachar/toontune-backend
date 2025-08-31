#!/usr/bin/env python3
"""
Compare sequential vs parallel processing performance
"""

import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.word_level_pipeline import create_word_level_video


def run_comparison(video_path: str, duration: float):
    """Run both sequential and parallel and compare"""
    
    print("=" * 70)
    print("SEQUENTIAL vs PARALLEL PERFORMANCE TEST")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Duration: {duration} seconds")
    print("=" * 70)
    
    # Test 1: Sequential
    print("\n📊 TEST 1: SEQUENTIAL PROCESSING")
    print("-" * 40)
    start_time = time.time()
    
    output_seq = create_word_level_video(
        input_video_path=video_path,
        duration_seconds=duration,
        output_name=f"ai_math1_sequential_{duration}s.mp4",
        use_parallel=False
    )
    
    sequential_time = time.time() - start_time
    print(f"⏱️ Sequential time: {sequential_time:.2f} seconds")
    
    # Test 2: Parallel
    print("\n📊 TEST 2: PARALLEL PROCESSING")
    print("-" * 40)
    start_time = time.time()
    
    output_par = create_word_level_video(
        input_video_path=video_path,
        duration_seconds=duration,
        output_name=f"ai_math1_parallel_{duration}s.mp4",
        use_parallel=True
    )
    
    parallel_time = time.time() - start_time
    print(f"⏱️ Parallel time: {parallel_time:.2f} seconds")
    
    # Results
    print("\n" + "=" * 70)
    print("🏁 RESULTS")
    print("=" * 70)
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Parallel:   {parallel_time:.2f}s")
    
    if parallel_time < sequential_time:
        speedup = sequential_time / parallel_time
        time_saved = sequential_time - parallel_time
        print(f"\n✅ Parallel is {speedup:.2f}x faster!")
        print(f"   Time saved: {time_saved:.2f} seconds")
    else:
        slowdown = parallel_time / sequential_time
        print(f"\n⚠️ Parallel is {slowdown:.2f}x slower")
        print("   This might be due to overhead for short videos")
    
    return sequential_time, parallel_time


if __name__ == "__main__":
    video_path = "uploads/assets/videos/ai_math1.mp4"
    
    # Test with full video
    duration = 206.0
    
    if len(sys.argv) > 1:
        duration = float(sys.argv[1])
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    seq_time, par_time = run_comparison(video_path, duration)