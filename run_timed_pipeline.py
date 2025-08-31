#!/usr/bin/env python3
"""
Run word-level pipeline with detailed timing measurements
"""

import time
import sys
import os
from datetime import datetime, timedelta

# Add to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.word_level_pipeline import create_word_level_video


def format_duration(seconds):
    """Format duration in human-readable format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{seconds:.2f}s"


def run_with_timing(video_path: str, duration: float):
    """Run pipeline with detailed timing"""
    
    print("=" * 70)
    print("WORD-LEVEL PIPELINE WITH TIMING MEASUREMENT")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Duration: {duration} seconds")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    # Track phase timings
    phase_times = {}
    
    # Phase 1: Initialization
    print("\n⏱️ PHASE 1: Initialization...")
    phase_start = time.time()
    
    # Run the pipeline
    try:
        output_file = create_word_level_video(
            input_video_path=video_path,
            duration_seconds=duration,
            output_name=f"ai_math1_timed_{duration}s.mp4"
        )
        
        # Record total time
        total_time = time.time() - start_time
        end_datetime = datetime.now()
        
        # Success report
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Output file: {output_file}")
        print(f"Start time: {start_datetime.strftime('%H:%M:%S')}")
        print(f"End time: {end_datetime.strftime('%H:%M:%S')}")
        print(f"Total duration: {format_duration(total_time)}")
        print("-" * 70)
        
        # Performance metrics
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"Output size: {file_size:.2f} MB")
            
            # Calculate processing speed
            fps = 25  # Assuming 25 fps
            total_frames = int(duration * fps)
            frames_per_second = total_frames / total_time
            realtime_factor = duration / total_time
            
            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"  • Total frames processed: {total_frames}")
            print(f"  • Processing speed: {frames_per_second:.1f} frames/second")
            print(f"  • Realtime factor: {realtime_factor:.2f}x")
            print(f"  • Time per frame: {(total_time / total_frames) * 1000:.1f} ms")
            
            if realtime_factor < 1:
                print(f"  • ⚠️ Processing is {1/realtime_factor:.1f}x slower than realtime")
            else:
                print(f"  • ✅ Processing is {realtime_factor:.1f}x faster than realtime")
        
        print("=" * 70)
        
        # Save timing report
        report_file = f"outputs/timing_report_{duration}s.txt"
        with open(report_file, 'w') as f:
            f.write(f"Pipeline Timing Report\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Duration: {duration} seconds\n")
            f.write(f"Start: {start_datetime}\n")
            f.write(f"End: {end_datetime}\n")
            f.write(f"Total time: {format_duration(total_time)}\n")
            f.write(f"Processing speed: {frames_per_second:.1f} fps\n")
            f.write(f"Realtime factor: {realtime_factor:.2f}x\n")
        
        print(f"\n📄 Timing report saved to: {report_file}")
        
        return total_time
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ Pipeline failed after {format_duration(total_time)}")
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    # Default to full video
    video_path = "uploads/assets/videos/ai_math1.mp4"
    duration = 206.0
    
    # Allow custom duration from command line
    if len(sys.argv) > 1:
        duration = float(sys.argv[1])
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    # Run with timing
    total_time = run_with_timing(video_path, duration)
    
    if total_time:
        print(f"\n⏱️ FINAL TIME: {format_duration(total_time)}")