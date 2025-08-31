#!/usr/bin/env python3
"""
Test parallel processing for word-level pipeline
Compares performance vs sequential processing
"""

import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.word_level_pipeline.main import create_word_level_video
from pipelines.word_level_pipeline.parallel_video_generator import ParallelVideoGenerator
from pipelines.word_level_pipeline.video_generator import VideoGenerator
from pipelines.word_level_pipeline.pipeline import WordLevelPipeline
from pipelines.word_level_pipeline.transcript_handler import TranscriptHandler
from pipelines.word_level_pipeline.scene_processor import SceneProcessor
from pipelines.word_level_pipeline.masking import ForegroundMaskExtractor


def test_parallel_processing(input_video_path: str, duration: float = 10.0):
    """Test parallel processing performance"""
    
    print("=" * 60)
    print("PARALLEL PROCESSING PERFORMANCE TEST")
    print("=" * 60)
    
    # Setup components
    video_name = Path(input_video_path).stem
    temp_segment = f"outputs/{video_name}_{duration}sec_test.mp4"
    
    # Initialize components
    transcript_handler = TranscriptHandler()
    mask_extractor = ForegroundMaskExtractor(input_video_path)
    pipeline = WordLevelPipeline(font_size=55)
    scene_processor = SceneProcessor()
    
    # Extract test segment
    print(f"\n📹 Extracting {duration}-second test segment...")
    video_gen_sequential = VideoGenerator(input_video_path)
    video_gen_sequential.extract_segment(input_video_path, duration, temp_segment)
    
    # Get transcript
    transcript_path = transcript_handler.get_transcript_path(input_video_path)
    if not transcript_path:
        print("❌ No transcript found")
        return
    
    # Load enriched phrases
    phrase_groups = transcript_handler.load_enriched_phrases(transcript_path, duration)
    
    # Extract sample frames
    sample_frames = video_gen_sequential.extract_sample_frames(temp_segment)
    
    # Extract masks
    print("   Extracting foreground masks...")
    foreground_masks = []
    for frame in sample_frames:
        mask = mask_extractor.extract_foreground_mask(frame)
        foreground_masks.append(mask)
    
    # Create word objects
    word_objects, sentence_fog_times = scene_processor.create_word_objects_from_scenes(
        transcript_handler, pipeline, phrase_groups, foreground_masks, 
        sample_frames, transcript_path, video_gen_sequential, temp_segment
    )
    
    if not word_objects:
        print("❌ No word objects created")
        return
    
    print(f"\n✅ Created {len(word_objects)} word objects")
    
    # Test 1: Sequential processing
    print("\n" + "=" * 40)
    print("TEST 1: SEQUENTIAL PROCESSING")
    print("=" * 40)
    
    output_sequential = f"outputs/{video_name}_sequential_test.mp4"
    start_time = time.time()
    
    video_gen_sequential.render_video(
        temp_segment, word_objects, sentence_fog_times, 
        output_sequential, duration
    )
    
    sequential_time = time.time() - start_time
    print(f"\n⏱️ Sequential time: {sequential_time:.2f} seconds")
    
    # Test 2: Parallel processing
    print("\n" + "=" * 40)
    print("TEST 2: PARALLEL PROCESSING")
    print("=" * 40)
    
    output_parallel = f"outputs/{video_name}_parallel_test.mp4"
    video_gen_parallel = ParallelVideoGenerator(input_video_path)
    
    start_time = time.time()
    
    video_gen_parallel.render_video(
        temp_segment, word_objects, sentence_fog_times, 
        output_parallel, duration
    )
    
    parallel_time = time.time() - start_time
    print(f"\n⏱️ Parallel time: {parallel_time:.2f} seconds")
    
    # Results
    print("\n" + "=" * 60)
    print("PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Parallel:   {parallel_time:.2f}s")
    speedup = sequential_time / parallel_time
    print(f"🚀 SPEEDUP: {speedup:.2f}x faster!")
    print(f"Time saved: {sequential_time - parallel_time:.2f} seconds")
    
    # Cleanup
    if os.path.exists(temp_segment):
        os.remove(temp_segment)
    
    return speedup


if __name__ == "__main__":
    # Test with ai_math1 video
    video_path = "uploads/assets/videos/ai_math1.mp4"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    # Run test with 10-second segment
    speedup = test_parallel_processing(video_path, duration=10.0)
    
    if speedup and speedup > 1.5:
        print("\n✅ Parallel processing is significantly faster!")
        print("   Consider using it for production.")
    else:
        print("\n⚠️ Parallel processing didn't provide expected speedup.")
        print("   This might be due to I/O bottlenecks or small test size.")