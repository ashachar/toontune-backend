"""
Parallel video generator for word-level pipeline
Processes multiple frames concurrently for massive speedup
"""

import cv2
import numpy as np
import os
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
import pickle
import tempfile
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipelines.word_level_pipeline.video_generator import VideoGenerator
from pipelines.word_level_pipeline.frame_processor import FrameProcessor
from pipelines.word_level_pipeline.word_factory import WordObject
from pipelines.word_level_pipeline.debug_renderer import DebugRenderer, IS_DEBUG_MODE


class ParallelVideoGenerator(VideoGenerator):
    """Parallel version of VideoGenerator using multiprocessing"""
    
    def __init__(self, video_path: str = None):
        super().__init__(video_path)
        self.num_workers = min(cpu_count() - 1, 8)  # Leave one CPU free, max 8 workers
        print(f"🚀 Parallel processing with {self.num_workers} workers")
    
    def render_video(self, input_video_path: str, word_objects: List[WordObject],
                    sentence_fog_times: List[Tuple[float, float]], 
                    output_path: str, duration_seconds: float) -> str:
        """Render video using parallel frame processing"""
        
        # Get video info
        cap = cv2.VideoCapture(input_video_path)
        fps, width, height, total_frames = self.get_video_info(input_video_path)
        cap.release()
        
        print(f"\nVideo: {width}x{height} @ {fps:.2f} fps")
        print(f"Total frames: {total_frames}")
        print(f"🚀 Using {self.num_workers} parallel workers")
        
        # Create temp directory for frame chunks
        temp_dir = tempfile.mkdtemp(prefix="word_pipeline_")
        
        # Split frames into chunks for parallel processing
        frames_per_chunk = max(50, total_frames // (self.num_workers * 4))  # Smaller chunks for better load balancing
        chunks = []
        
        for start_frame in range(0, total_frames, frames_per_chunk):
            end_frame = min(start_frame + frames_per_chunk, total_frames)
            chunks.append((start_frame, end_frame))
        
        print(f"📦 Split into {len(chunks)} chunks of ~{frames_per_chunk} frames")
        
        # Serialize objects for multiprocessing
        serialized_words = pickle.dumps(word_objects)
        serialized_fog = pickle.dumps(sentence_fog_times)
        
        # Process chunks in parallel
        chunk_files = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i, (start, end) in enumerate(chunks):
                chunk_file = f"{temp_dir}/chunk_{i:04d}.mp4"
                chunk_files.append(chunk_file)
                
                # Pass original video path for mask loading
                original_video = self.video_path if hasattr(self, 'video_path') and self.video_path else input_video_path
                
                future = executor.submit(
                    process_chunk,
                    input_video_path,
                    original_video,  # Pass original for mask loading
                    chunk_file,
                    start,
                    end,
                    serialized_words,
                    serialized_fog,
                    fps,
                    width,
                    height,
                    duration_seconds
                )
                futures.append(future)
            
            # Monitor progress
            for i, future in enumerate(futures):
                future.result()  # Wait for completion
                progress = (i + 1) / len(futures)
                print(f"  Chunk {i+1}/{len(chunks)} complete ({progress*100:.1f}%)")
        
        # Concatenate all chunks using FFmpeg
        print("\n🎬 Concatenating chunks...")
        concat_list = f"{temp_dir}/concat.txt"
        with open(concat_list, 'w') as f:
            for chunk_file in chunk_files:
                f.write(f"file '{chunk_file}'\n")
        
        # Use FFmpeg to concatenate (much faster than OpenCV)
        concat_cmd = (
            f"ffmpeg -f concat -safe 0 -i {concat_list} "
            f"-c copy {output_path} -y 2>/dev/null"
        )
        os.system(concat_cmd)
        
        # Cleanup temp files
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
        os.remove(concat_list)
        
        # Generate debug video if debug mode is enabled
        print(f"\n🔍 Checking IS_DEBUG_MODE: {IS_DEBUG_MODE}")
        if IS_DEBUG_MODE:
            print("\n🐛 DEBUG MODE: Creating debug video with mask overlay...")
            # Use final output path if available, otherwise use temp path
            if hasattr(self, 'final_output_path'):
                debug_output_path = self.final_output_path.replace('.mp4', '_debug.mp4')
            else:
                debug_output_path = output_path.replace('.mp4', '_debug.mp4')
            print(f"   Debug output path: {debug_output_path}")
            # Create new temp dir for debug since we're about to delete the original
            debug_temp_dir = tempfile.mkdtemp(prefix="word_pipeline_debug_")
            self._render_debug_video(input_video_path, word_objects, sentence_fog_times, 
                                    debug_output_path, duration_seconds, debug_temp_dir)
            print(f"✅ Debug video created: {debug_output_path}")
        
        os.rmdir(temp_dir)
        
        return output_path
    
    def _render_debug_video(self, input_video_path: str, word_objects: List[WordObject],
                           sentence_fog_times: List[Tuple[float, float]], 
                           output_path: str, duration_seconds: float, temp_dir: str):
        """Render debug video with mask overlay and bounding boxes"""
        
        # Get video info
        cap = cv2.VideoCapture(input_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Debug video has double height
        debug_height = height * 2
        
        # Use the original video path for mask loading (if available)
        # The video_path attribute is set in __init__ and contains the original video
        original_video_path = self.video_path if hasattr(self, 'video_path') and self.video_path else input_video_path
        
        # Create debug renderer and frame processor with original video path for mask
        debug_renderer = DebugRenderer(original_video_path)
        frame_processor = FrameProcessor(original_video_path)
        
        # Process chunks in parallel for debug video
        frames_per_chunk = max(50, total_frames // (self.num_workers * 4))
        chunks = []
        for start_frame in range(0, total_frames, frames_per_chunk):
            end_frame = min(start_frame + frames_per_chunk, total_frames)
            chunks.append((start_frame, end_frame))
        
        # Serialize objects
        import pickle
        serialized_words = pickle.dumps(word_objects)
        serialized_fog = pickle.dumps(sentence_fog_times)
        
        # Process debug chunks in parallel
        debug_chunk_files = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i, (start, end) in enumerate(chunks):
                chunk_file = f"{temp_dir}/debug_chunk_{i:04d}.mp4"
                debug_chunk_files.append(chunk_file)
                
                future = executor.submit(
                    process_debug_chunk,
                    input_video_path,
                    original_video_path,  # Pass original for mask loading
                    chunk_file,
                    start,
                    end,
                    serialized_words,
                    serialized_fog,
                    fps,
                    width,
                    debug_height,
                    duration_seconds
                )
                futures.append(future)
            
            # Wait for completion
            for i, future in enumerate(futures):
                future.result()
                print(f"  Debug chunk {i+1}/{len(chunks)} complete")
        
        # Concatenate debug chunks
        concat_list = f"{temp_dir}/debug_concat.txt"
        with open(concat_list, 'w') as f:
            for chunk_file in debug_chunk_files:
                f.write(f"file '{chunk_file}'\n")
        
        # Use FFmpeg to concatenate
        concat_cmd = (
            f"ffmpeg -f concat -safe 0 -i {concat_list} "
            f"-c:v libx264 -preset fast -crf 23 {output_path} -y 2>/dev/null"
        )
        os.system(concat_cmd)
        
        # Cleanup debug chunks
        for chunk_file in debug_chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
        os.remove(concat_list)
        os.rmdir(temp_dir)
        
        cap.release()


def process_debug_chunk(input_video_path: str, original_video_path: str, output_path: str, 
                       start_frame: int, end_frame: int,
                       serialized_words: bytes, serialized_fog: bytes,
                       fps: float, width: int, height: int, 
                       duration_seconds: float) -> str:
    """Process a debug chunk with mask overlay and bounding boxes"""
    
    # Deserialize objects
    word_objects = pickle.loads(serialized_words)
    sentence_fog_times = pickle.loads(serialized_fog)
    
    # Create processors for this worker
    # Use original video path for mask loading
    frame_processor = FrameProcessor(original_video_path)
    debug_renderer = DebugRenderer(original_video_path)
    
    # Open video and seek to start
    cap = cv2.VideoCapture(input_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Create output for this chunk (debug video has double height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames in this chunk
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        time_seconds = frame_num / fps
        
        # First process the frame normally
        animated_frame = frame_processor.process_frame(
            frame, time_seconds, word_objects, 
            sentence_fog_times, frame_number=frame_num
        )
        
        # Then create debug visualization
        debug_frame = debug_renderer.render_debug_frame(
            animated_frame, time_seconds, word_objects, frame_number=frame_num
        )
        
        if debug_frame is not None:
            out.write(debug_frame)
    
    out.release()
    cap.release()
    
    return output_path


def process_chunk(input_video_path: str, original_video_path: str, output_path: str, 
                 start_frame: int, end_frame: int,
                 serialized_words: bytes, serialized_fog: bytes,
                 fps: float, width: int, height: int, 
                 duration_seconds: float) -> str:
    """Process a chunk of frames (runs in separate process)"""
    
    # Deserialize objects
    word_objects = pickle.loads(serialized_words)
    sentence_fog_times = pickle.loads(serialized_fog)
    
    # Create processor for this worker - use original video path for mask loading!
    frame_processor = FrameProcessor(original_video_path)
    
    # Open video and seek to start
    cap = cv2.VideoCapture(input_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Create output for this chunk
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames in this chunk
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        time_seconds = frame_num / fps
        
        # Process frame
        animated_frame = frame_processor.process_frame(
            frame, time_seconds, word_objects, 
            sentence_fog_times, frame_number=frame_num
        )
        
        # Add info overlay
        cv2.putText(animated_frame, 
                   f"Parallel Processing | {time_seconds:.2f}s / {duration_seconds}s", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1)
        
        out.write(animated_frame)
    
    out.release()
    cap.release()
    
    return output_path


class OptimizedFrameProcessor(FrameProcessor):
    """Optimized frame processor with caching"""
    
    def __init__(self):
        super().__init__()
        self.word_image_cache = {}
        self.active_words_cache = {}
    
    def get_active_words(self, word_objects: List[WordObject], 
                        time_seconds: float) -> Tuple[List[WordObject], List[WordObject]]:
        """Get only active words for current time (with caching)"""
        
        # Cache key based on time (rounded to frame)
        cache_key = round(time_seconds, 3)
        
        if cache_key in self.active_words_cache:
            return self.active_words_cache[cache_key]
        
        # Filter to only active words
        active_behind = []
        active_front = []
        
        for word in word_objects:
            # Skip if outside time range
            animation_start = word.start_time - word.rise_duration
            if time_seconds < animation_start or time_seconds > word.end_time:
                continue
            
            if word.is_behind:
                active_behind.append(word)
            else:
                active_front.append(word)
        
        # Cache result
        self.active_words_cache[cache_key] = (active_behind, active_front)
        
        return active_behind, active_front