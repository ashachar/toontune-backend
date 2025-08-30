"""
Video generation and processing logic for word-level pipeline
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple

from .frame_processor import FrameProcessor
from .models import WordObject


class VideoGenerator:
    """Handles video reading, processing, and output generation"""
    
    def __init__(self, video_path=None):
        """Initialize video generator with optional video path for cached masks
        
        Args:
            video_path: Path to video file for cached mask lookup
        """
        self.frame_processor = FrameProcessor(video_path)
        self.video_path = video_path
    
    def extract_segment(self, input_video_path: str, duration_seconds: float, 
                       output_path: str) -> None:
        """Extract a segment from the input video with audio"""
        print(f"Extracting {duration_seconds}-second segment with audio...")
        os.system(f"ffmpeg -i {input_video_path} -t {duration_seconds} -c:v libx264 -preset fast -crf 18 -c:a copy {output_path} -y 2>/dev/null")
    
    def get_video_info(self, video_path: str) -> Tuple[float, int, int, int]:
        """Get video information"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps, width, height, total_frames
    
    def extract_sample_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract sample frames for mask testing"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        sample_frames = []
        sample_interval = max(1, total_frames // 30)  # Sample up to 30 frames
        for i in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                sample_frames.append(frame)
            if len(sample_frames) >= 10:  # Limit to 10 samples for speed
                break
        cap.release()
        return sample_frames
    
    def extract_scene_frames(self, video_path: str, start_time: float, end_time: float, 
                            num_samples: int = 15) -> List[np.ndarray]:
        """Extract sample frames from a specific scene time range
        
        Args:
            video_path: Path to video file
            start_time: Scene start time in seconds
            end_time: Scene end time in seconds  
            num_samples: Number of frames to sample from the scene
            
        Returns:
            List of frames sampled throughout the scene duration
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame range for the scene
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_scene_frames = end_frame - start_frame
        
        # Sample evenly throughout the scene
        sample_frames = []
        if total_scene_frames <= num_samples:
            # If scene is short, sample every frame
            for frame_num in range(start_frame, end_frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    sample_frames.append(frame)
        else:
            # Sample evenly distributed frames
            sample_interval = total_scene_frames / num_samples
            for i in range(num_samples):
                frame_num = start_frame + int(i * sample_interval)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    sample_frames.append(frame)
        
        cap.release()
        print(f"      Extracted {len(sample_frames)} frames from scene [{start_time:.2f}s - {end_time:.2f}s]")
        return sample_frames
    
    def render_video(self, input_video_path: str, word_objects: List[WordObject],
                    sentence_fog_times: List[Tuple[float, float]], 
                    output_path: str, duration_seconds: float) -> str:
        """Render the final video with word animations"""
        # Open video
        cap = cv2.VideoCapture(input_video_path)
        fps, width, height, total_frames = self.get_video_info(input_video_path)
        
        print(f"\nVideo: {width}x{height} @ {fps:.2f} fps")
        print(f"Total frames: {total_frames}")
        
        # Create output video (temporary, without audio)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("\nRendering word-level animation with fog dissolve...")
        
        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            time_seconds = frame_num / fps
            
            # Process frame with word-level rendering
            animated_frame = self.frame_processor.process_frame(frame, time_seconds, 
                                                               word_objects, sentence_fog_times,
                                                               frame_number=frame_num)
            
            # Add info overlay
            cv2.putText(animated_frame, 
                       f"Word-Level with Stripe Layout | {time_seconds:.2f}s / {duration_seconds}s", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (200, 200, 200), 1)
            
            # Show current active words count
            active_words = sum(1 for w in word_objects 
                             if w.start_time <= time_seconds <= w.end_time)
            phase = f"Active words: {active_words}"
            
            cv2.putText(animated_frame, phase, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (100, 255, 100), 2)
            
            out.write(animated_frame)
            
            if frame_num % 25 == 0:
                progress = frame_num / total_frames
                print(f"  Frame {frame_num}/{total_frames} ({progress*100:.1f}%)")
        
        out.release()
        cap.release()
        return output_path
    
    def merge_audio_and_convert(self, temp_video_path: str, source_video_path: str,
                               final_output_path: str) -> str:
        """Merge audio from source and convert to H.264"""
        print("\nMerging with audio and converting to H.264...")
        # Copy audio from original segment and video from processed version
        os.system(f"ffmpeg -i {temp_video_path} -i {source_video_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -c:a copy -map 0:v:0 -map 1:a:0? -movflags +faststart {final_output_path} -y 2>/dev/null")
        return final_output_path
    
    def cleanup_temp_files(self, *file_paths: str) -> None:
        """Clean up temporary files"""
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)