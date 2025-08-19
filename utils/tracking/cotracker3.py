#!/usr/bin/env python3
"""
CoTracker3 Utility for Video Point Tracking
Optimized for M3 Max with MPS acceleration
"""

import torch
import imageio.v3 as iio
import numpy as np
import cv2
import time
from pathlib import Path
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

class CoTracker3:
    def __init__(self, model_type: str = "cotracker3_online", device: Optional[str] = None):
        """
        Initialize CoTracker3 for point tracking.
        
        Args:
            model_type: Either 'cotracker3_online' (faster) or 'cotracker3_offline' (more accurate)
            device: Device to use ('mps', 'cuda', 'cpu'). Auto-detects if None.
        """
        # Auto-detect best device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"🎯 CoTracker3 initializing on device: {self.device}")
        
        # Load model
        print(f"📦 Loading model: {model_type}")
        self.model_type = model_type
        self.model = torch.hub.load(
            "facebookresearch/co-tracker", 
            model_type
        ).to(self.device)
        self.model.eval()
        
        print("✅ CoTracker3 ready!")
    
    def load_video(self, video_path: str, max_frames: Optional[int] = None) -> torch.Tensor:
        """
        Load and preprocess video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load
            
        Returns:
            Video tensor of shape [1, T, 3, H, W]
        """
        print(f"📹 Loading video: {video_path}")
        
        # Read video frames
        frames = iio.imread(video_path, plugin="FFMPEG")
        
        if max_frames and len(frames) > max_frames:
            frames = frames[:max_frames]
            
        # Convert to tensor [T, H, W, 3] -> [1, T, 3, H, W]
        video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float()
        video = video.to(self.device)
        
        print(f"  Shape: {video.shape}")
        print(f"  Frames: {video.shape[1]}")
        print(f"  Resolution: {video.shape[3]}x{video.shape[4]}")
        
        return video
    
    def track_points(
        self, 
        video: torch.Tensor, 
        points: Optional[List[Tuple[int, float, float]]] = None,
        grid_size: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Track points through video.
        
        Args:
            video: Video tensor [1, T, 3, H, W]
            points: List of (frame_idx, x, y) tuples for specific points
            grid_size: Grid size for automatic point selection (if points is None)
            
        Returns:
            tracks: Point trajectories [1, T, N, 2]
            visibility: Point visibility [1, T, N]
            elapsed_time: Processing time in seconds
        """
        print(f"🔍 Tracking points...")
        start_time = time.time()
        
        if self.model_type == "cotracker3_online":
            # Online tracking mode
            if points:
                # Track specific points
                queries = torch.tensor(points).float()[None].to(self.device)
                self.model(video_chunk=video, is_first_step=True, queries=queries)
            else:
                # Track grid
                self.model(video_chunk=video, is_first_step=True, grid_size=grid_size)
            
            # Process in chunks
            tracks_list = []
            vis_list = []
            
            step = self.model.step
            for ind in range(0, video.shape[1] - step, step):
                chunk = video[:, ind : ind + step * 2]
                tracks, visibility = self.model(video_chunk=chunk)
                tracks_list.append(tracks)
                vis_list.append(visibility)
            
            # Concatenate results
            pred_tracks = torch.cat(tracks_list, dim=1) if tracks_list else torch.empty(1, 0, 0, 2)
            pred_visibility = torch.cat(vis_list, dim=1) if vis_list else torch.empty(1, 0, 0)
            
        else:
            # Offline tracking mode
            if points:
                queries = torch.tensor(points).float()[None].to(self.device)
                pred_tracks, pred_visibility = self.model(video, queries=queries)
            else:
                pred_tracks, pred_visibility = self.model(video, grid_size=grid_size)
        
        # Synchronize for accurate timing
        if self.device == 'mps':
            torch.mps.synchronize()
        elif self.device == 'cuda':
            torch.cuda.synchronize()
            
        elapsed_time = time.time() - start_time
        
        print(f"  Points tracked: {pred_tracks.shape[2]}")
        print(f"  Processing time: {elapsed_time:.2f}s")
        print(f"  FPS: {video.shape[1] / elapsed_time:.1f}")
        
        return pred_tracks, pred_visibility, elapsed_time
    
    def visualize_tracks(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        output_path: str,
        point_size: int = 5,
        trail_length: int = 10
    ):
        """
        Create visualization video with tracked points.
        
        Args:
            video: Original video tensor
            tracks: Point tracks
            visibility: Point visibility
            output_path: Path to save output video
            point_size: Size of tracked points
            trail_length: Length of motion trail
        """
        print(f"🎨 Creating visualization...")
        
        # Convert tensors to numpy
        video_np = video[0].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        tracks_np = tracks[0].cpu().numpy()
        vis_np = visibility[0].cpu().numpy()
        
        # Setup video writer
        height, width = video_np.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        # Colors for different points
        colors = plt.cm.rainbow(np.linspace(0, 1, tracks_np.shape[1]))[:, :3] * 255
        
        for frame_idx in range(len(video_np)):
            frame = video_np[frame_idx].copy()
            
            # Draw each tracked point
            for point_idx in range(tracks_np.shape[1]):
                if vis_np[frame_idx, point_idx] > 0.5:  # Point is visible
                    x, y = tracks_np[frame_idx, point_idx]
                    x, y = int(x), int(y)
                    
                    # Draw point
                    cv2.circle(frame, (x, y), point_size, colors[point_idx].tolist(), -1)
                    cv2.circle(frame, (x, y), point_size + 2, (255, 255, 255), 2)
                    
                    # Draw trail
                    for trail_idx in range(max(0, frame_idx - trail_length), frame_idx):
                        if vis_np[trail_idx, point_idx] > 0.5:
                            x_prev, y_prev = tracks_np[trail_idx, point_idx]
                            x_next, y_next = tracks_np[trail_idx + 1, point_idx]
                            alpha = (trail_idx - (frame_idx - trail_length)) / trail_length
                            color_trail = (colors[point_idx] * alpha).astype(int).tolist()
                            cv2.line(frame, (int(x_prev), int(y_prev)), 
                                   (int(x_next), int(y_next)), color_trail, 2)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Visualization saved to: {output_path}")
    
    def find_stable_background_segment(
        self,
        video_path: str,
        segment_duration: float = 5.0,
        sample_rate: int = 5
    ) -> Tuple[float, float, float]:
        """
        Find a video segment with minimal background variation.
        
        Args:
            video_path: Path to video
            segment_duration: Duration of segment in seconds
            sample_rate: Sample every N frames for efficiency
            
        Returns:
            start_time: Best segment start time
            end_time: Best segment end time
            stability_score: Lower is more stable
        """
        print(f"🔍 Analyzing video for stable background...")
        
        # Load video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        segment_frames = int(segment_duration * fps)
        
        print(f"  Video: {total_frames} frames @ {fps:.1f} fps")
        print(f"  Segment: {segment_frames} frames ({segment_duration}s)")
        
        # Analyze stability across the video
        best_start = 0
        best_score = float('inf')
        
        # Sample positions to check
        step = segment_frames // 2  # Check every half segment
        positions = range(0, total_frames - segment_frames, step)
        
        for start_frame in positions:
            # Sample frames in this segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            prev_gray = None
            total_diff = 0
            samples = 0
            
            for i in range(0, segment_frames, sample_rate):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (128, 128))  # Downsample for speed
                
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    total_diff += np.mean(diff)
                    samples += 1
                
                prev_gray = gray
                
                # Skip ahead
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i + sample_rate)
            
            if samples > 0:
                avg_diff = total_diff / samples
                if avg_diff < best_score:
                    best_score = avg_diff
                    best_start = start_frame
        
        cap.release()
        
        start_time = best_start / fps
        end_time = (best_start + segment_frames) / fps
        
        print(f"✅ Found stable segment:")
        print(f"  Time: {start_time:.2f}s - {end_time:.2f}s")
        print(f"  Stability score: {best_score:.2f}")
        
        return start_time, end_time, best_score
    
    def extract_video_segment(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        duration: float
    ):
        """
        Extract a segment from video using ffmpeg.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            start_time: Start time in seconds
            duration: Duration in seconds
        """
        import subprocess
        
        print(f"✂️ Extracting segment: {start_time:.2f}s for {duration}s")
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', input_path,
            '-t', str(duration),
            '-c', 'copy',
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"✅ Segment saved to: {output_path}")
    
    def find_background_edge_point(
        self,
        video_path: str,
        frame_idx: int = 0
    ) -> Tuple[float, float]:
        """
        Find a point along a background edge using edge detection.
        
        Args:
            video_path: Path to video
            frame_idx: Frame to analyze
            
        Returns:
            x, y coordinates of edge point
        """
        print(f"🔍 Finding background edge point...")
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find edge points
        edge_points = np.column_stack(np.where(edges > 0))
        
        if len(edge_points) == 0:
            # Fallback to center point
            h, w = frame.shape[:2]
            return w // 2, h // 2
        
        # Select a point from the upper portion (likely background)
        upper_points = edge_points[edge_points[:, 0] < frame.shape[0] // 3]
        
        if len(upper_points) > 0:
            # Select a random edge point from upper region
            idx = np.random.randint(len(upper_points))
            y, x = upper_points[idx]
        else:
            # Fallback to any edge point
            idx = np.random.randint(len(edge_points))
            y, x = edge_points[idx]
        
        print(f"✅ Selected edge point: ({x}, {y})")
        
        return float(x), float(y)


def main():
    """
    Main function for testing CoTracker3
    """
    print("🚀 CoTracker3 Test Pipeline Starting...")
    
    # Initialize tracker
    tracker = CoTracker3(model_type="cotracker3_online")
    
    # Input video
    input_video = "uploads/assets/videos/do_re_mi/scenes/downsampled/scene_001.mp4"
    
    # Find stable segment
    start_time, end_time, stability = tracker.find_stable_background_segment(
        input_video,
        segment_duration=5.0
    )
    
    # Extract segment
    output_segment = "tests/tracking_test.mov"
    tracker.extract_video_segment(
        input_video,
        output_segment,
        start_time,
        duration=5.0
    )
    
    # Find edge point
    edge_x, edge_y = tracker.find_background_edge_point(output_segment)
    
    # Load video
    video = tracker.load_video(output_segment)
    
    # Track the edge point
    points = [(0, edge_x, edge_y)]  # Track from first frame
    tracks, visibility, elapsed = tracker.track_points(video, points=points)
    
    # Create visualization
    output_video = "tests/tracking_test_tracked.mp4"
    tracker.visualize_tracks(video, tracks, visibility, output_video)
    
    print("\n" + "="*50)
    print("📊 RESULTS:")
    print(f"  Stable segment: {start_time:.2f}s - {end_time:.2f}s")
    print(f"  Edge point: ({edge_x:.1f}, {edge_y:.1f})")
    print(f"  Tracking time: {elapsed:.2f} seconds")
    print(f"  Output video: {output_video}")
    print("="*50)
    
    return elapsed


if __name__ == "__main__":
    main()