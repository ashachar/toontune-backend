"""
Base utilities for video editing effects.
"""

import cv2
import numpy as np
import subprocess
import tempfile
import shutil
import os
from pathlib import Path
from typing import Union, Tuple, Optional, List
import replicate
from dotenv import load_dotenv

load_dotenv()

class VideoProcessor:
    """Base class for video processing utilities."""
    
    def __init__(self, input_path: Union[str, Path]):
        """
        Initialize video processor.
        
        Parameters:
        -----------
        input_path : str or Path
            Path to input video file
        """
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Get video properties
        self.cap = cv2.VideoCapture(str(self.input_path))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.release()
        
    def extract_frames(self, temp_dir: Path, start_frame: int = 0, end_frame: Optional[int] = None) -> List[Path]:
        """Extract frames from video."""
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        end_frame = end_frame or self.frame_count
        
        cmd = [
            'ffmpeg',
            '-i', str(self.input_path),
            '-vf', f'select=between(n\\,{start_frame}\\,{end_frame})',
            '-vsync', '0',
            str(frames_dir / 'frame_%04d.png')
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        return sorted(list(frames_dir.glob('*.png')))
    
    def create_video_from_frames(self, frames: List[Path], output_path: Path, fps: Optional[int] = None) -> Path:
        """Create video from frame sequence."""
        fps = fps or self.fps
        
        # Create input file list
        list_file = output_path.parent / "frames.txt"
        with open(list_file, 'w') as f:
            for frame in frames:
                f.write(f"file '{frame}'\n")
                f.write(f"duration {1.0/fps}\n")
            # Add last frame
            if frames:
                f.write(f"file '{frames[-1]}'\n")
        
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_file),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-y',
            str(output_path)
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        list_file.unlink()
        
        return output_path
    
    def apply_audio_from_original(self, video_path: Path, output_path: Path) -> Path:
        """Copy audio from original video to processed video."""
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-i', str(self.input_path),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-y',
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path
        except subprocess.CalledProcessError:
            # No audio in original, just copy video
            shutil.copy(video_path, output_path)
            return output_path


def segment_subject(image_path: Path, use_replicate: bool = True) -> np.ndarray:
    """
    Segment the main subject from an image using SAM2.
    
    Parameters:
    -----------
    image_path : Path
        Path to input image
    use_replicate : bool
        Whether to use Replicate API for segmentation
        
    Returns:
    --------
    np.ndarray
        Binary mask of the subject
    """
    if use_replicate and os.environ.get("REPLICATE_API_TOKEN"):
        try:
            with open(image_path, 'rb') as f:
                prediction = replicate.run(
                    'meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83',
                    input={
                        'image': f,
                        'points_per_side': 32,
                        'pred_iou_thresh': 0.88,
                        'stability_score_thresh': 0.95,
                        'use_m2m': True,
                        'multimask_output': False
                    }
                )
            
            # Get the largest mask
            import requests
            from PIL import Image
            from io import BytesIO
            
            masks = prediction.get('individual_masks', [])
            if masks:
                largest_mask = None
                largest_area = 0
                
                for mask_url in masks[:5]:  # Check top 5 masks
                    response = requests.get(str(mask_url))
                    mask_img = Image.open(BytesIO(response.content))
                    mask_array = np.array(mask_img)
                    
                    if len(mask_array.shape) == 3:
                        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
                    
                    area = np.sum(mask_array > 128)
                    if area > largest_area:
                        largest_area = area
                        largest_mask = mask_array
                
                if largest_mask is not None:
                    return (largest_mask > 128).astype(np.uint8) * 255
        except Exception as e:
            print(f"Replicate segmentation failed: {e}, using fallback")
    
    # Fallback: simple foreground extraction using GrabCut
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    # Initialize mask with probable foreground in center
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    return mask2


def estimate_depth(image_path: Path, use_replicate: bool = True) -> np.ndarray:
    """
    Estimate depth map from an image.
    
    Parameters:
    -----------
    image_path : Path
        Path to input image
    use_replicate : bool
        Whether to use Replicate API for depth estimation
        
    Returns:
    --------
    np.ndarray
        Depth map (0-255)
    """
    if use_replicate and os.environ.get("REPLICATE_API_TOKEN"):
        try:
            with open(image_path, 'rb') as f:
                output = replicate.run(
                    "cjwbw/midas:eaa2bda4e758507af1c321dcaaab0ed66e90afb5b1e983e82a7cdbe419b20b89",
                    input={"image": f}
                )
            
            import requests
            from PIL import Image
            from io import BytesIO
            
            response = requests.get(str(output))
            depth_img = Image.open(BytesIO(response.content))
            depth_array = np.array(depth_img)
            
            if len(depth_array.shape) == 3:
                depth_array = cv2.cvtColor(depth_array, cv2.COLOR_RGB2GRAY)
            
            return depth_array
            
        except Exception as e:
            print(f"Replicate depth estimation failed: {e}, using fallback")
    
    # Fallback: simple depth approximation based on Y position
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    # Create gradient depth map (closer at bottom, farther at top)
    depth_map = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        depth_value = int(255 * (1 - y / h))
        depth_map[y, :] = depth_value
    
    # Apply some blur for smoothness
    depth_map = cv2.GaussianBlur(depth_map, (21, 21), 0)
    
    return depth_map


def track_object_in_video(video_path: Path, initial_bbox: Tuple[int, int, int, int], 
                         use_replicate: bool = True) -> List[Tuple[int, int, int, int]]:
    """
    Track an object throughout a video.
    
    Parameters:
    -----------
    video_path : Path
        Path to video file
    initial_bbox : Tuple
        Initial bounding box (x, y, width, height)
    use_replicate : bool
        Whether to use Replicate API for tracking
        
    Returns:
    --------
    List[Tuple]
        List of bounding boxes for each frame
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize tracker
    tracker = cv2.TrackerCSRT_create()
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []
    
    tracker.init(frame, initial_bbox)
    
    bboxes = [initial_bbox]
    
    for _ in range(1, frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        success, bbox = tracker.update(frame)
        if success:
            bboxes.append(tuple(map(int, bbox)))
        else:
            # If tracking fails, use last known position
            bboxes.append(bboxes[-1])
    
    cap.release()
    return bboxes