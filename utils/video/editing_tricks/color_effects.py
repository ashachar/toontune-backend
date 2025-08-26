"""
Color effects for video editing.
"""

import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Union, Tuple, Optional, List
try:
    from .base import VideoProcessor, segment_subject
except ImportError:
    from base import VideoProcessor, segment_subject


def apply_color_splash(
    input_video: Union[str, Path],
    target_color: Tuple[int, int, int] = (255, 0, 0),
    tolerance: float = 30.0,
    segment_subject: bool = True,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Apply color splash effect - keep specific color while making rest grayscale.
    
    This effect creates a dramatic visual where only objects of a specific color
    remain colored while everything else becomes black and white. Perfect for
    highlighting red roses, blue skies, or any distinctive colored element.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file or cv2.VideoCapture object
    target_color : Tuple[int, int, int]
        RGB color to preserve (default: red)
    tolerance : float
        Color matching tolerance (0-100, default: 30)
        Lower = exact match only, Higher = broader color range
    segment_subject : bool
        If True, tries to detect and preserve the main subject's colors
    output_path : str or Path, optional
        Output video path. If None, creates temp file
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Keep only red colors in video
    >>> output = apply_color_splash("input.mp4", target_color=(255, 0, 0))
    
    >>> # Keep blue with wider tolerance
    >>> output = apply_color_splash("input.mp4", target_color=(0, 0, 255), tolerance=50)
    """
    # Initialize processor
    processor = VideoProcessor(input_video)
    
    # Setup paths
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix='color_splash_'))
    
    try:
        # Extract frames
        print("Extracting frames...")
        frames = processor.extract_frames(temp_dir)
        
        # Process each frame
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        # If segment_subject is True, detect subject in first frame
        subject_mask = None
        if segment_subject and frames:
            print("Detecting subject...")
            subject_mask = segment_subject(frames[0])
            if subject_mask is not None:
                # Resize mask to frame dimensions
                frame = cv2.imread(str(frames[0]))
                h, w = frame.shape[:2]
                subject_mask = cv2.resize(subject_mask, (w, h))
        
        print(f"Processing {len(frames)} frames...")
        for i, frame_path in enumerate(frames):
            # Read frame
            frame = cv2.imread(str(frame_path))
            result = apply_color_splash_to_frame(
                frame, 
                target_color, 
                tolerance,
                subject_mask
            )
            
            # Save processed frame
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Processed {i}/{len(frames)} frames")
        
        # Create video from processed frames
        print("Creating output video...")
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        
        # Add audio from original
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Color splash effect applied: {output_path}")
        return output_path
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_color_splash_to_frame(
    frame: np.ndarray,
    target_color: Tuple[int, int, int],
    tolerance: float,
    subject_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply color splash effect to a single frame.
    
    Parameters:
    -----------
    frame : np.ndarray
        Input frame (BGR)
    target_color : Tuple[int, int, int]
        RGB color to preserve
    tolerance : float
        Color matching tolerance
    subject_mask : np.ndarray, optional
        Mask of subject to preserve colors
        
    Returns:
    --------
    np.ndarray
        Processed frame
    """
    # Convert to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV for better color matching
    frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    
    # Convert target color to HSV
    target_rgb = np.uint8([[target_color]])
    target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV)[0][0]
    
    # Create color mask
    lower_bound = np.array([
        max(0, target_hsv[0] - tolerance),
        max(0, target_hsv[1] - tolerance),
        max(0, target_hsv[2] - tolerance)
    ])
    upper_bound = np.array([
        min(179, target_hsv[0] + tolerance),
        min(255, target_hsv[1] + tolerance),
        min(255, target_hsv[2] + tolerance)
    ])
    
    # Create mask for target color
    color_mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)
    
    # If subject mask provided, combine masks
    if subject_mask is not None:
        color_mask = cv2.bitwise_or(color_mask, subject_mask)
    
    # Apply morphology to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    
    # Create grayscale version
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Apply Gaussian blur to mask for smoother transition
    color_mask = cv2.GaussianBlur(color_mask, (9, 9), 2)
    
    # Normalize mask to 0-1 range
    mask_norm = color_mask.astype(np.float32) / 255.0
    mask_norm = np.stack([mask_norm] * 3, axis=-1)
    
    # Blend color and grayscale based on mask
    result = (frame * mask_norm + gray_3channel * (1 - mask_norm)).astype(np.uint8)
    
    return result


def apply_selective_color(
    input_video: Union[str, Path],
    color_adjustments: List[dict],
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Apply selective color adjustments to specific color ranges.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    color_adjustments : List[dict]
        List of color adjustment configs:
        [{
            'target': (R, G, B),  # Target color
            'tolerance': 30,      # Color range tolerance
            'hue_shift': 0,       # Hue adjustment (-180 to 180)
            'saturation': 1.0,    # Saturation multiplier
            'brightness': 1.0     # Brightness multiplier
        }]
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='selective_color_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        print(f"Applying selective color to {len(frames)} frames...")
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            # Apply each color adjustment
            result = frame.copy()
            for adj in color_adjustments:
                result = apply_selective_color_to_frame(result, adj)
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_selective_color_to_frame(frame: np.ndarray, adjustment: dict) -> np.ndarray:
    """Apply selective color adjustment to a single frame."""
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Convert target color to HSV
    target_rgb = np.uint8([[adjustment['target']]])
    target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV)[0][0]
    
    tolerance = adjustment.get('tolerance', 30)
    
    # Create mask for target color range
    lower = np.array([
        max(0, target_hsv[0] - tolerance),
        max(0, target_hsv[1] - tolerance * 2),
        max(0, target_hsv[2] - tolerance * 2)
    ])
    upper = np.array([
        min(179, target_hsv[0] + tolerance),
        min(255, target_hsv[1] + tolerance * 2),
        min(255, target_hsv[2] + tolerance * 2)
    ])
    
    mask = cv2.inRange(frame_hsv, lower, upper)
    
    # Apply adjustments
    if 'hue_shift' in adjustment:
        frame_hsv[:, :, 0] = np.where(
            mask > 0,
            (frame_hsv[:, :, 0] + adjustment['hue_shift']) % 180,
            frame_hsv[:, :, 0]
        )
    
    if 'saturation' in adjustment:
        frame_hsv[:, :, 1] = np.where(
            mask > 0,
            np.clip(frame_hsv[:, :, 1] * adjustment['saturation'], 0, 255),
            frame_hsv[:, :, 1]
        )
    
    if 'brightness' in adjustment:
        frame_hsv[:, :, 2] = np.where(
            mask > 0,
            np.clip(frame_hsv[:, :, 2] * adjustment['brightness'], 0, 255),
            frame_hsv[:, :, 2]
        )
    
    result = cv2.cvtColor(frame_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result