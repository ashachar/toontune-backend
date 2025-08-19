"""
Text effects for video editing.
"""

import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict
try:
    from .base import VideoProcessor, segment_subject, track_object_in_video
except ImportError:
    from base import VideoProcessor, segment_subject, track_object_in_video


def apply_text_behind_subject(
    input_video: Union[str, Path],
    text: str,
    position: Tuple[int, int] = None,
    font_scale: float = 2.0,
    font_color: Tuple[int, int, int] = (255, 255, 255),
    font_thickness: int = 3,
    segment_foreground: bool = True,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Place text behind the main subject in the video.
    
    Creates a layered effect where text appears to be behind the subject,
    perfect for title sequences and creative overlays. The subject is
    automatically detected and remains in the foreground.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    text : str
        Text to display behind subject
    position : Tuple[int, int], optional
        Text position (x, y). If None, centers the text
    font_scale : float
        Font size scale factor (default: 2.0)
    font_color : Tuple[int, int, int]
        RGB color of text (default: white)
    font_thickness : int
        Thickness of text (default: 3)
    segment_foreground : bool
        If True, automatically segments the foreground subject
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Add title behind person
    >>> output = apply_text_behind_subject("interview.mp4", "INTERVIEW 2024")
    
    >>> # Custom position and color
    >>> output = apply_text_behind_subject(
    ...     "video.mp4",
    ...     "BACKGROUND TEXT",
    ...     position=(100, 200),
    ...     font_color=(255, 0, 0)
    ... )
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='text_behind_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        # Get text dimensions for positioning
        font = cv2.FONT_HERSHEY_SIMPLEX  # Use SIMPLEX with thickness for bold effect
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        # Default position: center
        if position is None:
            position = (
                processor.width // 2 - text_width // 2,
                processor.height // 2 + text_height // 2
            )
        
        # Segment subject in first frame if needed
        subject_masks = []
        if segment_foreground and frames:
            print("Segmenting subject in frames...")
            # For efficiency, segment every 5th frame and interpolate
            key_frames_idx = list(range(0, len(frames), 5))
            key_masks = []
            
            for idx in key_frames_idx:
                mask = segment_subject(frames[idx])
                if mask is not None:
                    mask = cv2.resize(mask, (processor.width, processor.height))
                    key_masks.append((idx, mask))
            
            # Interpolate masks for all frames
            for i in range(len(frames)):
                # Find surrounding key frames
                prev_idx, prev_mask = None, None
                next_idx, next_mask = None, None
                
                for idx, mask in key_masks:
                    if idx <= i:
                        prev_idx, prev_mask = idx, mask
                    if idx >= i and next_idx is None:
                        next_idx, next_mask = idx, mask
                
                if prev_mask is not None and next_mask is not None and prev_idx != next_idx:
                    # Interpolate between masks
                    alpha = (i - prev_idx) / (next_idx - prev_idx)
                    mask = cv2.addWeighted(
                        prev_mask, 1 - alpha,
                        next_mask, alpha,
                        0
                    )
                elif prev_mask is not None:
                    mask = prev_mask
                elif next_mask is not None:
                    mask = next_mask
                else:
                    mask = np.zeros((processor.height, processor.width), dtype=np.uint8)
                
                subject_masks.append(mask)
        
        print(f"Processing {len(frames)} frames...")
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            # Create text layer
            text_layer = frame.copy()
            cv2.putText(
                text_layer, text, position,
                font, font_scale, font_color, font_thickness,
                cv2.LINE_AA
            )
            
            # Apply mask if available
            if subject_masks and i < len(subject_masks):
                mask = subject_masks[i]
                # Dilate mask slightly for better coverage
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask = cv2.dilate(mask, kernel, iterations=1)
                
                # Normalize mask
                mask_norm = mask.astype(np.float32) / 255.0
                mask_3ch = np.stack([mask_norm] * 3, axis=-1)
                
                # Composite: text layer in background, original frame in foreground
                result = text_layer * (1 - mask_3ch) + frame * mask_3ch
                result = result.astype(np.uint8)
            else:
                result = text_layer
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Processed {i}/{len(frames)} frames")
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Text behind subject effect applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_motion_tracking_text(
    input_video: Union[str, Path],
    text: str,
    track_point: Optional[Tuple[int, int]] = None,
    offset: Tuple[int, int] = (0, -50),
    font_scale: float = 1.0,
    font_color: Tuple[int, int, int] = (255, 255, 255),
    font_thickness: int = 2,
    background_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Add text that follows a tracked object in the video.
    
    Automatically tracks an object or point in the video and attaches text
    that follows its movement. Perfect for labeling moving objects, creating
    callouts, or adding dynamic annotations.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    text : str
        Text to display
    track_point : Tuple[int, int], optional
        Initial point to track (x, y). If None, tracks center object
    offset : Tuple[int, int]
        Offset from tracked point (default: (0, -50) for above object)
    font_scale : float
        Font size scale factor (default: 1.0)
    font_color : Tuple[int, int, int]
        RGB color of text (default: white)
    font_thickness : int
        Thickness of text (default: 2)
    background_color : Tuple[int, int, int], optional
        Background color for text. None for no background
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Track and label a moving car
    >>> output = apply_motion_tracking_text(
    ...     "traffic.mp4",
    ...     "Tesla Model 3",
    ...     track_point=(300, 400)
    ... )
    
    >>> # Track with custom styling
    >>> output = apply_motion_tracking_text(
    ...     "sports.mp4",
    ...     "Player #10",
    ...     font_color=(255, 255, 0),
    ...     background_color=(0, 0, 0)
    ... )
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='motion_text_'))
    
    try:
        # Setup tracking
        cap = cv2.VideoCapture(str(input_video))
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read video")
        
        # Initialize tracking point
        if track_point is None:
            # Default to center of frame
            track_point = (processor.width // 2, processor.height // 2)
        
        # Define initial bounding box around track point
        bbox_size = 60
        initial_bbox = (
            max(0, track_point[0] - bbox_size // 2),
            max(0, track_point[1] - bbox_size // 2),
            bbox_size,
            bbox_size
        )
        
        # Track object through video
        print("Tracking object...")
        bboxes = track_object_in_video(Path(input_video), initial_bbox)
        cap.release()
        
        # Extract frames
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        print(f"Applying motion tracking text to {len(frames)} frames...")
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            if i < len(bboxes):
                bbox = bboxes[i]
                # Calculate text position from bbox center
                center_x = bbox[0] + bbox[2] // 2
                center_y = bbox[1] + bbox[3] // 2
                text_x = center_x + offset[0]
                text_y = center_y + offset[1]
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, font_thickness
                )
                
                # Draw background if specified
                if background_color is not None:
                    padding = 5
                    cv2.rectangle(
                        frame,
                        (text_x - padding, text_y - text_height - padding),
                        (text_x + text_width + padding, text_y + padding),
                        background_color,
                        -1
                    )
                
                # Draw text
                cv2.putText(
                    frame, text,
                    (text_x, text_y),
                    font, font_scale, font_color, font_thickness,
                    cv2.LINE_AA
                )
                
                # Optional: Draw tracking box for debugging
                # cv2.rectangle(frame, (bbox[0], bbox[1]), 
                #              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                #              (0, 255, 0), 2)
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), frame)
            processed_frames.append(output_frame)
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Motion tracking text applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_animated_subtitle(
    input_video: Union[str, Path],
    subtitles: List[Dict[str, any]],
    font_scale: float = 1.0,
    font_color: Tuple[int, int, int] = (255, 255, 255),
    background_color: Tuple[int, int, int] = (0, 0, 0),
    animation_type: str = "fade",
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Add animated subtitles with various entrance/exit effects.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    subtitles : List[Dict]
        List of subtitle entries:
        [{
            'text': 'Subtitle text',
            'start_time': 1.0,  # seconds
            'end_time': 3.0,    # seconds
            'position': 'bottom'  # or 'top', 'center', or (x, y)
        }]
    font_scale : float
        Font size scale factor
    font_color : Tuple[int, int, int]
        RGB color of text
    background_color : Tuple[int, int, int]
        Background color for subtitle bar
    animation_type : str
        Animation type: 'fade', 'slide', 'typewriter'
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
    
    temp_dir = Path(tempfile.mkdtemp(prefix='animated_subtitle_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        print(f"Adding animated subtitles to {len(frames)} frames...")
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            current_time = i / processor.fps
            
            # Find active subtitle
            for subtitle in subtitles:
                if subtitle['start_time'] <= current_time <= subtitle['end_time']:
                    # Calculate animation progress
                    duration = subtitle['end_time'] - subtitle['start_time']
                    progress = (current_time - subtitle['start_time']) / duration
                    
                    # Apply animation
                    frame = apply_subtitle_animation(
                        frame, subtitle['text'],
                        subtitle.get('position', 'bottom'),
                        progress, animation_type,
                        font, font_scale, font_color, background_color
                    )
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), frame)
            processed_frames.append(output_frame)
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_subtitle_animation(
    frame: np.ndarray,
    text: str,
    position: Union[str, Tuple[int, int]],
    progress: float,
    animation_type: str,
    font: int,
    font_scale: float,
    font_color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int]
) -> np.ndarray:
    """Apply animated subtitle to a single frame."""
    h, w = frame.shape[:2]
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, 2
    )
    
    # Determine position
    if isinstance(position, str):
        if position == 'bottom':
            x = w // 2 - text_width // 2
            y = h - 50
        elif position == 'top':
            x = w // 2 - text_width // 2
            y = 50 + text_height
        else:  # center
            x = w // 2 - text_width // 2
            y = h // 2
    else:
        x, y = position
    
    # Apply animation
    if animation_type == 'fade':
        # Fade in/out
        if progress < 0.2:
            alpha = progress / 0.2
        elif progress > 0.8:
            alpha = (1.0 - progress) / 0.2
        else:
            alpha = 1.0
    elif animation_type == 'slide':
        # Slide in from bottom
        if progress < 0.2:
            offset = int((1.0 - progress / 0.2) * 100)
            y += offset
            alpha = 1.0
        else:
            alpha = 1.0
    elif animation_type == 'typewriter':
        # Typewriter effect
        chars_to_show = int(len(text) * min(progress * 2, 1.0))
        text = text[:chars_to_show]
        alpha = 1.0
    else:
        alpha = 1.0
    
    if alpha > 0 and text:
        # Draw background
        overlay = frame.copy()
        padding = 10
        cv2.rectangle(
            overlay,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + padding),
            bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            overlay, text,
            (x, y),
            font, font_scale, font_color, 2,
            cv2.LINE_AA
        )
        
        # Blend with original
        frame = cv2.addWeighted(frame, 1 - alpha * 0.7, overlay, alpha * 0.7, 0)
    
    return frame