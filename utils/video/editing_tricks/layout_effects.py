"""
Layout effects for video editing.
"""

import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict
try:
    from .base import VideoProcessor, segment_subject
except ImportError:
    from base import VideoProcessor, segment_subject


def apply_highlight_focus(
    input_video: Union[str, Path],
    focus_area: Optional[Tuple[int, int, int, int]] = None,
    blur_strength: int = 21,
    vignette: bool = True,
    track_subject: bool = False,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Blur background while keeping focus area sharp (portrait mode effect).
    
    Creates a professional depth-of-field effect by blurring everything
    except the focus area. Can automatically track and focus on the main
    subject or use a fixed focus region.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    focus_area : Tuple[int, int, int, int], optional
        Focus rectangle (x, y, width, height). If None, auto-detects subject
    blur_strength : int
        Blur kernel size (must be odd, default: 21)
    vignette : bool
        Add vignette effect around edges (default: True)
    track_subject : bool
        Continuously track and focus on subject (default: False)
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Auto-focus on subject with blur
    >>> output = apply_highlight_focus("portrait.mp4", blur_strength=31)
    
    >>> # Focus on specific area
    >>> output = apply_highlight_focus(
    ...     "interview.mp4",
    ...     focus_area=(200, 100, 400, 500),
    ...     vignette=True
    ... )
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    # Ensure blur_strength is odd
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    temp_dir = Path(tempfile.mkdtemp(prefix='focus_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        # Auto-detect focus area if not provided
        if focus_area is None and frames:
            print("Detecting subject for focus...")
            first_frame = cv2.imread(str(frames[0]))
            mask = segment_subject(frames[0])
            
            if mask is not None:
                mask = cv2.resize(mask, (processor.width, processor.height))
                # Find bounding box of mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    # Expand focus area slightly
                    padding = 50
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(processor.width - x, w + 2 * padding)
                    h = min(processor.height - y, h + 2 * padding)
                    focus_area = (x, y, w, h)
            
            if focus_area is None:
                # Fallback to center focus
                focus_area = (
                    processor.width // 4,
                    processor.height // 4,
                    processor.width // 2,
                    processor.height // 2
                )
        
        print(f"Applying highlight focus to {len(frames)} frames...")
        print(f"Focus area: {focus_area}")
        
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            # Create blurred version
            blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
            
            # Create focus mask
            mask = np.zeros((processor.height, processor.width), dtype=np.float32)
            
            if focus_area:
                x, y, w, h = focus_area
                # Create gradient mask for smooth transition
                cv2.rectangle(mask, (x, y), (x + w, y + h), 1.0, -1)
                
                # Apply Gaussian blur to mask for smooth edges
                transition_size = 50
                mask = cv2.GaussianBlur(mask, (transition_size * 2 + 1, transition_size * 2 + 1), 0)
            
            # Apply vignette if requested
            if vignette:
                vignette_mask = create_vignette_mask(processor.width, processor.height)
                mask = mask * vignette_mask
            
            # Normalize mask
            mask = np.clip(mask, 0, 1)
            mask_3ch = np.stack([mask] * 3, axis=-1)
            
            # Composite sharp and blurred based on mask
            result = (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Processed {i}/{len(frames)} frames")
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Highlight focus effect applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def add_progress_bar(
    input_video: Union[str, Path],
    bar_height: int = 5,
    bar_color: Tuple[int, int, int] = (0, 255, 0),
    background_color: Tuple[int, int, int] = (50, 50, 50),
    position: str = "bottom",
    style: str = "solid",
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Add a progress bar showing video playback position.
    
    Adds a visual progress indicator that shows how much of the video
    has been played. Useful for social media content, tutorials, or
    any video where duration awareness is important.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    bar_height : int
        Height of progress bar in pixels (default: 5)
    bar_color : Tuple[int, int, int]
        RGB color of progress bar (default: green)
    background_color : Tuple[int, int, int]
        RGB color of bar background (default: dark gray)
    position : str
        Position: 'top', 'bottom', 'both'
    style : str
        Style: 'solid', 'gradient', 'glow', 'segmented'
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Simple progress bar at bottom
    >>> output = add_progress_bar("tutorial.mp4")
    
    >>> # Custom styled progress bar
    >>> output = add_progress_bar(
    ...     "video.mp4",
    ...     bar_height=8,
    ...     bar_color=(255, 0, 255),
    ...     style="gradient"
    ... )
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='progress_bar_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        total_frames = len(frames)
        
        print(f"Adding progress bar to {total_frames} frames...")
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            # Calculate progress
            progress = (i + 1) / total_frames
            
            # Draw progress bar
            if position in ["bottom", "both"]:
                frame = draw_progress_bar(
                    frame, progress, bar_height, bar_color,
                    background_color, "bottom", style
                )
            
            if position in ["top", "both"]:
                frame = draw_progress_bar(
                    frame, progress, bar_height, bar_color,
                    background_color, "top", style
                )
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), frame)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Processed {i}/{total_frames} frames ({progress*100:.1f}%)")
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Progress bar added: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_video_in_text(
    input_video: Union[str, Path],
    text: str,
    font_scale: float = 10.0,
    font_thickness: int = 30,
    position: Optional[Tuple[int, int]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Create text filled with video content (video mask effect).
    
    Creates a striking effect where the video plays inside text characters,
    with the text acting as a mask. Perfect for titles, intros, and
    creative typography effects.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    text : str
        Text to use as mask
    font_scale : float
        Font size scale factor (default: 10.0 for large text)
    font_thickness : int
        Thickness of text (default: 30 for bold)
    position : Tuple[int, int], optional
        Text position (x, y). If None, centers the text
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Video playing inside text
    >>> output = apply_video_in_text("ocean.mp4", "OCEAN")
    
    >>> # Custom positioning
    >>> output = apply_video_in_text(
    ...     "city.mp4",
    ...     "NYC",
    ...     font_scale=15.0,
    ...     position=(100, 300)
    ... )
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='video_text_'))
    
    try:
        # Create text mask
        font = cv2.FONT_HERSHEY_SIMPLEX  # Use SIMPLEX for bold effect with thickness
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        # Default position: center
        if position is None:
            position = (
                processor.width // 2 - text_width // 2,
                processor.height // 2 + text_height // 2
            )
        
        # Create text mask
        text_mask = np.zeros((processor.height, processor.width), dtype=np.uint8)
        cv2.putText(
            text_mask, text, position,
            font, font_scale, 255, font_thickness,
            cv2.LINE_AA
        )
        
        # Apply dilation for better coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)
        
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        print(f"Applying video-in-text effect to {len(frames)} frames...")
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            # Create black background
            result = np.zeros_like(frame)
            
            # Apply text mask
            mask_3ch = np.stack([text_mask] * 3, axis=-1) / 255.0
            result = (frame * mask_3ch).astype(np.uint8)
            
            # Optional: Add border around text
            border_mask = cv2.dilate(text_mask, kernel, iterations=3) - text_mask
            border_color = (255, 255, 255)  # White border
            border_3ch = np.stack([border_mask] * 3, axis=-1) / 255.0
            result = result + (np.array(border_color) * border_3ch).astype(np.uint8)
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Processed {i}/{len(frames)} frames")
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Video-in-text effect applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def create_vignette_mask(width: int, height: int, strength: float = 0.7) -> np.ndarray:
    """Create vignette mask for darkening edges."""
    mask = np.ones((height, width), dtype=np.float32)
    
    # Create radial gradient
    cx, cy = width // 2, height // 2
    max_dist = np.sqrt(cx * cx + cy * cy)
    
    y_coords, x_coords = np.ogrid[:height, :width]
    distances = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    
    # Apply vignette formula
    mask = 1.0 - (distances / max_dist) * strength
    mask = np.clip(mask, 0, 1)
    
    return mask


def draw_progress_bar(
    frame: np.ndarray,
    progress: float,
    height: int,
    bar_color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int],
    position: str,
    style: str
) -> np.ndarray:
    """Draw progress bar on frame."""
    h, w = frame.shape[:2]
    
    # Determine bar position
    if position == "top":
        y_start = 0
        y_end = height
    else:  # bottom
        y_start = h - height
        y_end = h
    
    # Draw background
    frame[y_start:y_end, :] = bg_color
    
    # Calculate progress width
    progress_width = int(w * progress)
    
    if style == "solid":
        # Simple solid bar
        frame[y_start:y_end, :progress_width] = bar_color
        
    elif style == "gradient":
        # Gradient effect
        for x in range(progress_width):
            factor = x / progress_width
            color = [
                int(bg_color[i] * (1 - factor) + bar_color[i] * factor)
                for i in range(3)
            ]
            frame[y_start:y_end, x] = color
            
    elif style == "glow":
        # Add glow effect
        frame[y_start:y_end, :progress_width] = bar_color
        # Add bright edge
        if progress_width > 0:
            edge_color = [min(255, c + 100) for c in bar_color]
            frame[y_start:y_end, max(0, progress_width-2):progress_width] = edge_color
            
    elif style == "segmented":
        # Segmented bar (like loading bars)
        segment_width = 20
        gap_width = 2
        
        for x in range(0, progress_width, segment_width + gap_width):
            segment_end = min(x + segment_width, progress_width)
            frame[y_start:y_end, x:segment_end] = bar_color
    
    return frame


def apply_split_screen(
    videos: List[Union[str, Path]],
    layout: str = "horizontal",
    border_width: int = 2,
    border_color: Tuple[int, int, int] = (255, 255, 255),
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Combine multiple videos in split-screen layout.
    
    Parameters:
    -----------
    videos : List[str or Path]
        List of video paths (2-4 videos)
    layout : str
        Layout type: 'horizontal', 'vertical', 'grid'
    border_width : int
        Width of border between videos
    border_color : Tuple[int, int, int]
        RGB color of border
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
    """
    if len(videos) < 2 or len(videos) > 4:
        raise ValueError("Split screen requires 2-4 videos")
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    # Get properties from first video
    processor = VideoProcessor(videos[0])
    target_fps = processor.fps
    
    # Determine output dimensions
    if layout == "horizontal":
        out_width = processor.width * len(videos) + border_width * (len(videos) - 1)
        out_height = processor.height
    elif layout == "vertical":
        out_width = processor.width
        out_height = processor.height * len(videos) + border_width * (len(videos) - 1)
    else:  # grid
        grid_cols = 2
        grid_rows = (len(videos) + 1) // 2
        out_width = processor.width * grid_cols + border_width * (grid_cols - 1)
        out_height = processor.height * grid_rows + border_width * (grid_rows - 1)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='split_screen_'))
    
    try:
        # Extract frames from all videos
        all_frames = []
        min_frame_count = float('inf')
        
        for video in videos:
            proc = VideoProcessor(video)
            frames = proc.extract_frames(temp_dir / f"video_{len(all_frames)}")
            all_frames.append(frames)
            min_frame_count = min(min_frame_count, len(frames))
        
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        print(f"Creating split screen with {len(videos)} videos...")
        for i in range(min_frame_count):
            # Create output frame
            output_frame = np.full((out_height, out_width, 3), border_color, dtype=np.uint8)
            
            for v_idx, frames in enumerate(all_frames):
                if i < len(frames):
                    frame = cv2.imread(str(frames[i]))
                    frame = cv2.resize(frame, (processor.width, processor.height))
                    
                    # Calculate position based on layout
                    if layout == "horizontal":
                        x = v_idx * (processor.width + border_width)
                        y = 0
                    elif layout == "vertical":
                        x = 0
                        y = v_idx * (processor.height + border_width)
                    else:  # grid
                        col = v_idx % 2
                        row = v_idx // 2
                        x = col * (processor.width + border_width)
                        y = row * (processor.height + border_width)
                    
                    # Place frame
                    output_frame[y:y+processor.height, x:x+processor.width] = frame
            
            output_path_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_path_frame), output_frame)
            processed_frames.append(output_path_frame)
        
        # Create final video
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video, fps=target_fps)
        
        # Try to add audio from first video
        processor.apply_audio_from_original(temp_video, output_path)
        
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)