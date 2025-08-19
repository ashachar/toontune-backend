"""
Motion effects for video editing.
"""

import cv2
import numpy as np
import tempfile
import shutil
import math
from pathlib import Path
from typing import Union, Tuple, Optional, List
try:
    from .base import VideoProcessor, estimate_depth
except ImportError:
    from base import VideoProcessor, estimate_depth


def apply_floating_effect(
    input_video: Union[str, Path],
    amplitude: float = 20.0,
    frequency: float = 0.5,
    direction: str = "vertical",
    phase_shift: float = 0.0,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Apply a smooth floating/hovering animation effect to video.
    
    Creates a gentle floating motion that makes objects appear to hover
    or bob in place. Perfect for logos, text overlays, or creating a
    dreamy, weightless effect.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    amplitude : float
        Maximum displacement in pixels (default: 20.0)
    frequency : float
        Oscillation frequency in Hz (default: 0.5)
        Lower = slower float, Higher = faster float
    direction : str
        Float direction: 'vertical', 'horizontal', 'circular', 'figure8'
    phase_shift : float
        Starting phase offset in radians (default: 0.0)
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Gentle vertical floating
    >>> output = apply_floating_effect("logo.mp4", amplitude=15, frequency=0.3)
    
    >>> # Circular floating pattern
    >>> output = apply_floating_effect("object.mp4", direction="circular")
    
    >>> # Figure-8 motion
    >>> output = apply_floating_effect("text.mp4", direction="figure8", amplitude=30)
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='floating_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        print(f"Applying floating effect to {len(frames)} frames...")
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            # Calculate displacement based on time
            t = i / processor.fps
            
            if direction == "vertical":
                dx = 0
                dy = amplitude * math.sin(2 * math.pi * frequency * t + phase_shift)
            elif direction == "horizontal":
                dx = amplitude * math.sin(2 * math.pi * frequency * t + phase_shift)
                dy = 0
            elif direction == "circular":
                dx = amplitude * math.cos(2 * math.pi * frequency * t + phase_shift)
                dy = amplitude * math.sin(2 * math.pi * frequency * t + phase_shift)
            elif direction == "figure8":
                dx = amplitude * math.sin(2 * math.pi * frequency * t + phase_shift)
                dy = amplitude * math.sin(4 * math.pi * frequency * t + phase_shift) / 2
            else:
                dx, dy = 0, 0
            
            # Create transformation matrix
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            
            # Apply transformation
            result = cv2.warpAffine(
                frame, M,
                (frame.shape[1], frame.shape[0]),
                borderMode=cv2.BORDER_REPLICATE
            )
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Processed {i}/{len(frames)} frames")
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Floating effect applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_smooth_zoom(
    input_video: Union[str, Path],
    zoom_factor: float = 1.5,
    zoom_center: Optional[Tuple[int, int]] = None,
    zoom_type: str = "in",
    easing: str = "ease_in_out",
    hold_frames: int = 0,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Apply smooth zoom in/out effect with easing.
    
    Creates professional zoom effects with various easing functions.
    Can zoom in, out, or create a zoom-hold-return effect. Supports
    different easing curves for natural-looking motion.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    zoom_factor : float
        Maximum zoom level (1.0 = no zoom, 2.0 = 2x zoom)
    zoom_center : Tuple[int, int], optional
        Center point for zoom (x, y). If None, uses frame center
    zoom_type : str
        Type: 'in', 'out', 'in_out' (zoom in then out), 'pulse'
    easing : str
        Easing function: 'linear', 'ease_in', 'ease_out', 'ease_in_out', 'bounce'
    hold_frames : int
        Number of frames to hold at max zoom (for in_out type)
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Smooth zoom in with ease
    >>> output = apply_smooth_zoom("scene.mp4", zoom_factor=1.8, easing="ease_in_out")
    
    >>> # Zoom to specific point
    >>> output = apply_smooth_zoom("product.mp4", zoom_center=(400, 300))
    
    >>> # Pulse effect
    >>> output = apply_smooth_zoom("logo.mp4", zoom_type="pulse", zoom_factor=1.2)
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    if zoom_center is None:
        zoom_center = (processor.width // 2, processor.height // 2)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='zoom_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        total_frames = len(frames)
        
        print(f"Applying smooth zoom to {total_frames} frames...")
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            # Calculate zoom progress
            if zoom_type == "in":
                progress = i / total_frames
            elif zoom_type == "out":
                progress = 1.0 - (i / total_frames)
            elif zoom_type == "in_out":
                if i < total_frames // 2 - hold_frames // 2:
                    progress = (i / (total_frames // 2 - hold_frames // 2))
                elif i > total_frames // 2 + hold_frames // 2:
                    progress = 1.0 - ((i - total_frames // 2 - hold_frames // 2) / 
                                     (total_frames // 2 - hold_frames // 2))
                else:
                    progress = 1.0
            elif zoom_type == "pulse":
                progress = abs(math.sin(2 * math.pi * i / total_frames * 2))
            else:
                progress = 0
            
            # Apply easing
            progress = apply_easing(progress, easing)
            
            # Calculate current zoom
            current_zoom = 1.0 + (zoom_factor - 1.0) * progress
            
            # Apply zoom transformation
            result = apply_zoom_to_frame(frame, current_zoom, zoom_center)
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Processed {i}/{total_frames} frames (zoom: {current_zoom:.2f}x)")
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Smooth zoom effect applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_3d_photo_effect(
    input_video: Union[str, Path],
    parallax_strength: float = 30.0,
    num_layers: int = 5,
    movement_type: str = "horizontal",
    use_depth_estimation: bool = True,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Create 3D photo/parallax effect from video.
    
    Simulates depth by creating multiple layers that move at different
    speeds, creating a 3D illusion. Can use AI depth estimation for
    more realistic effects.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    parallax_strength : float
        Maximum parallax displacement in pixels (default: 30.0)
    num_layers : int
        Number of depth layers to create (default: 5)
    movement_type : str
        Type: 'horizontal', 'vertical', 'circular', 'zoom'
    use_depth_estimation : bool
        Use AI depth estimation if available (default: True)
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Basic 3D photo effect
    >>> output = apply_3d_photo_effect("landscape.mp4")
    
    >>> # Strong parallax with circular motion
    >>> output = apply_3d_photo_effect(
    ...     "portrait.mp4",
    ...     parallax_strength=50,
    ...     movement_type="circular"
    ... )
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='3d_photo_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        
        # Get depth map for first frame
        if frames and use_depth_estimation:
            print("Estimating depth...")
            depth_map = estimate_depth(frames[0], use_replicate=True)
            depth_map = cv2.resize(depth_map, (processor.width, processor.height))
        else:
            # Create simple gradient depth map
            depth_map = create_gradient_depth(processor.height, processor.width)
        
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        print(f"Applying 3D photo effect to {len(frames)} frames...")
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            # Calculate movement based on frame number
            t = i / processor.fps
            
            if movement_type == "horizontal":
                base_dx = parallax_strength * math.sin(t)
                base_dy = 0
            elif movement_type == "vertical":
                base_dx = 0
                base_dy = parallax_strength * math.sin(t)
            elif movement_type == "circular":
                base_dx = parallax_strength * math.cos(t)
                base_dy = parallax_strength * math.sin(t) * 0.5
            elif movement_type == "zoom":
                zoom = 1.0 + 0.1 * math.sin(t)
                base_dx = 0
                base_dy = 0
            else:
                base_dx, base_dy = 0, 0
            
            # Apply parallax effect
            result = apply_parallax_effect(
                frame, depth_map, base_dx, base_dy, num_layers
            )
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Processed {i}/{len(frames)} frames")
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"3D photo effect applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_zoom_to_frame(frame: np.ndarray, zoom: float, center: Tuple[int, int]) -> np.ndarray:
    """Apply zoom transformation to a single frame."""
    h, w = frame.shape[:2]
    cx, cy = center
    
    # Calculate zoom matrix
    M = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
    
    # Adjust translation to keep center point fixed
    M[0, 2] += (1 - zoom) * cx
    M[1, 2] += (1 - zoom) * cy
    
    # Apply transformation
    result = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
    
    return result


def apply_easing(t: float, easing_type: str) -> float:
    """Apply easing function to progress value."""
    if easing_type == "linear":
        return t
    elif easing_type == "ease_in":
        return t * t
    elif easing_type == "ease_out":
        return t * (2 - t)
    elif easing_type == "ease_in_out":
        if t < 0.5:
            return 2 * t * t
        else:
            return -1 + (4 - 2 * t) * t
    elif easing_type == "bounce":
        if t < 0.5:
            return 8 * t * t * t * t
        else:
            p = t - 1
            return 1 + 8 * p * p * p * p
    else:
        return t


def create_gradient_depth(height: int, width: int) -> np.ndarray:
    """Create a simple gradient depth map."""
    depth = np.zeros((height, width), dtype=np.uint8)
    
    # Create radial gradient (center is closer)
    cx, cy = width // 2, height // 2
    max_dist = math.sqrt(cx * cx + cy * cy)
    
    for y in range(height):
        for x in range(width):
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            depth[y, x] = int(255 * (1 - dist / max_dist))
    
    return depth


def apply_parallax_effect(
    frame: np.ndarray,
    depth_map: np.ndarray,
    dx: float,
    dy: float,
    num_layers: int
) -> np.ndarray:
    """Apply parallax effect based on depth map."""
    h, w = frame.shape[:2]
    result = np.zeros_like(frame)
    
    # Create depth layers
    depth_levels = np.linspace(0, 255, num_layers + 1)
    
    for i in range(num_layers):
        # Create mask for this depth layer
        mask = ((depth_map >= depth_levels[i]) & 
                (depth_map < depth_levels[i + 1])).astype(np.uint8)
        
        # Calculate displacement for this layer
        layer_factor = i / num_layers
        layer_dx = dx * layer_factor
        layer_dy = dy * layer_factor
        
        # Create transformation matrix
        M = np.float32([[1, 0, layer_dx], [0, 1, layer_dy]])
        
        # Transform this layer
        layer = cv2.warpAffine(frame, M, (w, h))
        
        # Apply mask
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = np.where(mask_3ch > 0, layer, result)
    
    return result


def apply_rotation_effect(
    input_video: Union[str, Path],
    rotation_speed: float = 30.0,
    rotation_axis: str = "z",
    pivot_point: Optional[Tuple[int, int]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Apply smooth rotation effect to video.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    rotation_speed : float
        Rotation speed in degrees per second
    rotation_axis : str
        Axis of rotation: 'z' (2D), 'x' (3D horizontal), 'y' (3D vertical)
    pivot_point : Tuple[int, int], optional
        Rotation pivot point. If None, uses center
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
    
    if pivot_point is None:
        pivot_point = (processor.width // 2, processor.height // 2)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='rotation_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            # Calculate rotation angle
            angle = (i / processor.fps) * rotation_speed
            
            if rotation_axis == "z":
                # Simple 2D rotation
                M = cv2.getRotationMatrix2D(pivot_point, angle, 1.0)
                result = cv2.warpAffine(frame, M, (processor.width, processor.height))
            else:
                # 3D rotation (simplified)
                result = apply_3d_rotation(frame, angle, rotation_axis, pivot_point)
            
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


def apply_3d_rotation(
    frame: np.ndarray,
    angle: float,
    axis: str,
    pivot: Tuple[int, int]
) -> np.ndarray:
    """Apply simplified 3D rotation effect."""
    h, w = frame.shape[:2]
    
    # Create perspective transformation
    if axis == "x":
        # Rotation around horizontal axis
        skew = math.sin(math.radians(angle)) * 0.3
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [0, h * skew],
            [w, h * skew],
            [0, h * (1 - skew)],
            [w, h * (1 - skew)]
        ])
    else:  # axis == "y"
        # Rotation around vertical axis
        skew = math.sin(math.radians(angle)) * 0.3
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [w * skew, 0],
            [w * (1 - skew), 0],
            [w * skew, h],
            [w * (1 - skew), h]
        ])
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, M, (w, h))
    
    return result


def apply_dolly_zoom(
    input_video: Union[str, Path],
    dolly_speed: float = 0.02,
    dolly_direction: str = "in",
    smooth_acceleration: bool = True,
    crop_to_original: bool = True,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Apply dolly zoom (camera push/pull) effect.
    
    Creates the cinematic effect of a camera physically moving forward or backward,
    different from a simple zoom which only changes focal length.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    dolly_speed : float
        Speed of dolly movement (0.01 = slow, 0.05 = fast)
    dolly_direction : str
        Direction: 'in' (push forward), 'out' (pull back)
    smooth_acceleration : bool
        Use smooth acceleration/deceleration
    crop_to_original : bool
        Maintain original dimensions by cropping
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Slow push in
    >>> output = apply_dolly_zoom("scene.mp4", dolly_speed=0.015, dolly_direction="in")
    
    >>> # Fast pull out
    >>> output = apply_dolly_zoom("scene.mp4", dolly_speed=0.04, dolly_direction="out")
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='dolly_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        total_frames = len(frames)
        
        print(f"Applying dolly zoom to {total_frames} frames...")
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            # Calculate zoom factor
            progress = i / total_frames
            
            if smooth_acceleration:
                # Ease in-out
                progress = apply_easing(progress, "ease_in_out")
            
            if dolly_direction == "in":
                zoom = 1.0 + (dolly_speed * total_frames * progress)
            else:
                zoom = 1.0 + (dolly_speed * total_frames * (1 - progress))
            
            # Apply zoom with center focus
            center = (processor.width // 2, processor.height // 2)
            result = apply_zoom_to_frame(frame, zoom, center)
            
            # Crop to original size if needed
            if crop_to_original and zoom > 1.0:
                h, w = processor.height, processor.width
                cy, cx = result.shape[0] // 2, result.shape[1] // 2
                y1 = max(0, cy - h // 2)
                x1 = max(0, cx - w // 2)
                result = result[y1:y1+h, x1:x1+w]
                result = cv2.resize(result, (w, h))
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Processed {i}/{total_frames} frames")
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Dolly zoom effect applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_rack_focus(
    input_video: Union[str, Path],
    focus_points: List[Tuple[int, int]],
    focus_timings: Optional[List[float]] = None,
    blur_strength: float = 15.0,
    transition_duration: float = 1.0,
    use_depth: bool = False,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Apply rack focus effect (shifting focus between subjects).
    
    Simulates the cinematic technique of changing focus from one subject to another,
    creating depth and directing viewer attention.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    focus_points : List[Tuple[int, int]]
        List of (x, y) coordinates to focus on
    focus_timings : List[float], optional
        When to shift focus (in seconds). If None, distributes evenly
    blur_strength : float
        Maximum blur for out-of-focus areas (default: 15.0)
    transition_duration : float
        Duration of focus transition in seconds
    use_depth : bool
        Use depth estimation for realistic blur
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Focus shift between two subjects
    >>> output = apply_rack_focus(
    ...     "dialogue.mp4",
    ...     focus_points=[(200, 300), (600, 300)],
    ...     focus_timings=[0, 3, 6]
    ... )
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    # Calculate focus schedule
    if focus_timings is None:
        duration = processor.frame_count / processor.fps
        focus_timings = [i * duration / len(focus_points) for i in range(len(focus_points))]
    
    temp_dir = Path(tempfile.mkdtemp(prefix='rack_focus_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        
        # Get depth map if needed
        depth_map = None
        if use_depth and frames:
            print("Estimating depth...")
            depth_map = estimate_depth(frames[0], use_replicate=True)
            depth_map = cv2.resize(depth_map, (processor.width, processor.height))
        
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        print(f"Applying rack focus to {len(frames)} frames...")
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            # Determine current focus point
            current_time = i / processor.fps
            focus_idx = 0
            for j, timing in enumerate(focus_timings):
                if current_time >= timing:
                    focus_idx = min(j, len(focus_points) - 1)
            
            # Calculate transition progress
            if focus_idx < len(focus_timings) - 1:
                next_timing = focus_timings[focus_idx + 1]
                if current_time + transition_duration >= next_timing:
                    # In transition
                    transition_progress = (current_time - (next_timing - transition_duration)) / transition_duration
                    transition_progress = min(1.0, max(0.0, transition_progress))
                    
                    # Interpolate focus point
                    if focus_idx < len(focus_points) - 1:
                        x1, y1 = focus_points[focus_idx]
                        x2, y2 = focus_points[focus_idx + 1]
                        focus_x = int(x1 + (x2 - x1) * transition_progress)
                        focus_y = int(y1 + (y2 - y1) * transition_progress)
                    else:
                        focus_x, focus_y = focus_points[focus_idx]
                else:
                    focus_x, focus_y = focus_points[focus_idx]
            else:
                focus_x, focus_y = focus_points[min(focus_idx, len(focus_points) - 1)]
            
            # Apply selective blur
            result = apply_selective_blur(frame, (focus_x, focus_y), blur_strength, depth_map)
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Processed {i}/{len(frames)} frames")
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Rack focus effect applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_handheld_shake(
    input_video: Union[str, Path],
    shake_intensity: float = 5.0,
    shake_frequency: float = 2.0,
    rotation_amount: float = 1.0,
    smooth_motion: bool = True,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Apply handheld camera shake for realistic footage.
    
    Simulates natural camera movement from handheld shooting,
    adding organic motion and documentary-style realism.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    shake_intensity : float
        Maximum shake displacement in pixels (default: 5.0)
    shake_frequency : float
        Shake frequency in Hz (default: 2.0)
    rotation_amount : float
        Maximum rotation in degrees (default: 1.0)
    smooth_motion : bool
        Use smooth, realistic motion vs jerky shake
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Subtle handheld motion
    >>> output = apply_handheld_shake("interview.mp4", shake_intensity=3.0)
    
    >>> # Documentary-style shake
    >>> output = apply_handheld_shake("scene.mp4", shake_intensity=8.0, rotation_amount=2.0)
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='handheld_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        # Generate smooth noise for realistic motion
        total_frames = len(frames)
        if smooth_motion:
            # Use Perlin-like noise for smooth motion
            x_motion = generate_smooth_noise(total_frames, shake_frequency) * shake_intensity
            y_motion = generate_smooth_noise(total_frames, shake_frequency, seed=42) * shake_intensity
            r_motion = generate_smooth_noise(total_frames, shake_frequency * 0.7, seed=84) * rotation_amount
        else:
            # Random shake
            x_motion = np.random.randn(total_frames) * shake_intensity
            y_motion = np.random.randn(total_frames) * shake_intensity
            r_motion = np.random.randn(total_frames) * rotation_amount
        
        print(f"Applying handheld shake to {total_frames} frames...")
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            # Get motion for this frame
            dx = x_motion[i]
            dy = y_motion[i]
            angle = r_motion[i]
            
            # Create transformation matrix
            center = (processor.width // 2, processor.height // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += dx
            M[1, 2] += dy
            
            # Apply transformation with border replication
            result = cv2.warpAffine(
                frame, M,
                (processor.width, processor.height),
                borderMode=cv2.BORDER_REPLICATE
            )
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Processed {i}/{total_frames} frames")
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Handheld shake effect applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_speed_ramp(
    input_video: Union[str, Path],
    speed_points: List[Tuple[float, float]],
    interpolation: str = "smooth",
    maintain_duration: bool = False,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Apply speed ramping (variable slow/fast motion) effect.
    
    Creates dramatic timing changes by smoothly transitioning between
    different playback speeds, perfect for action sequences or emphasis.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    speed_points : List[Tuple[float, float]]
        List of (time_in_seconds, speed_multiplier) pairs
        e.g., [(0, 1.0), (2, 0.3), (3, 1.0)] for slow-mo at 2 seconds
    interpolation : str
        Interpolation type: 'smooth', 'linear', 'instant'
    maintain_duration : bool
        Keep original video duration (compress/extend as needed)
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Slow motion at key moment
    >>> output = apply_speed_ramp(
    ...     "action.mp4",
    ...     speed_points=[(0, 1.0), (2, 0.2), (2.5, 0.2), (3, 1.0)]
    ... )
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='speed_ramp_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        total_frames = len(frames)
        duration = total_frames / processor.fps
        
        # Build speed curve
        frame_indices = []
        output_frame_count = 0
        
        print(f"Calculating speed ramp for {total_frames} frames...")
        
        for i in range(total_frames):
            current_time = i / processor.fps
            
            # Find speed at current time
            speed = 1.0
            for j in range(len(speed_points) - 1):
                t1, s1 = speed_points[j]
                t2, s2 = speed_points[j + 1]
                
                if t1 <= current_time <= t2:
                    # Interpolate speed
                    if interpolation == "smooth":
                        t = (current_time - t1) / (t2 - t1)
                        t = apply_easing(t, "ease_in_out")
                        speed = s1 + (s2 - s1) * t
                    elif interpolation == "linear":
                        t = (current_time - t1) / (t2 - t1)
                        speed = s1 + (s2 - s1) * t
                    else:  # instant
                        speed = s1
                    break
                elif current_time > t2:
                    speed = s2
            
            # Calculate frame sampling
            if speed > 0:
                # For slow motion, repeat frames; for fast motion, skip frames
                if speed < 1.0:
                    # Slow motion - repeat frames
                    repeat_count = int(1.0 / speed)
                    for _ in range(repeat_count):
                        frame_indices.append(i)
                        output_frame_count += 1
                elif speed > 1.0:
                    # Fast motion - sample frames
                    if output_frame_count % int(speed) == 0:
                        frame_indices.append(i)
                        output_frame_count += 1
                else:
                    frame_indices.append(i)
                    output_frame_count += 1
        
        # Process frames according to speed curve
        print(f"Processing {len(frame_indices)} output frames...")
        for out_idx, frame_idx in enumerate(frame_indices):
            if frame_idx < len(frames):
                frame_path = frames[frame_idx]
                frame = cv2.imread(str(frame_path))
                
                output_frame = frames_dir / f"frame_{out_idx:04d}.png"
                cv2.imwrite(str(output_frame), frame)
                processed_frames.append(output_frame)
                
                if out_idx % 30 == 0:
                    print(f"  Processed {out_idx}/{len(frame_indices)} frames")
        
        # Adjust output FPS if maintaining duration
        output_fps = processor.fps
        if maintain_duration:
            output_fps = len(processed_frames) / duration
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video, fps=int(output_fps))
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Speed ramp effect applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_bloom_effect(
    input_video: Union[str, Path],
    threshold: float = 200,
    bloom_intensity: float = 1.5,
    blur_radius: int = 21,
    color_shift: Optional[Tuple[float, float, float]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Apply bloom/glow effect to bright areas.
    
    Creates a soft, dreamy glow around bright elements, perfect for
    romantic scenes, magic effects, or enhancing highlights.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    threshold : float
        Brightness threshold for bloom (0-255, default: 200)
    bloom_intensity : float
        Strength of bloom effect (default: 1.5)
    blur_radius : int
        Blur radius for glow (must be odd, default: 21)
    color_shift : Tuple[float, float, float], optional
        RGB multipliers for colored bloom (e.g., (1.2, 1.0, 0.8) for warm)
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Standard bloom
    >>> output = apply_bloom_effect("sunset.mp4", threshold=180)
    
    >>> # Warm golden bloom
    >>> output = apply_bloom_effect(
    ...     "magic.mp4",
    ...     bloom_intensity=2.0,
    ...     color_shift=(1.3, 1.1, 0.7)
    ... )
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='bloom_'))
    
    # Ensure blur radius is odd
    if blur_radius % 2 == 0:
        blur_radius += 1
    
    try:
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        print(f"Applying bloom effect to {len(frames)} frames...")
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            
            # Extract bright areas
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Create bloom layer
            bloom = cv2.bitwise_and(frame, frame, mask=bright_mask)
            
            # Apply gaussian blur for glow
            bloom = cv2.GaussianBlur(bloom, (blur_radius, blur_radius), 0)
            
            # Apply color shift if specified
            if color_shift:
                b, g, r = cv2.split(bloom)
                b = (b * color_shift[2]).astype(np.uint8)
                g = (g * color_shift[1]).astype(np.uint8)
                r = (r * color_shift[0]).astype(np.uint8)
                bloom = cv2.merge([b, g, r])
            
            # Blend bloom with original
            bloom = (bloom * bloom_intensity).astype(np.uint8)
            result = cv2.addWeighted(frame, 1.0, bloom, 1.0, 0)
            
            # Prevent overexposure
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Processed {i}/{len(frames)} frames")
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Bloom effect applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_ken_burns(
    input_image: Union[str, Path],
    duration: float = 5.0,
    fps: int = 30,
    start_scale: float = 1.0,
    end_scale: float = 1.3,
    start_position: Optional[Tuple[int, int]] = None,
    end_position: Optional[Tuple[int, int]] = None,
    easing: str = "ease_in_out",
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Apply Ken Burns effect (pan and zoom) to a still image.
    
    Creates dynamic video from static images by combining smooth
    panning and zooming movements, commonly used in documentaries.
    
    Parameters:
    -----------
    input_image : str or Path
        Path to input image file
    duration : float
        Duration of output video in seconds
    fps : int
        Frames per second for output video
    start_scale : float
        Initial zoom level (1.0 = original size)
    end_scale : float
        Final zoom level
    start_position : Tuple[int, int], optional
        Starting pan position (x, y). If None, uses center
    end_position : Tuple[int, int], optional
        Ending pan position (x, y). If None, uses center
    easing : str
        Easing function: 'linear', 'ease_in', 'ease_out', 'ease_in_out'
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Zoom in on a photo
    >>> output = apply_ken_burns(
    ...     "landscape.jpg",
    ...     duration=8.0,
    ...     start_scale=1.0,
    ...     end_scale=1.5
    ... )
    
    >>> # Pan across image while zooming
    >>> output = apply_ken_burns(
    ...     "group_photo.jpg",
    ...     start_position=(100, 200),
    ...     end_position=(800, 200),
    ...     end_scale=1.2
    ... )
    """
    input_image = Path(input_image)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    # Load image
    image = cv2.imread(str(input_image))
    if image is None:
        raise ValueError(f"Could not load image: {input_image}")
    
    h, w = image.shape[:2]
    
    # Set default positions
    if start_position is None:
        start_position = (w // 2, h // 2)
    if end_position is None:
        end_position = (w // 2, h // 2)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='ken_burns_'))
    
    try:
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        total_frames = int(duration * fps)
        processed_frames = []
        
        print(f"Creating Ken Burns effect with {total_frames} frames...")
        
        for i in range(total_frames):
            # Calculate progress
            progress = i / total_frames
            progress = apply_easing(progress, easing)
            
            # Interpolate scale
            current_scale = start_scale + (end_scale - start_scale) * progress
            
            # Interpolate position
            current_x = int(start_position[0] + (end_position[0] - start_position[0]) * progress)
            current_y = int(start_position[1] + (end_position[1] - start_position[1]) * progress)
            
            # Apply transformation
            M = cv2.getRotationMatrix2D((current_x, current_y), 0, current_scale)
            
            # Adjust translation to keep focus point centered
            M[0, 2] += (w / 2) - current_x
            M[1, 2] += (h / 2) - current_y
            
            # Apply transformation
            result = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Generated {i}/{total_frames} frames")
        
        # Create video from frames
        processor = VideoProcessor.__new__(VideoProcessor)
        processor.fps = fps
        processor.width = w
        processor.height = h
        
        processor.create_video_from_frames(processed_frames, output_path, fps=fps)
        
        print(f"Ken Burns effect applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_light_sweep(
    input_video: Union[str, Path],
    sweep_duration: float = 1.0,
    sweep_width: int = 100,
    sweep_angle: float = 45.0,
    sweep_color: Tuple[int, int, int] = (255, 255, 200),
    sweep_intensity: float = 0.5,
    repeat_interval: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Apply light sweep/shimmer effect across the video.
    
    Creates a moving light reflection that sweeps across the frame,
    simulating light hitting a reflective surface or adding sparkle.
    
    Parameters:
    -----------
    input_video : str or Path
        Path to input video file
    sweep_duration : float
        Duration of one sweep in seconds
    sweep_width : int
        Width of the light sweep in pixels
    sweep_angle : float
        Angle of the sweep in degrees (0 = vertical, 90 = horizontal)
    sweep_color : Tuple[int, int, int]
        RGB color of the sweep (default: warm white)
    sweep_intensity : float
        Opacity/intensity of the sweep (0.0 to 1.0)
    repeat_interval : float, optional
        Seconds between sweeps. If None, sweeps continuously
    output_path : str or Path, optional
        Output video path
        
    Returns:
    --------
    Path
        Path to the output video file
        
    Example:
    --------
    >>> # Golden shimmer on text
    >>> output = apply_light_sweep(
    ...     "title.mp4",
    ...     sweep_color=(255, 215, 0),
    ...     sweep_intensity=0.7
    ... )
    
    >>> # Periodic light sweep
    >>> output = apply_light_sweep(
    ...     "logo.mp4",
    ...     repeat_interval=3.0,
    ...     sweep_width=150
    ... )
    """
    processor = VideoProcessor(input_video)
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.mp4'))
    else:
        output_path = Path(output_path)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='light_sweep_'))
    
    try:
        frames = processor.extract_frames(temp_dir)
        processed_frames = []
        frames_dir = temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        
        total_frames = len(frames)
        duration = total_frames / processor.fps
        sweep_frames = int(sweep_duration * processor.fps)
        
        print(f"Applying light sweep to {total_frames} frames...")
        
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            h, w = frame.shape[:2]
            
            # Calculate sweep progress
            current_time = i / processor.fps
            
            if repeat_interval:
                # Periodic sweep
                cycle_time = current_time % repeat_interval
                if cycle_time <= sweep_duration:
                    sweep_progress = cycle_time / sweep_duration
                else:
                    sweep_progress = -1  # No sweep
            else:
                # Continuous sweep
                sweep_progress = (i % sweep_frames) / sweep_frames
            
            if sweep_progress >= 0:
                # Create sweep overlay
                overlay = np.zeros_like(frame)
                
                # Calculate sweep position
                angle_rad = math.radians(sweep_angle)
                
                # Sweep moves diagonally across frame
                sweep_distance = w + h
                current_pos = sweep_distance * sweep_progress
                
                # Create gradient mask for sweep
                mask = np.zeros((h, w), dtype=np.float32)
                
                for y in range(h):
                    for x in range(w):
                        # Calculate distance from sweep line
                        line_dist = x * math.cos(angle_rad) + y * math.sin(angle_rad)
                        
                        # Calculate intensity based on distance from sweep center
                        dist_from_sweep = abs(line_dist - current_pos)
                        
                        if dist_from_sweep < sweep_width:
                            intensity = 1.0 - (dist_from_sweep / sweep_width)
                            intensity = intensity ** 2  # Falloff curve
                            mask[y, x] = intensity
                
                # Apply color to sweep
                overlay[:, :, 0] = sweep_color[2] * mask  # B
                overlay[:, :, 1] = sweep_color[1] * mask  # G
                overlay[:, :, 2] = sweep_color[0] * mask  # R
                
                # Blend with original
                overlay = (overlay * sweep_intensity).astype(np.uint8)
                result = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
                result = np.clip(result, 0, 255).astype(np.uint8)
            else:
                result = frame
            
            output_frame = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(output_frame), result)
            processed_frames.append(output_frame)
            
            if i % 30 == 0:
                print(f"  Processed {i}/{total_frames} frames")
        
        temp_video = temp_dir / "temp_output.mp4"
        processor.create_video_from_frames(processed_frames, temp_video)
        processor.apply_audio_from_original(temp_video, output_path)
        
        print(f"Light sweep effect applied: {output_path}")
        return output_path
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# Helper functions

def apply_selective_blur(
    frame: np.ndarray,
    focus_point: Tuple[int, int],
    max_blur: float,
    depth_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """Apply selective blur based on distance from focus point."""
    h, w = frame.shape[:2]
    fx, fy = focus_point
    
    if depth_map is not None:
        # Use depth map for realistic blur
        focus_depth = depth_map[fy, fx] if 0 <= fy < h and 0 <= fx < w else 128
        
        # Calculate blur amount based on depth difference
        depth_diff = np.abs(depth_map.astype(float) - focus_depth)
        blur_map = (depth_diff / 255.0) * max_blur
    else:
        # Create radial blur based on distance from focus point
        blur_map = np.zeros((h, w), dtype=np.float32)
        max_dist = math.sqrt(w*w + h*h)
        
        for y in range(h):
            for x in range(w):
                dist = math.sqrt((x - fx)**2 + (y - fy)**2)
                blur_amount = (dist / max_dist) * max_blur
                blur_map[y, x] = blur_amount
    
    # Apply variable blur
    result = frame.copy()
    
    # Use multiple blur levels for smooth transition
    blur_levels = 5
    for level in range(1, blur_levels + 1):
        blur_radius = int(max_blur * level / blur_levels)
        if blur_radius % 2 == 0:
            blur_radius += 1
        if blur_radius < 3:
            continue
            
        # Create mask for this blur level
        min_blur = max_blur * (level - 1) / blur_levels
        max_blur_level = max_blur * level / blur_levels
        mask = ((blur_map >= min_blur) & (blur_map < max_blur_level)).astype(np.uint8)
        
        if np.any(mask):
            blurred = cv2.GaussianBlur(frame, (blur_radius, blur_radius), 0)
            mask_3ch = np.stack([mask] * 3, axis=-1)
            result = np.where(mask_3ch > 0, blurred, result)
    
    return result


def generate_smooth_noise(
    length: int,
    frequency: float,
    amplitude: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate smooth noise using sine wave combinations."""
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, length / 30, length)  # Assuming 30 fps
    noise = np.zeros(length)
    
    # Combine multiple sine waves for organic motion
    for i in range(3):
        freq = frequency * (1 + i * 0.3)
        phase = np.random.random() * 2 * np.pi
        amp = amplitude / (i + 1)
        noise += amp * np.sin(2 * np.pi * freq * t + phase)
    
    return noise