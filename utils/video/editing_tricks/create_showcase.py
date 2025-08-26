#!/usr/bin/env python3
"""
Create a showcase video combining all video effects with labels.
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil


def add_title_to_video(input_path: Path, title: str, output_path: Path):
    """Add a title overlay to a video."""
    cap = cv2.VideoCapture(str(input_path))
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add semi-transparent background for title
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add title text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, title, (20, 45), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
    
    cap.release()
    out.release()


def create_grid_showcase(videos: list, output_path: Path):
    """Create a grid layout showcase of multiple videos."""
    # Parameters
    grid_cols = 3
    grid_rows = 4
    tile_width = 426  # 1280 / 3
    tile_height = 270  # 1080 / 4
    output_width = 1280
    output_height = 1080
    fps = 30
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
    
    # Open all video captures
    caps = []
    for video_info in videos:
        if video_info['path'].exists():
            cap = cv2.VideoCapture(str(video_info['path']))
            caps.append({
                'cap': cap,
                'title': video_info['title'],
                'position': video_info['position']
            })
    
    # Process frames
    frame_count = 0
    max_frames = 90  # 3 seconds at 30fps
    
    while frame_count < max_frames:
        # Create black canvas
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        for cap_info in caps:
            cap = cap_info['cap']
            title = cap_info['title']
            row, col = cap_info['position']
            
            ret, frame = cap.read()
            if not ret:
                # Loop video if it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            
            if ret:
                # Resize frame to fit tile
                frame = cv2.resize(frame, (tile_width, tile_height))
                
                # Add title overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (5, 5), (tile_width - 5, 35), (0, 0, 0), -1)
                frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
                
                # Add title text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(frame, title, (10, 25), font, font_scale, 
                           (255, 255, 255), 1, cv2.LINE_AA)
                
                # Place in grid
                y_start = row * tile_height
                y_end = y_start + tile_height
                x_start = col * tile_width
                x_end = x_start + tile_width
                
                canvas[y_start:y_end, x_start:x_end] = frame
        
        # Add main title
        main_title = "Video Editing Tricks Showcase"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(main_title, font, 1.5, 2)[0]
        text_x = (output_width - text_size[0]) // 2
        
        # Add shadow effect
        cv2.putText(canvas, main_title, (text_x + 2, output_height - 18), 
                   font, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, main_title, (text_x, output_height - 20), 
                   font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(canvas)
        frame_count += 1
    
    # Cleanup
    for cap_info in caps:
        cap_info['cap'].release()
    out.release()
    
    print(f"Created grid showcase: {output_path}")


def create_sequential_showcase(videos: list, output_path: Path):
    """Create a sequential showcase with smooth transitions."""
    temp_dir = Path(tempfile.mkdtemp(prefix='showcase_'))
    
    try:
        # Create labeled versions
        labeled_videos = []
        for i, video_info in enumerate(videos):
            if not video_info['path'].exists():
                continue
            
            labeled_path = temp_dir / f"labeled_{i:02d}.mp4"
            add_title_to_video(video_info['path'], video_info['title'], labeled_path)
            labeled_videos.append(str(labeled_path))
        
        # Create concat list
        list_file = temp_dir / "videos.txt"
        with open(list_file, 'w') as f:
            for video in labeled_videos:
                f.write(f"file '{video}'\n")
        
        # Concatenate with ffmpeg
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
        print(f"Created sequential showcase: {output_path}")
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    """Create showcase videos."""
    print("\n" + "=" * 60)
    print("CREATING VIDEO EFFECTS SHOWCASE")
    print("=" * 60)
    
    # Define videos with their positions in grid
    test_dir = Path("test_output")
    videos = [
        # Row 0
        {'path': test_dir / 'test_input.mp4', 'title': 'Original', 'position': (0, 0)},
        {'path': test_dir / 'selective_color.mp4', 'title': 'Selective Color', 'position': (0, 1)},
        {'path': test_dir / 'floating.mp4', 'title': 'Floating Effect', 'position': (0, 2)},
        
        # Row 1
        {'path': test_dir / 'smooth_zoom.mp4', 'title': 'Smooth Zoom', 'position': (1, 0)},
        {'path': test_dir / '3d_photo.mp4', 'title': '3D Photo Effect', 'position': (1, 1)},
        {'path': test_dir / 'rotation.mp4', 'title': 'Rotation', 'position': (1, 2)},
        
        # Row 2
        {'path': test_dir / 'motion_text.mp4', 'title': 'Motion Tracking Text', 'position': (2, 0)},
        {'path': test_dir / 'animated_subtitles.mp4', 'title': 'Animated Subtitles', 'position': (2, 1)},
        {'path': test_dir / 'highlight_focus.mp4', 'title': 'Highlight Focus', 'position': (2, 2)},
        
        # Row 3
        {'path': test_dir / 'progress_bar.mp4', 'title': 'Progress Bar', 'position': (3, 0)},
        # Leave two spots empty for now
        {'path': test_dir / 'test_input.mp4', 'title': 'More Effects...', 'position': (3, 1)},
        {'path': test_dir / 'test_input.mp4', 'title': 'Coming Soon!', 'position': (3, 2)},
    ]
    
    # Create grid showcase
    print("\nCreating grid showcase...")
    grid_output = test_dir / 'showcase_grid.mp4'
    create_grid_showcase(videos, grid_output)
    
    # Create sequential showcase
    print("\nCreating sequential showcase...")
    seq_output = test_dir / 'showcase_sequential.mp4'
    create_sequential_showcase(videos[:10], seq_output)
    
    # Convert to H.264
    print("\nConverting to H.264...")
    for video in [grid_output, seq_output]:
        h264_output = video.parent / f"{video.stem}_h264.mp4"
        cmd = [
            'ffmpeg',
            '-i', str(video),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-y',
            str(h264_output)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"  ✓ {h264_output.name}")
    
    print("\n" + "=" * 60)
    print("SHOWCASE VIDEOS CREATED")
    print("=" * 60)
    print(f"\n📺 Grid Layout: {test_dir}/showcase_grid_h264.mp4")
    print(f"📺 Sequential: {test_dir}/showcase_sequential_h264.mp4")
    print("\nThese videos demonstrate all implemented effects!")


if __name__ == "__main__":
    main()