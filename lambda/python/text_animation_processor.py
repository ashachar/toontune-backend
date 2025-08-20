#!/usr/bin/env python3
"""
Text Animation Processor for Lambda
Applies text animation effects to videos
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import animation modules
from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve


def process_video(input_path, text, output_path):
    """
    Process video with text animation combo.
    """
    print(f"Processing: {input_path}")
    print(f"Text: {text}")
    print(f"Output: {output_path}")
    
    # Open video to get properties
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Animation parameters
    center_position = (width // 2, int(height * 0.45))
    font_size = int(min(150, height * 0.28))
    
    # Phase durations (in frames)
    phase1_frames = 30  # Shrinking
    phase2_frames = 20  # Moving behind
    phase3_frames = 40  # Stable behind
    dissolve_frames = 60  # Dissolve
    
    total_animation_frames = phase1_frames + phase2_frames + phase3_frames + dissolve_frames
    
    print(f"Animation setup: center={center_position}, font_size={font_size}")
    print(f"Animation phases: shrink={phase1_frames}, behind={phase2_frames}, stable={phase3_frames}, dissolve={dissolve_frames}")
    
    # Create TextBehindSegment animation
    text_animator = TextBehindSegment(
        element_path=input_path,
        background_path=input_path,
        position=center_position,
        text=text,
        font_size=font_size,
        text_color=(255, 220, 0),  # Yellow
        start_scale=2.0,
        end_scale=1.0,
        phase1_duration=phase1_frames / fps,
        phase2_duration=phase2_frames / fps,
        phase3_duration=phase3_frames / fps,
        center_position=center_position,
        fps=fps
    )
    
    # Get handoff data after stable phase
    handoff_frame_idx = phase1_frames + phase2_frames + phase3_frames - 1
    
    # Read a frame to establish handoff
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Render the handoff frame to freeze the text state
        _ = text_animator.render_text_frame(frame, handoff_frame_idx)
    
    handoff_data = text_animator.get_handoff_data()
    print(f"Handoff data acquired: {len(handoff_data.get('final_letter_positions', []))} letters")
    
    # Create WordDissolve animation with handoff
    word_dissolver = WordDissolve(
        element_path=input_path,
        background_path=input_path,
        position=center_position,
        word=text,
        font_size=font_size,
        text_color=(255, 220, 0),
        stable_duration=0.17,  # 5 frames at 30fps
        dissolve_duration=2.0,  # 60 frames at 30fps
        dissolve_stagger=0.5,
        float_distance=30,
        randomize_order=False,
        maintain_kerning=True,
        center_position=center_position,
        handoff_data=handoff_data,
        fps=fps
    )
    
    # Open video for processing
    cap = cv2.VideoCapture(input_path)
    
    # Setup video writer with H.264 codec
    # Use a temporary file first, then convert to H.264
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")
    
    frame_idx = 0
    frames_processed = 0
    
    print("Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply animations based on frame index
        if frame_idx < phase1_frames + phase2_frames + phase3_frames:
            # TextBehindSegment phase
            frame_rgb = text_animator.render_text_frame(frame_rgb, frame_idx)
        elif frame_idx < total_animation_frames:
            # WordDissolve phase
            dissolve_frame = frame_idx - (phase1_frames + phase2_frames + phase3_frames)
            frame_rgb = word_dissolver.render_word_frame(frame_rgb, dissolve_frame)
        
        # Convert back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        frames_processed += 1
        frame_idx += 1
        
        # Stop after animation completes + 1 second buffer
        if frame_idx >= total_animation_frames + fps:
            break
        
        # Progress indicator
        if frames_processed % 30 == 0:
            print(f"  Processed {frames_processed} frames...")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"✓ Processing complete! {frames_processed} frames written")
    
    # Convert to H.264 for web compatibility using imageio-ffmpeg
    print("Converting to H.264 for web compatibility...")
    import imageio_ffmpeg as ffmpeg
    import subprocess
    
    # Get the ffmpeg executable from imageio-ffmpeg
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    
    # Use ffmpeg to convert to H.264 with web-compatible settings
    ffmpeg_cmd = [
        ffmpeg_exe,
        '-i', temp_output,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',  # Enable fast start for web streaming
        '-y',  # Overwrite output
        output_path
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        print("✓ Successfully converted to H.264")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        # If ffmpeg fails, just rename the temp file
        os.rename(temp_output, output_path)
        print("Warning: Using fallback format (may not be web-compatible)")
    except Exception as e:
        print(f"Conversion error: {str(e)}")
        # If anything fails, just rename the temp file
        if os.path.exists(temp_output):
            os.rename(temp_output, output_path)
        print("Warning: Using fallback format (may not be web-compatible)")
    else:
        # Remove temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)
    
    # Verify output file
    if not os.path.exists(output_path):
        raise ValueError(f"Output file was not created: {output_path}")
    
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Output file size: {output_size:.2f} MB")


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) != 4:
        print("Usage: python text_animation_processor.py <input_video> <text> <output_video>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    text = sys.argv[2].upper()
    output_path = sys.argv[3]
    
    if not os.path.exists(input_path):
        print(f"Error: Input video not found: {input_path}")
        sys.exit(1)
    
    try:
        process_video(input_path, text, output_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()