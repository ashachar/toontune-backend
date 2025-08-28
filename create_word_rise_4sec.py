"""
Create a 4-second video showcasing the WordRiseSequence3D animation
"""

import cv2
import numpy as np
import sys
import os

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/word_3d')

from base_3d_text_animation import Animation3DConfig
from word_3d import WordRiseSequence3D


def create_word_rise_video():
    """Create a 4-second video with words rising from below"""
    
    print("Creating 4-second Word Rise Animation")
    print("=" * 60)
    
    # Load input video (we'll use just 4 seconds)
    input_video = "uploads/assets/videos/ai_math1.mp4"
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Error: Could not open {input_video}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} fps")
    
    # Calculate total frames for 4 seconds
    duration_seconds = 4
    total_frames = int(fps * duration_seconds)
    
    print(f"Creating {duration_seconds}-second video ({total_frames} frames)")
    
    # Create output video
    output_path = "outputs/word_rise_4sec.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Configure the animation
    # Use a meaningful phrase that works well with sequential reveal
    text = "AMAZING WORDS RISE UP"
    num_words = len(text.split())
    
    # Calculate optimal timing for 4 seconds with very gentle, slow motion
    # We want all words to be fully visible by 3.5 seconds, leaving 0.5s to admire
    animation_duration_ms = 3500
    
    # Use much longer rise duration for very smooth, gentle motion
    rise_duration_ms = 1200  # Each word takes 1.2 seconds to rise (very slow and gentle)
    # Calculate word spacing so all words appear within the duration
    word_spacing_ms = 600  # Increased spacing for more dramatic effect
    
    print(f"\nAnimation configuration:")
    print(f"  Text: '{text}'")
    print(f"  Words: {num_words}")
    print(f"  Word spacing: {word_spacing_ms}ms")
    print(f"  Rise duration: {rise_duration_ms}ms per word")
    print(f"  Total animation: {animation_duration_ms}ms")
    
    # Create animation configuration
    config = Animation3DConfig(
        text=text,
        duration_ms=4000,  # Full 4 seconds
        position=(640, 360, 0),  # Center position
        font_size=85,  # Larger for impact
        font_color=(255, 255, 255),  # Pure white
        depth_color=(200, 200, 200),  # Light gray depth
        stagger_ms=0,  # No letter stagger, we handle word timing
        enable_shadows=True,
        shadow_distance=10,
        shadow_opacity=0.7
    )
    
    # Create the word rise animation with very gentle settings
    animation = WordRiseSequence3D(
        config,
        word_spacing_ms=word_spacing_ms,
        rise_distance=200,  # Further reduced distance for gentler motion
        rise_duration_ms=rise_duration_ms,
        overshoot=0.0,  # No overshoot to prevent any jitter
        fade_in=True,  # Smooth fade as they rise
        stack_mode=False  # All words go to same vertical position
    )
    
    print("\nRendering frames...")
    
    # Process frames
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            # If video ends, loop back to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                # Create black frame if video fails
                frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply animation
        animated_frame = animation.apply_frame(frame, frame_num, fps)
        
        # Add progress indicator
        progress = frame_num / total_frames
        time_seconds = frame_num / fps
        
        # Add subtle info overlay
        info_text = f"Word Rise Animation | {time_seconds:.1f}s / {duration_seconds}s"
        cv2.putText(animated_frame, info_text, 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (150, 150, 150), 1)
        
        # Add progress bar
        bar_width = int(width * 0.3)
        bar_height = 4
        bar_x = width - bar_width - 10
        bar_y = height - 30
        # Background
        cv2.rectangle(animated_frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        # Progress
        progress_width = int(bar_width * progress)
        if progress_width > 0:
            cv2.rectangle(animated_frame, (bar_x, bar_y), 
                         (bar_x + progress_width, bar_y + bar_height), 
                         (100, 200, 100), -1)
        
        out.write(animated_frame)
        
        # Progress indicator
        if frame_num % 10 == 0:
            print(f"  Frame {frame_num}/{total_frames} ({progress*100:.1f}%)")
    
    out.release()
    cap.release()
    
    # Convert to H.264
    print("\nConverting to H.264...")
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    result = os.system(convert_cmd)
    
    if result == 0:
        os.remove(output_path)
        print(f"\n✅ Success! Video created: {h264_output}")
    else:
        print(f"\n⚠️ H.264 conversion failed, keeping original: {output_path}")
        h264_output = output_path
    
    print("\n" + "=" * 60)
    print("WORD RISE ANIMATION COMPLETE!")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Output: {h264_output}")
    print("=" * 60)
    
    return h264_output


if __name__ == "__main__":
    print("WORD RISE SEQUENCE 3D - 4 SECOND SHOWCASE")
    print("=" * 60)
    print("This creates a focused 4-second video where:")
    print("  • Each word rises from below, one after another")
    print("  • Words fade in as they rise to the center")
    print("  • Smooth motion with subtle bounce at the top")
    print("  • Perfect timing for a 4-second showcase")
    print()
    
    output = create_word_rise_video()
    
    if output:
        print(f"\nVideo ready to view: {output}")
        print("\nAnimation breakdown:")
        print("  0.0s - 0.5s: First word rises")
        print("  0.5s - 1.0s: Second word rises")
        print("  1.0s - 1.5s: Third word rises")
        print("  1.5s - 2.0s: Fourth word rises")
        print("  2.0s - 4.0s: All words visible and settled")