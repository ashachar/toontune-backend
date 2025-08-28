"""
Test all text animation families sequentially on AI_Math1.mp4
Each animation appears one after another for clear visibility
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/opacity_family')
sys.path.append('utils/animations/motion_family') 
sys.path.append('utils/animations/scale_family')
sys.path.append('utils/animations/progressive_family')
sys.path.append('utils/animations/compound_family')

from base_text_animation import AnimationConfig, EasingType
from opacity_animation import SimpleFadeAnimation, BlurFadeAnimation, GlowFadeAnimation
from motion_animation import SlideInAnimation, FloatUpAnimation, BounceInAnimation
from scale_animation import ZoomInAnimation, Rotate3DAnimation
from progressive_animation import TypewriterAnimation, WordRevealAnimation, LineStaggerAnimation
from compound_animation import FadeSlideAnimation, ScaleBlurAnimation


def create_sequential_animations():
    """Create all animations to run sequentially"""
    
    animations = []
    duration = 2000  # 2 seconds per animation
    y_position = 200  # Center position
    
    # Calculate sequential start times (3 seconds apart for visibility)
    time_offset = 3.0  # seconds between animation starts
    current_time = 0.0
    
    # ============ OPACITY FAMILY ============
    
    # 1. Simple Fade
    config = AnimationConfig(
        text="1. Simple Fade In",
        duration_ms=duration,
        position=(100, y_position),
        font_size=48,
        font_color=(255, 255, 255),
        easing=EasingType.EASE_IN_OUT
    )
    animations.append({
        "name": "Simple Fade",
        "animation": SimpleFadeAnimation(config, start_opacity=0, end_opacity=1),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    current_time += time_offset
    
    # 2. Blur Fade
    config = AnimationConfig(
        text="2. Blur to Focus",
        duration_ms=duration,
        position=(100, y_position),
        font_size=48,
        font_color=(255, 220, 100),
        easing=EasingType.EASE_OUT
    )
    animations.append({
        "name": "Blur Fade",
        "animation": BlurFadeAnimation(config, start_blur=15, end_blur=0),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    current_time += time_offset
    
    # 3. Glow Fade
    config = AnimationConfig(
        text="3. Glowing Text",
        duration_ms=duration,
        position=(100, y_position),
        font_size=48,
        font_color=(100, 255, 255),
        easing=EasingType.LINEAR
    )
    animations.append({
        "name": "Glow Fade",
        "animation": GlowFadeAnimation(config, glow_radius=8, pulse_count=2),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    current_time += time_offset
    
    # ============ MOTION FAMILY ============
    
    # 4. Slide In from Left
    config = AnimationConfig(
        text="4. Slide From Left",
        duration_ms=duration,
        position=(300, y_position),
        font_size=48,
        font_color=(255, 200, 200),
        easing=EasingType.EASE_OUT
    )
    animations.append({
        "name": "Slide Left",
        "animation": SlideInAnimation(config, direction="left", distance=250),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    current_time += time_offset
    
    # 5. Slide In from Right
    config = AnimationConfig(
        text="5. Slide From Right",
        duration_ms=duration,
        position=(100, y_position),
        font_size=48,
        font_color=(200, 255, 200),
        easing=EasingType.EASE_OUT
    )
    animations.append({
        "name": "Slide Right",
        "animation": SlideInAnimation(config, direction="right", distance=250),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    current_time += time_offset
    
    # 6. Slide In from Top
    config = AnimationConfig(
        text="6. Slide From Top",
        duration_ms=duration,
        position=(100, y_position),
        font_size=48,
        font_color=(200, 200, 255),
        easing=EasingType.EASE_OUT
    )
    animations.append({
        "name": "Slide Top",
        "animation": SlideInAnimation(config, direction="top", distance=150),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    current_time += time_offset
    
    # 7. Slide In from Bottom
    config = AnimationConfig(
        text="7. Slide From Bottom",
        duration_ms=duration,
        position=(100, y_position),
        font_size=48,
        font_color=(255, 255, 200),
        easing=EasingType.EASE_OUT
    )
    animations.append({
        "name": "Slide Bottom",
        "animation": SlideInAnimation(config, direction="bottom", distance=150),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    current_time += time_offset
    
    # 8. Float Up
    config = AnimationConfig(
        text="8. Floating Upward",
        duration_ms=duration,
        position=(100, y_position + 50),
        font_size=48,
        font_color=(255, 200, 100),
        easing=EasingType.EASE_IN_OUT
    )
    animations.append({
        "name": "Float Up",
        "animation": FloatUpAnimation(config, float_distance=50),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    current_time += time_offset
    
    # 9. Bounce In
    config = AnimationConfig(
        text="9. Bounce Effect!",
        duration_ms=duration + 500,  # Longer for bounce effect
        position=(100, y_position),
        font_size=52,
        font_color=(150, 255, 150),
        easing=EasingType.ELASTIC
    )
    animations.append({
        "name": "Bounce",
        "animation": BounceInAnimation(config, direction="bottom", distance=200),
        "start_time": current_time,
        "duration": (duration + 500) / 1000.0
    })
    current_time += time_offset
    
    # ============ SCALE FAMILY ============
    
    # 10. Zoom In
    config = AnimationConfig(
        text="10. ZOOM IN",
        duration_ms=duration,
        position=(200, y_position),
        font_size=56,
        font_color=(255, 100, 100),
        easing=EasingType.EASE_OUT
    )
    animations.append({
        "name": "Zoom In",
        "animation": ZoomInAnimation(config, start_scale=2.0, end_scale=1.0),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    current_time += time_offset
    
    # 11. 3D Rotation Y-axis
    config = AnimationConfig(
        text="11. 3D ROTATE-Y",
        duration_ms=duration,
        position=(150, y_position),
        font_size=50,
        font_color=(200, 200, 255),
        easing=EasingType.EASE_IN_OUT
    )
    animations.append({
        "name": "3D Rotate Y",
        "animation": Rotate3DAnimation(config, rotation_axis="Y", start_rotation=90),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    current_time += time_offset
    
    # 12. 3D Rotation X-axis
    config = AnimationConfig(
        text="12. 3D ROTATE-X",
        duration_ms=duration,
        position=(150, y_position),
        font_size=50,
        font_color=(255, 200, 255),
        easing=EasingType.EASE_IN_OUT
    )
    animations.append({
        "name": "3D Rotate X",
        "animation": Rotate3DAnimation(config, rotation_axis="X", start_rotation=90),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    current_time += time_offset
    
    # ============ PROGRESSIVE FAMILY ============
    
    # 13. Typewriter
    config = AnimationConfig(
        text="13. Typewriter effect...",
        duration_ms=duration + 1000,  # Longer for typing effect
        position=(100, y_position),
        font_size=44,
        font_color=(255, 255, 150),
        easing=EasingType.LINEAR
    )
    animations.append({
        "name": "Typewriter",
        "animation": TypewriterAnimation(config, cursor_visible=True, cursor_blink=True),
        "start_time": current_time,
        "duration": (duration + 1000) / 1000.0
    })
    current_time += time_offset
    
    # 14. Word Reveal
    config = AnimationConfig(
        text="14. Word by word reveal",
        duration_ms=duration,
        position=(100, y_position),
        font_size=44,
        font_color=(150, 255, 200),
        easing=EasingType.EASE_IN
    )
    animations.append({
        "name": "Word Reveal",
        "animation": WordRevealAnimation(config, fade_words=True),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    current_time += time_offset
    
    # 15. Line Stagger
    config = AnimationConfig(
        text="15. First line\nSecond line\nThird line",
        duration_ms=duration + 1000,
        position=(100, y_position - 30),
        font_size=40,
        font_color=(255, 150, 255),
        easing=EasingType.EASE_OUT
    )
    animations.append({
        "name": "Line Stagger",
        "animation": LineStaggerAnimation(config, line_delay_ms=400),
        "start_time": current_time,
        "duration": (duration + 1000) / 1000.0
    })
    current_time += time_offset
    
    # ============ COMPOUND FAMILY ============
    
    # 16. Fade + Slide
    config = AnimationConfig(
        text="16. Fade + Slide",
        duration_ms=duration,
        position=(100, y_position),
        font_size=48,
        font_color=(255, 255, 100),
        easing=EasingType.EASE_IN_OUT
    )
    animations.append({
        "name": "Fade Slide",
        "animation": FadeSlideAnimation(config, slide_direction="top", slide_distance=60),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    current_time += time_offset
    
    # 17. Scale + Blur
    config = AnimationConfig(
        text="17. Scale + Blur",
        duration_ms=duration,
        position=(200, y_position),
        font_size=52,
        font_color=(100, 200, 255),
        easing=EasingType.EASE_OUT
    )
    animations.append({
        "name": "Scale Blur",
        "animation": ScaleBlurAnimation(config, start_scale=1.8, start_blur=12),
        "start_time": current_time,
        "duration": duration / 1000.0
    })
    
    return animations


def apply_sequential_test(input_video: str, output_path: str = "outputs/"):
    """Apply all animations sequentially to the video"""
    
    print(f"Loading video: {input_video}")
    
    # Create all animations
    animations = create_sequential_animations()
    total_duration = animations[-1]["start_time"] + animations[-1]["duration"] + 1.0
    
    print(f"Total animation sequence duration: {total_duration:.1f} seconds")
    print(f"Total animations to test: {len(animations)}")
    
    # Extract video segment
    temp_segment = f"{output_path}temp_segment.mp4"
    extract_cmd = f"ffmpeg -i {input_video} -t {total_duration} -c:v libx264 -preset fast -crf 18 {temp_segment} -y"
    print(f"Extracting {total_duration:.1f} second segment...")
    os.system(extract_cmd)
    
    # Open video
    cap = cv2.VideoCapture(temp_segment)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} @ {fps:.2f} fps")
    
    # Create output video writer
    temp_output = f"{output_path}sequential_test_temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    frame_number = 0
    active_animations = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_number / fps
        
        # Check for newly active animations
        for anim_data in animations:
            if (anim_data["start_time"] <= current_time < anim_data["start_time"] + anim_data["duration"] 
                and anim_data not in active_animations):
                active_animations.append(anim_data)
                print(f"  [{current_time:.1f}s] Starting: {anim_data['name']}")
        
        # Remove expired animations
        active_animations = [a for a in active_animations 
                           if current_time < a["start_time"] + a["duration"]]
        
        # Apply active animations
        for anim_data in active_animations:
            relative_time = current_time - anim_data["start_time"]
            relative_frame = int(relative_time * fps)
            frame = anim_data["animation"].apply_frame(frame, relative_frame, fps)
        
        # Add frame counter and current animation name
        cv2.putText(frame, f"Frame: {frame_number} | Time: {current_time:.1f}s", 
                   (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if active_animations:
            anim_names = ", ".join([a["name"] for a in active_animations])
            cv2.putText(frame, f"Active: {anim_names}", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        out.write(frame)
        frame_number += 1
        
        if frame_number % int(fps * 5) == 0:
            print(f"  Progress: {current_time:.1f}s / {total_duration:.1f}s")
    
    cap.release()
    out.release()
    
    # Convert to H.264
    final_output = f"{output_path}all_animations_sequential_test.mp4"
    print("\nConverting to H.264...")
    convert_cmd = f"ffmpeg -i {temp_output} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {final_output} -y"
    os.system(convert_cmd)
    
    # Clean up temp files
    os.remove(temp_segment)
    os.remove(temp_output)
    
    print(f"\n✅ Sequential test complete: {final_output}")
    
    # Create animation list summary
    summary_path = f"{output_path}animation_sequence_summary.txt"
    with open(summary_path, "w") as f:
        f.write("ANIMATION SEQUENCE SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Duration: {total_duration:.1f} seconds\n")
        f.write(f"Total Animations: {len(animations)}\n\n")
        f.write("Timeline:\n")
        f.write("-" * 60 + "\n")
        
        for i, anim in enumerate(animations, 1):
            f.write(f"{i:2d}. {anim['name']:15s} | Start: {anim['start_time']:5.1f}s | Duration: {anim['duration']:.1f}s\n")
    
    print(f"Animation timeline saved to: {summary_path}")
    
    return final_output


def main():
    # Check if AI_Math1.mp4 exists
    video_path = "uploads/assets/videos/ai_math1.mp4"
    
    if not Path(video_path).exists():
        print(f"Error: {video_path} not found!")
        return
    
    print("=" * 60)
    print("SEQUENTIAL TEXT ANIMATION TEST")
    print("=" * 60)
    print("\nThis test will display each animation one after another")
    print("for clear visibility and comparison.\n")
    
    # Run the sequential test
    output_file = apply_sequential_test(video_path)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"\n✨ Output video: {output_file}")
    print("\nEach animation appears sequentially with a 3-second interval.")
    print("The video includes frame counter and active animation labels.")


if __name__ == "__main__":
    main()