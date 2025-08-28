"""
Test word-based 3D animations where words appear sequentially
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
from word_3d import WordRiseSequence3D, WordDropIn3D, WordWave3D


def test_word_animations():
    """Test word-based sequential animations"""
    
    print("Testing Word Sequential 3D Animations")
    print("=" * 60)
    
    # Load input video
    input_video = "uploads/assets/videos/ai_math1.mp4"
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Error: Could not open {input_video}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} fps")
    
    # Create output video
    output_path = "outputs/word_animations_3d.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Define test animations
    animations = [
        # Word rise from below - main request
        ("WORDS RISE FROM BELOW", WordRiseSequence3D, {
            "word_spacing_ms": 400,
            "rise_distance": 300,
            "rise_duration_ms": 500,
            "overshoot": 0.15,
            "fade_in": True,
            "stack_mode": False
        }, 4000),
        
        # Words rise and stack
        ("WORDS STACK VERTICALLY", WordRiseSequence3D, {
            "word_spacing_ms": 300,
            "rise_distance": 250,
            "rise_duration_ms": 400,
            "overshoot": 0.1,
            "fade_in": True,
            "stack_mode": True
        }, 3500),
        
        # Fast word rise
        ("QUICK WORD SEQUENCE", WordRiseSequence3D, {
            "word_spacing_ms": 200,
            "rise_distance": 200,
            "rise_duration_ms": 300,
            "overshoot": 0.05,
            "fade_in": False,
            "stack_mode": False
        }, 3000),
        
        # Words drop from above
        ("WORDS DROP WITH BOUNCE", WordDropIn3D, {
            "word_spacing_ms": 350,
            "drop_height": 400,
            "drop_duration_ms": 500,
            "bounce_count": 3,
            "bounce_damping": 0.5
        }, 4000),
        
        # Wave pattern
        ("CENTER OUTWARD WAVE", WordWave3D, {
            "wave_speed_ms": 150,
            "rise_distance": 200,
            "scale_effect": True
        }, 3500),
    ]
    
    print(f"\nRendering {len(animations)} word animations...")
    
    frame_count = 0
    animation_index = 0
    current_animation = None
    animation_start_frame = 0
    
    while animation_index <= len(animations):
        ret, frame = cap.read()
        if not ret:
            if animation_index < len(animations):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                break
        
        # Start new animation when needed
        if animation_index < len(animations) and (current_animation is None or 
            (frame_count - animation_start_frame) * 1000 / fps >= animations[animation_index][3]):
            
            text, anim_class, params, duration = animations[animation_index]
            print(f"\n[{animation_index + 1}/{len(animations)}] {text}")
            
            config = Animation3DConfig(
                text=text,
                duration_ms=duration,
                position=(640, 360, 0),
                font_size=70,
                font_color=(255, 255, 255),
                depth_color=(180, 180, 180),
                stagger_ms=0,  # No letter stagger, we handle word timing
                enable_shadows=True,
                shadow_distance=8,
                shadow_opacity=0.6
            )
            
            current_animation = anim_class(config, **params)
            animation_start_frame = frame_count
            animation_index += 1
        
        # Apply animation
        if current_animation and animation_index > 0:
            animation_frame_num = frame_count - animation_start_frame
            animated_frame = current_animation.apply_frame(frame, animation_frame_num, fps)
            
            # Add labels
            if animation_index <= len(animations):
                anim_info = animations[animation_index-1]
                progress = min(1.0, (animation_frame_num / fps) * 1000 / anim_info[3])
                
                # Show animation type
                cv2.putText(animated_frame, f"[{animation_index}/{len(animations)}] {anim_info[0]}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
                
                # Show word timing info
                num_words = len(anim_info[0].split())
                current_word = min(num_words - 1, int(progress * num_words * 1.5))
                cv2.putText(animated_frame, f"Word {current_word + 1}/{num_words} | Progress: {progress:.0%}", 
                           (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            
            out.write(animated_frame)
        else:
            out.write(frame)
        
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Frame {frame_count}")
    
    out.release()
    cap.release()
    
    # Convert to H.264
    print("\nConverting to H.264...")
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    os.system(convert_cmd)
    os.remove(output_path)
    
    print(f"\n" + "=" * 60)
    print("WORD ANIMATIONS COMPLETE!")
    print(f"Total animations: {len(animations)}")
    print(f"Duration: {frame_count / fps:.1f} seconds")
    print(f"Output: {h264_output}")
    print("=" * 60)
    
    return h264_output


if __name__ == "__main__":
    print("WORD SEQUENTIAL 3D ANIMATIONS TEST")
    print("=" * 60)
    print("Features:")
    print("  • WordRiseSequence3D - Words rise from below one after another")
    print("  • WordDropIn3D - Words drop from above with bounce")
    print("  • WordWave3D - Words appear in wave pattern from center")
    print()
    
    output = test_word_animations()
    
    if output:
        print(f"\n✅ Success! Video created: {output}")
        print("\nAnimation highlights:")
        print("  • Words appear sequentially, not all at once")
        print("  • Each word maintains its own timing")
        print("  • Smooth rise/drop motions with physics")
        print("  • Optional stacking for multi-line effects")
        print("  • Center-outward wave patterns")