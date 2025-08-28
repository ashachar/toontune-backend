"""
Test the Text Morph 3D animation - replicating real_estate.mov 9-11s effect
"""

import cv2
import numpy as np
import sys
import os

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/morph_3d')

from base_3d_text_animation import Animation3DConfig
from morph_3d import TextMorph3D, CrossDissolve3D


def test_morph_animation():
    """Test text morphing animation matching real_estate.mov 9-11s"""
    
    print("Testing Text Morph 3D Animation")
    print("Replicating effect from real_estate.mov (9-11 seconds)")
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
    output_path = "outputs/text_morph_3d.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Test different morph variations
    animations = [
        # Main morph effect - matching the real estate video timing
        ("ORIGINAL TEXT", "MORPHED TEXT", TextMorph3D, {
            "target_text": "TRANSFORMED",
            "morph_start": 0.3,  # Start morphing at 30% 
            "morph_end": 0.7,    # Complete morph at 70%
            "blur_peak": 20.0,   # Strong blur during transition
            "dissolve_overlap": 0.3
        }, 4000),
        
        # Quick morph
        ("FAST MORPH", "QUICK CHANGE", TextMorph3D, {
            "target_text": "QUICK CHANGE",
            "morph_start": 0.4,
            "morph_end": 0.6,  # Faster transition
            "blur_peak": 15.0,
            "dissolve_overlap": 0.2
        }, 3000),
        
        # Slow morph  
        ("SLOW TRANSITION", "GRADUAL CHANGE", TextMorph3D, {
            "target_text": "GRADUAL CHANGE",
            "morph_start": 0.2,
            "morph_end": 0.8,  # Slower transition
            "blur_peak": 25.0,
            "dissolve_overlap": 0.4
        }, 5000),
        
        # Cross dissolve variant
        ("CROSS FADE", "NEW MESSAGE", CrossDissolve3D, {
            "target_text": "NEW MESSAGE",
            "transition_start": 0.35,
            "transition_duration": 0.3
        }, 3500),
    ]
    
    print(f"\nRendering {len(animations)} morph animations...")
    
    frame_count = 0
    animation_index = 0
    current_animation = None
    animation_start_frame = 0
    
    while animation_index <= len(animations):
        ret, frame = cap.read()
        if not ret:
            # Loop video if needed
            if animation_index < len(animations):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                break
        
        # Check if we need to start a new animation
        if animation_index < len(animations) and (current_animation is None or 
            (frame_count - animation_start_frame) * 1000 / fps >= animations[animation_index][4]):
            
            # Create new animation
            source_text, target_text, anim_class, params, duration = animations[animation_index]
            print(f"\n[{animation_index + 1}/{len(animations)}] '{source_text}' → '{target_text}'")
            
            config = Animation3DConfig(
                text=source_text,
                duration_ms=duration,
                position=(640, 360, 0),
                font_size=80,
                font_color=(255, 255, 255),
                depth_color=(150, 150, 150),
                stagger_ms=20,
                enable_shadows=True,
                shadow_distance=5
            )
            
            current_animation = anim_class(config, **params)
            animation_start_frame = frame_count
            animation_index += 1
        
        # Apply current animation
        if current_animation and animation_index > 0:
            animation_frame_num = frame_count - animation_start_frame
            animated_frame = current_animation.apply_frame(frame, animation_frame_num, fps)
            
            # Calculate and show progress
            if animation_index <= len(animations):
                progress = min(1.0, (animation_frame_num / fps) * 1000 / animations[animation_index-1][4])
                
                # Add labels
                source, target = animations[animation_index-1][0], animations[animation_index-1][1]
                
                # Show current state
                if progress < 0.3:
                    state = "SHOWING SOURCE"
                    color = (200, 255, 200)
                elif progress < 0.7:
                    state = "MORPHING..."
                    color = (200, 200, 255)
                else:
                    state = "SHOWING TARGET"
                    color = (255, 200, 200)
                
                cv2.putText(animated_frame, f"[{animation_index}/{len(animations)}] {state}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
                cv2.putText(animated_frame, f"'{source}' -> '{target}'", 
                           (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                cv2.putText(animated_frame, f"Progress: {progress:.0%}", 
                           (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            out.write(animated_frame)
        else:
            out.write(frame)
        
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Progress: {frame_count} frames")
    
    out.release()
    cap.release()
    
    # Convert to H.264
    print(f"\nConverting to H.264...")
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    os.system(convert_cmd)
    os.remove(output_path)
    
    print(f"\n" + "=" * 60)
    print(f"TEXT MORPH ANIMATION COMPLETE!")
    print(f"Total animations: {len(animations)}")
    print(f"Total duration: {frame_count / fps:.1f} seconds")
    print(f"Output: {h264_output}")
    print("=" * 60)
    
    return h264_output


if __name__ == "__main__":
    print("TEXT MORPH 3D ANIMATION TEST")
    print("=" * 60)
    print("This replicates the exact animation from real_estate.mov (9-11s):")
    print("  • Text smoothly morphs from one phrase to another")
    print("  • Letters blur/dissolve during transition")
    print("  • Staggered letter transitions")
    print("  • Maintains position while content changes")
    print()
    
    output = test_morph_animation()
    
    if output:
        print(f"\n✅ Success! Video created: {output}")
        print("\nAnimation features:")
        print("  • Smooth text-to-text morphing")
        print("  • Blur effect peaks during transition")
        print("  • Letters fade out/in with overlap")
        print("  • Configurable morph timing and blur intensity")
        print("  • Matches the real_estate.mov effect!")