"""
Comprehensive test suite for all 3D text animations
Tests each 3D animation family on AI_Math1.mp4
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/opacity_3d')
sys.path.append('utils/animations/3d_animations/motion_3d')
sys.path.append('utils/animations/3d_animations/scale_3d')
sys.path.append('utils/animations/3d_animations/progressive_3d')
sys.path.append('utils/animations/3d_animations/compound_3d')

from base_3d_text_animation import Base3DTextAnimation, Animation3DConfig
from opacity_3d import Fade3D, BlurFade3D, GlowPulse3D, Dissolve3D, Materialize3D
from motion_3d import Slide3D, Float3D, Bounce3D, Orbit3D, Swarm3D
from scale_3d import Zoom3D, Rotate3DAxis, FlipCard3D, Tumble3D, Unfold3D, Explode3D
from progressive_3d import Typewriter3D, WordReveal3D, WaveReveal3D, Cascade3D, Build3D
from compound_3d import FadeSlide3D, ZoomBlur3D, RotateExplode3D, WaveFloat3D, TypewriterBounce3D, SpiralMaterialize3D, ElasticPop3D


def create_3d_animation_showcase():
    """Create all 3D animations for testing"""
    
    animations = []
    duration = 3000  # 3 seconds per animation
    position_3d = (640, 360, 0)  # Center of 1280x720 frame
    
    # ============ OPACITY 3D FAMILY ============
    
    # 1. Fade 3D with wave mode
    config = Animation3DConfig(
        text="3D FADE WAVE",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(255, 255, 255),
        depth_color=(180, 180, 180),
        stagger_ms=50
    )
    animations.append({
        "name": "Fade3D Wave",
        "animation": Fade3D(config, fade_mode="wave"),
        "start_time": 0
    })
    
    # 2. Blur Fade 3D
    config = Animation3DConfig(
        text="BLUR DEPTH",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(255, 220, 100)
    )
    animations.append({
        "name": "BlurFade3D",
        "animation": BlurFade3D(config, blur_varies_with_depth=True),
        "start_time": 3.5
    })
    
    # 3. Glow Pulse 3D
    config = Animation3DConfig(
        text="GLOW PULSE",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(100, 255, 255),
        enable_glow=True
    )
    animations.append({
        "name": "GlowPulse3D",
        "animation": GlowPulse3D(config, pulse_count=3, glow_cascade=True),
        "start_time": 7
    })
    
    # 4. Dissolve 3D
    config = Animation3DConfig(
        text="DISSOLVE 3D",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(255, 200, 200)
    )
    animations.append({
        "name": "Dissolve3D",
        "animation": Dissolve3D(config, dissolve_direction="spiral"),
        "start_time": 10.5
    })
    
    # 5. Materialize 3D
    config = Animation3DConfig(
        text="MATERIALIZE",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(200, 255, 200)
    )
    animations.append({
        "name": "Materialize3D",
        "animation": Materialize3D(config, materialize_from="edges"),
        "start_time": 14
    })
    
    # ============ MOTION 3D FAMILY ============
    
    # 6. Slide 3D
    config = Animation3DConfig(
        text="SLIDE 3D",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(255, 255, 200)
    )
    animations.append({
        "name": "Slide3D",
        "animation": Slide3D(config, slide_direction="front", curved_path=True),
        "start_time": 17.5
    })
    
    # 7. Float 3D
    config = Animation3DConfig(
        text="FLOATING",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(255, 200, 100)
    )
    animations.append({
        "name": "Float3D",
        "animation": Float3D(config, float_pattern="wave"),
        "start_time": 21
    })
    
    # 8. Bounce 3D
    config = Animation3DConfig(
        text="BOUNCE!",
        duration_ms=duration,
        position=position_3d,
        font_size=65,
        font_color=(150, 255, 150)
    )
    animations.append({
        "name": "Bounce3D",
        "animation": Bounce3D(config, bounce_count=3, spin_on_bounce=True),
        "start_time": 24.5
    })
    
    # 9. Orbit 3D
    config = Animation3DConfig(
        text="ORBITAL",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(200, 200, 255)
    )
    animations.append({
        "name": "Orbit3D",
        "animation": Orbit3D(config, orbit_axis="y", elliptical=True),
        "start_time": 28
    })
    
    # 10. Swarm 3D
    config = Animation3DConfig(
        text="SWARM",
        duration_ms=duration + 1000,  # Longer for swarm effect
        position=position_3d,
        font_size=60,
        font_color=(255, 150, 255)
    )
    animations.append({
        "name": "Swarm3D",
        "animation": Swarm3D(config, swarm_speed=3.0),
        "start_time": 31.5
    })
    
    # ============ SCALE 3D FAMILY ============
    
    # 11. Zoom 3D
    config = Animation3DConfig(
        text="ZOOM 3D",
        duration_ms=duration,
        position=position_3d,
        font_size=70,
        font_color=(255, 100, 100)
    )
    animations.append({
        "name": "Zoom3D",
        "animation": Zoom3D(config, spiral_zoom=True, pulsate=True),
        "start_time": 36
    })
    
    # 12. Rotate 3D Axis
    config = Animation3DConfig(
        text="ROTATE-Y",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(100, 100, 255)
    )
    animations.append({
        "name": "Rotate3DAxis",
        "animation": Rotate3DAxis(config, rotation_axis="all", wobble=True),
        "start_time": 39.5
    })
    
    # 13. Flip Card 3D
    config = Animation3DConfig(
        text="FLIP CARD",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(255, 255, 100),
        depth_color=(100, 100, 50)
    )
    animations.append({
        "name": "FlipCard3D",
        "animation": FlipCard3D(config, flip_sequential=True, show_back=True),
        "start_time": 43
    })
    
    # 14. Tumble 3D
    config = Animation3DConfig(
        text="TUMBLE",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(255, 200, 255)
    )
    animations.append({
        "name": "Tumble3D",
        "animation": Tumble3D(config, gravity_effect=True),
        "start_time": 46.5
    })
    
    # 15. Unfold 3D
    config = Animation3DConfig(
        text="UNFOLD",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(200, 255, 255)
    )
    animations.append({
        "name": "Unfold3D",
        "animation": Unfold3D(config, unfold_origin="center"),
        "start_time": 50
    })
    
    # 16. Explode 3D
    config = Animation3DConfig(
        text="EXPLODE!",
        duration_ms=duration,
        position=position_3d,
        font_size=65,
        font_color=(255, 150, 100)
    )
    animations.append({
        "name": "Explode3D",
        "animation": Explode3D(config, explosion_force=300),
        "start_time": 53.5
    })
    
    # ============ PROGRESSIVE 3D FAMILY ============
    
    # 17. Typewriter 3D
    config = Animation3DConfig(
        text="TYPE 3D...",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(255, 255, 150)
    )
    animations.append({
        "name": "Typewriter3D",
        "animation": Typewriter3D(config, key_bounce=True),
        "start_time": 57
    })
    
    # 18. Word Reveal 3D
    config = Animation3DConfig(
        text="WORD BY WORD",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(150, 255, 200)
    )
    animations.append({
        "name": "WordReveal3D",
        "animation": WordReveal3D(config, reveal_style="flip"),
        "start_time": 60.5
    })
    
    # 19. Wave Reveal 3D
    config = Animation3DConfig(
        text="WAVE REVEAL",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(255, 200, 150)
    )
    animations.append({
        "name": "WaveReveal3D",
        "animation": WaveReveal3D(config, wave_direction="horizontal"),
        "start_time": 64
    })
    
    # 20. Cascade 3D
    config = Animation3DConfig(
        text="CASCADE",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(150, 200, 255)
    )
    animations.append({
        "name": "Cascade3D",
        "animation": Cascade3D(config, cascade_style="fountain"),
        "start_time": 67.5
    })
    
    # 21. Build 3D
    config = Animation3DConfig(
        text="BUILD 3D",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(255, 150, 200)
    )
    animations.append({
        "name": "Build3D",
        "animation": Build3D(config, build_style="assemble"),
        "start_time": 71
    })
    
    # ============ COMPOUND 3D FAMILY ============
    
    # 22. Fade Slide 3D
    config = Animation3DConfig(
        text="FADE+SLIDE",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(255, 255, 100)
    )
    animations.append({
        "name": "FadeSlide3D",
        "animation": FadeSlide3D(config, spiral_slide=True),
        "start_time": 74.5
    })
    
    # 23. Zoom Blur 3D
    config = Animation3DConfig(
        text="ZOOM+BLUR",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(100, 200, 255)
    )
    animations.append({
        "name": "ZoomBlur3D",
        "animation": ZoomBlur3D(config, zoom_origin="spiral"),
        "start_time": 78
    })
    
    # 24. Rotate Explode 3D
    config = Animation3DConfig(
        text="ROTATE+EXP",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(255, 100, 150)
    )
    animations.append({
        "name": "RotateExplode3D",
        "animation": RotateExplode3D(config, implosion_first=True),
        "start_time": 81.5
    })
    
    # 25. Wave Float 3D
    config = Animation3DConfig(
        text="WAVE+FLOAT",
        duration_ms=duration,
        position=position_3d,
        font_size=60,
        font_color=(150, 255, 150)
    )
    animations.append({
        "name": "WaveFloat3D",
        "animation": WaveFloat3D(config),
        "start_time": 85
    })
    
    # 26. Elastic Pop 3D
    config = Animation3DConfig(
        text="ELASTIC POP!",
        duration_ms=duration,
        position=position_3d,
        font_size=65,
        font_color=(255, 200, 100)
    )
    animations.append({
        "name": "ElasticPop3D",
        "animation": ElasticPop3D(config, overshoot_scale=1.8),
        "start_time": 88.5
    })
    
    return animations


def apply_3d_animations_test(input_video: str, output_path: str = "outputs/"):
    """Apply all 3D animations to the video"""
    
    print(f"Loading video: {input_video}")
    
    # Create all animations
    animations = create_3d_animation_showcase()
    total_duration = animations[-1]["start_time"] + 4  # Last animation + buffer
    
    print(f"Total 3D animation sequence duration: {total_duration:.1f} seconds")
    print(f"Total 3D animations to test: {len(animations)}")
    
    # Extract video segment
    temp_segment = f"{output_path}temp_3d_segment.mp4"
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
    temp_output = f"{output_path}3d_animations_temp.mp4"
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
            start_time = anim_data["start_time"]
            duration = anim_data["animation"].config.duration_ms / 1000.0
            
            if (start_time <= current_time < start_time + duration 
                and anim_data not in active_animations):
                active_animations.append(anim_data)
                print(f"  [{current_time:.1f}s] Starting: {anim_data['name']}")
        
        # Remove expired animations
        active_animations = [a for a in active_animations 
                           if current_time < a["start_time"] + a["animation"].config.duration_ms / 1000.0]
        
        # Apply active animations
        for anim_data in active_animations:
            relative_time = current_time - anim_data["start_time"]
            relative_frame = int(relative_time * fps)
            frame = anim_data["animation"].apply_frame(frame, relative_frame, fps)
        
        # Add labels
        cv2.putText(frame, "3D Text Animations Demo", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Time: {current_time:.1f}s", 
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
    final_output = f"{output_path}all_3d_animations_test.mp4"
    print("\nConverting to H.264...")
    convert_cmd = f"ffmpeg -i {temp_output} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {final_output} -y"
    os.system(convert_cmd)
    
    # Clean up temp files
    os.remove(temp_segment)
    os.remove(temp_output)
    
    print(f"\n✅ 3D animation test complete: {final_output}")
    
    # Create summary
    summary_path = f"{output_path}3d_animations_summary.txt"
    with open(summary_path, "w") as f:
        f.write("3D TEXT ANIMATIONS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Duration: {total_duration:.1f} seconds\n")
        f.write(f"Total 3D Animations: {len(animations)}\n\n")
        f.write("Animation Timeline:\n")
        f.write("-" * 60 + "\n")
        
        for i, anim in enumerate(animations, 1):
            f.write(f"{i:2d}. {anim['name']:18s} | Start: {anim['start_time']:5.1f}s\n")
    
    print(f"Summary saved to: {summary_path}")
    
    return final_output


def main():
    # Check if AI_Math1.mp4 exists
    video_path = "uploads/assets/videos/ai_math1.mp4"
    
    if not Path(video_path).exists():
        print(f"Error: {video_path} not found!")
        return
    
    print("=" * 60)
    print("3D TEXT ANIMATIONS TEST SUITE")
    print("=" * 60)
    print("\nThis test demonstrates all 3D text animation effects")
    print("with individual letter control and depth.\n")
    
    # Run the test
    output_file = apply_3d_animations_test(video_path)
    
    print("\n" + "=" * 60)
    print("3D ANIMATION TEST COMPLETE")
    print("=" * 60)
    print(f"\n✨ Output video: {output_file}")
    print("\nThe video showcases 26 different 3D text animations including:")
    print("- Opacity 3D: Fade, Blur, Glow, Dissolve, Materialize")
    print("- Motion 3D: Slide, Float, Bounce, Orbit, Swarm")
    print("- Scale 3D: Zoom, Rotate, Flip, Tumble, Unfold, Explode")
    print("- Progressive 3D: Typewriter, Word Reveal, Wave, Cascade, Build")
    print("- Compound 3D: Combined effects with multiple animations")


if __name__ == "__main__":
    main()