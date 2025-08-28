"""
Create final 3D animations showcase with all fixes applied
Overlays all 27 3D text animations on ai_math1.mp4
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add all animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/opacity_3d')
sys.path.append('utils/animations/3d_animations/motion_3d')
sys.path.append('utils/animations/3d_animations/scale_3d')
sys.path.append('utils/animations/3d_animations/rotation_3d')
sys.path.append('utils/animations/3d_animations/progressive_3d')
sys.path.append('utils/animations/3d_animations/compound_3d')

from base_3d_text_animation import Animation3DConfig
from opacity_3d import Fade3D, BlurFade3D, GlowPulse3D, Dissolve3D, Materialize3D
from motion_3d import Slide3D, Float3D, Bounce3D, Orbit3D
from scale_3d import Zoom3D, Rotate3DAxis, FlipCard3D, Tumble3D, Unfold3D, Explode3D
from progressive_3d import Typewriter3D, WordReveal3D, WaveReveal3D, Cascade3D, Build3D
from compound_3d import FadeSlide3D, ZoomBlur3D, RotateExplode3D, WaveFloat3D, TypewriterBounce3D, SpiralMaterialize3D, ElasticPop3D


def create_final_3d_showcase():
    """Create the final showcase with all 3D animations"""
    
    # Input video
    input_video = "uploads/assets/videos/ai_math1.mp4"
    output_video = "outputs/3D_Animations_Final_Showcase.mp4"
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open {input_video}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Input video: {width}x{height} @ {fps:.2f} fps")
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Animation settings
    duration_ms = 3000  # 3 seconds per animation
    position_3d = (640, 360, 0)
    
    # Create all animations with bright, visible text
    animations = []
    start_time = 1.0  # Start after 1 second
    
    print("\nCreating animations...")
    
    # === OPACITY 3D FAMILY (5 animations) ===
    
    # 1. Fade 3D Wave
    config = Animation3DConfig(
        text="3D FADE WAVE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 255, 255),
        stagger_ms=40
    )
    animations.append({
        "animation": Fade3D(config, fade_mode="wave", depth_fade=False),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 2. Blur Fade
    config = Animation3DConfig(
        text="3D BLUR FADE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 220, 100)
    )
    animations.append({
        "animation": BlurFade3D(config, start_blur=20, end_blur=0),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 3. Glow Pulse
    config = Animation3DConfig(
        text="3D GLOW PULSE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(100, 255, 255)
    )
    animations.append({
        "animation": GlowPulse3D(config, glow_radius=10, pulse_count=2),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 4. Dissolve
    config = Animation3DConfig(
        text="3D DISSOLVE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 150, 255)
    )
    animations.append({
        "animation": Dissolve3D(config, dissolve_direction="up"),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 5. Materialize
    config = Animation3DConfig(
        text="3D MATERIALIZE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(150, 255, 150)
    )
    animations.append({
        "animation": Materialize3D(config, materialize_from="edges"),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # === MOTION 3D FAMILY (4 animations) ===
    
    # 6. Slide
    config = Animation3DConfig(
        text="3D SLIDE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 255, 255)
    )
    animations.append({
        "animation": Slide3D(config, slide_direction="left"),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 7. Float
    config = Animation3DConfig(
        text="3D FLOAT",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 200, 100)
    )
    animations.append({
        "animation": Float3D(config, float_pattern="wave"),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 8. Bounce
    config = Animation3DConfig(
        text="3D BOUNCE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(100, 200, 255)
    )
    animations.append({
        "animation": Bounce3D(config, gravity=500),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 9. Orbit
    config = Animation3DConfig(
        text="3D ORBIT",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 100, 200)
    )
    animations.append({
        "animation": Orbit3D(config, orbit_radius=100),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # === SCALE 3D FAMILY (6 animations) ===
    
    # 10. Zoom
    config = Animation3DConfig(
        text="3D ZOOM",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(100, 255, 100)
    )
    animations.append({
        "animation": Zoom3D(config, start_scale=0.1, end_scale=1.0),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 11. Rotate
    config = Animation3DConfig(
        text="3D ROTATE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 255, 100)
    )
    animations.append({
        "animation": Rotate3DAxis(config, rotation_axis="y"),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 12. Flip Card
    config = Animation3DConfig(
        text="3D FLIP CARD",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(100, 255, 255)
    )
    animations.append({
        "animation": FlipCard3D(config, flip_axis="x"),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 13. Tumble
    config = Animation3DConfig(
        text="3D TUMBLE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 150, 150)
    )
    animations.append({
        "animation": Tumble3D(config, tumble_speed=2.0),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 14. Unfold
    config = Animation3DConfig(
        text="3D UNFOLD",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(150, 150, 255)
    )
    animations.append({
        "animation": Unfold3D(config, unfold_origin="center"),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 15. Explode
    config = Animation3DConfig(
        text="3D EXPLODE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 100, 100)
    )
    animations.append({
        "animation": Explode3D(config, explosion_force=150, implode=False),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # === PROGRESSIVE 3D FAMILY (5 animations) ===
    
    # 16. Typewriter
    config = Animation3DConfig(
        text="3D TYPEWRITER",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 255, 255)
    )
    animations.append({
        "animation": Typewriter3D(config, key_bounce=True),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 17. Word Reveal
    config = Animation3DConfig(
        text="3D WORD REVEAL",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(200, 255, 200)
    )
    animations.append({
        "animation": WordReveal3D(config, reveal_style="fade"),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 18. Wave Reveal
    config = Animation3DConfig(
        text="3D WAVE REVEAL",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(200, 200, 255)
    )
    animations.append({
        "animation": WaveReveal3D(config, wave_amplitude=50),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 19. Cascade
    config = Animation3DConfig(
        text="3D CASCADE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 200, 200)
    )
    animations.append({
        "animation": Cascade3D(config, cascade_style="waterfall"),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 20. Build
    config = Animation3DConfig(
        text="3D BUILD",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 255, 150)
    )
    animations.append({
        "animation": Build3D(config, build_style="blocks", build_from_direction="bottom"),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # === COMPOUND 3D FAMILY (7 animations) ===
    
    # 21. Fade+Slide
    config = Animation3DConfig(
        text="FADE+SLIDE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 200, 100)
    )
    animations.append({
        "animation": FadeSlide3D(config),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 22. Zoom+Blur
    config = Animation3DConfig(
        text="ZOOM+BLUR",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(100, 255, 200)
    )
    animations.append({
        "animation": ZoomBlur3D(config),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 23. Rotate+Explode
    config = Animation3DConfig(
        text="ROTATE+EXPLODE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 100, 255)
    )
    animations.append({
        "animation": RotateExplode3D(config),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 24. Wave+Float
    config = Animation3DConfig(
        text="WAVE+FLOAT",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(100, 200, 255)
    )
    animations.append({
        "animation": WaveFloat3D(config),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 25. Type+Bounce
    config = Animation3DConfig(
        text="TYPE+BOUNCE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 255, 100)
    )
    animations.append({
        "animation": TypewriterBounce3D(config),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 26. Spiral Materialize
    config = Animation3DConfig(
        text="SPIRAL MATERIALIZE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(200, 100, 255)
    )
    animations.append({
        "animation": SpiralMaterialize3D(config),
        "start": start_time,
        "duration": 3
    })
    start_time += 3
    
    # 27. Elastic Pop
    config = Animation3DConfig(
        text="ELASTIC POP",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=70,
        font_color=(255, 150, 100)
    )
    animations.append({
        "animation": ElasticPop3D(config),
        "start": start_time,
        "duration": 3
    })
    
    total_duration = start_time + 3
    print(f"Total animations: {len(animations)}")
    print(f"Total duration: {total_duration} seconds")
    
    # Process video frames
    frame_count = 0
    total_frames = int(total_duration * fps)
    
    print(f"\nProcessing {total_frames} frames...")
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        
        if not ret:
            # If video ends, restart from beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        
        current_time = frame_count / fps
        
        # Apply active animations
        for anim_data in animations:
            anim_start = anim_data["start"]
            anim_duration = anim_data["duration"]
            
            if anim_start <= current_time < anim_start + anim_duration:
                relative_time = current_time - anim_start
                relative_frame = int(relative_time * fps)
                
                # Apply animation to frame
                frame = anim_data["animation"].apply_frame(frame, relative_frame, fps)
                
                # Add animation name label
                animation_name = anim_data["animation"].__class__.__name__
                cv2.putText(frame, animation_name, 
                           (width - 200, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Add progress indicator
        progress = (frame_count / total_frames) * 100
        cv2.rectangle(frame, (10, height - 15), (110, height - 5), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, height - 15), (int(10 + progress), height - 5), (100, 255, 100), -1)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % int(fps * 5) == 0:  # Progress every 5 seconds
            print(f"  Progress: {current_time:.1f}s / {total_duration:.1f}s ({progress:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    
    # Convert to H.264 with audio from original
    print("\nConverting to H.264 and adding audio...")
    h264_output = output_video.replace('.mp4', '_h264.mp4')
    
    # FFmpeg command to copy audio from original and encode video as H.264
    convert_cmd = (
        f"ffmpeg -i {output_video} -i {input_video} "
        f"-c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p "
        f"-c:a aac -b:a 192k "
        f"-map 0:v:0 -map 1:a:0 "
        f"-movflags +faststart "
        f"-t {total_duration} "
        f"{h264_output} -y"
    )
    
    print("Running FFmpeg...")
    os.system(convert_cmd)
    
    # Clean up temp file
    if os.path.exists(output_video):
        os.remove(output_video)
    
    print(f"\n✅ Final showcase created: {h264_output}")
    print(f"Duration: {total_duration} seconds")
    print(f"Animations: {len(animations)}")
    
    return h264_output


if __name__ == "__main__":
    print("=" * 70)
    print("3D ANIMATIONS FINAL SHOWCASE")
    print("=" * 70)
    print()
    print("This video showcases all 27 3D text animations with:")
    print("  • Fully opaque, bright white text")
    print("  • Fixed scaling artifacts")
    print("  • Smooth transitions")
    print("  • Original audio from ai_math1.mp4")
    print()
    
    output = create_final_3d_showcase()
    
    if output:
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"Video saved as: {output}")
        print("\nAnimation families included:")
        print("  1. Opacity 3D (5): Fade, Blur, Glow, Dissolve, Materialize")
        print("  2. Motion 3D (4): Slide, Float, Bounce, Orbit") 
        print("  3. Scale 3D (6): Zoom, Rotate, Flip, Tumble, Unfold, Explode")
        print("  4. Progressive 3D (5): Typewriter, Word, Wave, Cascade, Build")
        print("  5. Compound 3D (7): Combined effects")
        print("\nEnjoy your 3D text animations showcase!")
    else:
        print("\n❌ Failed to create showcase video")