"""
Create a comprehensive showcase video of all 3D text animations
Applied to AI_Math1.mp4 video with proper timing and labels
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

from base_3d_text_animation import Animation3DConfig
from opacity_3d import Fade3D, BlurFade3D, GlowPulse3D, Dissolve3D, Materialize3D
from motion_3d import Slide3D, Float3D, Bounce3D, Orbit3D
from scale_3d import Zoom3D, Rotate3DAxis, FlipCard3D, Tumble3D, Unfold3D, Explode3D
from progressive_3d import Typewriter3D, WordReveal3D, WaveReveal3D, Cascade3D, Build3D
from compound_3d import FadeSlide3D, ZoomBlur3D, RotateExplode3D, WaveFloat3D, TypewriterBounce3D, SpiralMaterialize3D, ElasticPop3D


def create_3d_showcase_animations():
    """Create all 3D animations with proper timing for showcase"""
    
    animations = []
    duration_ms = 2500  # 2.5 seconds per animation
    position_3d = (640, 360, 0)  # Center position
    
    # Time offset between animations
    time_offset = 3.0  # seconds
    current_time = 1.0  # Start at 1 second
    
    # ============ OPACITY 3D FAMILY (5 animations) ============
    print("Creating Opacity 3D animations...")
    
    # 1. Fade 3D Wave
    config = Animation3DConfig(
        text="01. 3D FADE WAVE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(255, 255, 255),
        depth_color=(150, 150, 150),
        stagger_ms=60
    )
    animations.append({
        "name": "3D Fade Wave",
        "family": "Opacity 3D",
        "animation": Fade3D(config, fade_mode="wave", depth_fade=True),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 2. Blur Fade 3D
    config = Animation3DConfig(
        text="02. BLUR FADE 3D",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(255, 220, 100),
        stagger_ms=50
    )
    animations.append({
        "name": "3D Blur Fade",
        "family": "Opacity 3D",
        "animation": BlurFade3D(config, start_blur=20, blur_varies_with_depth=True),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 3. Glow Pulse 3D
    config = Animation3DConfig(
        text="03. GLOW PULSE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(100, 255, 255),
        enable_glow=True,
        glow_radius=8
    )
    animations.append({
        "name": "3D Glow Pulse",
        "family": "Opacity 3D",
        "animation": GlowPulse3D(config, pulse_count=2, glow_cascade=True),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 4. Dissolve 3D
    config = Animation3DConfig(
        text="04. DISSOLVE 3D",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(255, 200, 200),
        stagger_ms=40
    )
    animations.append({
        "name": "3D Dissolve",
        "family": "Opacity 3D",
        "animation": Dissolve3D(config, dissolve_direction="spiral", float_distance=80),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 5. Materialize 3D
    config = Animation3DConfig(
        text="05. MATERIALIZE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(200, 255, 200),
        stagger_ms=45
    )
    animations.append({
        "name": "3D Materialize",
        "family": "Opacity 3D",
        "animation": Materialize3D(config, materialize_from="edges", particle_spread=150),
        "start_time": current_time
    })
    current_time += time_offset
    
    # ============ MOTION 3D FAMILY (4 animations) ============
    print("Creating Motion 3D animations...")
    
    # 6. Slide 3D
    config = Animation3DConfig(
        text="06. SLIDE 3D",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(255, 255, 200),
        stagger_ms=50
    )
    animations.append({
        "name": "3D Slide",
        "family": "Motion 3D",
        "animation": Slide3D(config, slide_direction="left", curved_path=True, slide_distance=250),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 7. Float 3D
    config = Animation3DConfig(
        text="07. FLOATING 3D",
        duration_ms=duration_ms + 500,  # Longer for float effect
        position=position_3d,
        font_size=55,
        font_color=(255, 200, 100),
        stagger_ms=60
    )
    animations.append({
        "name": "3D Float",
        "family": "Motion 3D",
        "animation": Float3D(config, float_pattern="wave", bob_amplitude=25),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 8. Bounce 3D
    config = Animation3DConfig(
        text="08. BOUNCE!",
        duration_ms=duration_ms + 500,
        position=position_3d,
        font_size=60,
        font_color=(150, 255, 150),
        stagger_ms=40
    )
    animations.append({
        "name": "3D Bounce",
        "family": "Motion 3D",
        "animation": Bounce3D(config, bounce_count=2, spin_on_bounce=True),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 9. Orbit 3D
    config = Animation3DConfig(
        text="09. ORBIT 3D",
        duration_ms=duration_ms + 500,
        position=position_3d,
        font_size=55,
        font_color=(200, 200, 255),
        stagger_ms=70
    )
    animations.append({
        "name": "3D Orbit",
        "family": "Motion 3D",
        "animation": Orbit3D(config, orbit_axis="y", elliptical=True, orbit_radius=120),
        "start_time": current_time
    })
    current_time += time_offset
    
    # ============ SCALE 3D FAMILY (6 animations) ============
    print("Creating Scale 3D animations...")
    
    # 10. Zoom 3D
    config = Animation3DConfig(
        text="10. ZOOM 3D",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=65,
        font_color=(255, 100, 100),
        stagger_ms=50
    )
    animations.append({
        "name": "3D Zoom",
        "family": "Scale 3D",
        "animation": Zoom3D(config, start_scale=0.1, spiral_zoom=True, pulsate=False),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 11. Rotate 3D
    config = Animation3DConfig(
        text="11. ROTATE 3D",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(100, 100, 255),
        stagger_ms=60
    )
    animations.append({
        "name": "3D Rotate",
        "family": "Scale 3D",
        "animation": Rotate3DAxis(config, rotation_axis="y", wobble=False),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 12. Flip Card 3D
    config = Animation3DConfig(
        text="12. FLIP CARD",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(255, 255, 100),
        depth_color=(180, 180, 70),
        stagger_ms=80
    )
    animations.append({
        "name": "3D Flip Card",
        "family": "Scale 3D",
        "animation": FlipCard3D(config, flip_sequential=True, show_back=True),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 13. Tumble 3D
    config = Animation3DConfig(
        text="13. TUMBLE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(255, 200, 255),
        stagger_ms=50
    )
    animations.append({
        "name": "3D Tumble",
        "family": "Scale 3D",
        "animation": Tumble3D(config, gravity_effect=False, tumble_chaos=0.3),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 14. Unfold 3D
    config = Animation3DConfig(
        text="14. UNFOLD 3D",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(200, 255, 255),
        stagger_ms=60
    )
    animations.append({
        "name": "3D Unfold",
        "family": "Scale 3D",
        "animation": Unfold3D(config, unfold_origin="center", unfold_layers=2),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 15. Explode 3D
    config = Animation3DConfig(
        text="15. EXPLODE!",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=60,
        font_color=(255, 150, 100),
        stagger_ms=30
    )
    animations.append({
        "name": "3D Explode",
        "family": "Scale 3D",
        "animation": Explode3D(config, explosion_force=200, implode=False),
        "start_time": current_time
    })
    current_time += time_offset
    
    # ============ PROGRESSIVE 3D FAMILY (5 animations) ============
    print("Creating Progressive 3D animations...")
    
    # 16. Typewriter 3D
    config = Animation3DConfig(
        text="16. TYPE 3D",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(255, 255, 150)
    )
    animations.append({
        "name": "3D Typewriter",
        "family": "Progressive 3D",
        "animation": Typewriter3D(config, key_bounce=True, key_press_depth=15),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 17. Word Reveal 3D
    config = Animation3DConfig(
        text="17. WORD REVEAL",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(150, 255, 200)
    )
    animations.append({
        "name": "3D Word Reveal",
        "family": "Progressive 3D",
        "animation": WordReveal3D(config, reveal_style="flip"),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 18. Wave Reveal 3D
    config = Animation3DConfig(
        text="18. WAVE REVEAL",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(255, 200, 150),
        stagger_ms=40
    )
    animations.append({
        "name": "3D Wave Reveal",
        "family": "Progressive 3D",
        "animation": WaveReveal3D(config, wave_direction="horizontal", wave_amplitude=60),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 19. Cascade 3D
    config = Animation3DConfig(
        text="19. CASCADE",
        duration_ms=duration_ms + 500,
        position=position_3d,
        font_size=55,
        font_color=(150, 200, 255)
    )
    animations.append({
        "name": "3D Cascade",
        "family": "Progressive 3D",
        "animation": Cascade3D(config, cascade_style="waterfall"),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 20. Build 3D
    config = Animation3DConfig(
        text="20. BUILD 3D",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(255, 150, 200)
    )
    animations.append({
        "name": "3D Build",
        "family": "Progressive 3D",
        "animation": Build3D(config, build_style="blocks"),
        "start_time": current_time
    })
    current_time += time_offset
    
    # ============ COMPOUND 3D FAMILY (7 animations) ============
    print("Creating Compound 3D animations...")
    
    # 21. Fade Slide 3D
    config = Animation3DConfig(
        text="21. FADE+SLIDE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(255, 255, 100),
        stagger_ms=50
    )
    animations.append({
        "name": "3D Fade+Slide",
        "family": "Compound 3D",
        "animation": FadeSlide3D(config, spiral_slide=True, slide_distance=150),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 22. Zoom Blur 3D
    config = Animation3DConfig(
        text="22. ZOOM+BLUR",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(100, 200, 255),
        stagger_ms=60
    )
    animations.append({
        "name": "3D Zoom+Blur",
        "family": "Compound 3D",
        "animation": ZoomBlur3D(config, zoom_origin="center", max_zoom=3.0),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 23. Rotate Explode 3D
    config = Animation3DConfig(
        text="23. ROTATE+EXP",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(255, 100, 150),
        stagger_ms=40
    )
    animations.append({
        "name": "3D Rotate+Explode",
        "family": "Compound 3D",
        "animation": RotateExplode3D(config, implosion_first=True, explosion_force=200),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 24. Wave Float 3D
    config = Animation3DConfig(
        text="24. WAVE+FLOAT",
        duration_ms=duration_ms + 500,
        position=position_3d,
        font_size=55,
        font_color=(150, 255, 150),
        stagger_ms=70
    )
    animations.append({
        "name": "3D Wave+Float",
        "family": "Compound 3D",
        "animation": WaveFloat3D(config, wave_amplitude=40, float_height=80),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 25. Typewriter Bounce
    config = Animation3DConfig(
        text="25. TYPE+BOUNCE",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=55,
        font_color=(255, 200, 100)
    )
    animations.append({
        "name": "3D Type+Bounce",
        "family": "Compound 3D",
        "animation": TypewriterBounce3D(config, bounce_height=40),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 26. Spiral Materialize
    config = Animation3DConfig(
        text="26. SPIRAL MAT",
        duration_ms=duration_ms + 500,
        position=position_3d,
        font_size=55,
        font_color=(200, 150, 255),
        stagger_ms=50
    )
    animations.append({
        "name": "3D Spiral Materialize",
        "family": "Compound 3D",
        "animation": SpiralMaterialize3D(config, spiral_radius=100, spiral_rotations=2),
        "start_time": current_time
    })
    current_time += time_offset
    
    # 27. Elastic Pop
    config = Animation3DConfig(
        text="27. ELASTIC POP!",
        duration_ms=duration_ms,
        position=position_3d,
        font_size=60,
        font_color=(255, 200, 50),
        stagger_ms=60
    )
    animations.append({
        "name": "3D Elastic Pop",
        "family": "Compound 3D",
        "animation": ElasticPop3D(config, overshoot_scale=1.6, pop_rotation=True),
        "start_time": current_time
    })
    
    return animations


def create_3d_showcase_video(input_video: str, output_path: str = "outputs/"):
    """Create the comprehensive 3D showcase video"""
    
    print("=" * 70)
    print("3D TEXT ANIMATIONS SHOWCASE VIDEO GENERATOR")
    print("=" * 70)
    print(f"\nInput video: {input_video}")
    
    # Create all animations
    animations = create_3d_showcase_animations()
    total_duration = animations[-1]["start_time"] + 4  # Last animation + buffer
    
    print(f"\nTotal animations: {len(animations)}")
    print(f"Total duration: {total_duration:.1f} seconds")
    print(f"Animation families: 5 (Opacity, Motion, Scale, Progressive, Compound)")
    
    # Extract video segment
    temp_segment = f"{output_path}temp_3d_showcase.mp4"
    extract_cmd = f"ffmpeg -i {input_video} -t {total_duration} -c:v libx264 -preset fast -crf 18 {temp_segment} -y"
    print(f"\nExtracting {total_duration:.1f} second segment from video...")
    os.system(extract_cmd)
    
    # Open video
    cap = cv2.VideoCapture(temp_segment)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps:.2f} fps")
    print(f"Total frames to process: {total_frames}")
    
    # Create output video writer
    temp_output = f"{output_path}3d_showcase_temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    frame_number = 0
    active_animations = []
    current_family = ""
    
    print("\nProcessing frames...")
    
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
                
                # Update current family
                if anim_data["family"] != current_family:
                    current_family = anim_data["family"]
                    print(f"\n[{current_time:.1f}s] === {current_family} ===")
                
                print(f"  [{current_time:.1f}s] Starting: {anim_data['name']}")
        
        # Remove expired animations
        active_animations = [a for a in active_animations 
                           if current_time < a["start_time"] + a["animation"].config.duration_ms / 1000.0]
        
        # Apply active animations
        for anim_data in active_animations:
            relative_time = current_time - anim_data["start_time"]
            relative_frame = int(relative_time * fps)
            try:
                frame = anim_data["animation"].apply_frame(frame, relative_frame, fps)
            except Exception as e:
                print(f"  Warning: Error in {anim_data['name']}: {str(e)}")
        
        # Add overlays
        # Title bar
        cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 0), -1)
        cv2.putText(frame, "3D TEXT ANIMATIONS SHOWCASE", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Current family indicator
        if current_family:
            family_text = f"Family: {current_family}"
            cv2.putText(frame, family_text, 
                       (width - 300, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 1)
        
        # Bottom info bar
        cv2.rectangle(frame, (0, height - 60), (width, height), (0, 0, 0), -1)
        
        # Progress bar
        progress = current_time / total_duration
        bar_width = int((width - 40) * progress)
        cv2.rectangle(frame, (20, height - 45), (20 + bar_width, height - 35), (100, 255, 100), -1)
        cv2.rectangle(frame, (20, height - 45), (width - 20, height - 35), (100, 100, 100), 1)
        
        # Time and animation info
        time_text = f"Time: {current_time:.1f}s / {total_duration:.1f}s"
        cv2.putText(frame, time_text, 
                   (20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if active_animations:
            anim_text = f"Active: {active_animations[0]['name']}"
            cv2.putText(frame, anim_text, 
                       (250, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        # Animation counter
        completed = sum(1 for a in animations if a["start_time"] + a["animation"].config.duration_ms / 1000.0 < current_time)
        counter_text = f"Animation: {completed + len(active_animations)}/{len(animations)}"
        cv2.putText(frame, counter_text, 
                   (width - 200, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        
        out.write(frame)
        frame_number += 1
        
        # Progress indicator
        if frame_number % int(fps * 5) == 0:
            progress_percent = (frame_number / total_frames) * 100
            print(f"  Progress: {progress_percent:.1f}% ({frame_number}/{total_frames} frames)")
    
    cap.release()
    out.release()
    
    print("\nConverting to H.264...")
    
    # Convert to H.264
    final_output = f"{output_path}3D_Text_Animations_Showcase.mp4"
    convert_cmd = f"ffmpeg -i {temp_output} -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart {final_output} -y"
    os.system(convert_cmd)
    
    # Clean up temp files
    os.remove(temp_segment)
    os.remove(temp_output)
    
    print(f"\n✅ 3D showcase video created: {final_output}")
    
    # Create summary document
    summary_path = f"{output_path}3D_Animations_Showcase_Summary.txt"
    with open(summary_path, "w") as f:
        f.write("3D TEXT ANIMATIONS SHOWCASE - SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Animations: {len(animations)}\n")
        f.write(f"Total Duration: {total_duration:.1f} seconds\n")
        f.write(f"Video Resolution: {width}x{height} @ {fps:.2f} fps\n\n")
        
        f.write("ANIMATION FAMILIES:\n")
        f.write("-" * 70 + "\n\n")
        
        current_family = ""
        for i, anim in enumerate(animations, 1):
            if anim["family"] != current_family:
                current_family = anim["family"]
                f.write(f"\n### {current_family} ###\n\n")
            
            f.write(f"{i:2d}. {anim['name']:25s} | Start: {anim['start_time']:5.1f}s\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("\nKEY FEATURES:\n")
        f.write("- Individual letter control in 3D space\n")
        f.write("- Depth effects and perspective projection\n")
        f.write("- Physics simulation (gravity, bounce, swarm)\n")
        f.write("- Complex motions (orbits, spirals, waves)\n")
        f.write("- Visual effects (blur, glow based on depth)\n")
        f.write("- Stagger timing for sequential effects\n")
    
    print(f"Summary saved to: {summary_path}")
    
    return final_output


def main():
    # Check if AI_Math1.mp4 exists
    video_path = "uploads/assets/videos/ai_math1.mp4"
    
    if not Path(video_path).exists():
        print(f"Error: {video_path} not found!")
        return
    
    # Create the showcase video
    output_file = create_3d_showcase_video(video_path)
    
    print("\n" + "=" * 70)
    print("3D SHOWCASE VIDEO COMPLETE!")
    print("=" * 70)
    print(f"\n✨ Output video: {output_file}")
    print("\nThe video showcases all 27 3D text animations organized by family:")
    print("  • Opacity 3D: 5 animations (fade, blur, glow, dissolve, materialize)")
    print("  • Motion 3D: 4 animations (slide, float, bounce, orbit)")
    print("  • Scale 3D: 6 animations (zoom, rotate, flip, tumble, unfold, explode)")
    print("  • Progressive 3D: 5 animations (typewriter, word, wave, cascade, build)")
    print("  • Compound 3D: 7 animations (combined multi-effect animations)")
    print("\nEach animation demonstrates individual letter control in 3D space")
    print("with depth effects, physics simulation, and complex motion patterns.")


if __name__ == "__main__":
    main()