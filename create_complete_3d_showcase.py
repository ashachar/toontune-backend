"""
Create comprehensive showcase of all 27 3D text animations
"""

import cv2
import numpy as np
import sys
import os

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/opacity_3d')
sys.path.append('utils/animations/3d_animations/scale_3d')
sys.path.append('utils/animations/3d_animations/motion_3d')
sys.path.append('utils/animations/3d_animations/progressive_3d')
sys.path.append('utils/animations/3d_animations/compound_3d')

# Import base config
from base_3d_text_animation import Animation3DConfig

# Import opacity animations
from opacity_3d import (
    Fade3D, BlurFade3D, GlowPulse3D, Dissolve3D, Materialize3D
)

# Import scale/transform animations  
from scale_3d import (
    Zoom3D, Rotate3DAxis, FlipCard3D, Tumble3D, Unfold3D, Explode3D
)

# Import motion animations
from motion_3d import (
    Slide3D, Float3D, Bounce3D, Orbit3D, Swarm3D
)

# Import progressive animations
from progressive_3d import (
    Typewriter3D, WordReveal3D, WaveReveal3D, Cascade3D, Build3D
)

# Import compound animations
from compound_3d import (
    FadeSlide3D, ZoomBlur3D, RotateExplode3D, WaveFloat3D, 
    TypewriterBounce3D, SpiralMaterialize3D, ElasticPop3D
)


def create_animation_showcase():
    """Create a comprehensive showcase of all 3D animations"""
    
    print("Creating Complete 3D Animation Showcase")
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
    total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height} @ {fps:.2f} fps, {total_input_frames} frames")
    
    # Create output video
    output_path = "outputs/complete_3d_showcase.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Define all animations with their configurations
    animations = [
        # Opacity Family (5 animations)
        ("3D FADE WAVE", Fade3D, {"fade_mode": "wave", "depth_fade": False}, 3000),
        ("BLUR FADE 3D", BlurFade3D, {"blur_varies_with_depth": True}, 3000),
        ("GLOW PULSE", GlowPulse3D, {"pulse_count": 3, "glow_cascade": True}, 3000),
        ("DISSOLVE UP", Dissolve3D, {"dissolve_direction": "up", "float_distance": 150}, 3000),
        ("MATERIALIZE", Materialize3D, {"materialize_from": "edges", "particle_spread": 250}, 3000),
        
        # Scale/Transform Family (6 animations)
        ("ZOOM IN 3D", Zoom3D, {"start_scale": 0.1, "end_scale": 1.0}, 3000),
        ("ROTATE AXIS", Rotate3DAxis, {"rotation_axis": "y", "rotations": 2}, 3000),
        ("FLIP CARD", FlipCard3D, {"flip_axis": "x"}, 3000),
        ("TUMBLE 3D", Tumble3D, {"tumble_speed": 2.0}, 3000),
        ("UNFOLD 3D", Unfold3D, {"unfold_origin": "center"}, 3000),
        ("EXPLODE 3D", Explode3D, {"explosion_force": 300, "implode": True}, 3000),
        
        # Motion Family (5 animations)
        ("SLIDE IN", Slide3D, {"slide_direction": "left"}, 3000),
        ("FLOAT 3D", Float3D, {"bob_amplitude": 30, "bob_frequency": 2.0}, 3000),
        ("BOUNCE 3D", Bounce3D, {"bounce_count": 3, "bounce_height": 100}, 3000),
        ("ORBIT 3D", Orbit3D, {"orbit_radius": 150, "orbit_speed": 2.0}, 3000),
        ("SWARM IN", Swarm3D, {"swarm_radius": 300}, 3000),
        
        # Progressive Family (5 animations)
        ("TYPEWRITER", Typewriter3D, {}, 3000),
        ("WORD REVEAL", WordReveal3D, {"reveal_style": "flip"}, 3000),
        ("WAVE REVEAL", WaveReveal3D, {"wave_frequency": 2.0}, 3000),
        ("CASCADE 3D", Cascade3D, {"cascade_speed": 2.0}, 3000),
        ("BUILD 3D", Build3D, {"build_style": "blocks"}, 3000),
        
        # Compound Family (7 animations)
        ("FADE SLIDE", FadeSlide3D, {}, 3000),
        ("ZOOM BLUR", ZoomBlur3D, {"max_zoom": 5.0, "blur_intensity": 20}, 3000),
        ("ROTATE EXPLODE", RotateExplode3D, {"explosion_force": 300}, 3000),
        ("WAVE FLOAT", WaveFloat3D, {"wave_amplitude": 50, "float_height": 30}, 3000),
        ("TYPE BOUNCE", TypewriterBounce3D, {"typing_speed": 50, "bounce_height": 50}, 3000),
        ("SPIRAL MATERIALIZE", SpiralMaterialize3D, {"spiral_radius": 200}, 3000),
        ("ELASTIC POP", ElasticPop3D, {"overshoot_scale": 1.5}, 3000),
    ]
    
    print(f"\nTotal animations to render: {len(animations)}")
    
    frame_count = 0
    animation_index = 0
    current_animation = None
    animation_start_frame = 0
    
    # Process video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video if we need more frames
            if animation_index < len(animations):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                break
        
        # Check if we need to start a new animation
        if animation_index >= len(animations):
            break
            
        if current_animation is None or (frame_count - animation_start_frame) * 1000 / fps >= animations[animation_index][3]:
            
            # Create new animation
            text, anim_class, params, duration = animations[animation_index]
            print(f"\n[{animation_index + 1}/{len(animations)}] Starting: {text}")
            
            config = Animation3DConfig(
                text=text,
                duration_ms=duration,
                position=(640, 360, 0),
                font_size=70,
                font_color=(255, 255, 255),
                depth_color=(100, 100, 100),
                stagger_ms=30
            )
            
            current_animation = anim_class(config, **params)
            animation_start_frame = frame_count
            animation_index += 1
        
        # Apply current animation
        if current_animation:
            animation_frame_num = frame_count - animation_start_frame
            animated_frame = current_animation.apply_frame(frame, animation_frame_num, fps)
            
            # Add animation name overlay
            cv2.putText(animated_frame, f"{animation_index}/{len(animations)}: {animations[animation_index-1][0]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            out.write(animated_frame)
        else:
            out.write(frame)
        
        frame_count += 1
        
        # Progress indicator
        if frame_count % 100 == 0:
            elapsed_seconds = frame_count / fps
            print(f"  Frame {frame_count} ({elapsed_seconds:.1f}s)")
    
    out.release()
    cap.release()
    
    # Convert to H.264
    print(f"\nConverting to H.264...")
    h264_output = output_path.replace('.mp4', '_h264.mp4')
    convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
    os.system(convert_cmd)
    os.remove(output_path)
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"SHOWCASE COMPLETE!")
    print(f"Total animations: {len(animations)}")
    print(f"Total duration: {frame_count / fps:.1f} seconds")
    print(f"Output: {h264_output}")
    print("=" * 60)
    
    return h264_output


if __name__ == "__main__":
    print("COMPLETE 3D TEXT ANIMATION SHOWCASE")
    print("=" * 60)
    print("Featuring all 28 animations across 5 families")
    print("With fixed Fade animation (no flashing!)")
    print()
    
    output = create_animation_showcase()
    
    if output:
        print(f"\n✅ Success! Video created: {output}")
        print("\nAnimation families included:")
        print("  • Opacity (5): Fade, BlurFade, GlowPulse, Dissolve, Materialize")
        print("  • Scale/Transform (6): Zoom, Rotate3DAxis, FlipCard, Tumble, Unfold, Explode")
        print("  • Motion (5): Slide, Float, Bounce, Orbit, Swarm")
        print("  • Progressive (5): Typewriter, WordReveal, WaveReveal, Cascade, Build")
        print("  • Compound (7): FadeSlide, ZoomBlur, RotateExplode, WaveFloat, TypeBounce, SpiralMaterialize, ElasticPop")