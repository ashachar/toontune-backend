"""
Test all text animation families on AI_Math1.mp4
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


def create_test_animations():
    """Create a set of test animations for each family"""
    
    animations = []
    
    # 1. OPACITY FAMILY - Simple fade in
    config1 = AnimationConfig(
        text="Simple Fade In",
        duration_ms=2000,
        position=(50, 100),
        font_size=48,
        font_color=(255, 255, 255),
        easing=EasingType.EASE_IN_OUT
    )
    animations.append(("opacity_fade", SimpleFadeAnimation(config1), 0, 60))
    
    # 2. OPACITY FAMILY - Blur fade
    config2 = AnimationConfig(
        text="Blur to Focus",
        duration_ms=2500,
        position=(50, 200),
        font_size=52,
        font_color=(255, 220, 0),
        easing=EasingType.EASE_OUT
    )
    animations.append(("opacity_blur", BlurFadeAnimation(config2, start_blur=20), 30, 90))
    
    # 3. OPACITY FAMILY - Glow fade
    config3 = AnimationConfig(
        text="Glowing Text",
        duration_ms=3000,
        position=(50, 300),
        font_size=56,
        font_color=(0, 255, 255),
        easing=EasingType.LINEAR
    )
    animations.append(("opacity_glow", GlowFadeAnimation(config3, pulse_count=3), 60, 120))
    
    # 4. MOTION FAMILY - Slide from left
    config4 = AnimationConfig(
        text="Slide From Left",
        duration_ms=1500,
        position=(200, 150),
        font_size=44,
        font_color=(255, 255, 255),
        easing=EasingType.EASE_OUT
    )
    animations.append(("motion_slide", SlideInAnimation(config4, direction="left", distance=200), 0, 45))
    
    # 5. MOTION FAMILY - Float up
    config5 = AnimationConfig(
        text="Floating Up",
        duration_ms=2000,
        position=(100, 250),
        font_size=40,
        font_color=(255, 200, 100),
        easing=EasingType.EASE_IN_OUT
    )
    animations.append(("motion_float", FloatUpAnimation(config5, float_distance=40), 30, 75))
    
    # 6. MOTION FAMILY - Bounce in
    config6 = AnimationConfig(
        text="Bounce Effect!",
        duration_ms=2500,
        position=(150, 350),
        font_size=50,
        font_color=(100, 255, 100),
        easing=EasingType.ELASTIC
    )
    animations.append(("motion_bounce", BounceInAnimation(config6, direction="bottom"), 60, 120))
    
    # 7. SCALE FAMILY - Zoom in
    config7 = AnimationConfig(
        text="ZOOM IN",
        duration_ms=1800,
        position=(250, 200),
        font_size=60,
        font_color=(255, 100, 100),
        easing=EasingType.EASE_OUT
    )
    animations.append(("scale_zoom", ZoomInAnimation(config7, start_scale=2.0), 0, 54))
    
    # 8. SCALE FAMILY - 3D Rotation
    config8 = AnimationConfig(
        text="3D ROTATION",
        duration_ms=2000,
        position=(200, 300),
        font_size=48,
        font_color=(200, 200, 255),
        easing=EasingType.EASE_IN_OUT
    )
    animations.append(("scale_3d", Rotate3DAnimation(config8, rotation_axis="Y"), 45, 90))
    
    # 9. PROGRESSIVE FAMILY - Typewriter
    config9 = AnimationConfig(
        text="Typing animation...",
        duration_ms=3000,
        position=(50, 400),
        font_size=38,
        font_color=(255, 255, 200),
        easing=EasingType.LINEAR
    )
    animations.append(("progressive_type", TypewriterAnimation(config9, cursor_visible=True), 0, 90))
    
    # 10. PROGRESSIVE FAMILY - Word reveal
    config10 = AnimationConfig(
        text="Word by word reveal",
        duration_ms=2500,
        position=(50, 450),
        font_size=40,
        font_color=(200, 255, 200),
        easing=EasingType.EASE_IN
    )
    animations.append(("progressive_word", WordRevealAnimation(config10), 30, 90))
    
    # 11. PROGRESSIVE FAMILY - Line stagger
    config11 = AnimationConfig(
        text="First line appears\\nThen second line\\nFinally third",
        duration_ms=3500,
        position=(50, 500),
        font_size=36,
        font_color=(255, 200, 255),
        easing=EasingType.EASE_OUT
    )
    animations.append(("progressive_line", LineStaggerAnimation(config11), 60, 120))
    
    # 12. COMPOUND FAMILY - Fade + Slide
    config12 = AnimationConfig(
        text="Fade + Slide",
        duration_ms=2000,
        position=(300, 100),
        font_size=46,
        font_color=(255, 255, 100),
        easing=EasingType.EASE_IN_OUT
    )
    animations.append(("compound_fade_slide", FadeSlideAnimation(config12, slide_direction="top"), 0, 60))
    
    # 13. COMPOUND FAMILY - Scale + Blur
    config13 = AnimationConfig(
        text="Scale + Blur",
        duration_ms=2200,
        position=(300, 250),
        font_size=50,
        font_color=(100, 200, 255),
        easing=EasingType.EASE_OUT
    )
    animations.append(("compound_scale_blur", ScaleBlurAnimation(config13), 45, 105))
    
    return animations


def apply_animations_to_video(input_video: str, output_prefix: str = "outputs/"):
    """Apply all test animations to the input video"""
    
    print(f"Loading video: {input_video}")
    
    # Create test animations
    animations = create_test_animations()
    
    # Group animations by time ranges
    animation_groups = {
        "group1": [a for a in animations if a[0].startswith(("opacity", "motion_slide", "scale_zoom", "progressive_type", "compound_fade"))],
        "group2": [a for a in animations if a[0].startswith(("motion_float", "motion_bounce", "scale_3d", "progressive_word", "progressive_line", "compound_scale"))]
    }
    
    # Process each group
    for group_name, group_animations in animation_groups.items():
        if not group_animations:
            continue
            
        print(f"\nProcessing animation group: {group_name}")
        print(f"  Animations in group: {[a[0] for a in group_animations]}")
        
        # Get time range for this group
        start_frame = min(a[2] for a in group_animations)
        end_frame = max(a[3] for a in group_animations)
        duration_seconds = (end_frame - start_frame) / 30  # Assuming 30 fps
        
        # Extract video segment
        segment_path = f"{output_prefix}segment_{group_name}.mp4"
        extract_cmd = f"ffmpeg -i {input_video} -ss {start_frame/30} -t {duration_seconds} -c:v libx264 -preset fast -crf 18 {segment_path} -y"
        os.system(extract_cmd)
        
        # Open video segment
        cap = cv2.VideoCapture(segment_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer
        output_path = f"{output_prefix}animated_{group_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply animations for this frame
            for anim_name, animation, anim_start, anim_end in group_animations:
                global_frame = start_frame + frame_count
                if anim_start <= global_frame < anim_end:
                    # Calculate relative frame for this animation
                    relative_frame = global_frame - anim_start
                    frame = animation.apply_frame(frame, relative_frame, fps)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"    Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        
        # Convert to H.264
        h264_output = f"{output_prefix}animated_{group_name}_h264.mp4"
        convert_cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart {h264_output} -y"
        os.system(convert_cmd)
        
        print(f"  Created: {h264_output}")
    
    # Create a combined demo video
    print("\nCreating combined demo video...")
    combine_cmd = f"ffmpeg -i {output_prefix}animated_group1_h264.mp4 -i {output_prefix}animated_group2_h264.mp4 -filter_complex '[0:v][1:v]concat=n=2:v=1[outv]' -map '[outv]' {output_prefix}all_animations_demo.mp4 -y"
    os.system(combine_cmd)
    
    print(f"\n✅ Created combined demo: {output_prefix}all_animations_demo.mp4")


if __name__ == "__main__":
    # Check if AI_Math1.mp4 exists
    video_path = "uploads/assets/videos/ai_math1.mp4"
    
    if not Path(video_path).exists():
        print(f"Error: {video_path} not found!")
        sys.exit(1)
    
    print("=" * 60)
    print("TESTING ALL TEXT ANIMATION FAMILIES")
    print("=" * 60)
    
    # Apply animations
    apply_animations_to_video(video_path)
    
    print("\n" + "=" * 60)
    print("ANIMATION TEST COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - outputs/animated_group1_h264.mp4")
    print("  - outputs/animated_group2_h264.mp4")
    print("  - outputs/all_animations_demo.mp4")
    print("\nEach video demonstrates different animation families applied to AI_Math1.mp4")