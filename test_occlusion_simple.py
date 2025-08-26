#!/usr/bin/env python3
"""Simple test to verify occlusion is working in dissolve."""

from utils.animations.apply_3d_text_animation import apply_animation_to_video

# Create a video with dissolve starting immediately
# This should show occlusion from the start if it's working
result = apply_animation_to_video(
    video_path="uploads/assets/videos/do_re_mi/scenes/test_pipeline/cartoon_only.mp4",
    text="Hello",
    font_size=50,
    position=(640, 340),  # Position text right on the person
    motion_duration=0.1,  # Minimal motion, go straight to dissolve
    dissolve_duration=2.0,
    debug=False
)

print(f"✅ Created: {result}")
print("\nIf occlusion is working, text should be partially hidden behind the person.")
print("If not working, text will be fully visible on top of the person.")