#!/usr/bin/env python3
"""Test that occlusion now works with is_behind=True."""

from utils.animations.apply_3d_text_animation import apply_animation_to_video

# Use the same video from earlier tests
result = apply_animation_to_video(
    video_path="uploads/assets/videos/do_re_mi/scenes/test_pipeline/cartoon_only.mp4",
    text="Hello World",
    font_size=40,
    position=(640, 300),
    motion_duration=0.75,
    dissolve_duration=1.5,
    output_path="outputs/test_occlusion_fixed.mp4",
    debug=False
)

print(f"✅ Created: {result}")
print("\nOcclusion should now work - letters should be hidden behind the person during dissolve.")