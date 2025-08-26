#!/usr/bin/env python3
"""Final test of occlusion fix with our test video."""

from utils.animations.apply_3d_text_animation import apply_animation_to_video

result = apply_animation_to_video(
    video_path="test_person_h264.mp4",
    text="Hello World",
    font_size=60,
    position=(640, 350),  # Position text where person will be
    motion_duration=0.75,
    dissolve_duration=1.25,
    output_path="outputs/test_occlusion_final_fixed.mp4",
    final_opacity=0.7,
    supersample=2,  # Low quality for fast testing
    debug=False
)

print(f"✅ Created: {result}")
print("\nWith is_behind=True fix, letters should now be hidden behind the moving person.")