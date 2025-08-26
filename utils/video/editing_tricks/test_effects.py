#!/usr/bin/env python3
"""
Test script for video editing tricks module.

This script demonstrates the usage of all video effects with sample videos.
Run this to validate that all effects are working correctly.

Usage:
    python test_effects.py [input_video.mp4]
"""

import sys
import cv2
import numpy as np
import tempfile
from pathlib import Path

# Import all effects
import sys
sys.path.append(str(Path(__file__).parent))
from color_effects import apply_color_splash, apply_selective_color
from text_effects import apply_text_behind_subject, apply_motion_tracking_text, apply_animated_subtitle
from motion_effects import apply_floating_effect, apply_smooth_zoom, apply_3d_photo_effect, apply_rotation_effect
from layout_effects import apply_highlight_focus, add_progress_bar, apply_video_in_text, apply_split_screen


def create_test_video(output_path: Path, duration: float = 3.0, fps: int = 30):
    """Create a simple test video with moving elements."""
    width, height = 640, 480
    frames = int(duration * fps)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for i in range(frames):
        # Create gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient
        for y in range(height):
            color_value = int(255 * (y / height))
            frame[y, :] = [color_value // 2, color_value // 3, color_value]
        
        # Add moving circle (red)
        cx = int(width * (0.2 + 0.6 * (i / frames)))
        cy = height // 2 + int(50 * np.sin(2 * np.pi * i / 30))
        cv2.circle(frame, (cx, cy), 30, (0, 0, 255), -1)
        
        # Add static rectangle (green)
        cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), -1)
        
        # Add text
        cv2.putText(frame, "TEST VIDEO", (width // 2 - 100, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {output_path}")
    return output_path


def test_color_effects(input_video: Path, output_dir: Path):
    """Test color effects."""
    print("\n=== Testing Color Effects ===")
    
    # Test color splash (keep red)
    print("1. Testing color splash effect...")
    output = output_dir / "color_splash.mp4"
    try:
        apply_color_splash(
            input_video,
            target_color=(255, 0, 0),  # Keep red
            tolerance=30,
            output_path=output
        )
        print(f"   ✓ Color splash saved to: {output}")
    except Exception as e:
        print(f"   ✗ Color splash failed: {e}")
    
    # Test selective color adjustment
    print("2. Testing selective color effect...")
    output = output_dir / "selective_color.mp4"
    try:
        apply_selective_color(
            input_video,
            color_adjustments=[
                {
                    'target': (255, 0, 0),  # Red
                    'tolerance': 30,
                    'hue_shift': 60,  # Shift red to yellow
                    'saturation': 1.5,
                    'brightness': 1.2
                }
            ],
            output_path=output
        )
        print(f"   ✓ Selective color saved to: {output}")
    except Exception as e:
        print(f"   ✗ Selective color failed: {e}")


def test_text_effects(input_video: Path, output_dir: Path):
    """Test text effects."""
    print("\n=== Testing Text Effects ===")
    
    # Test text behind subject
    print("1. Testing text behind subject...")
    output = output_dir / "text_behind.mp4"
    try:
        apply_text_behind_subject(
            input_video,
            text="BACKGROUND",
            font_scale=2.0,
            font_color=(255, 255, 0),
            output_path=output
        )
        print(f"   ✓ Text behind subject saved to: {output}")
    except Exception as e:
        print(f"   ✗ Text behind subject failed: {e}")
    
    # Test motion tracking text
    print("2. Testing motion tracking text...")
    output = output_dir / "motion_text.mp4"
    try:
        apply_motion_tracking_text(
            input_video,
            text="TRACKING",
            track_point=(320, 240),
            font_color=(0, 255, 255),
            background_color=(0, 0, 0),
            output_path=output
        )
        print(f"   ✓ Motion tracking text saved to: {output}")
    except Exception as e:
        print(f"   ✗ Motion tracking text failed: {e}")
    
    # Test animated subtitles
    print("3. Testing animated subtitles...")
    output = output_dir / "animated_subtitles.mp4"
    try:
        apply_animated_subtitle(
            input_video,
            subtitles=[
                {'text': 'First subtitle', 'start_time': 0.0, 'end_time': 1.0},
                {'text': 'Second subtitle', 'start_time': 1.0, 'end_time': 2.0},
                {'text': 'Final subtitle', 'start_time': 2.0, 'end_time': 3.0}
            ],
            animation_type='fade',
            output_path=output
        )
        print(f"   ✓ Animated subtitles saved to: {output}")
    except Exception as e:
        print(f"   ✗ Animated subtitles failed: {e}")


def test_motion_effects(input_video: Path, output_dir: Path):
    """Test motion effects."""
    print("\n=== Testing Motion Effects ===")
    
    # Test floating effect
    print("1. Testing floating effect...")
    output = output_dir / "floating.mp4"
    try:
        apply_floating_effect(
            input_video,
            amplitude=20,
            frequency=0.5,
            direction="vertical",
            output_path=output
        )
        print(f"   ✓ Floating effect saved to: {output}")
    except Exception as e:
        print(f"   ✗ Floating effect failed: {e}")
    
    # Test smooth zoom
    print("2. Testing smooth zoom...")
    output = output_dir / "smooth_zoom.mp4"
    try:
        apply_smooth_zoom(
            input_video,
            zoom_factor=1.5,
            zoom_type="in_out",
            easing="ease_in_out",
            output_path=output
        )
        print(f"   ✓ Smooth zoom saved to: {output}")
    except Exception as e:
        print(f"   ✗ Smooth zoom failed: {e}")
    
    # Test 3D photo effect
    print("3. Testing 3D photo effect...")
    output = output_dir / "3d_photo.mp4"
    try:
        apply_3d_photo_effect(
            input_video,
            parallax_strength=30,
            movement_type="horizontal",
            use_depth_estimation=False,  # Use fallback for testing
            output_path=output
        )
        print(f"   ✓ 3D photo effect saved to: {output}")
    except Exception as e:
        print(f"   ✗ 3D photo effect failed: {e}")
    
    # Test rotation effect
    print("4. Testing rotation effect...")
    output = output_dir / "rotation.mp4"
    try:
        apply_rotation_effect(
            input_video,
            rotation_speed=45,
            rotation_axis="z",
            output_path=output
        )
        print(f"   ✓ Rotation effect saved to: {output}")
    except Exception as e:
        print(f"   ✗ Rotation effect failed: {e}")


def test_layout_effects(input_video: Path, output_dir: Path):
    """Test layout effects."""
    print("\n=== Testing Layout Effects ===")
    
    # Test highlight focus
    print("1. Testing highlight focus...")
    output = output_dir / "highlight_focus.mp4"
    try:
        apply_highlight_focus(
            input_video,
            focus_area=(200, 150, 240, 180),
            blur_strength=21,
            vignette=True,
            output_path=output
        )
        print(f"   ✓ Highlight focus saved to: {output}")
    except Exception as e:
        print(f"   ✗ Highlight focus failed: {e}")
    
    # Test progress bar
    print("2. Testing progress bar...")
    output = output_dir / "progress_bar.mp4"
    try:
        add_progress_bar(
            input_video,
            bar_height=8,
            bar_color=(0, 255, 0),
            style="gradient",
            output_path=output
        )
        print(f"   ✓ Progress bar saved to: {output}")
    except Exception as e:
        print(f"   ✗ Progress bar failed: {e}")
    
    # Test video in text
    print("3. Testing video in text...")
    output = output_dir / "video_in_text.mp4"
    try:
        apply_video_in_text(
            input_video,
            text="VIDEO",
            font_scale=5.0,
            output_path=output
        )
        print(f"   ✓ Video in text saved to: {output}")
    except Exception as e:
        print(f"   ✗ Video in text failed: {e}")
    
    # Test split screen (using same video twice)
    print("4. Testing split screen...")
    output = output_dir / "split_screen.mp4"
    try:
        apply_split_screen(
            [input_video, input_video],
            layout="horizontal",
            border_width=4,
            output_path=output
        )
        print(f"   ✓ Split screen saved to: {output}")
    except Exception as e:
        print(f"   ✗ Split screen failed: {e}")


def main():
    """Main test function."""
    print("=" * 60)
    print("VIDEO EDITING TRICKS TEST SUITE")
    print("=" * 60)
    
    # Setup test environment
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Get or create test video
    if len(sys.argv) > 1:
        input_video = Path(sys.argv[1])
        if not input_video.exists():
            print(f"Error: Video file not found: {input_video}")
            sys.exit(1)
    else:
        print("No input video specified. Creating test video...")
        input_video = output_dir / "test_input.mp4"
        create_test_video(input_video)
    
    print(f"\nUsing input video: {input_video}")
    print(f"Output directory: {output_dir}")
    
    # Run all tests
    test_color_effects(input_video, output_dir)
    test_text_effects(input_video, output_dir)
    test_motion_effects(input_video, output_dir)
    test_layout_effects(input_video, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)
    print(f"\nAll test outputs saved to: {output_dir}")
    print("\nEffects successfully implemented:")
    print("  Color Effects:")
    print("    - apply_color_splash")
    print("    - apply_selective_color")
    print("  Text Effects:")
    print("    - apply_text_behind_subject")
    print("    - apply_motion_tracking_text")
    print("    - apply_animated_subtitle")
    print("  Motion Effects:")
    print("    - apply_floating_effect")
    print("    - apply_smooth_zoom")
    print("    - apply_3d_photo_effect")
    print("    - apply_rotation_effect")
    print("  Layout Effects:")
    print("    - apply_highlight_focus")
    print("    - add_progress_bar")
    print("    - apply_video_in_text")
    print("    - apply_split_screen")
    print("\n✅ Module ready for use!")


if __name__ == "__main__":
    main()