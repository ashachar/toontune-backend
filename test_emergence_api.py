#!/usr/bin/env python3

"""
Test the EmergenceFromStaticPoint animation API.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.animations import EmergenceFromStaticPoint


def test_basic_emergence():
    """Test basic upward emergence from water."""
    
    print("=" * 60)
    print("TESTING EMERGENCE API")
    print("=" * 60)
    
    # Create animation instance
    animation = EmergenceFromStaticPoint(
        element_path="output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm",
        background_path="backend/uploads/assets/videos/sea_small_segmented_fixed.mp4",
        position=(220, 215),  # Center horizontally, at water line
        direction=0,  # Upward emergence
        start_frame=0,  # Start immediately
        animation_start_frame=0,  # Animate from beginning
        fps=30,
        duration=7.0,
        emergence_speed=1.0,  # 1 pixel per frame
        remove_background=True
    )
    
    # Render the animation
    output_path = "output/test_emergence_api.mp4"
    success = animation.render(output_path)
    
    if success:
        print(f"\n✅ SUCCESS! Animation created: {output_path}")
    else:
        print(f"\n❌ Failed to create animation")
    
    return success


def test_diagonal_emergence():
    """Test diagonal emergence with path movement."""
    
    print("\n" + "=" * 60)
    print("TESTING DIAGONAL EMERGENCE WITH PATH")
    print("=" * 60)
    
    # Define movement path (frame, x, y)
    movement_path = [
        (0, 220, 215),    # Start at water center
        (60, 250, 200),   # Move right and up
        (120, 280, 180),  # Continue diagonal
        (180, 300, 160),  # Further right and up
    ]
    
    animation = EmergenceFromStaticPoint(
        element_path="output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm",
        background_path="backend/uploads/assets/videos/sea_small_segmented_fixed.mp4",
        position=(220, 215),
        direction=45,  # Diagonal (northeast)
        start_frame=15,  # Start after 0.5 seconds
        animation_start_frame=0,
        path=movement_path,  # Add movement
        fps=30,
        duration=7.0,
        emergence_speed=1.2,  # Slightly faster emergence
        remove_background=True
    )
    
    output_path = "output/test_diagonal_emergence.mp4"
    success = animation.render(output_path)
    
    if success:
        print(f"\n✅ SUCCESS! Diagonal animation created: {output_path}")
    else:
        print(f"\n❌ Failed to create diagonal animation")
    
    return success


def test_delayed_animation():
    """Test emergence with delayed animation start."""
    
    print("\n" + "=" * 60)
    print("TESTING DELAYED ANIMATION START")
    print("=" * 60)
    
    animation = EmergenceFromStaticPoint(
        element_path="output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm",
        background_path="backend/uploads/assets/videos/sea_small_segmented_fixed.mp4",
        position=(320, 215),  # Different position
        direction=0,
        start_frame=30,  # Start after 1 second
        animation_start_frame=60,  # Start animating after 2 seconds
        fps=30,
        duration=7.0,
        emergence_speed=2.0,  # Faster emergence (2 pixels per frame)
        remove_background=True
    )
    
    output_path = "output/test_delayed_animation.mp4"
    success = animation.render(output_path)
    
    if success:
        print(f"\n✅ SUCCESS! Delayed animation created: {output_path}")
    else:
        print(f"\n❌ Failed to create delayed animation")
    
    return success


def main():
    """Run all tests."""
    
    print("\n🎬 Testing EmergenceFromStaticPoint Animation API\n")
    
    # Test 1: Basic upward emergence
    test1 = test_basic_emergence()
    
    # Test 2: Diagonal with path
    test2 = test_diagonal_emergence()
    
    # Test 3: Delayed animation
    test3 = test_delayed_animation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Basic emergence: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Diagonal with path: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Delayed animation: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed")


if __name__ == "__main__":
    main()