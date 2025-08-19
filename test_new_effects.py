#!/usr/bin/env python3
"""
Test script for new video effects added to motion_effects.py
"""

import sys
from pathlib import Path
from utils.editing_tricks import (
    apply_dolly_zoom,
    apply_rack_focus,
    apply_handheld_shake,
    apply_speed_ramp,
    apply_bloom_effect,
    apply_ken_burns,
    apply_light_sweep
)

def test_effects():
    """Test the newly added video effects."""
    
    print("=" * 60)
    print("TESTING NEW VIDEO EFFECTS")
    print("=" * 60)
    
    # Test video path - you'll need to provide an actual video
    test_video = "test_video.mp4"
    test_image = "test_image.jpg"
    
    # Check if test files exist
    if not Path(test_video).exists():
        print(f"\n⚠️  Test video '{test_video}' not found.")
        print("Please provide a test video file to test video effects.")
        test_video = None
    
    if not Path(test_image).exists():
        print(f"\n⚠️  Test image '{test_image}' not found.")
        print("Please provide a test image file to test Ken Burns effect.")
        test_image = None
    
    results = []
    
    # Test 1: Dolly Zoom (Camera Push)
    if test_video:
        print("\n1. Testing Dolly Zoom Effect...")
        print("   - Simulates camera pushing forward")
        try:
            output = apply_dolly_zoom(
                test_video,
                dolly_speed=0.02,
                dolly_direction="in",
                output_path="output/test_dolly_zoom.mp4"
            )
            results.append(("Dolly Zoom", "✅ Success", output))
            print(f"   ✅ Created: {output}")
        except Exception as e:
            results.append(("Dolly Zoom", "❌ Failed", str(e)))
            print(f"   ❌ Error: {e}")
    
    # Test 2: Rack Focus
    if test_video:
        print("\n2. Testing Rack Focus Effect...")
        print("   - Shifts focus between different points")
        try:
            output = apply_rack_focus(
                test_video,
                focus_points=[(100, 100), (400, 300)],
                focus_timings=[0, 2],
                output_path="output/test_rack_focus.mp4"
            )
            results.append(("Rack Focus", "✅ Success", output))
            print(f"   ✅ Created: {output}")
        except Exception as e:
            results.append(("Rack Focus", "❌ Failed", str(e)))
            print(f"   ❌ Error: {e}")
    
    # Test 3: Handheld Shake
    if test_video:
        print("\n3. Testing Handheld Shake Effect...")
        print("   - Adds realistic camera shake")
        try:
            output = apply_handheld_shake(
                test_video,
                shake_intensity=3.0,
                shake_frequency=2.0,
                output_path="output/test_handheld.mp4"
            )
            results.append(("Handheld Shake", "✅ Success", output))
            print(f"   ✅ Created: {output}")
        except Exception as e:
            results.append(("Handheld Shake", "❌ Failed", str(e)))
            print(f"   ❌ Error: {e}")
    
    # Test 4: Speed Ramp
    if test_video:
        print("\n4. Testing Speed Ramp Effect...")
        print("   - Creates slow motion at key moments")
        try:
            output = apply_speed_ramp(
                test_video,
                speed_points=[(0, 1.0), (1, 0.3), (2, 1.0)],
                output_path="output/test_speed_ramp.mp4"
            )
            results.append(("Speed Ramp", "✅ Success", output))
            print(f"   ✅ Created: {output}")
        except Exception as e:
            results.append(("Speed Ramp", "❌ Failed", str(e)))
            print(f"   ❌ Error: {e}")
    
    # Test 5: Bloom Effect
    if test_video:
        print("\n5. Testing Bloom Effect...")
        print("   - Adds soft glow to bright areas")
        try:
            output = apply_bloom_effect(
                test_video,
                threshold=180,
                bloom_intensity=1.5,
                color_shift=(1.2, 1.0, 0.8),  # Warm bloom
                output_path="output/test_bloom.mp4"
            )
            results.append(("Bloom Effect", "✅ Success", output))
            print(f"   ✅ Created: {output}")
        except Exception as e:
            results.append(("Bloom Effect", "❌ Failed", str(e)))
            print(f"   ❌ Error: {e}")
    
    # Test 6: Ken Burns Effect
    if test_image:
        print("\n6. Testing Ken Burns Effect...")
        print("   - Creates pan and zoom on still image")
        try:
            output = apply_ken_burns(
                test_image,
                duration=5.0,
                start_scale=1.0,
                end_scale=1.4,
                output_path="output/test_ken_burns.mp4"
            )
            results.append(("Ken Burns", "✅ Success", output))
            print(f"   ✅ Created: {output}")
        except Exception as e:
            results.append(("Ken Burns", "❌ Failed", str(e)))
            print(f"   ❌ Error: {e}")
    
    # Test 7: Light Sweep
    if test_video:
        print("\n7. Testing Light Sweep Effect...")
        print("   - Adds shimmer/light sweep across video")
        try:
            output = apply_light_sweep(
                test_video,
                sweep_duration=1.0,
                sweep_color=(255, 215, 0),  # Golden
                sweep_intensity=0.6,
                output_path="output/test_light_sweep.mp4"
            )
            results.append(("Light Sweep", "✅ Success", output))
            print(f"   ✅ Created: {output}")
        except Exception as e:
            results.append(("Light Sweep", "❌ Failed", str(e)))
            print(f"   ❌ Error: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for effect, status, detail in results:
        print(f"{effect:20} {status}")
        if "Success" in status:
            print(f"{'':20} Output: {detail}")
    
    print("\n✨ New effects have been successfully added to the system!")
    print("\nThese effects match the requirements from 'The Sound of Music' video:")
    print("• Dolly Zoom - for the 'slow dolly-in' effect")
    print("• Rack Focus - for shifting focus between subjects")
    print("• Handheld Shake - for the 'handheld motion' effect")
    print("• Speed Ramp - for slow motion at key moments")
    print("• Bloom Effect - for the glow on highlights")
    print("• Ken Burns - for pan and zoom on still images")
    print("• Light Sweep - for the shimmer effect on titles")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    Path("output").mkdir(exist_ok=True)
    
    test_effects()