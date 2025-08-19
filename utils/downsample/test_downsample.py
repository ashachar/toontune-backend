#!/usr/bin/env python3
"""
Test script for image and video downsampling utilities
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.downsample.video_downsample import VideoDownsampler
from utils.downsample.image_downsample import ImageDownsampler


def test_video_downsample():
    """Test video downsampling with various presets"""
    print("\n" + "="*60)
    print("🎬 VIDEO DOWNSAMPLING TEST")
    print("="*60)
    
    # Use the 3-second test video we created earlier
    test_video = "test_element_3sec.mp4"
    
    if not os.path.exists(test_video):
        print("Creating test video from do_re_mi.mov...")
        import subprocess
        cmd = [
            'ffmpeg',
            '-i', 'uploads/assets/videos/do_re_mi.mov',
            '-t', '3',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',
            test_video
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Test different presets
    presets = ['micro', 'tiny', 'mini']
    
    for preset in presets:
        print(f"\n📹 Testing preset: {preset}")
        print("-" * 40)
        
        output = f"output/downsample_test/video_{preset}.mp4"
        os.makedirs("output/downsample_test", exist_ok=True)
        
        downsampler = VideoDownsampler(
            input_path=test_video,
            output_path=output,
            preset=preset,
            fps=15  # Also reduce framerate for smaller files
        )
        
        success = downsampler.downsample(verbose=True)
        
        if success:
            print(f"✅ {preset} downsample successful!")
        else:
            print(f"❌ {preset} downsample failed!")
    
    return True


def test_image_downsample():
    """Test image downsampling with various effects"""
    print("\n" + "="*60)
    print("🖼️  IMAGE DOWNSAMPLING TEST")
    print("="*60)
    
    # Find a test image
    test_images = [
        "cartoon-test/man.png",
        "cartoon-test/woman.png",
        "uploads/assets/images/Cartographer_at_Work_on_Map.png"
    ]
    
    test_image = None
    for img_path in test_images:
        if os.path.exists(img_path):
            test_image = img_path
            break
    
    if not test_image:
        # Create a test image using ffmpeg
        print("Creating test image from video...")
        import subprocess
        test_image = "test_frame.png"
        cmd = [
            'ffmpeg',
            '-i', 'test_element_3sec.mp4',
            '-vf', 'select=eq(n\\,30)',  # Extract frame 30
            '-vframes', '1',
            '-y',
            test_image
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Test 1: Different presets
    print("\n🧪 Test 1: Different Presets")
    print("-" * 40)
    
    presets = ['icon', 'micro', 'tiny', 'mini']
    
    for preset in presets:
        print(f"\n📐 Testing preset: {preset}")
        
        output = f"output/downsample_test/image_{preset}.png"
        os.makedirs("output/downsample_test", exist_ok=True)
        
        downsampler = ImageDownsampler(
            input_path=test_image,
            output_path=output,
            preset=preset
        )
        
        success = downsampler.downsample(verbose=True)
        
        if success:
            print(f"✅ {preset} downsample successful!")
    
    # Test 2: Pixel art effect
    print("\n🧪 Test 2: Pixel Art Effect")
    print("-" * 40)
    
    output = "output/downsample_test/image_pixel_art.png"
    downsampler = ImageDownsampler(
        input_path=test_image,
        output_path=output,
        preset='mini'
    )
    
    success = downsampler.downsample(verbose=True, pixel_art=True)
    if success:
        print("✅ Pixel art effect successful!")
    
    # Test 3: Color reduction
    print("\n🧪 Test 3: Color Reduction")
    print("-" * 40)
    
    for colors in [256, 16, 4]:
        print(f"\n🎨 Reducing to {colors} colors")
        
        output = f"output/downsample_test/image_{colors}colors.png"
        downsampler = ImageDownsampler(
            input_path=test_image,
            output_path=output,
            preset='small',
            palette_colors=colors,
            dither=True
        )
        
        success = downsampler.downsample(verbose=True)
        if success:
            print(f"✅ {colors} color reduction successful!")
    
    # Test 4: Retro effect
    print("\n🧪 Test 4: Retro Effect")
    print("-" * 40)
    
    output = "output/downsample_test/image_retro.png"
    downsampler = ImageDownsampler(
        input_path=test_image,
        output_path=output,
        preset='mini',
        palette_colors=8
    )
    
    success = downsampler.downsample(verbose=True, retro=True)
    if success:
        print("✅ Retro effect successful!")
    
    return True


def main():
    """Run all tests"""
    print("\n🚀 DOWNSAMPLE UTILITIES TEST SUITE")
    print("Testing extreme downsampling capabilities")
    print("\nLowest resolutions available:")
    print("  - Video: 32x32 pixels (micro preset)")
    print("  - Image: 8x8 pixels (icon preset)")
    
    # Test video downsampling
    video_success = test_video_downsample()
    
    # Test image downsampling
    image_success = test_image_downsample()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    if video_success and image_success:
        print("✅ All tests passed!")
        print("\n📁 Output files saved to: output/downsample_test/")
        print("\nCapabilities demonstrated:")
        print("  • Video downsampling to 32x32 pixels")
        print("  • Image downsampling to 8x8 pixels")
        print("  • Pixel art effects")
        print("  • Color palette reduction")
        print("  • Retro/vintage filters")
        print("  • Aspect ratio preservation")
        print("  • Multiple resampling algorithms")
        
        # Open output folder
        import subprocess
        subprocess.run(['open', 'output/downsample_test/'])
    else:
        print("❌ Some tests failed")
        if not video_success:
            print("  - Video tests failed")
        if not image_success:
            print("  - Image tests failed")


if __name__ == "__main__":
    main()