#!/usr/bin/env python3
"""
Test the photorealistic burn animation with real fire and volumetric smoke.
Production-quality demonstration.
"""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.letter_3d_burn.photorealistic_burn import PhotorealisticLetterBurn


def create_photorealistic_burn_demo():
    """Create a demonstration of photorealistic burning letters."""
    
    print("="*80)
    print("PHOTOREALISTIC BURN ANIMATION - PRODUCTION QUALITY")
    print("="*80)
    print("\nGenerating realistic fire and volumetric smoke...")
    print("This is computationally intensive for maximum quality.\n")
    
    # Video properties - higher resolution for quality
    width, height = 1920, 1080
    fps = 30
    duration = 5.0  # Longer to show full effect
    
    # Create photorealistic burn animation
    burn = PhotorealisticLetterBurn(
        duration=duration,
        fps=fps,
        resolution=(width, height),
        text="BURN",
        font_size=300,  # Large text for dramatic effect
        text_color=(200, 200, 200),  # Light gray (paper-like)
        
        # Fire properties
        flame_height=200,
        flame_intensity=1.0,
        flame_spread_speed=3.0,
        
        # Timing
        stable_duration=0.5,  # Hold before burning
        ignite_duration=0.5,  # Ignition phase
        burn_duration=2.5,    # Main burning
        burn_stagger=0.4,     # Delay between letters
        
        # Physics
        heat_propagation=0.85,
        char_threshold=0.7,
        
        # Quality
        supersample_factor=2,
        debug=True
    )
    
    # Create gradient background (dark at bottom, slightly lighter at top)
    background = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        # Dark brown/black gradient
        factor = (1.0 - y / height) * 0.3
        background[y, :] = [
            int(20 + 30 * factor),  # B
            int(15 + 25 * factor),  # G  
            int(10 + 20 * factor)   # R
        ]
    
    # Add some subtle noise for texture
    noise = np.random.randint(-5, 5, (height, width, 3))
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Output path
    output_path = "outputs/photorealistic_burn_demo.mp4"
    os.makedirs("outputs", exist_ok=True)
    
    # Video writer - use high quality codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    print(f"Rendering {total_frames} frames at {width}x{height}...")
    print("Features being rendered:")
    print("  • Realistic flames with Perlin noise turbulence")
    print("  • Volumetric smoke with fluid dynamics")
    print("  • Heat propagation through material")
    print("  • Glowing embers and sparks")
    print("  • Progressive charring and burning")
    print("\nProgress:")
    
    for frame_num in range(total_frames):
        # Generate frame
        frame = burn.generate_frame(frame_num, background.copy())
        
        # Add frame counter for reference
        cv2.putText(frame, f"Frame: {frame_num}", (50, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        # Write frame
        writer.write(frame)
        
        # Progress bar
        if frame_num % 5 == 0:
            progress = (frame_num / total_frames) * 100
            bar_length = 40
            filled = int(bar_length * frame_num / total_frames)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r  [{bar}] {progress:.1f}%", end='', flush=True)
    
    print("\n")
    writer.release()
    print(f"✅ Raw video saved to: {output_path}")
    
    # Convert to H.264 with high quality
    h264_output = "outputs/photorealistic_burn_demo_h264.mp4"
    cmd = (f"ffmpeg -i {output_path} "
          f"-c:v libx264 -preset slow -crf 18 "
          f"-pix_fmt yuv420p -movflags +faststart "
          f"-y {h264_output}")
    
    print("Converting to H.264 for compatibility...")
    os.system(cmd + " 2>/dev/null")
    print(f"✅ H.264 version saved to: {h264_output}")
    
    return h264_output


def create_comparison_with_simple():
    """Create comparison between simple and photorealistic burn."""
    
    from utils.animations.letter_3d_burn.burn import Letter3DBurn
    
    print("\nCreating comparison video...")
    
    # Common parameters  
    width, height = 960, 540
    fps = 30
    duration = 4.0
    text = "FIRE"
    font_size = 150
    
    # Create both animations
    simple_burn = Letter3DBurn(
        duration=duration,
        fps=fps,
        resolution=(width, height),
        text=text,
        font_size=font_size,
        text_color=(255, 220, 0),
        initial_position=(width // 2, height // 3),
        supersample_factor=2
    )
    
    photo_burn = PhotorealisticLetterBurn(
        duration=duration,
        fps=fps,
        resolution=(width, height),
        text=text,
        font_size=font_size,
        text_color=(200, 200, 200),
        initial_position=(width // 2, 2 * height // 3),
        flame_height=120,
        supersample_factor=2
    )
    
    # Output
    output_path = "outputs/burn_comparison.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Dark background
    background = np.ones((height, width, 3), dtype=np.uint8) * 30
    
    total_frames = int(duration * fps)
    
    for frame_num in range(total_frames):
        frame = background.copy()
        
        # Labels
        cv2.putText(frame, "SIMPLE BURN", (50, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        cv2.putText(frame, "PHOTOREALISTIC", (50, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        
        # Draw divider
        cv2.line(frame, (0, height // 2), (width, height // 2), (60, 60, 60), 2)
        
        # Generate animations
        frame = simple_burn.generate_frame(frame_num, frame)
        frame = photo_burn.generate_frame(frame_num, frame)
        
        writer.write(frame)
        
        if frame_num % 10 == 0:
            print(f"  Comparison frame {frame_num}/{total_frames}")
    
    writer.release()
    
    # Convert to H.264
    h264_output = "outputs/burn_comparison_h264.mp4"
    cmd = f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart -y {h264_output} 2>/dev/null"
    os.system(cmd)
    
    print(f"✅ Comparison saved to: {h264_output}")
    return h264_output


def create_showcase_frames():
    """Extract key frames to show the burn progression."""
    
    print("\nExtracting showcase frames...")
    
    # Create a simple burn for frame extraction
    burn = PhotorealisticLetterBurn(
        duration=4.0,
        fps=30,
        resolution=(1920, 1080),
        text="INFERNO",
        font_size=250,
        text_color=(220, 220, 220),
        flame_height=180,
        burn_stagger=0.3
    )
    
    # Dark background
    background = np.ones((1080, 1920, 3), dtype=np.uint8) * 25
    
    # Key frames to extract
    key_frames = {
        0: "Initial",
        30: "Ignition Starting",
        60: "Active Burning", 
        90: "Intense Fire",
        110: "Smoke Rising"
    }
    
    showcase_dir = "outputs/burn_showcase"
    os.makedirs(showcase_dir, exist_ok=True)
    
    for frame_num, description in key_frames.items():
        frame = burn.generate_frame(frame_num, background.copy())
        
        # Add description
        cv2.putText(frame, f"{description} (Frame {frame_num})", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
        
        # Save frame
        filename = f"{showcase_dir}/frame_{frame_num:03d}_{description.lower().replace(' ', '_')}.png"
        cv2.imwrite(filename, frame)
        print(f"  Saved: {filename}")
    
    print(f"✅ Showcase frames saved to: {showcase_dir}/")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHOTOREALISTIC BURN EFFECT DEMONSTRATION")
    print("="*80)
    print("\nThis creates production-quality burning text with:")
    print("  • Realistic fire using Perlin noise turbulence")
    print("  • Volumetric smoke with physics simulation")
    print("  • Heat propagation and material charring")
    print("  • Glowing embers and flying sparks")
    print("  • Progressive burn from edges")
    print("="*80 + "\n")
    
    # Create main demo
    demo_video = create_photorealistic_burn_demo()
    
    # Create comparison
    # comparison_video = create_comparison_with_simple()
    
    # Create showcase frames
    create_showcase_frames()
    
    print("\n" + "="*80)
    print("PHOTOREALISTIC BURN COMPLETE!")
    print("="*80)
    print(f"\n📹 Main Demo: {demo_video}")
    # print(f"📹 Comparison: {comparison_video}")
    print(f"🖼️  Showcase Frames: outputs/burn_showcase/")
    print("\n✨ Features Demonstrated:")
    print("  • Letters catch actual fire with realistic flames")
    print("  • Thick volumetric smoke that billows and rises")
    print("  • Material burns progressively from ignition points")
    print("  • Glowing edges and flying embers")
    print("  • Professional production quality")
    print("="*80)