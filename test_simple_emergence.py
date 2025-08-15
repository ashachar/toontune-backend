#!/usr/bin/env python3

"""
Simple test of the EmergenceFromStaticPoint animation API.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.animations import EmergenceFromStaticPoint


def main():
    """Test basic sailor emergence using the API."""
    
    print("=" * 60)
    print("TESTING EMERGENCE API - SIMPLE")
    print("=" * 60)
    
    # Create animation with exact same parameters as before
    animation = EmergenceFromStaticPoint(
        element_path="output/bulk_batch_images_transparent_bg_20250815_145806/Sailor_Salute_in_Cartoon_Style.png.webm",
        background_path="backend/uploads/assets/videos/sea_small_segmented_fixed.mp4",
        position=(220, 215),  # x=220, y=215 (at water line)
        direction=0,  # 0 degrees = upward
        start_frame=0,  # Start from first frame
        animation_start_frame=0,  # Animation starts immediately
        fps=30,
        duration=7.0,
        emergence_speed=1.0,  # 1 pixel per frame
        remove_background=True,
        background_color='0x000000',
        background_similarity=0.15
    )
    
    print("\n📍 Configuration:")
    print(f"   Position: (220, 215) - center of water")
    print(f"   Direction: 0° (upward)")
    print(f"   Speed: 1 pixel/frame")
    print(f"   Duration: 7 seconds at 30fps = 210 frames")
    print()
    
    # Render the animation
    output_path = "output/sailor_api_emergence.mp4"
    success = animation.render(output_path)
    
    if success:
        print(f"\n✅ SUCCESS!")
        print(f"📹 Animation created: {output_path}")
        print("\nThe sailor should emerge from the water:")
        print("  • Frame 0: 1 pixel visible")
        print("  • Frame 30: 31 pixels (hat)")
        print("  • Frame 60: 61 pixels (head)")
        print("  • Frame 120: 121 pixels (upper body)")
        print("  • Frame 210: 211 pixels (3/4 body)")
    else:
        print(f"\n❌ Failed to create animation")


if __name__ == "__main__":
    main()