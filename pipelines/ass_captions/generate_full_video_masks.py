#!/usr/bin/env python3
"""
Generate RVM and SAM2 masks for the full ai_math1 video.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent dirs to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def generate_rvm_mask(input_video, output_mask):
    """Generate RVM mask using cached_rvm."""
    # Import with correct path
    import sys
    sys.path.insert(0, '../..')
    from utils.video.background.cached_rvm import CachedRobustVideoMatting
    
    print(f"Generating RVM mask for {input_video}...")
    rvm = CachedRobustVideoMatting()
    
    # Force regeneration by deleting cache if it exists
    cache_path = Path(output_mask)
    if cache_path.exists() and cache_path.stat().st_size < 1000000:  # If less than 1MB, probably wrong
        print(f"Removing incomplete mask: {cache_path}")
        cache_path.unlink()
    
    result = rvm.get_rvm_output(input_video)
    
    # Move to desired location if different
    if result != output_mask:
        Path(result).rename(output_mask)
    
    print(f"RVM mask saved to: {output_mask}")
    return output_mask


def main():
    # Paths
    input_video = "../../uploads/assets/videos/ai_math1.mp4"
    
    # Create output directory
    output_dir = Path("../../uploads/assets/videos/ai_math1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input exists
    if not Path(input_video).exists():
        print(f"Error: Input video not found: {input_video}")
        print("Available videos:")
        videos_dir = Path("../../uploads/assets/videos")
        for f in videos_dir.glob("ai_math*.mp4"):
            print(f"  {f}")
        return 1
    
    # Generate RVM mask
    rvm_mask = output_dir / "ai_math1_full_rvm_mask.mp4"
    
    try:
        generate_rvm_mask(input_video, str(rvm_mask))
        print(f"✅ RVM mask generated: {rvm_mask}")
    except Exception as e:
        print(f"❌ Failed to generate RVM mask: {e}")
        return 1
    
    # Note: SAM2 mask will be generated automatically by the main script
    print("\nNote: SAM2 head mask will be generated automatically when running the main script.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())