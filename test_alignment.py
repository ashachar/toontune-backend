#!/usr/bin/env python3
"""Test alignment of Runway video with original video"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from pipelines.person_animation.main import PersonAnimationPipeline

def test_alignment():
    # Initialize pipeline
    output_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/runway_experiment"
    pipeline = PersonAnimationPipeline(output_dir=output_dir)
    
    # Paths to videos
    original_video = "uploads/assets/runway_experiment/runway_demo_input.mp4"
    runway_video = "uploads/assets/runway_experiment/runway_act_two_output.mp4"
    
    # Run alignment
    aligned_video = pipeline.align_runway_video(original_video, runway_video, timestamp=0.5)
    
    print(f"\nAligned video saved to: {aligned_video}")
    print(f"Check visualization at: {output_dir}/alignment_visualization.png")

if __name__ == "__main__":
    test_alignment()