"""
Main entry point for word-level pipeline with CLI support
"""

import sys

from .main import create_word_level_video


if __name__ == "__main__":
    print("WORD-LEVEL TEXT ANIMATION PIPELINE")
    print("=" * 60)
    print("Golden standard with fog dissolve effect")
    print()
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python -m pipelines.word_level_pipeline <video_path> [duration_seconds]")
        print("Example: python -m pipelines.word_level_pipeline uploads/assets/videos/sample.mp4 10")
        sys.exit(1)
    
    video_path = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    
    output = create_word_level_video(video_path, duration)
    
    if output:
        print(f"\n✨ Success! Video ready: {output}")
        print("\nThis is the golden standard implementation with:")
        print("  - AI-powered transcript enrichment")
        print("  - Face detection and avoidance using OpenCV")
        print("  - Stripe-based layout with smart positioning")
        print("  - Text placed between face and frame edges")
        print("  - Intelligent visibility testing (15% threshold)")
        print("  - Foreground/background text placement")
        print("  - Dynamic visual styling based on importance")
        print("  - Gentle word rise animations")
        print("  - Clean fog dissolve effect")
        print("  - Proper word-level tracking")
        print("  - No reappearing after dissolve")