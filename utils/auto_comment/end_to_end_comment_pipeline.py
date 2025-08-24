#!/usr/bin/env python3
"""
End-to-end comment pipeline entry point.

This script adds contextual comments to videos at detected silence gaps.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.auto_comment.pipeline import EndToEndCommentPipeline


def main():
    """Main entry point for the comment pipeline."""
    parser = argparse.ArgumentParser(
        description="Add contextual comments to videos at silence gaps"
    )
    parser.add_argument(
        "video",
        help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        help="Output directory (default: same as video)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Validate input video
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return 1
    
    # Run pipeline
    try:
        pipeline = EndToEndCommentPipeline(args.video)
        output = pipeline.run_pipeline()
        
        print(f"\n🎯 Done! Output: {output}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())