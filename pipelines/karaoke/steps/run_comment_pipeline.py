#!/usr/bin/env python
"""Simple wrapper to run the comment pipeline."""

import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.auto_comment.pipeline.main_pipeline import EndToEndCommentPipeline

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_comment_pipeline.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    pipeline = EndToEndCommentPipeline(video_path)
    pipeline.run_pipeline()