#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick comparison script showing both ASS and text-behind-head approaches.
"""

import os
from main import create_ass_file, burn_subtitles
from render_text_behind_head import TextBehindHeadRenderer


def generate_both_versions():
    """Generate both ASS subtitle and text-behind-head versions."""
    
    # Common paths
    video_path = "ai_math1_6sec.mp4"
    mask_video_path = "../../uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4"
    transcript_path = "../../uploads/assets/videos/ai_math1/transcript_enriched_partial.json"
    
    print("Generating ASS subtitle version...")
    print("="*50)
    
    # Generate ASS version
    ass_file = "comparison_ass_captions.ass"
    ass_output = "comparison_ass_with_captions.mp4"
    
    create_ass_file(transcript_path, video_path, mask_video_path, ass_file)
    burn_subtitles(video_path, ass_file, ass_output)
    
    print(f"\n✓ ASS version created: {ass_output}")
    
    print("\nGenerating text-behind-head version...")
    print("="*50)
    
    # Generate text-behind-head version
    behind_output = "comparison_text_behind_head.mp4"
    
    renderer = TextBehindHeadRenderer(video_path, mask_video_path, transcript_path)
    renderer.process_video(behind_output)
    
    behind_h264 = behind_output.replace('.mp4', '_h264.mp4')
    print(f"\n✓ Text-behind-head version created: {behind_h264}")
    
    # Compare file sizes
    print("\nFile size comparison:")
    print("="*30)
    
    original_size = os.path.getsize(video_path)
    ass_size = os.path.getsize(ass_output)
    behind_size = os.path.getsize(behind_h264)
    
    print(f"Original video:     {original_size:7,} bytes ({original_size/1024/1024:.1f} MB)")
    print(f"ASS subtitles:      {ass_size:7,} bytes ({ass_size/1024/1024:.1f} MB) [+{ass_size-original_size:,}]")
    print(f"Text behind head:   {behind_size:7,} bytes ({behind_size/1024/1024:.1f} MB) [+{behind_size-original_size:,}]")
    
    print(f"\nOutputs to compare:")
    print(f"  1. ASS subtitles (text over video): {ass_output}")
    print(f"  2. Text behind head (true masking): {behind_h264}")


if __name__ == '__main__':
    generate_both_versions()