#!/usr/bin/env python3
"""
Visual verification of head avoidance in two-position layout
Extracts frames and overlays detected head regions
"""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.word_level_pipeline.masking import ForegroundMaskExtractor
from utils.text_placement.two_position_layout import TwoPositionLayoutManager

def verify_head_avoidance(video_path: str, frame_times: list = [1.0, 3.0, 5.0]):
    """
    Extract frames and visualize head detection zones
    """
    print("=" * 80)
    print("HEAD AVOIDANCE VERIFICATION")
    print("=" * 80)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    mask_extractor = ForegroundMaskExtractor()
    layout_manager = TwoPositionLayoutManager()
    
    for time_sec in frame_times:
        frame_num = int(time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Could not read frame at {time_sec}s")
            continue
            
        print(f"\n⏱️ Frame at {time_sec}s:")
        
        # Extract foreground mask
        mask = mask_extractor.extract_foreground_mask(frame)
        
        # Detect head regions at top and bottom positions
        top_y = layout_manager.default_top_y
        bottom_y = layout_manager.default_bottom_y
        
        print(f"   Checking top position (y={top_y}):")
        top_heads = layout_manager._detect_head_regions([mask], top_y)
        for head_x, head_width in top_heads:
            print(f"      Head detected: x={head_x}, width={head_width}")
            # Draw head zone on frame
            cv2.rectangle(frame, 
                         (head_x - head_width//2, top_y - 50),
                         (head_x + head_width//2, top_y + 50),
                         (0, 0, 255), 2)
            cv2.putText(frame, "HEAD", 
                       (head_x - 20, top_y - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        print(f"   Checking bottom position (y={bottom_y}):")
        bottom_heads = layout_manager._detect_head_regions([mask], bottom_y)
        for head_x, head_width in bottom_heads:
            print(f"      Head detected: x={head_x}, width={head_width}")
            # Draw head zone on frame
            cv2.rectangle(frame, 
                         (head_x - head_width//2, bottom_y - 50),
                         (head_x + head_width//2, bottom_y + 50),
                         (0, 255, 0), 2)
            cv2.putText(frame, "HEAD", 
                       (head_x - 20, bottom_y - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw text position lines
        cv2.line(frame, (0, top_y), (frame.shape[1], top_y), (255, 255, 0), 1)
        cv2.line(frame, (0, bottom_y), (frame.shape[1], bottom_y), (255, 255, 0), 1)
        
        # Save frame for inspection
        output_path = f"outputs/head_verify_{time_sec}s.jpg"
        cv2.imwrite(output_path, frame)
        print(f"   💾 Saved visualization: {output_path}")
    
    cap.release()
    
    print("\n" + "=" * 80)
    print("✅ Verification complete!")
    print("   Check the saved images to see head detection zones")
    print("   Red boxes = heads at top position")
    print("   Green boxes = heads at bottom position")
    print("   Yellow lines = text placement positions")
    print("=" * 80)

if __name__ == "__main__":
    video_path = "outputs/ai_math1_two_position.mp4"
    if os.path.exists(video_path):
        verify_head_avoidance(video_path)
    else:
        print(f"❌ Video not found: {video_path}")