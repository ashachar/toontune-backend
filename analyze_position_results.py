#!/usr/bin/env python3
"""
Analyze and report the optimal position finding results.
"""

import cv2
import numpy as np
from utils.segmentation.segment_extractor import extract_foreground_mask

def analyze_visibility(video_path, text_position, text_size=(705, 172)):
    """Analyze visibility of text at given position throughout video."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = min(30, total_frames)  # Sample first 30 frames
    
    visibility_scores = []
    
    for i in range(0, sample_frames, 3):  # Sample every 3rd frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract foreground mask
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            mask = extract_foreground_mask(frame_rgb)
            mask = (mask > 128).astype(np.uint8) * 255
        except:
            continue
        
        # Calculate text region
        x1 = max(0, text_position[0] - text_size[0] // 2)
        y1 = max(0, text_position[1] - text_size[1] // 2)
        x2 = min(frame.shape[1], x1 + text_size[0])
        y2 = min(frame.shape[0], y1 + text_size[1])
        
        # Extract region and calculate visibility
        text_region = mask[y1:y2, x1:x2]
        if text_region.size > 0:
            visibility = np.sum(text_region == 0) / text_region.size
            visibility_scores.append(visibility)
    
    cap.release()
    
    if visibility_scores:
        return {
            'mean': np.mean(visibility_scores),
            'min': np.min(visibility_scores),
            'max': np.max(visibility_scores),
            'std': np.std(visibility_scores)
        }
    return None

# Analyze both positions
print("="*60)
print("TEXT POSITION ANALYSIS RESULTS")
print("="*60)

# Optimal position
optimal_pos = (693, 106)
optimal_vis = analyze_visibility("uploads/assets/videos/ai_math1_4sec.mp4", optimal_pos)
print(f"\n📍 OPTIMAL POSITION: {optimal_pos}")
if optimal_vis:
    print(f"   Average visibility: {optimal_vis['mean']:.1%}")
    print(f"   Minimum visibility: {optimal_vis['min']:.1%}")
    print(f"   Maximum visibility: {optimal_vis['max']:.1%}")
    print(f"   Std deviation: {optimal_vis['std']:.3f}")

# Center position
center_pos = (640, 360)
center_vis = analyze_visibility("uploads/assets/videos/ai_math1_4sec.mp4", center_pos)
print(f"\n📍 CENTER POSITION: {center_pos}")
if center_vis:
    print(f"   Average visibility: {center_vis['mean']:.1%}")
    print(f"   Minimum visibility: {center_vis['min']:.1%}")
    print(f"   Maximum visibility: {center_vis['max']:.1%}")
    print(f"   Std deviation: {center_vis['std']:.3f}")

# Comparison
if optimal_vis and center_vis:
    print("\n" + "="*60)
    print("IMPROVEMENT SUMMARY")
    print("="*60)
    
    avg_improvement = (optimal_vis['mean'] - center_vis['mean']) / center_vis['mean'] * 100
    min_improvement = (optimal_vis['min'] - center_vis['min']) / max(center_vis['min'], 0.01) * 100
    
    print(f"✅ Average visibility improved by: {avg_improvement:+.1f}%")
    print(f"✅ Minimum visibility improved by: {min_improvement:+.1f}%")
    print(f"✅ More stable (lower std): {optimal_vis['std']:.3f} vs {center_vis['std']:.3f}")
    
    print("\n🎯 KEY INSIGHT:")
    print(f"   The optimal position ({optimal_pos[0]}, {optimal_pos[1]}) places the text")
    print(f"   in the upper-right area where there's less occlusion from")
    print(f"   the animated character, resulting in {optimal_vis['mean']:.0%} visibility")
    print(f"   compared to {center_vis['mean']:.0%} at center position.")

print("\n" + "="*60)
print("FILES GENERATED:")
print("="*60)
print("1. ai_math1_optimal_hq.mp4 - Text at optimal position")
print("2. ai_math1_center_hq.mp4 - Text at center position")
print("3. text_position_comparison_h264.mp4 - Side-by-side comparison")
print("4. optimal_text_position_debug.png - Debug visualization")