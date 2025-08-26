#!/usr/bin/env python3
"""
Debug the specific issue where 'd' in "Hello World" shows stale mask occlusion.
Find the exact frame where the foreground has moved but 'd' still shows old occlusion.
"""

import cv2
import numpy as np
import sys
import os

print("="*80)
print("🔍 DEBUGGING 'd' STALE MASK ISSUE")
print("="*80)

# Video to analyze
VIDEO = "outputs/hello_world_2m20s_FIXED_compatible.mp4"
if not os.path.exists(VIDEO):
    VIDEO = "outputs/hello_world_2m20s_both_animations_compatible.mp4"

print(f"\n📹 Analyzing: {VIDEO}")

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"❌ Cannot open {VIDEO}")
    sys.exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"  FPS: {fps}, Total frames: {total_frames}")
print(f"  Duration: {total_frames/fps:.2f}s")

# Based on timing:
# - Motion: 0-20 frames (0-0.8s)
# - Hold: 20-32 frames (0.8-1.3s)  
# - Dissolve: 32+ frames (1.3s+)
# The 'd' is the last letter, so it should dissolve last

print("\n🎯 Looking for frame where 'd' shows stale occlusion...")
print("  'd' should start dissolving around frame 50-60")
print("  Looking for frames where foreground moves but 'd' occlusion doesn't update")

# Import segmentation module to extract masks
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils.video.segmentation.segment_extractor import extract_foreground_mask
except:
    print("⚠️ Cannot import segment_extractor, will analyze visually")
    extract_foreground_mask = None

# Analyze frames where 'd' is visible and potentially affected
critical_frames = range(45, 70)  # Frames where 'd' is likely dissolving

prev_mask = None
mask_changes = []

for frame_idx in critical_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break
    
    # Extract mask if possible
    if extract_foreground_mask:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = extract_foreground_mask(frame_rgb)
        
        # Find mask center and bounds
        mask_pixels = np.where(mask > 128)
        if len(mask_pixels[0]) > 0:
            mask_top = mask_pixels[0].min()
            mask_bottom = mask_pixels[0].max()
            mask_left = mask_pixels[1].min()
            mask_right = mask_pixels[1].max()
            mask_center_y = (mask_top + mask_bottom) // 2
            
            # Compare with previous frame
            if prev_mask is not None:
                prev_pixels = np.where(prev_mask > 128)
                if len(prev_pixels[0]) > 0:
                    prev_center_y = (prev_pixels[0].min() + prev_pixels[0].max()) // 2
                    movement = mask_center_y - prev_center_y
                    
                    if abs(movement) > 5:  # Significant movement
                        print(f"\n🔴 Frame {frame_idx}: Foreground moved {movement:+d} pixels vertically")
                        print(f"   Mask bounds: y=[{mask_top}-{mask_bottom}], center_y={mask_center_y}")
                        
                        # Check 'd' region (approximate - right side of "World")
                        # 'd' should be around x=850-900, y=320-380
                        d_region_x = (850, 900)
                        d_region_y = (320, 380)
                        
                        # Check if mask overlaps with 'd' region
                        d_mask_region = mask[d_region_y[0]:d_region_y[1], 
                                            d_region_x[0]:d_region_x[1]]
                        d_occluded_pixels = np.sum(d_mask_region > 128)
                        
                        print(f"   'd' region ({d_region_x[0]}-{d_region_x[1]}, "
                              f"{d_region_y[0]}-{d_region_y[1]}): "
                              f"{d_occluded_pixels} occluded pixels")
                        
                        if d_occluded_pixels > 0:
                            print(f"   ⚠️ 'd' is being occluded!")
                        
                        # Save visualization
                        vis_frame = frame.copy()
                        # Draw mask bounds
                        cv2.rectangle(vis_frame, (mask_left, mask_top), 
                                    (mask_right, mask_bottom), (0, 255, 0), 2)
                        # Draw 'd' region
                        cv2.rectangle(vis_frame, d_region_x, d_region_y, 
                                    (0, 0, 255), 2)
                        cv2.putText(vis_frame, f"Frame {frame_idx}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(vis_frame, f"Mask moved {movement:+d}px", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        cv2.imwrite(f"outputs/debug_frame_{frame_idx}.png", vis_frame)
                        print(f"   💾 Saved debug image: outputs/debug_frame_{frame_idx}.png")
                        
                        mask_changes.append((frame_idx, movement, d_occluded_pixels))
            
            prev_mask = mask.copy()
    else:
        # Visual analysis without mask extraction
        time_s = frame_idx / fps
        if frame_idx % 5 == 0:
            print(f"Frame {frame_idx} (t={time_s:.2f}s): Checking 'd' visibility...")
            
            # Save frame for manual inspection
            cv2.imwrite(f"outputs/frame_{frame_idx}.png", frame)

cap.release()

print("\n" + "="*80)
print("📊 ANALYSIS RESULTS")
print("="*80)

if mask_changes:
    print("\n🔍 Frames with significant mask movement:")
    for frame, movement, d_pixels in mask_changes:
        time_s = frame / fps
        print(f"  Frame {frame} (t={time_s:.2f}s): "
              f"Mask moved {movement:+d}px, 'd' occluded: {d_pixels} pixels")
    
    # Find problematic frame
    print("\n🎯 Likely problematic frames:")
    for i in range(1, len(mask_changes)):
        prev_frame, prev_move, prev_d = mask_changes[i-1]
        curr_frame, curr_move, curr_d = mask_changes[i]
        
        # If mask moved but 'd' occlusion didn't change appropriately
        if abs(curr_move) > 5 and abs(curr_d - prev_d) < 10:
            print(f"  ⚠️ Frame {curr_frame}: Mask moved {curr_move}px but "
                  f"'d' occlusion barely changed ({prev_d} → {curr_d})")
else:
    print("\n📌 Key frames to check manually:")
    print("  Frame 50-55: 'd' should be visible and dissolving")
    print("  Frame 55-60: Foreground likely moves")
    print("  Frame 60-65: Check if 'd' occlusion updates")

print("\n💡 Next steps:")
print("  1. Check the saved debug images")
print("  2. Look for frames where green box (mask) moves but red box ('d') occlusion doesn't update")
print("  3. Focus on frames 50-65 where 'd' is dissolving")
print("="*80)