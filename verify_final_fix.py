#!/usr/bin/env python3
"""Final verification of artifact fixes in W dissolve animation."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

video_path = "hello_world_fixed.mp4"
cap = cv2.VideoCapture(video_path)

# Define W region of interest
w_roi = (520, 140, 780, 360)  # x1, y1, x2, y2

# Get background frame (before W starts dissolving)
cap.set(cv2.CAP_PROP_POS_FRAMES, 85)
ret, bg_frame = cap.read()
bg_roi = bg_frame[w_roi[1]:w_roi[3], w_roi[0]:w_roi[2]]

print("=" * 60)
print("FINAL ARTIFACT VERIFICATION REPORT")
print("=" * 60)
print()

# Analyze frames during W dissolve
critical_frames = [96, 100, 110, 120, 130, 140, 150]
artifact_data = []

for frame_idx in critical_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if ret:
        roi = frame[w_roi[1]:w_roi[3], w_roi[0]:w_roi[2]]
        
        # Calculate difference from background
        diff = cv2.absdiff(roi, bg_roi)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Find all non-background pixels
        non_bg_mask = gray_diff > 10
        
        # Identify yellow W pixels (high green and red, low blue in BGR)
        yellow_mask = (roi[:,:,1] > 180) & (roi[:,:,2] > 180) & (roi[:,:,0] < 100)
        
        # Potential artifacts are non-background, non-yellow pixels
        artifact_mask = non_bg_mask & ~yellow_mask
        
        # Focus on the critical areas where artifacts were reported
        # Left side of W (x: 50-70 in ROI)
        left_artifact = artifact_mask[:, 50:70]
        left_count = np.sum(left_artifact)
        
        # Right side of W (x: 210-230 in ROI)  
        right_artifact = artifact_mask[:, 210:230]
        right_count = np.sum(right_artifact)
        
        # Top area above W (y: 40-50 in ROI)
        top_artifact = artifact_mask[40:50, :]
        top_count = np.sum(top_artifact)
        
        total_artifacts = np.sum(artifact_mask)
        
        artifact_data.append({
            'frame': frame_idx,
            'total': total_artifacts,
            'left': left_count,
            'right': right_count,
            'top': top_count
        })
        
        print(f"Frame {frame_idx:3d}: Total artifacts: {total_artifacts:5d} | "
              f"Left: {left_count:4d} | Right: {right_count:4d} | Top: {top_count:4d}")
        
        # Save artifact visualization for frames with significant artifacts
        if total_artifacts > 100:
            vis = roi.copy()
            # Highlight artifacts in red
            vis[artifact_mask] = [0, 0, 255]
            cv2.imwrite(f"final_artifact_frame_{frame_idx:03d}.png", vis)

cap.release()

print()
print("ANALYSIS SUMMARY:")
print("-" * 40)

# Check if artifacts are decreasing over time (as expected during dissolve)
total_artifacts = [d['total'] for d in artifact_data]
left_artifacts = [d['left'] for d in artifact_data]
right_artifacts = [d['right'] for d in artifact_data]
top_artifacts = [d['top'] for d in artifact_data]

print(f"Maximum artifacts detected: {max(total_artifacts)}")
print(f"Average artifacts per frame: {np.mean(total_artifacts):.1f}")
print()

# Check specific issues
print("SPECIFIC ISSUE STATUS:")
print("-" * 40)

# 1. Space character artifact (should be near 0)
space_artifact_fixed = max(top_artifacts) < 50
print(f"1. Space character artifact (top area): {'✓ FIXED' if space_artifact_fixed else '✗ PRESENT'}")
print(f"   Max top artifacts: {max(top_artifacts)}")

# 2. Left side gray line
left_artifact_fixed = max(left_artifacts) < 100
print(f"2. Left side gray line: {'✓ FIXED' if left_artifact_fixed else '✗ PRESENT'}")
print(f"   Max left artifacts: {max(left_artifacts)}")

# 3. Right side gray line
right_artifact_fixed = max(right_artifacts) < 100
print(f"3. Right side gray line: {'✓ FIXED' if right_artifact_fixed else '✗ PRESENT'}")
print(f"   Max right artifacts: {max(right_artifacts)}")

print()
print("FIXES APPLIED:")
print("-" * 40)
print("✓ Persistent dead_mask to prevent letter reappearance")
print("✓ Premultiplied alpha scaling for proper RGBA handling")
print("✓ Space character sprites skipped entirely")
print("✓ Connected components analysis to remove disconnected pixels")
print("✓ Alpha threshold to remove faint edge pixels")

print()
print("CONCLUSION:")
print("-" * 40)

all_fixed = space_artifact_fixed and left_artifact_fixed and right_artifact_fixed

if all_fixed:
    print("✅ ALL ARTIFACTS HAVE BEEN SUCCESSFULLY RESOLVED!")
    print("The word dissolve animation is now working correctly.")
else:
    print("⚠️  Some artifacts may still be present.")
    print("Further investigation needed for:")
    if not space_artifact_fixed:
        print("  - Space character artifact above letters")
    if not left_artifact_fixed:
        print("  - Left side gray line artifact")
    if not right_artifact_fixed:
        print("  - Right side gray line artifact")

# Create a visual graph
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
frames = [d['frame'] for d in artifact_data]
plt.plot(frames, total_artifacts, 'b-', label='Total', linewidth=2)
plt.plot(frames, left_artifacts, 'r--', label='Left side', linewidth=1.5)
plt.plot(frames, right_artifacts, 'g--', label='Right side', linewidth=1.5)
plt.plot(frames, top_artifacts, 'm--', label='Top area', linewidth=1.5)
plt.xlabel('Frame')
plt.ylabel('Artifact Pixel Count')
plt.title('Artifact Detection During W Dissolve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
categories = ['Left\nSide', 'Right\nSide', 'Top\nArea']
max_values = [max(left_artifacts), max(right_artifacts), max(top_artifacts)]
colors = ['green' if v < 100 else 'red' for v in max_values]
bars = plt.bar(categories, max_values, color=colors, alpha=0.7)
plt.axhline(y=100, color='r', linestyle='--', label='Threshold')
plt.ylabel('Maximum Artifact Pixels')
plt.title('Peak Artifact Levels by Region')
plt.legend()

# Add value labels on bars
for bar, val in zip(bars, max_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('final_verification_report.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to final_verification_report.png")