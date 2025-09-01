#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path

# Check both the lossless video and PNG frames
lossless_video = "../../uploads/assets/videos/ai_math1b/raw_video_6sec_mask_lossless.mp4"
frames_dir = Path("../../uploads/assets/videos/ai_math1b/mask_frames")

print("="*60)
print("CHECKING LOSSLESS VIDEO")
print("="*60)

cap = cv2.VideoCapture(lossless_video)
all_colors = set()

# Check first 10 frames
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get unique colors
    unique = np.unique(frame.reshape(-1, 3), axis=0)
    for color in unique:
        all_colors.add(tuple(color))

cap.release()

print(f"Total unique colors in lossless video: {len(all_colors)}")

# Find green-ish colors
green_target = np.array([154, 254, 119])
greens = []
for color in all_colors:
    color_arr = np.array(color)
    diff = np.abs(color_arr.astype(int) - green_target.astype(int))
    if np.max(diff) <= 10:
        greens.append(color_arr)

print(f"Green-ish colors (within ±10): {len(greens)}")
if greens:
    for i, color in enumerate(greens[:10]):
        b, g, r = color
        print(f"  BGR [{b:3}, {g:3}, {r:3}]")

print("\n" + "="*60)
print("CHECKING PNG FRAMES (truly uncompressed)")
print("="*60)

# Check first PNG frame
first_frame = frames_dir / "frame_0001.png"
if first_frame.exists():
    img = cv2.imread(str(first_frame))
    
    # Get unique colors
    unique = np.unique(img.reshape(-1, 3), axis=0)
    print(f"Unique colors in first PNG frame: {len(unique)}")
    
    # Check corners (should be green)
    h, w = img.shape[:2]
    corners = [
        ("Top-left", img[0, 0]),
        ("Top-right", img[0, w-1]),
        ("Bottom-left", img[h-1, 0]),
        ("Bottom-right", img[h-1, w-1])
    ]
    
    print("\nCorner pixels (should be green):")
    for name, pixel in corners:
        b, g, r = pixel
        print(f"  {name:15} BGR: [{b:3}, {g:3}, {r:3}]")
    
    # Find the most common color
    colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    most_common_idx = np.argmax(counts)
    most_common = colors[most_common_idx]
    
    print(f"\nMost common color: BGR {most_common}")
    print(f"Appears in {counts[most_common_idx]} pixels ({100*counts[most_common_idx]/img.size*3:.1f}%)")
    
    # Check if there's variation
    green_colors = []
    for color in colors:
        diff = np.abs(color.astype(int) - most_common.astype(int))
        if np.max(diff) <= 5:  # Within 5 of most common
            green_colors.append(color)
    
    print(f"\nColors within ±5 of most common: {len(green_colors)}")
    if len(green_colors) > 1:
        print("GREEN VARIATION DETECTED even in PNG!")
        for color in green_colors[:10]:
            b, g, r = color
            print(f"  BGR [{b:3}, {g:3}, {r:3}]")
    else:
        print("✅ NO VARIATION - Single green color in PNG!")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if len(greens) > 1 or len(green_colors) > 1:
    print("❌ Green color STILL varies even in lossless/PNG!")
    print("This means the variation comes from Replicate's processing,")
    print("not from our video compression.")
else:
    print("✅ Green color is consistent in lossless version!")
    print("The variation was caused by video compression.")