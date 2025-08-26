#!/usr/bin/env python3
"""Test if resize is causing the stale mask bug."""

import cv2
import numpy as np

# Create a test mask
height, width = 360, 640
mask = np.zeros((height, width), dtype=np.uint8)

# Put a rectangle at x=200
mask[100:250, 200:300] = 255

print("Original mask:")
print(f"  Shape: {mask.shape} (height={height}, width={width})")
y_coords, x_coords = np.where(mask > 128)
if len(x_coords) > 0:
    print(f"  Mask at: x=[{x_coords.min()}-{x_coords.max()}]")

# Test the resize logic from dissolve
resolution = (width, height)  # This is how it's stored in dissolve

print(f"\nResolution tuple: {resolution}")
print(f"mask.shape[:2]: {mask.shape[:2]}")
print(f"(resolution[1], resolution[0]): {(resolution[1], resolution[0])}")

# Check if shapes match (this is the condition in the code)
if mask.shape[:2] != (resolution[1], resolution[0]):
    print("\n⚠️ Shapes don't match, will resize!")
    print(f"  Resizing to: {resolution}")
    
    # This is what happens in the code
    resized = cv2.resize(mask, resolution, interpolation=cv2.INTER_LINEAR)
    print(f"  Resized shape: {resized.shape}")
    
    # Check where the mask is after resize
    y_coords, x_coords = np.where(resized > 128)
    if len(x_coords) > 0:
        print(f"  Mask after resize at: x=[{x_coords.min()}-{x_coords.max()}]")
else:
    print("\n✅ Shapes match, no resize needed")

# Now test with a moved mask
print("\n" + "="*60)
print("Testing with moved mask:")

mask2 = np.zeros((height, width), dtype=np.uint8)
mask2[100:250, 400:500] = 255  # Rectangle at x=400

print("Mask 2 (moved):")
y_coords, x_coords = np.where(mask2 > 128)
if len(x_coords) > 0:
    print(f"  Mask at: x=[{x_coords.min()}-{x_coords.max()}]")

# Check resize
if mask2.shape[:2] != (resolution[1], resolution[0]):
    resized2 = cv2.resize(mask2, resolution, interpolation=cv2.INTER_LINEAR)
    y_coords, x_coords = np.where(resized2 > 128)
    if len(x_coords) > 0:
        print(f"  After resize: x=[{x_coords.min()}-{x_coords.max()}]")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("The resize check is comparing:")
print(f"  mask.shape[:2] = (height, width) = ({height}, {width})")
print(f"  (resolution[1], resolution[0]) = ({resolution[1]}, {resolution[0]})")
print(f"\nThese are equal, so no resize happens.")
print("The resize is NOT the bug.")