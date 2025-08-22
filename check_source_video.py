#!/usr/bin/env python3
"""Check if source video has text baked in."""

import cv2

# Check the source video used in test
cap = cv2.VideoCapture("test_element_3sec.mp4")

# Get a frame from the middle
cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
ret, frame = cap.read()

if ret:
    # Check for yellow text
    yellow_mask = (frame[:,:,1] > 180) & (frame[:,:,2] > 180) & (frame[:,:,0] < 100)
    yellow_count = sum(yellow_mask.flatten())
    
    print(f"Source video 'test_element_3sec.mp4' at frame 90:")
    print(f"  Yellow pixels: {yellow_count}")
    
    if yellow_count > 1000:
        print("  ⚠️  Source video appears to have yellow text baked in!")
        cv2.imwrite("source_video_frame.png", frame)
    else:
        print("  ✓ Source video appears clean")

cap.release()

# Also check at different positions
cap = cv2.VideoCapture("test_element_3sec.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nSource video info:")
print(f"  FPS: {fps}")
print(f"  Total frames: {total_frames}")
print(f"  Duration: {total_frames/fps:.1f} seconds")

# Sample several frames
test_frames = [0, 30, 60, 90, 120, 150]
for f in test_frames:
    if f < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if ret:
            yellow_mask = (frame[:,:,1] > 180) & (frame[:,:,2] > 180) & (frame[:,:,0] < 100)
            yellow_count = sum(yellow_mask.flatten())
            print(f"  Frame {f:3d}: {yellow_count:5d} yellow pixels")

cap.release()