#!/usr/bin/env python3
import cv2
import numpy as np

# Open both videos
cap_orig = cv2.VideoCapture("ai_math1_6sec.mp4")
cap_mask = cv2.VideoCapture("../../uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4")

# Check properties
fps_orig = cap_orig.get(cv2.CAP_PROP_FPS)
fps_mask = cap_mask.get(cv2.CAP_PROP_FPS)
frames_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
frames_mask = int(cap_mask.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video properties:")
print(f"Original: {fps_orig} fps, {frames_orig} frames")
print(f"Mask:     {fps_mask} fps, {frames_mask} frames")

if frames_orig != frames_mask:
    print(f"⚠️ WARNING: Frame count mismatch! Diff: {frames_orig - frames_mask}")

# Check a specific frame
frame_num = 62  # Around 2.5 seconds
cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
cap_mask.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

ret_o, frame_orig = cap_orig.read()
ret_m, frame_mask = cap_mask.read()

if ret_o and ret_m:
    # Try to detect if they're aligned by checking edges
    # Convert to grayscale for edge detection
    gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
    
    # For mask, first identify the person (non-green areas)
    green_color = np.array([154, 254, 119], dtype=np.uint8)
    # Use tolerance for green detection
    diff = np.abs(frame_mask.astype(int) - green_color.astype(int))
    is_green = np.all(diff <= 5, axis=2)  # 5 pixel tolerance
    person_mask = (~is_green).astype(np.uint8) * 255
    
    # Find edges in both
    edges_orig = cv2.Canny(gray_orig, 50, 150)
    edges_mask = cv2.Canny(person_mask, 50, 150)
    
    # Save for visual inspection
    cv2.imwrite("sync_orig_edges.png", edges_orig)
    cv2.imwrite("sync_mask_edges.png", edges_mask)
    cv2.imwrite("sync_person_mask.png", person_mask)
    
    # Overlay to check alignment
    overlay = np.zeros((720, 1280, 3), dtype=np.uint8)
    overlay[:,:,0] = edges_orig  # Blue channel
    overlay[:,:,1] = edges_mask  # Green channel
    cv2.imwrite("sync_overlay.png", overlay)
    
    print("\nSaved alignment check images:")
    print("- sync_orig_edges.png: Original video edges")
    print("- sync_mask_edges.png: Mask edges") 
    print("- sync_person_mask.png: Person mask")
    print("- sync_overlay.png: Overlay (blue=orig, green=mask, cyan=both)")
    print("\nIf edges don't align, videos are out of sync!")

cap_orig.release()
cap_mask.release()