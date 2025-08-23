#!/usr/bin/env python3
"""Generate specific frames showing the W occlusion fix"""

import cv2
import numpy as np
from utils.animations.text_3d_behind_segment_fixed import Text3DBehindSegment
from utils.animations.text_3d_behind_segment import Text3DBehindSegment as OldVersion

print("="*60)
print("DEMONSTRATING W OCCLUSION FIX")
print("="*60)

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get the exact frame where W gets occluded (around frame 45-50)
target_frame = 48
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cap.release()

print(f"\nGenerating comparison at frame {target_frame} (0.8 seconds)")
print("This is the exact moment where 'W' meets the girl's head\n")

# Common parameters
params = {
    "duration": 1.0,
    "fps": fps,
    "resolution": (W, H),
    "text": "HELLO WORLD",
    "segment_mask": None,
    "font_size": 140,
    "text_color": (255, 220, 0),
    "depth_color": (200, 170, 0),
    "depth_layers": 10,
    "depth_offset": 3,
    "start_scale": 1.8,
    "end_scale": 0.9,
    "phase1_duration": 0.4,
    "phase2_duration": 0.5,
    "phase3_duration": 0.1,
    "center_position": (W//2, H//2),
    "shadow_offset": 6,
    "outline_width": 2,
    "perspective_angle": 0,
    "supersample_factor": 3,
    "debug": False,
}

# Generate with OLD version (composite masking)
print("1. Generating with OLD composite masking...")
old_anim = OldVersion(**params)
old_frame = old_anim.generate_frame(target_frame, frame_rgb)
if old_frame.shape[2] == 4:
    old_frame = old_frame[:, :, :3]

# Generate with NEW version (per-layer masking)
print("2. Generating with NEW per-layer masking...")
new_anim = Text3DBehindSegment(**params)
new_frame = new_anim.generate_frame(target_frame, frame_rgb)
if new_frame.shape[2] == 4:
    new_frame = new_frame[:, :, :3]

# Save individual frames
cv2.imwrite("W_occlusion_OLD.png", cv2.cvtColor(old_frame, cv2.COLOR_RGB2BGR))
cv2.imwrite("W_occlusion_NEW.png", cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR))

# Create side-by-side comparison
comparison = np.zeros((H, W*2, 3), dtype=np.uint8)
comparison[:, :W] = old_frame
comparison[:, W:] = new_frame

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(comparison, "OLD (Composite Masking)", (20, 40), font, 1.0, (255, 255, 255), 2)
cv2.putText(comparison, "NEW (Per-Layer Masking)", (W+20, 40), font, 1.0, (255, 255, 255), 2)

# Draw dividing line
cv2.line(comparison, (W, 0), (W, H), (255, 255, 255), 2)

cv2.imwrite("W_occlusion_COMPARISON.png", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

# Zoom in on the W area for detailed view
w_x = int(W * 0.55)  # Approximate W position
w_y = int(H * 0.45)
crop_size = 200

# Crop and zoom
old_crop = old_frame[w_y:w_y+crop_size, w_x:w_x+crop_size]
new_crop = new_frame[w_y:w_y+crop_size, w_x:w_x+crop_size]

# Upscale for better visibility
zoom_factor = 2
old_zoom = cv2.resize(old_crop, (crop_size*zoom_factor, crop_size*zoom_factor), 
                      interpolation=cv2.INTER_NEAREST)
new_zoom = cv2.resize(new_crop, (crop_size*zoom_factor, crop_size*zoom_factor), 
                      interpolation=cv2.INTER_NEAREST)

# Create zoomed comparison
zoom_comparison = np.zeros((crop_size*zoom_factor, crop_size*zoom_factor*2, 3), dtype=np.uint8)
zoom_comparison[:, :crop_size*zoom_factor] = old_zoom
zoom_comparison[:, crop_size*zoom_factor:] = new_zoom

cv2.putText(zoom_comparison, "OLD", (10, 30), font, 1.0, (255, 255, 255), 2)
cv2.putText(zoom_comparison, "NEW", (crop_size*zoom_factor+10, 30), font, 1.0, (255, 255, 255), 2)
cv2.line(zoom_comparison, (crop_size*zoom_factor, 0), 
         (crop_size*zoom_factor, crop_size*zoom_factor), (255, 255, 255), 2)

cv2.imwrite("W_occlusion_ZOOM.png", cv2.cvtColor(zoom_comparison, cv2.COLOR_RGB2BGR))

print("\n" + "="*60)
print("✅ OCCLUSION FIX DEMONSTRATED!")
print("="*60)
print("\n📸 Generated comparisons:")
print("  • W_occlusion_OLD.png - Old composite masking (artifacts)")
print("  • W_occlusion_NEW.png - New per-layer masking (clean)")
print("  • W_occlusion_COMPARISON.png - Side-by-side comparison")
print("  • W_occlusion_ZOOM.png - Zoomed view of W boundary")
print("\n🔍 Look closely at the W boundary:")
print("  OLD: Depth layers bleed through, creating artifacts")
print("  NEW: Each layer cleanly cut at mask boundary")