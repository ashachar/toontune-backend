#!/usr/bin/env python3
"""
Debug why the algorithm thinks Maria's face is background.
"""

import cv2
import numpy as np
from PIL import Image
from rembg import remove
from pathlib import Path

def debug_mask_at_position():
    """Debug the exact mask values at the position."""
    
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Go to timestamp 10.1s (when "beginning" appears)
    timestamp = 10.1
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        print(f"ERROR: Could not read frame at {timestamp}s")
        return
    
    # Get background mask using rembg
    print("Applying rembg...")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    nobg_img = remove(pil_image)
    nobg_array = np.array(nobg_img)
    
    # Get alpha channel
    if nobg_array.shape[2] == 4:
        alpha = nobg_array[:, :, 3]
    else:
        alpha = np.ones(nobg_array.shape[:2], dtype=np.uint8) * 255
    
    # Background mask: where alpha < 128, it's background (255), else foreground (0)
    background_mask = np.where(alpha < 128, 255, 0).astype(np.uint8)
    
    # Position and text metrics
    x, y = 330, 170
    text = "beginning"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    print(f"\nText box (FFmpeg convention): ({x}, {y}) to ({x + text_width}, {y + text_height + baseline})")
    print(f"  Width: {text_width}px, Height: {text_height}px, Baseline: {baseline}px")
    
    # Extract the exact region
    y1 = y
    y2 = y + text_height + baseline
    x1 = x
    x2 = x + text_width
    
    print(f"\nExtracting mask region: [{y1}:{y2}, {x1}:{x2}]")
    
    if y1 >= 0 and y2 <= frame.shape[0] and x1 >= 0 and x2 <= frame.shape[1]:
        region = background_mask[y1:y2, x1:x2]
        
        print(f"Region shape: {region.shape}")
        print(f"Unique values in region: {np.unique(region)}")
        print(f"Value counts: 255 (background): {np.sum(region == 255)}, 0 (foreground): {np.sum(region == 0)}")
        
        bg_ratio = np.mean(region == 255)
        print(f"Background ratio: {bg_ratio:.1%}")
        
        # Save the region for inspection
        output_dir = Path("tests/debug_mask_region")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save the mask region
        cv2.imwrite(str(output_dir / "mask_region.png"), region)
        
        # Save the corresponding frame region
        frame_region = frame[y1:y2, x1:x2]
        cv2.imwrite(str(output_dir / "frame_region.png"), frame_region)
        
        # Save full mask with rectangle
        mask_viz = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(mask_viz, f"({x1}, {y1})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(output_dir / "full_mask_with_region.png"), mask_viz)
        
        # Save alpha channel for debugging
        cv2.imwrite(str(output_dir / "alpha_channel.png"), alpha)
        
        print(f"\nDebug images saved to {output_dir}")
        
        # Check a few specific pixels
        print("\nChecking specific pixels in the region:")
        for i in range(0, min(5, region.shape[0])):
            for j in range(0, min(5, region.shape[1])):
                pixel_y = y1 + i
                pixel_x = x1 + j
                mask_val = region[i, j]
                alpha_val = alpha[pixel_y, pixel_x]
                print(f"  Pixel ({pixel_x}, {pixel_y}): mask={mask_val}, alpha={alpha_val}")
    else:
        print(f"Region out of bounds!")
    
    cap.release()

if __name__ == "__main__":
    debug_mask_at_position()