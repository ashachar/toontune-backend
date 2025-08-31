#!/usr/bin/env python3
"""
Debug script for analyzing the "behind text" rendering issue.
This script uses MoviePy to process the video frame-by-frame and sets a breakpoint
at the problematic frame (3.4 seconds) where "be" text should appear behind the head.

TO DEBUG IN CURSOR:
1. Open this file in Cursor
2. Click on line 87 (the breakpoint() line) to set a breakpoint (red dot appears)
3. Press F5 or click "Run and Debug" in the sidebar
4. Select "Python File" when prompted
5. The script will stop at frame 85 (3.4s) where you can inspect all variables

WHAT TO INSPECT IN DEBUGGER:
- `mask_frame`: The green screen mask for this frame
- `bg_mask`: The calculated background mask (1.0 = show text, 0.0 = hide)
- `text_alpha`: The alpha channel of the text being rendered
- `final_alpha`: The result after masking (should be mostly 0 in head area)
- `visible_pixels`: Count of pixels that will be visible
"""

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Configuration
VIDEO_PATH = "outputs/ai_math1_word_level_h264.mp4"
MASK_PATH = "uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4"
PROBLEMATIC_FRAME = 85  # Frame at 3.4 seconds (25 fps * 3.4)
PROBLEMATIC_TIME = 3.4

# Text configuration for "be"
TEXT = "be"
TEXT_X = 576
TEXT_Y = 30
FONT_SIZE = 66  # 44 * 1.5 for behind text
TEXT_COLOR = (50, 50, 100)  # Dark blue for behind text

# Green screen parameters (from rendering.py)
TARGET_GREEN_BGR = np.array([154, 254, 119], dtype=np.float32)
TOLERANCE = 50

def create_text_image(text, font_size, color):
    """Create a text image with alpha channel."""
    padding = 100
    img = Image.new('RGBA', (300, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
    except:
        font = ImageFont.load_default()
    
    # Draw text with full opacity
    draw.text((padding, padding), text, fill=(*color, 255), font=font)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Crop to actual text bounds
    alpha = img_array[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if np.any(rows) and np.any(cols):
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return img_array[y_min:y_max+1, x_min:x_max+1]
    return img_array

def process_mask(mask_frame, text_x, text_y, text_width, text_height):
    """Process the green screen mask to get background mask."""
    # Extract region where text will be
    mask_region = mask_frame[text_y:text_y+text_height, text_x:text_x+text_width]
    
    # Calculate distance from target green
    diff = mask_region.astype(np.float32) - TARGET_GREEN_BGR
    distance = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Create background mask (1.0 = green/background, 0.0 = person/foreground)
    is_green = (distance < TOLERANCE)
    bg_mask = is_green.astype(np.float32)
    
    # Apply dilation to fix edge artifacts (from rendering.py)
    fg_mask = 1.0 - bg_mask
    kernel = np.ones((5, 5), np.uint8)
    fg_mask_binary = (fg_mask * 255).astype(np.uint8)
    fg_mask_binary = cv2.dilate(fg_mask_binary, kernel, iterations=1)
    bg_mask = 1.0 - (fg_mask_binary.astype(np.float32) / 255.0)
    
    return bg_mask, mask_region

def main():
    # Load videos
    video_clip = VideoFileClip(VIDEO_PATH)
    mask_cap = cv2.VideoCapture(MASK_PATH)
    
    # Create text image
    text_img = create_text_image(TEXT, FONT_SIZE, TEXT_COLOR)
    text_height, text_width = text_img.shape[:2]
    text_alpha = text_img[:, :, 3].astype(np.float32) / 255.0
    
    print(f"Text size: {text_width}x{text_height}")
    print(f"Text position: ({TEXT_X}, {TEXT_Y})")
    
    # Process frame by frame
    for frame_num, frame in enumerate(video_clip.iter_frames()):
        if frame_num == PROBLEMATIC_FRAME:
            print(f"\n🔴 BREAKPOINT: Frame {frame_num} (time={PROBLEMATIC_TIME}s)")
            print("The debugger will stop here. Inspect these variables:")
            print("  - mask_frame: The green screen mask")
            print("  - bg_mask: Background mask (1=show, 0=hide)")
            print("  - text_alpha: Original text alpha channel")
            print("  - final_alpha: Alpha after masking")
            print("  - visible_pixels: Number of visible pixels")
            
            # Get corresponding mask frame
            mask_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, mask_frame = mask_cap.read()
            
            if ret:
                # Process the mask
                bg_mask, mask_region = process_mask(mask_frame, TEXT_X, TEXT_Y, text_width, text_height)
                
                # Apply mask to text alpha
                final_alpha = text_alpha * bg_mask[:text_alpha.shape[0], :text_alpha.shape[1]]
                
                # Calculate statistics
                visible_pixels = (final_alpha > 0.1).sum()
                total_pixels = final_alpha.size
                visibility_percent = visible_pixels / total_pixels * 100
                
                # Visual debugging info
                print(f"\n📊 Statistics at frame {frame_num}:")
                print(f"  bg_mask shape: {bg_mask.shape}")
                print(f"  bg_mask mean: {bg_mask.mean():.3f}")
                print(f"  Visible pixels: {visible_pixels}/{total_pixels} ({visibility_percent:.1f}%)")
                print(f"  Text should be {100-visibility_percent:.1f}% hidden")
                
                # SET BREAKPOINT HERE - YOU CAN INSPECT ALL VARIABLES
                breakpoint()  # <-- Debugger will stop here!
                
                # Optional: Show visualization (uncomment if you want plots)
                # fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                # axes[0, 0].imshow(frame)
                # axes[0, 0].set_title("Original Frame")
                # axes[0, 1].imshow(mask_frame)
                # axes[0, 1].set_title("Green Screen Mask")
                # axes[0, 2].imshow(bg_mask, cmap='gray')
                # axes[0, 2].set_title("Background Mask (1=show, 0=hide)")
                # axes[1, 0].imshow(text_alpha, cmap='gray')
                # axes[1, 0].set_title("Text Alpha")
                # axes[1, 1].imshow(final_alpha, cmap='gray')
                # axes[1, 1].set_title("Final Alpha (After Masking)")
                # axes[1, 2].imshow(mask_region)
                # axes[1, 2].set_title("Mask Region at Text Position")
                # plt.tight_layout()
                # plt.show()
                
        # Optional: Process more frames
        if frame_num > PROBLEMATIC_FRAME + 5:
            break
    
    # Cleanup
    mask_cap.release()
    video_clip.close()
    
    print("\nDebug session complete!")

if __name__ == "__main__":
    main()