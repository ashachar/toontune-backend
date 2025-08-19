#!/usr/bin/env python3
"""
Check Maria's position across multiple frames during 'beginning' display.
"""

import cv2
import numpy as np
from PIL import Image
from rembg import remove
from pathlib import Path
import matplotlib.pyplot as plt

def check_frames_during_word():
    """Check multiple frames during the word 'beginning' display."""
    
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Word timing
    word_start = 10.1
    word_end = 11.479
    
    # Check 5 frames across the word duration
    timestamps = np.linspace(word_start, word_end, 5)
    
    # Position to check
    x, y = 330, 170
    text_width, text_height = 150, 32  # Approximate
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for i, timestamp in enumerate(timestamps):
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Apply rembg
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        nobg_img = remove(pil_image)
        nobg_array = np.array(nobg_img)
        
        # Get alpha channel
        if nobg_array.shape[2] == 4:
            alpha = nobg_array[:, :, 3]
        else:
            alpha = np.ones(nobg_array.shape[:2], dtype=np.uint8) * 255
        
        # Background mask
        background_mask = np.where(alpha < 128, 255, 0).astype(np.uint8)
        
        # Draw rectangle on frame
        frame_viz = frame.copy()
        cv2.rectangle(frame_viz, (x, y), (x + text_width, y + text_height), (0, 255, 0), 2)
        
        # Show frame
        axes[0, i].imshow(cv2.cvtColor(frame_viz, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f't={timestamp:.2f}s')
        axes[0, i].axis('off')
        
        # Show mask with rectangle
        mask_viz = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask_viz, (x, y), (x + text_width, y + text_height), (0, 255, 0), 2)
        axes[1, i].imshow(mask_viz)
        axes[1, i].set_title(f'Mask at {timestamp:.2f}s')
        axes[1, i].axis('off')
        
        # Calculate background ratio in the region
        region = background_mask[y:y+text_height, x:x+text_width]
        bg_ratio = np.mean(region == 255) if region.size > 0 else 0
        axes[1, i].text(0.5, -0.1, f'BG: {bg_ratio:.1%}', 
                        transform=axes[1, i].transAxes,
                        ha='center', fontsize=10)
    
    plt.suptitle("'beginning' position (330, 170) across different timestamps")
    plt.tight_layout()
    
    output_dir = Path("tests/debug_timing")
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "frames_during_word.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_dir}/frames_during_word.png")
    
    cap.release()
    
    # Also check if Maria is moving
    print("\nChecking Maria's position movement...")
    cap = cv2.VideoCapture(video_path)
    
    # Sample more densely
    timestamps = np.linspace(word_start, word_end, 10)
    maria_positions = []
    
    for timestamp in timestamps:
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Apply rembg
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        nobg_img = remove(pil_image)
        nobg_array = np.array(nobg_img)
        
        # Get alpha channel
        if nobg_array.shape[2] == 4:
            alpha = nobg_array[:, :, 3]
        else:
            alpha = np.ones(nobg_array.shape[:2], dtype=np.uint8) * 255
        
        # Find center of mass of foreground
        fg_mask = (alpha > 128).astype(np.uint8)
        M = cv2.moments(fg_mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            maria_positions.append((timestamp, cx, cy))
            print(f"  t={timestamp:.2f}s: Maria center at ({cx}, {cy})")
    
    cap.release()
    
    if maria_positions:
        # Check movement
        first_pos = maria_positions[0]
        last_pos = maria_positions[-1]
        dx = last_pos[1] - first_pos[1]
        dy = last_pos[2] - first_pos[2]
        print(f"\nMaria movement: Δx={dx}px, Δy={dy}px")

if __name__ == "__main__":
    check_frames_during_word()