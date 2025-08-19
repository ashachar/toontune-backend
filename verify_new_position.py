#!/usr/bin/env python3
"""
Verify the new position (650, 270) for 'beginning' stays in background.
"""

import cv2
import numpy as np
from PIL import Image
from rembg import remove
from pathlib import Path
import matplotlib.pyplot as plt

def verify_new_position():
    """Verify new position across multiple frames."""
    
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Word timing
    word_start = 10.1
    word_end = 11.479
    
    # New position vs old position
    new_x, new_y = 650, 270
    old_x, old_y = 330, 170
    text_width, text_height = 150, 32
    
    # Check 5 frames
    timestamps = np.linspace(word_start, word_end, 5)
    
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
        
        # Draw both positions on frame
        frame_viz = frame.copy()
        # Old position in RED
        cv2.rectangle(frame_viz, (old_x, old_y), (old_x + text_width, old_y + text_height), 
                     (0, 0, 255), 2)
        cv2.putText(frame_viz, "OLD", (old_x, old_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # New position in GREEN
        cv2.rectangle(frame_viz, (new_x, new_y), (new_x + text_width, new_y + text_height), 
                     (0, 255, 0), 2)
        cv2.putText(frame_viz, "NEW", (new_x, new_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show frame
        axes[0, i].imshow(cv2.cvtColor(frame_viz, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f't={timestamp:.2f}s')
        axes[0, i].axis('off')
        
        # Show mask with both positions
        mask_viz = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask_viz, (old_x, old_y), (old_x + text_width, old_y + text_height), 
                     (0, 0, 255), 2)
        cv2.rectangle(mask_viz, (new_x, new_y), (new_x + text_width, new_y + text_height), 
                     (0, 255, 0), 2)
        axes[1, i].imshow(mask_viz)
        axes[1, i].set_title(f'Mask at {timestamp:.2f}s')
        axes[1, i].axis('off')
        
        # Calculate background ratios
        old_region = background_mask[old_y:old_y+text_height, old_x:old_x+text_width]
        old_bg_ratio = np.mean(old_region == 255) if old_region.size > 0 else 0
        
        new_region = background_mask[new_y:new_y+text_height, new_x:new_x+text_width]
        new_bg_ratio = np.mean(new_region == 255) if new_region.size > 0 else 0
        
        axes[1, i].text(0.5, -0.1, f'OLD: {old_bg_ratio:.0%} | NEW: {new_bg_ratio:.0%}', 
                       transform=axes[1, i].transAxes,
                       ha='center', fontsize=10,
                       color='green' if new_bg_ratio > old_bg_ratio else 'red')
    
    plt.suptitle("'beginning' position comparison: OLD (330,170) in RED vs NEW (650,270) in GREEN")
    plt.tight_layout()
    
    output_dir = Path("tests/position_verification")
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "position_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_dir}/position_comparison.png")
    
    cap.release()
    
    # Create a final render simulation
    print("\nCreating final render simulation...")
    cap = cv2.VideoCapture(video_path)
    
    # Get frame at middle of word display
    mid_timestamp = (word_start + word_end) / 2
    frame_number = int(mid_timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if ret:
        # Simulate text rendering at both positions
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "beginning"
        
        # Old position render
        old_render = frame.copy()
        cv2.putText(old_render, text, (old_x, old_y + 22), font, 1.0, 
                   (255, 255, 255), 3)
        cv2.putText(old_render, text, (old_x, old_y + 22), font, 1.0, 
                   (0, 0, 0), 2)
        
        # New position render
        new_render = frame.copy()
        cv2.putText(new_render, text, (new_x, new_y + 22), font, 1.0, 
                   (255, 255, 255), 3)
        cv2.putText(new_render, text, (new_x, new_y + 22), font, 1.0, 
                   (0, 0, 0), 2)
        
        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].imshow(cv2.cvtColor(old_render, cv2.COLOR_BGR2RGB))
        axes[0].set_title('OLD Position (330, 170) - ON MARIA\'S FACE')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(new_render, cv2.COLOR_BGR2RGB))
        axes[1].set_title('NEW Position (650, 270) - SAFE IN BACKGROUND')
        axes[1].axis('off')
        
        plt.suptitle(f"'beginning' rendered at t={mid_timestamp:.2f}s")
        plt.tight_layout()
        plt.savefig(output_dir / "render_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Render comparison saved to {output_dir}/render_comparison.png")
    
    cap.release()

if __name__ == "__main__":
    verify_new_position()