#!/usr/bin/env python3
"""
Show all 17 masks individually from the full resolution SAM2 test
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~/sam2"))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def show_all_masks():
    """Re-run and show ALL masks individually"""
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load the full res frame we already extracted
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    frame_path = os.path.join(backend_dir, "test_frame_full_res.png")
    
    if not os.path.exists(frame_path):
        print("Extracting frame first...")
        video_path = os.path.join(backend_dir, "uploads/assets/videos/do_re_mi.mov")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 900)
        ret, frame = cap.read()
        cap.release()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(frame_rgb).save(frame_path)
    
    image = np.array(Image.open(frame_path))
    h, w = image.shape[:2]
    print(f"Image: {w}x{h}")
    
    # Load model and generate masks
    checkpoint = os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_small.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
    print("Generating masks...")
    sam2 = build_sam2(model_cfg, checkpoint, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        min_mask_region_area=100
    )
    
    with torch.inference_mode():
        masks = mask_generator.generate(image)
    
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    print(f"Generated {len(masks)} masks")
    
    # Create grid showing ALL masks
    num_masks = len(masks)
    cols = 6
    rows = (num_masks + cols - 1) // cols + 1  # +1 for header row
    
    fig = plt.figure(figsize=(24, 4*rows))
    
    # Show original in first position
    ax = plt.subplot(rows, cols, 1)
    ax.imshow(image)
    ax.set_title("ORIGINAL", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Show color legend
    ax = plt.subplot(rows, cols, 2)
    ax.axis('off')
    legend_text = "All 17 Masks:\n\n"
    for i in range(min(10, num_masks)):
        legend_text += f"#{i+1}: {masks[i]['area']} px\n"
    ax.text(0.1, 0.5, legend_text, fontsize=10, verticalalignment='center')
    ax.set_title("MASK SIZES", fontsize=12, fontweight='bold')
    
    # Show all masks combined
    ax = plt.subplot(rows, cols, 3)
    all_colored = np.zeros((h, w, 3), dtype=np.uint8)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_masks))
    for idx, mask in enumerate(masks):
        all_colored[mask['segmentation']] = (colors[idx][:3] * 255).astype(np.uint8)
    ax.imshow(all_colored)
    ax.set_title("ALL COMBINED", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Leave 3 spots empty for header row
    for i in range(4, 7):
        ax = plt.subplot(rows, cols, i)
        ax.axis('off')
    
    # Show each mask individually
    for idx in range(num_masks):
        ax = plt.subplot(rows, cols, idx + 7)  # Start from position 7
        
        # Show mask on black background with original image content
        mask_viz = np.zeros_like(image)
        mask_viz[masks[idx]['segmentation']] = image[masks[idx]['segmentation']]
        
        ax.imshow(mask_viz)
        
        # Add bounding box to see location
        bbox = masks[idx]['bbox']
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                            linewidth=1, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        
        # Determine what this might be based on location and size
        area = masks[idx]['area']
        cx, cy = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2  # Center
        
        # Guess what the mask might be
        label = f"#{idx+1}: {area}px"
        if area > 100000:
            label += "\n(Background)"
        elif 20000 < area < 40000 and cy < h/2:
            label += "\n(Person?)"
        elif 15000 < area < 25000 and cy < h/3:
            label += "\n(Face/Head?)"
        elif 5000 < area < 15000 and cy > h/2:
            label += "\n(Guitar?)"
        elif area < 2000 and cy < h/3:
            label += "\n(Hair/Detail?)"
        
        ax.set_title(label, fontsize=8)
        ax.axis('off')
    
    # Fill remaining empty spots
    total_spots = rows * cols
    for idx in range(num_masks + 7, total_spots):
        ax = plt.subplot(rows, cols, idx)
        ax.axis('off')
    
    plt.suptitle(f"ALL {num_masks} INDIVIDUAL MASKS - Full Resolution", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(backend_dir, "sam2_all_17_masks.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved to: sam2_all_17_masks.png")
    
    # Print detailed analysis
    print("\nDETAILED MASK ANALYSIS:")
    print("-" * 50)
    for i, mask in enumerate(masks):
        bbox = mask['bbox']
        cx, cy = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
        print(f"Mask {i+1}:")
        print(f"  Area: {mask['area']} pixels ({mask['area']*100/(h*w):.2f}% of image)")
        print(f"  Center: ({cx:.0f}, {cy:.0f})")
        print(f"  BBox: x={bbox[0]:.0f}, y={bbox[1]:.0f}, w={bbox[2]:.0f}, h={bbox[3]:.0f}")
        print(f"  IoU Score: {mask['predicted_iou']:.3f}")
        print()


if __name__ == "__main__":
    show_all_masks()