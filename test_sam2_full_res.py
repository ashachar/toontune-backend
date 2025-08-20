#!/usr/bin/env python3
"""
Test SAM2 on full resolution video frame
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt

# Add SAM2 to path
sys.path.insert(0, os.path.expanduser("~/sam2"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def test_full_resolution():
    """Test SAM2 on full resolution frame"""
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using {device}")
    
    # Extract full resolution frame
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    video_path = os.path.join(backend_dir, "uploads/assets/videos/do_re_mi.mov")
    
    print(f"Extracting frame from original video...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 900)  # Frame 900 for variety
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to extract frame")
        return
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    print(f"Full resolution frame: {w}x{h}")
    
    # Save for reference
    Image.fromarray(frame_rgb).save("test_frame_full_res.png")
    
    # Use small model (good balance of speed/quality)
    checkpoint = os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_small.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
    print("Loading model...")
    sam2 = build_sam2(model_cfg, checkpoint, device=device)
    
    # Test 1: Automatic segmentation
    print("\n1. Automatic segmentation on full resolution...")
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        min_mask_region_area=100
    )
    
    with torch.inference_mode():
        masks = mask_generator.generate(frame_rgb)
    
    print(f"Generated {len(masks)} masks")
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Visualize automatic results
    visualize_full_res_masks(frame_rgb, masks, "sam2_full_res_auto.png")
    
    # Test 2: Point prompts on person
    print("\n2. Point-based segmentation...")
    predictor = SAM2ImagePredictor(sam2)
    
    with torch.inference_mode():
        predictor.set_image(frame_rgb)
        
        # Points for person (adjust based on frame content)
        # Assuming person is roughly in center-left
        test_points = [
            [[w*0.3, h*0.3]],  # Head area
            [[w*0.3, h*0.5]],  # Body area
            [[w*0.25, h*0.6], [w*0.35, h*0.6]],  # Multiple body points
        ]
        
        point_results = []
        for points in test_points:
            points_np = np.array(points)
            labels_np = np.ones(len(points))
            
            masks, scores, _ = predictor.predict(
                point_coords=points_np,
                point_labels=labels_np,
                multimask_output=True
            )
            
            point_results.append({
                'masks': masks,
                'scores': scores,
                'points': points_np
            })
            print(f"  Points {points}: Best score = {scores.max():.3f}")
    
    visualize_point_comparison(frame_rgb, point_results, "sam2_full_res_points.png")
    
    print("\nDone! Check the output images.")
    return masks


def visualize_full_res_masks(image, masks, output_path):
    """Visualize masks on full resolution image"""
    h, w = image.shape[:2]
    
    fig = plt.figure(figsize=(20, 10))
    
    # Original
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(image)
    ax1.set_title("Original Full Resolution", fontweight='bold')
    ax1.axis('off')
    
    # All masks colored
    all_colored = np.zeros((h, w, 3), dtype=np.uint8)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(masks))))
    
    for idx, mask in enumerate(masks[:20]):
        color_idx = idx % len(colors)
        all_colored[mask['segmentation']] = (colors[color_idx][:3] * 255).astype(np.uint8)
    
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(all_colored)
    ax2.set_title(f"{len(masks)} Masks Total", fontweight='bold')
    ax2.axis('off')
    
    # Overlay
    overlay = cv2.addWeighted(image, 0.5, all_colored, 0.5, 0)
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(overlay)
    ax3.set_title("Overlay", fontweight='bold')
    ax3.axis('off')
    
    # Stats
    ax4 = plt.subplot(2, 4, 4)
    ax4.axis('off')
    stats_text = f"Total Masks: {len(masks)}\n\n"
    stats_text += "Top 5 by Area:\n"
    for i, m in enumerate(masks[:5]):
        stats_text += f"{i+1}. {m['area']} px ({m['area']*100/(h*w):.1f}%)\n"
    ax4.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    
    # Show top individual masks
    for idx in range(min(4, len(masks))):
        ax = plt.subplot(2, 4, idx + 5)
        mask_viz = np.zeros_like(image)
        mask_viz[masks[idx]['segmentation']] = image[masks[idx]['segmentation']]
        ax.imshow(mask_viz)
        ax.set_title(f"Mask {idx+1}: {masks[idx]['area']} px", fontsize=10)
        ax.axis('off')
    
    plt.suptitle(f"SAM2 on Full Resolution ({w}x{h})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    plt.savefig(os.path.join(backend_dir, output_path), dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")


def visualize_point_comparison(image, results, output_path):
    """Compare point-based results"""
    fig, axes = plt.subplots(len(results), 4, figsize=(16, 4*len(results)))
    
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, result in enumerate(results):
        # Show points
        ax = axes[row_idx, 0]
        ax.imshow(image)
        points = result['points']
        ax.scatter(points[:, 0], points[:, 1], c='red', s=200, marker='x')
        ax.set_title(f"Input Points ({len(points)})", fontweight='bold')
        ax.axis('off')
        
        # Show masks
        masks = result['masks']
        scores = result['scores']
        
        for mask_idx in range(min(3, len(masks))):
            ax = axes[row_idx, mask_idx + 1]
            
            overlay = image.copy()
            mask_colored = np.zeros_like(image)
            mask_bool = masks[mask_idx].astype(bool)
            mask_colored[mask_bool] = [255, 100, 0]  # Orange
            
            overlay = cv2.addWeighted(overlay, 0.6, mask_colored, 0.4, 0)
            ax.imshow(overlay)
            ax.set_title(f"Score: {scores[mask_idx]:.3f}")
            ax.axis('off')
    
    plt.suptitle("Point-based Segmentation on Full Resolution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    plt.savefig(os.path.join(backend_dir, output_path), dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    test_full_resolution()