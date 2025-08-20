#!/usr/bin/env python3
"""
SAM2 test focused on person segmentation with point prompts
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add SAM2 to path
sys.path.insert(0, os.path.expanduser("~/sam2"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def test_with_point_prompts():
    """Test SAM2 with specific point prompts on the person"""
    
    # Setup device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using {device}")
    
    # Use small model
    checkpoint = os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_small.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
    print("Loading model...")
    sam2 = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2)
    
    # Load image
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    frame_path = os.path.join(backend_dir, "test_frame_30.png")
    image = np.array(Image.open(frame_path))
    h, w = image.shape[:2]
    print(f"Image shape: {h}x{w}")
    
    # Set image
    with torch.inference_mode():
        predictor.set_image(image)
        
        # Define strategic points on the person
        # Based on the image, the person is in the center-left area
        points_sets = [
            # Face/head region
            {
                'points': [[w*0.35, h*0.25]],  # Head/face area
                'labels': [1],
                'name': 'Face/Head'
            },
            # Upper body/shirt
            {
                'points': [[w*0.35, h*0.45]],  # Chest area
                'labels': [1],
                'name': 'Upper Body/Shirt'
            },
            # Arms
            {
                'points': [[w*0.25, h*0.5]],  # Left arm area
                'labels': [1],
                'name': 'Arms'
            },
            # Guitar
            {
                'points': [[w*0.5, h*0.6]],  # Guitar body
                'labels': [1],
                'name': 'Guitar'
            },
            # Hair
            {
                'points': [[w*0.35, h*0.15]],  # Hair area
                'labels': [1],
                'name': 'Hair'
            },
            # Multiple points for whole person
            {
                'points': [
                    [w*0.35, h*0.25],  # Head
                    [w*0.35, h*0.45],  # Body
                    [w*0.3, h*0.6],    # Lower body
                ],
                'labels': [1, 1, 1],
                'name': 'Whole Person'
            }
        ]
        
        all_results = []
        
        for point_set in points_sets:
            points = np.array(point_set['points'])
            labels = np.array(point_set['labels'])
            
            # Generate masks
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True  # Get multiple mask options
            )
            
            all_results.append({
                'masks': masks,
                'scores': scores,
                'name': point_set['name'],
                'points': points
            })
            
            print(f"{point_set['name']}: Generated {len(masks)} masks, best score: {scores.max():.3f}")
    
    # Visualize results
    visualize_point_results(image, all_results)
    
    return all_results


def test_with_better_automatic():
    """Test automatic segmentation with very aggressive parameters"""
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using {device}")
    
    # Use large model for better quality
    checkpoint = os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    print("Loading LARGE model for better segmentation...")
    sam2 = build_sam2(model_cfg, checkpoint, device=device)
    
    # Very aggressive parameters
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=64,  # Maximum density
        points_per_batch=128,
        pred_iou_thresh=0.5,  # Very low threshold
        stability_score_thresh=0.7,  # Lower threshold
        stability_score_offset=0.5,
        crop_n_layers=2,  # More crop layers
        crop_overlap_ratio=0.6,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=10,  # Tiny minimum
        box_nms_thresh=0.3,  # Lower NMS threshold to keep more masks
        crop_nms_thresh=0.3
    )
    
    # Load image
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    frame_path = os.path.join(backend_dir, "test_frame_30.png")
    image = np.array(Image.open(frame_path))
    
    print("Generating masks with VERY aggressive parameters...")
    with torch.inference_mode():
        masks = mask_generator.generate(image)
    
    print(f"Generated {len(masks)} masks")
    
    # Sort by area
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Print detailed statistics
    print("\nDetailed Mask Statistics:")
    for i, mask in enumerate(masks[:20]):
        bbox = mask['bbox']  # x, y, w, h
        print(f"  Mask {i+1}: Area={mask['area']:5d}, IoU={mask['predicted_iou']:.3f}, "
              f"Bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f}x{bbox[3]:.0f})")
    
    # Visualize
    visualize_all_masks_detailed(image, masks)
    
    return masks


def visualize_point_results(image, results):
    """Visualize point-based segmentation results"""
    num_tests = len(results)
    fig, axes = plt.subplots(num_tests, 4, figsize=(16, 4*num_tests))
    
    if num_tests == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(results):
        # Original with points
        ax = axes[idx, 0]
        ax.imshow(image)
        points = result['points']
        ax.scatter(points[:, 0], points[:, 1], c='red', s=100, marker='x')
        ax.set_title(f"{result['name']} - Points", fontweight='bold')
        ax.axis('off')
        
        # Show up to 3 masks
        masks = result['masks']
        scores = result['scores']
        
        for mask_idx in range(min(3, len(masks))):
            ax = axes[idx, mask_idx + 1]
            
            # Create overlay
            overlay = image.copy()
            mask_colored = np.zeros_like(image)
            mask_bool = masks[mask_idx].astype(bool)
            mask_colored[mask_bool] = [255, 0, 0]  # Red mask
            
            overlay = cv2.addWeighted(overlay, 0.6, mask_colored, 0.4, 0)
            
            ax.imshow(overlay)
            ax.set_title(f"Mask {mask_idx+1} (score: {scores[mask_idx]:.3f})")
            ax.axis('off')
    
    plt.suptitle("SAM2 Point-based Segmentation - Looking for Person Parts", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    output_path = os.path.join(backend_dir, "sam2_person_points.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved point results to: {output_path}")


def visualize_all_masks_detailed(image, masks):
    """Detailed visualization of all masks"""
    h, w = image.shape[:2]
    
    # Create a figure showing masks by category
    fig = plt.figure(figsize=(20, 15))
    
    # Categorize masks by size
    large_masks = [m for m in masks if m['area'] > 5000]
    medium_masks = [m for m in masks if 500 < m['area'] <= 5000]
    small_masks = [m for m in masks if 100 < m['area'] <= 500]
    tiny_masks = [m for m in masks if m['area'] <= 100]
    
    print(f"\nMask Categories:")
    print(f"  Large (>5000px): {len(large_masks)} masks")
    print(f"  Medium (500-5000px): {len(medium_masks)} masks")
    print(f"  Small (100-500px): {len(small_masks)} masks")
    print(f"  Tiny (<100px): {len(tiny_masks)} masks")
    
    # Create color-coded visualization
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image)
    ax1.set_title("Original", fontweight='bold')
    ax1.axis('off')
    
    # All masks colored by size
    all_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color scheme: Large=Blue, Medium=Green, Small=Yellow, Tiny=Red
    for mask in large_masks:
        all_colored[mask['segmentation']] = [0, 0, 255]  # Blue
    for mask in medium_masks:
        all_colored[mask['segmentation']] = [0, 255, 0]  # Green
    for mask in small_masks:
        all_colored[mask['segmentation']] = [255, 255, 0]  # Yellow
    for mask in tiny_masks:
        all_colored[mask['segmentation']] = [255, 0, 0]  # Red
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(all_colored)
    ax2.set_title(f"All {len(masks)} Masks by Size\n(B=Large, G=Med, Y=Small, R=Tiny)", fontweight='bold')
    ax2.axis('off')
    
    # Overlay
    overlay = cv2.addWeighted(image, 0.5, all_colored, 0.5, 0)
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(overlay)
    ax3.set_title("Overlay", fontweight='bold')
    ax3.axis('off')
    
    # Show top medium masks (likely person parts)
    for i in range(min(3, len(medium_masks))):
        ax = plt.subplot(2, 3, i + 4)
        mask_viz = np.zeros_like(image)
        mask_viz[medium_masks[i]['segmentation']] = image[medium_masks[i]['segmentation']]
        ax.imshow(mask_viz)
        ax.set_title(f"Medium Mask {i+1} (Area: {medium_masks[i]['area']})")
        ax.axis('off')
    
    plt.suptitle(f"SAM2 Aggressive Segmentation - {len(masks)} Total Masks", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    output_path = os.path.join(backend_dir, "sam2_aggressive_masks.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved aggressive mask visualization to: {output_path}")
    
    # Create grid of individual masks
    create_detailed_grid(image, masks[:30])


def create_detailed_grid(image, masks):
    """Create detailed grid of individual masks"""
    num_masks = len(masks)
    cols = 6
    rows = (num_masks + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3*rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        
        if idx < num_masks:
            mask = masks[idx]['segmentation']
            # Show mask with original image content
            display = np.zeros_like(image)
            display[mask] = image[mask]
            
            # Add bounding box
            bbox = masks[idx]['bbox']
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                linewidth=2, edgecolor='r', facecolor='none')
            
            ax.imshow(display)
            ax.add_patch(rect)
            ax.set_title(f"#{idx+1} A:{masks[idx]['area']}", fontsize=8)
        
        ax.axis('off')
    
    plt.suptitle(f"Top {num_masks} Individual Masks with Bounding Boxes", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    output_path = os.path.join(backend_dir, "sam2_detailed_grid.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved detailed grid to: {output_path}")


def main():
    print("=" * 60)
    print("SAM2 Person Segmentation Investigation")
    print("=" * 60)
    
    # Test 1: Point prompts
    print("\n1. Testing with point prompts on person...")
    print("-" * 40)
    point_results = test_with_point_prompts()
    
    # Test 2: Aggressive automatic
    print("\n2. Testing with aggressive automatic segmentation...")
    print("-" * 40)
    auto_masks = test_with_better_automatic()
    
    print("\n" + "=" * 60)
    print("Investigation Complete!")
    print("\nGenerated files:")
    print("  - sam2_person_points.png (point-based attempts)")
    print("  - sam2_aggressive_masks.png (aggressive automatic)")
    print("  - sam2_detailed_grid.png (individual masks with bboxes)")
    print("=" * 60)


if __name__ == "__main__":
    main()