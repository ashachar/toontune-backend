#!/usr/bin/env python3
"""
Test SAM2 on video frame - runs from backend directory
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


def test_sam2_automatic():
    """Test SAM2 automatic segmentation"""
    
    # Setup device
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Model paths
    checkpoint = os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    print(f"Loading model from {checkpoint}")
    
    # Build model
    sam2 = build_sam2(model_cfg, checkpoint, device=device)
    
    # Create automatic mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=16,  # Fewer points for smaller image
        pred_iou_thresh=0.8,
        stability_score_thresh=0.85,
        crop_n_layers=0,
        min_mask_region_area=50  # Small minimum area for 256x256 image
    )
    
    # Use absolute paths
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    frame_path = os.path.join(backend_dir, "test_frame_30.png")
    video_path = os.path.join(backend_dir, "do_re_mi_with_music_256x256_downsampled.mov")
    
    # Load frame
    if not Path(frame_path).exists():
        print("Extracting frame from video...")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(frame_path)
    
    # Load image
    image = np.array(Image.open(frame_path))
    print(f"Image shape: {image.shape}")
    
    # Generate masks
    print("Generating automatic masks...")
    with torch.inference_mode():
        masks = mask_generator.generate(image)
    
    print(f"Generated {len(masks)} masks")
    
    # Sort by area
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Visualize top masks
    visualize_masks(image, masks[:8])
    
    return masks


def test_sam2_with_points():
    """Test SAM2 with point prompts"""
    
    # Setup device
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Model paths
    checkpoint = os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    print(f"Loading model from {checkpoint}")
    
    # Build model
    sam2 = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2)
    
    # Use absolute paths
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    frame_path = os.path.join(backend_dir, "test_frame_30.png")
    
    # Load image
    image = np.array(Image.open(frame_path))
    h, w = image.shape[:2]
    
    # Set image
    with torch.inference_mode():
        predictor.set_image(image)
        
        # Test with multiple points
        points = np.array([
            [w // 2, h // 3],      # Upper center (likely head)
            [w // 3, h // 2],      # Left middle
            [2 * w // 3, h // 2],  # Right middle
            [w // 2, 2 * h // 3],  # Lower center
        ])
        labels = np.array([1, 1, 1, 1])  # All positive points
        
        # Generate masks
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )
    
    print(f"Generated {len(masks)} masks with point prompts")
    
    # Visualize
    visualize_point_masks(image, masks, scores, points)
    
    return masks


def visualize_masks(image, masks):
    """Visualize automatic masks with colors"""
    h, w = image.shape[:2]
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # Original image
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(image)
    ax1.set_title("Original Frame", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Combined colored segments
    combined_colored = np.zeros((h, w, 3), dtype=np.uint8)
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(masks), 20)))
    
    for idx, (mask_data, color) in enumerate(zip(masks[:8], colors[:8])):
        mask = mask_data['segmentation']
        rgb_color = (color[:3] * 255).astype(np.uint8)
        combined_colored[mask] = rgb_color
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(combined_colored)
    ax2.set_title("All Segments (Colored)", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Overlay
    overlay = cv2.addWeighted(image, 0.5, combined_colored, 0.5, 0)
    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(overlay)
    ax3.set_title("Overlay on Original", fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Individual masks
    for idx in range(min(6, len(masks))):
        ax = plt.subplot(3, 3, idx + 4)
        
        mask_data = masks[idx]
        mask = mask_data['segmentation']
        
        # Create individual overlay
        individual_colored = np.zeros((h, w, 3), dtype=np.uint8)
        individual_colored[mask] = (colors[idx][:3] * 255).astype(np.uint8)
        
        result = cv2.addWeighted(image, 0.6, individual_colored, 0.4, 0)
        
        ax.imshow(result)
        ax.set_title(f"Mask {idx+1} (IoU: {mask_data['predicted_iou']:.2f})", fontsize=10)
        ax.axis('off')
    
    plt.suptitle("SAM2 Automatic Segmentation Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    output_path = os.path.join(backend_dir, "sam2_automatic_colored.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    # Save individual outputs
    Image.fromarray(combined_colored).save(os.path.join(backend_dir, "sam2_segments_colored.png"))
    Image.fromarray(overlay).save(os.path.join(backend_dir, "sam2_segments_overlay.png"))
    
    # plt.show()  # Comment out to avoid blocking


def visualize_point_masks(image, masks, scores, points):
    """Visualize masks from point prompts"""
    h, w = image.shape[:2]
    
    # Create figure
    fig, axes = plt.subplots(1, len(masks) + 1, figsize=(20, 5))
    
    # Original with points
    axes[0].imshow(image)
    axes[0].scatter(points[:, 0], points[:, 1], c='red', s=100, marker='x')
    axes[0].set_title("Original with Points", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Show masks
    colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))
    
    for idx, (mask, score, color) in enumerate(zip(masks, scores, colors)):
        ax = axes[idx + 1]
        
        # Create colored overlay
        mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
        mask_bool = mask.astype(bool)  # Ensure boolean indexing
        mask_colored[mask_bool] = (color[:3] * 255).astype(np.uint8)
        
        overlay = cv2.addWeighted(image, 0.6, mask_colored, 0.4, 0)
        
        ax.imshow(overlay)
        ax.set_title(f"Mask {idx+1} (score: {score:.2f})", fontsize=10)
        ax.axis('off')
    
    plt.suptitle("SAM2 Point-based Segmentation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    output_path = os.path.join(backend_dir, "sam2_points_colored.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved point visualization to: {output_path}")
    
    # plt.show()  # Comment out to avoid blocking


def main():
    """Main test function"""
    print("=" * 60)
    print("Testing SAM2 Segmentation")
    print("=" * 60)
    
    # Test automatic segmentation
    print("\n1. Testing Automatic Segmentation:")
    print("-" * 40)
    automatic_masks = test_sam2_automatic()
    
    # Test point-based segmentation
    print("\n2. Testing Point-based Segmentation:")
    print("-" * 40)
    point_masks = test_sam2_with_points()
    
    print("\n" + "=" * 60)
    print("SAM2 Test Complete!")
    print("Generated files:")
    print("  - sam2_automatic_colored.png")
    print("  - sam2_segments_colored.png")
    print("  - sam2_segments_overlay.png")
    print("  - sam2_points_colored.png")
    print("=" * 60)


if __name__ == "__main__":
    main()