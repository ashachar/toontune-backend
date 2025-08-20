#!/usr/bin/env python3
"""
Enhanced SAM2 test with better parameters for small images
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


def test_sam2_automatic_enhanced():
    """Test SAM2 with enhanced parameters for better segmentation"""
    
    # Setup device
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Model paths - try small model for better results
    checkpoint = os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_small.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
    print(f"Loading SMALL model from {checkpoint}")
    
    # Build model
    sam2 = build_sam2(model_cfg, checkpoint, device=device)
    
    # Create automatic mask generator with better parameters
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=32,  # More points for better coverage
        points_per_batch=64,
        pred_iou_thresh=0.7,  # Lower threshold for more masks
        stability_score_thresh=0.8,  # Lower for more masks
        stability_score_offset=1.0,
        crop_n_layers=1,  # Enable crop for better small object detection
        crop_overlap_ratio=0.5,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25  # Very small minimum area
    )
    
    # Load image
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    frame_path = os.path.join(backend_dir, "test_frame_30.png")
    image = np.array(Image.open(frame_path))
    print(f"Image shape: {image.shape}")
    
    # Generate masks
    print("Generating automatic masks with enhanced parameters...")
    with torch.inference_mode():
        masks = mask_generator.generate(image)
    
    print(f"Generated {len(masks)} masks")
    
    # Sort by area
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Print mask statistics
    print("\nMask Statistics:")
    for i, mask in enumerate(masks[:10]):
        print(f"  Mask {i+1}: Area={mask['area']}, IoU={mask['predicted_iou']:.3f}, Stability={mask['stability_score']:.3f}")
    
    # Visualize all masks
    visualize_enhanced_masks(image, masks)
    
    return masks


def visualize_enhanced_masks(image, masks):
    """Enhanced visualization showing more masks"""
    h, w = image.shape[:2]
    
    # Create larger figure for more masks
    num_display = min(12, len(masks))  # Show up to 12 masks
    cols = 4
    rows = (num_display + 3) // cols + 1  # +1 row for top images
    
    fig = plt.figure(figsize=(20, 5 * rows))
    
    # Original image
    ax1 = plt.subplot(rows, cols, 1)
    ax1.imshow(image)
    ax1.set_title("Original Frame", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # All segments with unique colors
    combined_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Use different colormap for more colors
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = np.vstack([colors, plt.cm.tab20b(np.linspace(0, 1, 20))])
    colors = np.vstack([colors, plt.cm.tab20c(np.linspace(0, 1, 20))])
    
    # Apply masks with transparency handling
    for idx, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        color_idx = idx % len(colors)
        rgb_color = (colors[color_idx][:3] * 255).astype(np.uint8)
        
        # Only color non-overlapping regions
        non_colored = combined_colored.sum(axis=2) == 0
        valid_mask = mask & non_colored
        combined_colored[valid_mask] = rgb_color
    
    ax2 = plt.subplot(rows, cols, 2)
    ax2.imshow(combined_colored)
    ax2.set_title(f"All {len(masks)} Segments", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Overlay
    overlay = cv2.addWeighted(image, 0.5, combined_colored, 0.5, 0)
    ax3 = plt.subplot(rows, cols, 3)
    ax3.imshow(overlay)
    ax3.set_title("Overlay on Original", fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Empty space
    ax4 = plt.subplot(rows, cols, 4)
    ax4.axis('off')
    
    # Individual masks
    for idx in range(num_display):
        ax = plt.subplot(rows, cols, idx + 5)
        
        mask_data = masks[idx]
        mask = mask_data['segmentation']
        
        # Create individual overlay
        individual_colored = np.zeros((h, w, 3), dtype=np.uint8)
        color_idx = idx % len(colors)
        individual_colored[mask] = (colors[color_idx][:3] * 255).astype(np.uint8)
        
        result = cv2.addWeighted(image, 0.6, individual_colored, 0.4, 0)
        
        ax.imshow(result)
        ax.set_title(f"#{idx+1} Area:{mask_data['area']} IoU:{mask_data['predicted_iou']:.2f}", fontsize=9)
        ax.axis('off')
    
    plt.suptitle(f"SAM2 Enhanced Segmentation - {len(masks)} Total Masks", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    output_path = os.path.join(backend_dir, "sam2_enhanced_colored.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved enhanced visualization to: {output_path}")
    
    # Save individual outputs
    Image.fromarray(combined_colored).save(os.path.join(backend_dir, "sam2_enhanced_segments.png"))
    Image.fromarray(overlay).save(os.path.join(backend_dir, "sam2_enhanced_overlay.png"))
    
    # Create a grid view of all masks
    create_mask_grid(image, masks, os.path.join(backend_dir, "sam2_mask_grid.png"))


def create_mask_grid(image, masks, output_path):
    """Create a grid showing all individual masks"""
    num_masks = len(masks)
    cols = 8
    rows = (num_masks + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2.5 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        
        if idx < num_masks:
            mask = masks[idx]['segmentation']
            # Show mask on black background
            display = np.zeros_like(image)
            display[mask] = image[mask]
            ax.imshow(display)
            ax.set_title(f"Mask {idx+1}", fontsize=8)
        
        ax.axis('off')
    
    plt.suptitle(f"All {num_masks} Individual Masks", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved mask grid to: {output_path}")


def main():
    """Main test function"""
    print("=" * 60)
    print("Enhanced SAM2 Segmentation Test")
    print("=" * 60)
    
    try:
        masks = test_sam2_automatic_enhanced()
        
        print("\n" + "=" * 60)
        print("Enhanced SAM2 Test Complete!")
        print(f"Successfully generated {len(masks)} masks")
        print("\nGenerated files:")
        print("  - sam2_enhanced_colored.png (main visualization)")
        print("  - sam2_enhanced_segments.png (segments only)")
        print("  - sam2_enhanced_overlay.png (overlay)")
        print("  - sam2_mask_grid.png (all masks grid)")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTrying with tiny model as fallback...")
        
        # Fallback to tiny model
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()