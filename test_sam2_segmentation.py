#!/usr/bin/env python3
"""
Test SAM2 segmentation on do_re_mi video frame
Falls back to simulation if SAM2 not installed
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def simulate_sam2_segmentation(image_path: str, num_segments: int = 5):
    """
    Simulate SAM2 segmentation using traditional methods
    This is a fallback when SAM2 is not installed
    """
    # Load image
    image = np.array(Image.open(image_path))
    h, w = image.shape[:2]
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply different thresholding techniques to create segments
    segments = []
    
    # Method 1: Otsu's thresholding
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segments.append(otsu_mask > 0)
    
    # Method 2: Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    segments.append(adaptive > 0)
    
    # Method 3: Edge detection + dilation
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    segments.append(dilated > 0)
    
    # Method 4: K-means clustering
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    kmeans_seg = labels.reshape(h, w)
    
    for i in range(3):
        segments.append(kmeans_seg == i)
    
    # Method 5: Contour-based regions
    contours, _ = cv2.findContours(otsu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
        segments.append(contour_mask > 0)
    
    # Select top segments
    selected_segments = segments[:num_segments]
    
    return image, selected_segments


def visualize_colored_segments(image, segments, output_path="colored_segments.png"):
    """
    Visualize segments with different colors
    """
    h, w = image.shape[:2]
    
    # Create figure with subplots
    num_segments = len(segments)
    cols = min(3, num_segments)
    rows = (num_segments + cols - 1) // cols
    
    fig, axes = plt.subplots(rows + 1, cols, figsize=(15, 5 * (rows + 1)))
    
    # Flatten axes array for easier indexing
    if rows + 1 == 1:
        axes = axes.reshape(1, -1)
    
    # Show original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Frame", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Create combined colored overlay
    combined_colored = np.zeros((h, w, 3), dtype=np.uint8)
    colors = plt.cm.tab20(np.linspace(0, 1, num_segments))
    
    for idx, (segment, color) in enumerate(zip(segments, colors)):
        # Individual segment visualization
        row = (idx + 1) // cols + 1
        col = (idx + 1) % cols
        
        # Create colored overlay for this segment
        overlay = image.copy()
        colored_mask = np.zeros_like(image)
        colored_mask[segment] = (color[:3] * 255).astype(np.uint8)
        
        # Blend with original
        result = cv2.addWeighted(overlay, 0.6, colored_mask, 0.4, 0)
        
        axes[row, col].imshow(result)
        axes[row, col].set_title(f"Segment {idx + 1}", fontsize=10)
        axes[row, col].axis('off')
        
        # Add to combined view
        combined_colored[segment] = (color[:3] * 255).astype(np.uint8)
    
    # Show combined colored segments
    axes[0, 1].imshow(combined_colored)
    axes[0, 1].set_title("All Segments (Colored)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Overlay combined on original
    overlay_combined = cv2.addWeighted(image, 0.5, combined_colored, 0.5, 0)
    axes[0, 2].imshow(overlay_combined)
    axes[0, 2].set_title("Overlay on Original", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Hide unused subplots
    for i in range(num_segments + 1, rows * cols):
        row = i // cols + 1
        col = i % cols
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].axis('off')
    
    plt.suptitle("SAM2 Segmentation Results (Simulated)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Colored segments saved to: {output_path}")
    
    # Also save individual images
    Image.fromarray(combined_colored).save("segments_colored_only.png")
    Image.fromarray(overlay_combined).save("segments_overlay.png")
    print("Additional outputs: segments_colored_only.png, segments_overlay.png")
    
    plt.show()
    return combined_colored


def test_with_sam2():
    """
    Test with actual SAM2 if available
    """
    try:
        # Import our SAM2 implementation
        from utils.segmentation.sam2_local import SAM2Local
        
        print("Testing with SAM2 implementation...")
        
        # Initialize SAM2
        sam2 = SAM2Local(model_size="tiny")
        
        # Load the extracted frame
        frame = np.array(Image.open("test_frame_30.png"))
        
        # Try automatic segmentation
        print("Running automatic segmentation...")
        segments = sam2.automatic_segmentation(
            frame,
            points_per_side=16,  # Fewer points for smaller image
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            min_mask_region_area=100
        )
        
        # Extract top 5 masks
        top_masks = []
        for i, seg in enumerate(segments[:5]):
            top_masks.append(seg['segmentation'])
        
        # Visualize
        visualize_colored_segments(frame, top_masks, "sam2_colored_segments.png")
        
        return True
        
    except Exception as e:
        print(f"SAM2 not available: {e}")
        return False


def main():
    """
    Main test function
    """
    # First check if test frame exists
    if not Path("test_frame_30.png").exists():
        print("Extracting frame from video...")
        cap = cv2.VideoCapture("do_re_mi_with_music_256x256_downsampled.mov")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save("test_frame_30.png")
            print("Frame extracted successfully")
    
    # Try with SAM2 first
    if not test_with_sam2():
        # Fallback to simulation
        print("\nUsing simulation fallback (SAM2 not installed)...")
        print("To use actual SAM2, install it with:")
        print("  git clone https://github.com/facebookresearch/sam2.git")
        print("  cd sam2 && pip install -e .")
        print()
        
        # Run simulation
        image, segments = simulate_sam2_segmentation("test_frame_30.png", num_segments=5)
        
        # Visualize colored segments
        visualize_colored_segments(image, segments, "simulated_colored_segments.png")
    
    print("\nSegmentation test complete!")


if __name__ == "__main__":
    main()