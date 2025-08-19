#!/usr/bin/env python3
"""
Test background extraction using rembg inverse.
Takes the subtracted pixels (background removed by rembg) as our background mask.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from rembg import remove
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract_background_using_rembg(frame: np.ndarray) -> np.ndarray:
    """
    Extract background by using rembg and taking the inverse.
    The pixels that rembg removes (makes transparent) are the background.
    """
    # Convert frame to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Apply rembg to remove background
    print("  Applying rembg to detect foreground...")
    nobg_img = remove(pil_image)
    
    # Convert to numpy array
    nobg_array = np.array(nobg_img)
    
    # Get the alpha channel (transparency mask)
    if nobg_array.shape[2] == 4:
        alpha = nobg_array[:, :, 3]
    else:
        # If no alpha channel, create one
        alpha = np.ones(nobg_array.shape[:2], dtype=np.uint8) * 255
    
    # The background is where alpha is low (transparent)
    # The foreground is where alpha is high (opaque)
    background_mask = np.where(alpha < 128, 255, 0).astype(np.uint8)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
    
    return background_mask


def find_text_positions_in_background(bg_mask: np.ndarray, width: int, height: int):
    """Find safe positions for text placement in background areas."""
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bg_mask, connectivity=8
    )
    
    safe_regions = []
    
    for i in range(1, num_labels):  # Skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Only consider large enough regions
        if area > 3000:  # Smaller threshold for more options
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Create a grid of potential positions within this region
            # This gives us multiple positions per region
            for dy in [0.25, 0.5, 0.75]:
                for dx in [0.25, 0.5, 0.75]:
                    cx = int(x + w * dx)
                    cy = int(y + h * dy)
                    
                    # Check if this position is actually in the background
                    if bg_mask[cy, cx] == 255:
                        # Keep within bounds with margin
                        cx = max(100, min(cx, width - 100))
                        cy = max(50, min(cy, height - 50))
                        safe_regions.append((cx, cy, area))
    
    # Sort by area (larger areas first) and take diverse positions
    safe_regions.sort(key=lambda p: p[2], reverse=True)
    
    # Select diverse positions
    selected = []
    min_distance = 150  # Minimum distance between text positions
    
    for cx, cy, area in safe_regions:
        # Check if far enough from existing positions
        too_close = False
        for sx, sy in selected:
            distance = np.sqrt((cx - sx)**2 + (cy - sy)**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            selected.append((cx, cy))
        
        if len(selected) >= 10:  # Limit to 10 positions
            break
    
    return selected if selected else [(width // 2, height // 3)]


def test_rembg_extraction():
    """Test background extraction using rembg inverse."""
    video_path = Path("uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4")
    
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps}fps")
    
    # Test on multiple timestamps
    test_times = [10.0, 24.0, 35.0]
    
    output_dir = Path('tests/background_extraction_results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for timestamp in test_times:
        print(f"\nProcessing frame at t={timestamp}s...")
        
        # Extract frame
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            print(f"  Could not read frame at {timestamp}s")
            continue
        
        # Extract background using rembg
        bg_mask = extract_background_using_rembg(frame)
        
        # Find text positions
        text_positions = find_text_positions_in_background(bg_mask, width, height)
        
        # Calculate statistics
        bg_percentage = (np.sum(bg_mask == 255) / bg_mask.size) * 100
        print(f"  Background percentage: {bg_percentage:.1f}%")
        print(f"  Found {len(text_positions)} text positions")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original frame
        axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Frame')
        axes[0, 0].axis('off')
        
        # Background mask
        axes[0, 1].imshow(bg_mask, cmap='gray')
        axes[0, 1].set_title('Background Mask (rembg inverse)')
        axes[0, 1].axis('off')
        
        # Overlay showing background highlighted
        overlay = frame.copy()
        # Darken foreground (non-background areas)
        overlay[bg_mask == 0] = overlay[bg_mask == 0] * 0.3
        # Tint background slightly green
        overlay[bg_mask == 255, 1] = np.minimum(overlay[bg_mask == 255, 1] + 30, 255)
        
        axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Background Highlighted')
        axes[1, 0].axis('off')
        
        # Show text positions
        text_overlay = frame.copy()
        # Darken foreground
        text_overlay[bg_mask == 0] = text_overlay[bg_mask == 0] * 0.5
        
        # Draw text positions
        for i, (x, y) in enumerate(text_positions):
            # Draw circle
            cv2.circle(text_overlay, (x, y), 20, (0, 255, 0), -1)
            # Add label
            cv2.putText(text_overlay, f"P{i+1}", (x-15, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            # Draw sample text
            sample_text = ["DOE", "DEER", "RAY", "DROP", "ME", "FAR", "SEW", "TEA"][i % 8]
            cv2.putText(text_overlay, sample_text, (x-30, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(text_overlay, sample_text, (x-30, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
        
        axes[1, 1].imshow(cv2.cvtColor(text_overlay, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Text Positions ({len(text_positions)} positions)')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Background Extraction using rembg (inverse) - t={timestamp}s')
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(output_dir / f'rembg_extraction_t{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save mask separately
        cv2.imwrite(str(output_dir / f'rembg_mask_t{timestamp}.png'), bg_mask)
        
        print(f"  Saved results to {output_dir}")
    
    cap.release()
    print("\nTest complete!")


if __name__ == "__main__":
    test_rembg_extraction()