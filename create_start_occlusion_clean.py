#!/usr/bin/env python3
"""
Create "START" text with clean occlusion effect using rembg for accurate background
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
from rembg import remove, new_session

# Add SAM2 to path
sys.path.insert(0, os.path.expanduser("~/sam2"))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def create_clean_occlusion():
    """Create START text with clean occlusion using rembg"""
    
    # Load the full resolution image
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    image_path = os.path.join(backend_dir, "test_frame_full_res.png")
    
    if not os.path.exists(image_path):
        print("Loading saved frame...")
        # Try the downsampled version if full res not available
        image_path = os.path.join(backend_dir, "test_frame_30.png")
        if not os.path.exists(image_path):
            print("No frame found, extracting...")
            video_path = os.path.join(backend_dir, "uploads/assets/videos/do_re_mi.mov")
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 900)
            ret, frame = cap.read()
            cap.release()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(image_path)
    
    # Load image
    image = Image.open(image_path)
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Step 1: Use rembg to get accurate foreground/background separation
    print("Removing background with rembg...")
    session = new_session('u2net')  # or 'u2netp' for lighter model
    
    # Get image with background removed (RGBA with transparency)
    img_no_bg = remove(image, session=session)
    img_no_bg_np = np.array(img_no_bg)
    
    # Extract the alpha channel as foreground mask
    if img_no_bg_np.shape[2] == 4:
        foreground_mask = img_no_bg_np[:, :, 3] > 128  # Binary mask of foreground
    else:
        # Fallback if no alpha channel
        foreground_mask = np.ones((h, w), dtype=bool)
    
    # Background mask is inverse of foreground
    background_mask = ~foreground_mask
    
    print(f"Foreground pixels: {np.sum(foreground_mask)}")
    print(f"Background pixels: {np.sum(background_mask)}")
    
    # Step 2: Generate SAM2 masks for additional detail (optional)
    print("Generating SAM2 masks for reference...")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    checkpoint = os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_small.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
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
        masks = mask_generator.generate(image_np)
    
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    print(f"Generated {len(masks)} SAM2 masks for reference")
    
    # Step 3: Create text layer
    text_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    
    # Use a bold font
    try:
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Avenir.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ]
        
        font = None
        font_size = min(180, int(h * 0.35))  # Scale with image size
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                print(f"Using font: {font_path}")
                break
        
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Position text
    text = "START"
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center-left positioning
    text_x = int(w * 0.15)  # 15% from left
    text_y = int(h * 0.35)  # 35% from top
    
    # Draw text with multiple effects for visibility
    # 1. Dark shadow for depth
    shadow_offset = 5
    draw.text((text_x + shadow_offset, text_y + shadow_offset), text, 
              font=font, fill=(0, 0, 0, 150))
    
    # 2. White outline for contrast
    outline_width = 3
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if abs(dx) == outline_width or abs(dy) == outline_width:
                draw.text((text_x + dx, text_y + dy), text, 
                         font=font, fill=(255, 255, 255, 200))
    
    # 3. Main text - bright yellow
    main_color = (255, 220, 0, 255)  # Golden yellow
    draw.text((text_x, text_y), text, font=font, fill=main_color)
    
    # Convert text layer to numpy
    text_layer_np = np.array(text_layer)
    
    # Step 4: Create final composite with CLEAN occlusion
    result = image_np.copy()
    
    # CRITICAL: Only apply text where it's in the BACKGROUND
    # This prevents any bleeding into foreground objects
    for c in range(3):  # RGB channels
        # Text appears ONLY in background areas
        text_visible_mask = (text_layer_np[:, :, 3] > 0) & background_mask
        result[text_visible_mask, c] = text_layer_np[text_visible_mask, c]
    
    # Step 5: Create enhanced version with subtle glow in background only
    result_enhanced = create_enhanced_version(image_np, text_layer_np, background_mask)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Original
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(image_np)
    ax1.set_title("Original", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Foreground mask from rembg
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(foreground_mask, cmap='gray')
    ax2.set_title("Foreground (rembg)", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Background mask
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(background_mask, cmap='gray')
    ax3.set_title("Background Mask", fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Text layer
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(text_layer)
    ax4.set_title("Text Layer", fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Final result - CLEAN
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(result)
    ax5.set_title("CLEAN: Text ONLY in Background", fontsize=14, fontweight='bold', color='green')
    ax5.axis('off')
    
    # Enhanced version
    ax6 = plt.subplot(2, 4, 6)
    ax6.imshow(result_enhanced)
    ax6.set_title("Enhanced with Glow", fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # Zoomed view of occlusion
    zoom_cx = int(w * 0.35)
    zoom_cy = int(h * 0.45)
    zoom_size = min(250, int(min(w, h) * 0.4))
    zoom_x1 = max(0, zoom_cx - zoom_size // 2)
    zoom_x2 = min(w, zoom_cx + zoom_size // 2)
    zoom_y1 = max(0, zoom_cy - zoom_size // 2)
    zoom_y2 = min(h, zoom_cy + zoom_size // 2)
    
    ax7 = plt.subplot(2, 4, 7)
    ax7.imshow(result[zoom_y1:zoom_y2, zoom_x1:zoom_x2])
    ax7.set_title("Zoomed: Clean Edges", fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # Comparison with bad version (if we didn't use rembg)
    bad_result = image_np.copy()
    # Apply text everywhere it's not covered by SAM2 person masks
    if len(masks) > 2:
        sam_person_mask = masks[1]['segmentation'] | masks[2]['segmentation']
        for c in range(3):
            bad_mask = (text_layer_np[:, :, 3] > 0) & (~sam_person_mask)
            bad_result[bad_mask, c] = text_layer_np[bad_mask, c]
    
    ax8 = plt.subplot(2, 4, 8)
    ax8.imshow(bad_result)
    ax8.set_title("BAD: Without rembg (bleeding)", fontsize=12, color='red')
    ax8.axis('off')
    
    plt.suptitle("CLEAN Occlusion Effect: Text Only in Background (using rembg)", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save results
    output_path = os.path.join(backend_dir, "start_clean_occlusion.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: start_clean_occlusion.png")
    
    # Save just the final clean image
    final_image = Image.fromarray(result)
    final_path = os.path.join(backend_dir, "start_clean_final.png")
    final_image.save(final_path)
    print(f"Saved clean final to: start_clean_final.png")
    
    # Save enhanced version
    enhanced_image = Image.fromarray(result_enhanced)
    enhanced_path = os.path.join(backend_dir, "start_clean_enhanced.png")
    enhanced_image.save(enhanced_path)
    print(f"Saved enhanced version to: start_clean_enhanced.png")
    
    return result


def create_enhanced_version(image_np, text_layer_np, background_mask):
    """Create version with subtle glow effect ONLY in background"""
    result = image_np.copy()
    
    # Extract text alpha channel
    text_alpha = text_layer_np[:, :, 3].astype(float) / 255
    
    # Create subtle glow
    glow = cv2.GaussianBlur(text_alpha, (15, 15), 5) * 200
    
    # Apply glow ONLY to background areas
    for c in range(3):
        # Glow in background only
        glow_mask = (glow > 30) & background_mask
        if c < 2:  # Yellowish glow
            result[glow_mask, c] = np.minimum(255, 
                                             result[glow_mask, c] + glow[glow_mask] * 0.2)
        
        # Main text in background only
        text_mask = (text_layer_np[:, :, 3] > 0) & background_mask
        result[text_mask, c] = text_layer_np[text_mask, c]
    
    return result


if __name__ == "__main__":
    import torch
    
    # Check if rembg is installed
    try:
        import rembg
        print("rembg is installed")
    except ImportError:
        print("Installing rembg...")
        os.system("pip install rembg")
    
    create_clean_occlusion()