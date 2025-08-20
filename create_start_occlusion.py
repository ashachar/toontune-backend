#!/usr/bin/env python3
"""
Create "START" text with occlusion effect using SAM2 masks
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt

# Add SAM2 to path
sys.path.insert(0, os.path.expanduser("~/sam2"))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def create_start_with_occlusion():
    """Create START text with person occlusion effect"""
    
    # Load the full resolution image
    backend_dir = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend"
    image_path = os.path.join(backend_dir, "test_frame_full_res.png")
    
    if not os.path.exists(image_path):
        print("Extracting frame...")
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
    
    # Generate masks using SAM2
    print("Generating masks...")
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
    print(f"Generated {len(masks)} masks")
    
    # Identify key masks
    # Mask 1: Background (largest)
    # Mask 2: Person body (29315 px)
    # Mask 3: Person head (23438 px)
    background_mask = masks[0]['segmentation']
    person_body_mask = masks[1]['segmentation']
    person_head_mask = masks[2]['segmentation']
    
    # Combine person masks for occlusion
    person_combined = person_body_mask | person_head_mask
    
    # Create text layer
    text_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    
    # Try to use a bold font, fallback to default if not available
    try:
        # Try different font paths for different systems
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Avenir.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 180)  # Large font size
                print(f"Using font: {font_path}")
                break
        
        if font is None:
            font = ImageFont.load_default()
            print("Using default font")
    except:
        font = ImageFont.load_default()
        print("Using default font")
    
    # Position text strategically
    text = "START"
    
    # Get text dimensions using textbbox
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position: middle-left area as suggested
    text_x = 250  # Start from left side
    text_y = 220  # Middle height
    
    # Draw text with effects
    # Add shadow/glow effect
    for offset in range(8, 0, -2):
        shadow_color = (0, 0, 0, 30)  # Semi-transparent black
        draw.text((text_x + offset, text_y + offset), text, font=font, fill=shadow_color)
    
    # Main text - bright yellow/white
    main_color = (255, 235, 100, 255)  # Bright yellow
    draw.text((text_x, text_y), text, font=font, fill=main_color)
    
    # Add outline for better visibility
    outline_width = 3
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((text_x + dx, text_y + dy), text, font=font, fill=(50, 50, 50, 200))
    draw.text((text_x, text_y), text, font=font, fill=main_color)
    
    # Convert text layer to numpy
    text_layer_np = np.array(text_layer)
    
    # Create final composite with occlusion
    result = image_np.copy()
    
    # Apply text only where there's no person (occlusion effect)
    for c in range(3):  # RGB channels
        # Where text exists and person doesn't occlude
        text_mask = (text_layer_np[:, :, 3] > 0) & (~person_combined)
        result[text_mask, c] = text_layer_np[text_mask, c]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Original
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Text layer
    axes[0, 1].imshow(text_layer)
    axes[0, 1].set_title("Text Layer (START)", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Person mask
    person_viz = np.zeros_like(image_np)
    person_viz[person_combined] = [255, 100, 100]
    axes[0, 2].imshow(person_viz)
    axes[0, 2].set_title("Person Mask (Occlusion)", fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Final result - large
    axes[1, 0].imshow(result)
    axes[1, 0].set_title("FINAL: Text Behind Person", fontsize=16, fontweight='bold', color='red')
    axes[1, 0].axis('off')
    
    # Zoomed in view
    zoom_x1, zoom_x2 = 200, 700
    zoom_y1, zoom_y2 = 150, 400
    axes[1, 1].imshow(result[zoom_y1:zoom_y2, zoom_x1:zoom_x2])
    axes[1, 1].set_title("Zoomed: Occlusion Detail", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Alternative style - with more glow
    result_glow = create_glow_version(image_np, text_layer_np, person_combined)
    axes[1, 2].imshow(result_glow)
    axes[1, 2].set_title("Alternative: With Glow Effect", fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle("START Text with Person Occlusion Effect", fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save results
    output_path = os.path.join(backend_dir, "start_occlusion_effect.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: start_occlusion_effect.png")
    
    # Save just the final image
    final_image = Image.fromarray(result)
    final_path = os.path.join(backend_dir, "start_final_only.png")
    final_image.save(final_path)
    print(f"Saved final only to: start_final_only.png")
    
    return result


def create_glow_version(image_np, text_layer_np, person_mask):
    """Create version with glow effect"""
    result = image_np.copy()
    h, w = image_np.shape[:2]
    
    # Create glow mask
    text_alpha = text_layer_np[:, :, 3].astype(float) / 255
    
    # Apply gaussian blur for glow
    glow = cv2.GaussianBlur(text_alpha, (21, 21), 10) * 255
    
    # Composite with glow
    for c in range(3):
        # Apply glow to background
        glow_mask = (glow > 20) & (~person_mask)
        if c < 2:  # Make glow yellowish
            result[glow_mask, c] = np.minimum(255, result[glow_mask, c] + glow[glow_mask] * 0.3)
        
        # Apply main text
        text_mask = (text_layer_np[:, :, 3] > 0) & (~person_mask)
        result[text_mask, c] = text_layer_np[text_mask, c]
    
    return result


if __name__ == "__main__":
    import torch
    create_start_with_occlusion()