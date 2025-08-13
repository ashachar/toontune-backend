#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 Head Detector Module
Integrated head detection using SAM2 with solid mask generation
"""

import os
import sys
import numpy as np
import cv2
import torch

# Try to import SAM2
SAM2_AVAILABLE = False
sam2_model = None
predictor = None

def initialize_sam2():
    """Initialize SAM2 model if available"""
    global SAM2_AVAILABLE, sam2_model, predictor
    
    if SAM2_AVAILABLE:
        return True
    
    try:
        # Add SAM2 paths
        sam2_root = os.path.join(os.path.dirname(__file__), '..', '..', 'cartoon-head-detection')
        sys.path.insert(0, sam2_root)
        sys.path.insert(0, os.path.join(sam2_root, 'sam2'))
        
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM2] Using device: {device}")
        
        # Try to find checkpoint
        checkpoint_paths = [
            os.path.join(sam2_root, 'checkpoints', 'sam2.1_hiera_small.pt'),
            os.path.join(sam2_root, 'sam2', 'checkpoints', 'sam2.1_hiera_small.pt'),
        ]
        
        checkpoint = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint = path
                break
        
        if not checkpoint:
            print("[SAM2] No checkpoint found")
            return False
        
        # Try to find config
        config = "configs/sam2.1/sam2.1_hiera_s.yaml"
        
        # Build model
        sam2_model = build_sam2(config, checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        SAM2_AVAILABLE = True
        print("[SAM2] Successfully initialized")
        return True
        
    except Exception as e:
        print(f"[SAM2] Failed to initialize: {e}")
        return False

def detect_head_with_sam2(image_bgr: np.ndarray, debug_name: str = "image") -> tuple:
    """
    Detect head using SAM2 and create solid mask
    Returns: (solid_mask, outline_mask, head_box)
    """
    global predictor
    
    H, W = image_bgr.shape[:2]
    
    # Initialize SAM2 if needed
    if not SAM2_AVAILABLE:
        if not initialize_sam2():
            return None, None, None
    
    if predictor is None:
        return None, None, None
    
    # Find head region
    head_box = find_head_region_color(image_bgr)
    if head_box is None:
        return None, None, None
    
    x0, y0, x1, y1 = head_box
    box_w = x1 - x0
    box_h = y1 - y0
    cx = (x0 + x1) / 2
    
    try:
        # Set image
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        
        # Create points for head
        points = np.array([
            [cx, y0 + 0.15 * box_h],  # Top (hair)
            [x0 + 0.3 * box_w, y0 + 0.15 * box_h],
            [x0 + 0.7 * box_w, y0 + 0.15 * box_h],
            [cx, y0 + 0.25 * box_h],  # Forehead
            [x0 + 0.3 * box_w, y0 + 0.4 * box_h],  # Eyes
            [x0 + 0.7 * box_w, y0 + 0.4 * box_h],
            [cx, y0 + 0.55 * box_h],  # Nose
            [cx, y0 + 0.7 * box_h],   # Mouth
            [cx, y0 + 0.85 * box_h],  # Chin
        ], dtype=np.float32)
        
        labels = np.ones(len(points), dtype=np.int32)
        
        # Add negative points below head
        negative_points = []
        if y1 + 50 < H:
            negative_points.extend([
                [cx, y1 + 50],
                [x0, y1 + 50],
                [x1, y1 + 50]
            ])
        
        if negative_points:
            neg_array = np.array(negative_points, dtype=np.float32)
            points = np.vstack([points, neg_array])
            neg_labels = np.zeros(len(negative_points), dtype=np.int32)
            labels = np.concatenate([labels, neg_labels])
        
        # Generate masks
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=head_box,
            multimask_output=True
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        mask = (masks[best_idx] * 255).astype(np.uint8)
        
        # Make mask solid (fill holes)
        mask = remove_noise_and_fill_holes(mask)
        
        # Create outline
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = cv2.erode(mask, kernel, iterations=1)
        outline = cv2.subtract(mask, eroded)
        
        print(f"[SAM2] Successfully detected head for {debug_name}")
        return mask, outline, head_box
        
    except Exception as e:
        print(f"[SAM2] Detection failed: {e}")
        return None, None, None

def find_head_region_color(image_bgr: np.ndarray) -> np.ndarray:
    """Find head region using color-based segmentation"""
    H, W = image_bgr.shape[:2]
    
    # Convert to HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Skin tone ranges
    lower_skin1 = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin1 = np.array([25, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    
    lower_skin2 = np.array([160, 20, 50], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    
    lower_skin3 = np.array([10, 30, 30], dtype=np.uint8)
    upper_skin3 = np.array([20, 150, 200], dtype=np.uint8)
    mask3 = cv2.inRange(hsv, lower_skin3, upper_skin3)
    
    # Combine masks
    skin_mask = cv2.bitwise_or(mask1, mask2)
    skin_mask = cv2.bitwise_or(skin_mask, mask3)
    
    # Focus on upper half
    skin_mask[int(H*0.5):, :] = 0
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 1000:
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Expand for hair
            expand_top = min(int(h * 0.8), y)
            expand_sides = int(w * 0.4)
            expand_bottom = int(h * 0.3)
            
            x = max(0, x - expand_sides)
            y = max(0, y - expand_top)
            w = min(W - x, w + 2 * expand_sides)
            h = min(H - y, h + expand_top + expand_bottom)
            
            if w < W * 0.8 and h < H * 0.8:
                return np.array([x, y, x+w, y+h], dtype=np.float32)
    
    # Fallback: upper-center region
    cx = W // 2
    cy = H // 4
    head_size = min(W, H) // 3
    x = max(0, cx - head_size // 2)
    y = max(0, cy - head_size // 2)
    w = head_size
    h = int(head_size * 1.3)
    
    return np.array([x, y, x + w, y + h], dtype=np.float32)

def remove_noise_and_fill_holes(mask: np.ndarray) -> np.ndarray:
    """Remove noise and fill ALL holes to create a solid mask"""
    if mask is None or np.sum(mask) == 0:
        return mask
    
    # Keep only largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        if len(sizes) > 0:
            largest_idx = np.argmax(sizes) + 1
            mask = np.zeros_like(mask)
            mask[labels == largest_idx] = 255
    
    # Progressive closing to fill holes
    for kernel_size in [15, 25, 35]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Fill contour completely
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, contours, -1, 255, -1)
    
    # Smooth edges
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel_erode, iterations=1)
    
    return mask