#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cartoon_face_mask.py
Purpose: Produce a pixel-accurate mask of a CARTOON face INCLUDING HAIR
from a colorful drawing using LBP cascade detector and SAM2
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch

print(f"[INFO] Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# SAM2 imports
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception as e:
    print("[ERR] SAM2 is not installed. Please follow the README section 'Install SAM2'.")
    raise

# Helper functions
def clip(val, lo, hi):
    return max(lo, min(hi, val))

def expand_head_box(bbox_xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Expand the face bbox to cover hair.
      - expand top by +60% face height (anime hair is often tall)
      - expand left/right by +20%
      - expand bottom by +20% (chin/neck margin)
    """
    x0, y0, x1, y1 = bbox_xyxy
    w = x1 - x0
    h = y1 - y0
    nx0 = clip(x0 - 0.20 * w, 0, img_w - 1)
    nx1 = clip(x1 + 0.20 * w, 0, img_w - 1)
    ny0 = clip(y0 - 0.60 * h, 0, img_h - 1)
    ny1 = clip(y1 + 0.20 * h, 0, img_h - 1)
    return np.array([nx0, ny0, nx1, ny1], dtype=np.float32)

def lbp_detector(image_bgr: np.ndarray) -> np.ndarray:
    """
    Returns bbox_xyxy from OpenCV LBP cascade detector, or None.
    """
    CASCADE_PATH = "lbpcascade_animeface.xml"
    if not os.path.isfile(CASCADE_PATH):
        print(f"[ERR] LBP cascade not found at {CASCADE_PATH}")
        return None
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24,24))
    
    if len(faces) == 0:
        print("[WARN] LBP cascade found no faces.")
        return None
    
    x, y, w, h = faces[0]
    return np.array([x, y, x+w, y+h], dtype=np.float32)

def find_sam2_config_path() -> str:
    """
    Locate 'sam2.1_hiera_s.yaml' inside the installed sam2 package.
    """
    import sam2
    cand = os.path.join(os.path.dirname(sam2.__file__), "configs", "sam2.1", "sam2.1_hiera_s.yaml")
    if os.path.isfile(cand):
        return cand
    # Fallback to relative path
    cand2 = os.path.join("sam2", "configs", "sam2.1", "sam2.1_hiera_s.yaml")
    if os.path.isfile(cand2):
        return cand2
    raise FileNotFoundError("Could not find SAM2 config 'sam2.1_hiera_s.yaml'.")

def run_sam2(image_bgr: np.ndarray, head_box_xyxy: np.ndarray,
             ckpt_path: str = "checkpoints/sam2.1_hiera_small.pt") -> np.ndarray:
    """
    Run SAM2 with a box prompt to get a binary mask for face+hair.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"SAM2 checkpoint not found at: {ckpt_path}")
    
    cfg_path = find_sam2_config_path()
    print(f"[INFO] Loading SAM2: cfg={cfg_path}, ckpt={ckpt_path}")
    
    sam2_model = build_sam2(cfg_path, ckpt_path)
    predictor = SAM2ImagePredictor(sam2_model)
    
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    
    box = head_box_xyxy.astype(np.float32)
    
    # Add some positive points to help guide SAM2 to include hair
    x0, y0, x1, y1 = box
    cx = 0.5 * (x0 + x1)
    top_y = y0 + 0.15 * (y1 - y0)
    left_x = x0 + 0.25 * (x1 - x0)
    right_x = x0 + 0.75 * (x1 - x0)
    
    # Create points for hair region
    points = np.array([
        [cx, top_y],          # top center (hair)
        [left_x, top_y],      # top left (hair)
        [right_x, top_y],     # top right (hair)
        [cx, (y0+y1)/2],      # center of face
    ], dtype=np.float32)
    
    labels = np.ones(len(points), dtype=np.int32)  # all positive points
    
    masks, scores, _ = predictor.predict(
        point_coords=points, 
        point_labels=labels, 
        box=box, 
        multimask_output=True
    )
    
    # Choose mask with highest score
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    
    mask = (mask * 255).astype(np.uint8)
    
    # Light post-process to smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (0,0), 0.8)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    
    return mask

def save_outputs(image_bgr: np.ndarray, mask: np.ndarray, head_box: np.ndarray, base_name: str):
    """Save mask outputs with proper naming"""
    # Save binary mask
    mask_path = f"{base_name}_mask.png"
    cv2.imwrite(mask_path, mask)
    
    # Save overlay
    overlay = image_bgr.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.6 + np.array([0, 0, 255]) * 0.4).astype(np.uint8)
    overlay_path = f"{base_name}_overlay.png"
    cv2.imwrite(overlay_path, overlay)
    
    # Save debug bbox
    dbg = image_bgr.copy()
    x0, y0, x1, y1 = head_box.astype(int)
    cv2.rectangle(dbg, (x0, y0), (x1, y1), (0,255,0), 2)
    bbox_path = f"{base_name}_bbox.png"
    cv2.imwrite(bbox_path, dbg)
    
    # Save RGBA cutout
    rgba = np.dstack([cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), mask])
    cutout_path = f"{base_name}_cutout.png"
    Image.fromarray(rgba, mode="RGBA").save(cutout_path)
    
    coverage = (mask > 0).mean() * 100.0
    print(f"[OK] Saved {mask_path}, {overlay_path}, {bbox_path}, {cutout_path} (coverage {coverage:.2f}%)")
    
    return mask_path, overlay_path

def process_image(image_path: str) -> tuple:
    """Process a single image and return mask path"""
    if not os.path.exists(image_path):
        print(f"[ERR] Image not found: {image_path}")
        return None, None
    
    # Get base name for outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"\n[INFO] Processing {image_path}...")
    image_bgr = cv2.imread(image_path, 1)
    if image_bgr is None:
        print(f"[ERR] Failed to read image: {image_path}")
        return None, None
    
    H, W = image_bgr.shape[:2]
    print(f"[INFO] Image size: {W}x{H}")
    
    # Detect face using LBP cascade
    bbox_xyxy = lbp_detector(image_bgr)
    if bbox_xyxy is None:
        print(f"[ERR] No face detected in {image_path}")
        return None, None
    
    print(f"[INFO] Face detected at: {bbox_xyxy}")
    
    # Expand box to include hair
    head_box = expand_head_box(bbox_xyxy, W, H)
    print(f"[INFO] Expanded head box: {head_box}")
    
    # Run SAM2 segmentation
    mask = run_sam2(image_bgr, head_box)
    
    # Save outputs
    mask_path, overlay_path = save_outputs(image_bgr, mask, head_box, base_name)
    
    return mask_path, overlay_path

def main():
    # Process both test images
    images = ["man.png", "woman.png"]
    results = []
    
    for img_path in images:
        mask_path, overlay_path = process_image(img_path)
        if mask_path:
            results.append((img_path, mask_path, overlay_path))
    
    # Display results
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    
    for img_path, mask_path, overlay_path in results:
        print(f"\n{img_path}:")
        print(f"  - Mask: {mask_path}")
        print(f"  - Overlay: {overlay_path}")
    
    # Open the masks for viewing
    if results:
        print("\n[INFO] Opening mask images for viewing...")
        for _, mask_path, _ in results:
            os.system(f"open {mask_path}")

if __name__ == "__main__":
    main()