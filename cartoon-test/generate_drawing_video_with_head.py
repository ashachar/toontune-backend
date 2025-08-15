#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_drawing_video_with_head.py
Purpose: Generate a video of an image being drawn from scratch, prioritizing the head
         (drawn from exterior to interior) followed by the rest of the body
"""

import cv2
import numpy as np
import os
import sys
import math
from sklearn.cluster import MeanShift, estimate_bandwidth

# Import SAM2 for head detection
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except:
    print("[WARNING] SAM2 not installed, head detection may not work optimally")

# ============================================================================
# HEAD DETECTION FUNCTIONS (from our developed algorithm)
# ============================================================================

def find_head_region(image_bgr: np.ndarray) -> np.ndarray:
    """Find head region using color-based segmentation"""
    H, W = image_bgr.shape[:2]
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Skin tone detection ranges
    lower_skin1 = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin1 = np.array([25, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    
    lower_skin2 = np.array([160, 20, 50], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    
    lower_skin3 = np.array([10, 30, 30], dtype=np.uint8)
    upper_skin3 = np.array([20, 150, 200], dtype=np.uint8)
    mask3 = cv2.inRange(hsv, lower_skin3, upper_skin3)
    
    # Combine skin masks
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
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 1000:
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Expand to include hair
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

def get_head_mask_sam2(image_bgr: np.ndarray, head_box: np.ndarray) -> np.ndarray:
    """Get head mask using SAM2"""
    try:
        # Initialize SAM2
        device = "cpu"
        ckpt_path = "../cartoon-head-detection/checkpoints/sam2.1_hiera_small.pt"
        
        if not os.path.exists(ckpt_path):
            print("[WARNING] SAM2 checkpoint not found, using simple mask")
            return create_simple_head_mask(image_bgr, head_box)
        
        sam2_model = build_sam2("../cartoon-head-detection/sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml", 
                                ckpt_path, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        
        x0, y0, x1, y1 = head_box
        box_w = x1 - x0
        box_h = y1 - y0
        cx = (x0 + x1) / 2
        
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
        
        # Post-process to fill holes
        mask = fill_mask_holes(mask)
        
        return mask
        
    except Exception as e:
        print(f"[WARNING] SAM2 failed: {e}")
        return create_simple_head_mask(image_bgr, head_box)

def create_simple_head_mask(image_bgr: np.ndarray, head_box: np.ndarray) -> np.ndarray:
    """Create simple elliptical head mask as fallback"""
    H, W = image_bgr.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    
    x0, y0, x1, y1 = head_box.astype(int)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    rx = (x1 - x0) // 2
    ry = (y1 - y0) // 2
    
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    
    return mask

def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in mask to make it solid"""
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
    
    # Final smoothing
    mask = cv2.GaussianBlur(mask, (7, 7), 2.0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    
    return mask

def create_head_drawing_layers(head_mask: np.ndarray) -> list:
    """
    Create drawing layers for the head, from exterior to interior
    Returns list of masks, each representing a layer to draw
    """
    layers = []
    
    # Layer 1: Hair/Head outline (outermost)
    outline_mask = head_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    eroded = cv2.erode(outline_mask, kernel, iterations=2)
    outline_layer = cv2.subtract(outline_mask, eroded)
    layers.append(('hair_outline', outline_layer))
    
    # Layer 2: Main hair/head area
    hair_mask = eroded.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    face_region = cv2.erode(hair_mask, kernel, iterations=2)
    hair_layer = cv2.subtract(hair_mask, face_region)
    layers.append(('hair_main', hair_layer))
    
    # Layer 3: Face outline
    face_outline = face_region.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    face_interior = cv2.erode(face_outline, kernel, iterations=1)
    face_outline_layer = cv2.subtract(face_outline, face_interior)
    layers.append(('face_outline', face_outline_layer))
    
    # Layer 4: Face interior (where features will be)
    layers.append(('face_interior', face_interior))
    
    return layers

# ============================================================================
# ORIGINAL VIDEO GENERATION FUNCTIONS (modified)
# ============================================================================

def load_hand_image(hand_path):
    hand = cv2.imread(hand_path, cv2.IMREAD_UNCHANGED)
    
    if hand is not None and hand.shape[2] == 4:
        bgr = hand[:, :, :3]
        alpha = hand[:, :, 3]
        return bgr, alpha
    else:
        return hand, None

def overlay_hand(background, hand, hand_alpha, position):
    x, y = position
    h, w = hand.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    x = max(0, min(x, bg_w - w))
    y = max(0, min(y, bg_h - h))
    
    if x + w > bg_w:
        w = bg_w - x
    if y + h > bg_h:
        h = bg_h - y
    
    if w <= 0 or h <= 0:
        return background
    
    roi = background[y:y+h, x:x+w]
    hand_crop = hand[:h, :w]
    
    if hand_alpha is not None:
        alpha_crop = hand_alpha[:h, :w]
        alpha = alpha_crop.astype(float) / 255
        alpha = np.expand_dims(alpha, axis=2)
        
        blended = (1 - alpha) * roi + alpha * hand_crop
        background[y:y+h, x:x+w] = blended.astype(np.uint8)
    else:
        background[y:y+h, x:x+w] = hand_crop
    
    return background

def create_spiral_path(mask, center, num_points=200):
    """Create a spiral path for drawing a masked region"""
    points = []
    max_radius = np.max(mask.shape) // 2
    
    for i in range(num_points):
        t = i / num_points
        radius = max_radius * t
        angle = 4 * np.pi * t  # 2 full rotations
        
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        
        # Only add points that are within the mask
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            if mask[y, x] > 0:
                points.append((x, y))
    
    return points

def create_circular_path(mask, center, radius_ratio=0.8, num_points=100):
    """Create a circular path for drawing a masked region"""
    points = []
    h, w = mask.shape
    
    # Calculate radius based on mask size
    avg_size = (w + h) / 4
    radius = avg_size * radius_ratio
    
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        
        # Only add points that are within the mask
        if 0 <= y < h and 0 <= x < w:
            if mask[y, x] > 0:
                points.append((x, y))
    
    # If we got too few points, try with smaller radius
    if len(points) < 20:
        return create_spiral_path(mask, center, num_points)
    
    return points

def segment_body_parts(image_bgr: np.ndarray, head_mask: np.ndarray) -> dict:
    """Segment the body into parts, excluding the head"""
    h, w = image_bgr.shape[:2]
    
    # Create mask for non-white pixels
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, content_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Remove head from content
    body_mask = cv2.bitwise_and(content_mask, cv2.bitwise_not(head_mask))
    
    # Find connected components in body
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(body_mask, connectivity=8)
    
    body_parts = []
    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] > 100:  # Minimum size threshold
            part_mask = (labels == i).astype(np.uint8) * 255
            center = (int(centroids[i][0]), int(centroids[i][1]))
            body_parts.append({
                'mask': part_mask,
                'center': center,
                'area': stats[i, cv2.CC_STAT_AREA],
                'top': stats[i, cv2.CC_STAT_TOP]
            })
    
    # Sort body parts from top to bottom
    body_parts.sort(key=lambda x: x['top'])
    
    return body_parts

def create_drawing_animation_with_head_priority():
    """Main function to create drawing animation with head-first priority"""
    
    # Input/output paths
    import sys
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
        base_name = os.path.splitext(os.path.basename(input_image))[0]
        output_video = f"../cartoon-test/drawing_animation_{base_name}_head_first.mp4"
    else:
        input_image = "../cartoon-test/man.png"  # default
        output_video = "../cartoon-test/drawing_animation_man_head_first.mp4"
    
    hand_path = "../cartoon-test/hand.png"  # You'll need to provide a hand image
    
    # Check if hand image exists, if not create a simple one
    if not os.path.exists(hand_path):
        print("[INFO] Creating placeholder hand image...")
        hand_img = np.ones((100, 100, 4), dtype=np.uint8) * 255
        cv2.circle(hand_img, (50, 50), 30, (100, 100, 100, 255), -1)
        cv2.imwrite(hand_path, hand_img)
    
    print(f"[INFO] Loading image: {input_image}")
    image_bgr = cv2.imread(input_image)
    if image_bgr is None:
        print(f"[ERROR] Could not load image: {input_image}")
        return
    
    height, width = image_bgr.shape[:2]
    print(f"[INFO] Image size: {width}x{height}")
    
    # Step 1: Detect head region
    print("[INFO] Detecting head region...")
    head_box = find_head_region(image_bgr)
    print(f"[INFO] Head box: {head_box}")
    
    # Step 2: Get head mask
    print("[INFO] Creating head mask...")
    head_mask = get_head_mask_sam2(image_bgr, head_box)
    
    # Save debug visualization
    debug_overlay = image_bgr.copy()
    colored_mask = np.zeros_like(image_bgr)
    colored_mask[:, :, 2] = head_mask
    debug_overlay = cv2.addWeighted(debug_overlay, 0.7, colored_mask, 0.3, 0)
    cv2.imwrite("../cartoon-test/debug_head_detection.png", debug_overlay)
    print("[INFO] Saved debug visualization to debug_head_detection.png")
    
    # Step 3: Create head drawing layers (exterior to interior)
    print("[INFO] Creating head drawing layers...")
    head_layers = create_head_drawing_layers(head_mask)
    print(f"[INFO] Created {len(head_layers)} head layers")
    
    # Step 4: Segment body parts
    print("[INFO] Segmenting body parts...")
    body_parts = segment_body_parts(image_bgr, head_mask)
    print(f"[INFO] Found {len(body_parts)} body parts")
    
    # Step 5: Load hand image
    print("[INFO] Loading hand image...")
    hand_img, hand_alpha = load_hand_image(hand_path)
    
    # Scale hand
    hand_scale = 0.12
    new_width = int(hand_img.shape[1] * hand_scale)
    new_height = int(hand_img.shape[0] * hand_scale)
    hand_img = cv2.resize(hand_img, (new_width, new_height))
    if hand_alpha is not None:
        hand_alpha = cv2.resize(hand_alpha, (new_width, new_height))
    
    # Step 6: Create video
    print("[INFO] Creating video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Initialize canvas and reveal mask
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    reveal_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Pen offset for hand
    pen_offset_x = new_width // 3
    pen_offset_y = new_height // 3
    
    # Total animation duration
    total_frames = 150  # 5 seconds at 30fps
    frame_count = 0
    
    # Calculate frames per component
    total_components = len(head_layers) + len(body_parts)
    frames_per_component = total_frames // total_components if total_components > 0 else total_frames
    
    print(f"[INFO] Total components: {total_components}, frames per component: {frames_per_component}")
    
    # Draw head layers first (exterior to interior)
    print("[INFO] Drawing head layers...")
    for layer_name, layer_mask in head_layers:
        print(f"  Drawing layer: {layer_name}")
        
        # Find center of this layer
        moments = cv2.moments(layer_mask)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            cx, cy = width // 2, height // 4
        
        # Create path for this layer
        if 'outline' in layer_name:
            # Use circular path for outlines
            path = create_circular_path(layer_mask, (cx, cy), radius_ratio=0.9)
        else:
            # Use spiral path for filled areas
            path = create_spiral_path(layer_mask, (cx, cy))
        
        if not path:
            continue
        
        # Draw this layer over multiple frames
        points_per_frame = max(1, len(path) // frames_per_component)
        path_idx = 0
        
        for _ in range(frames_per_component):
            if frame_count >= total_frames:
                break
            
            frame = canvas.copy()
            
            # Reveal points along the path
            for _ in range(points_per_frame):
                if path_idx < len(path):
                    px, py = path[path_idx]
                    # Use larger brush for head to ensure good coverage
                    cv2.circle(reveal_mask, (px, py), 15, 255, -1)
                    path_idx += 1
            
            # Apply revealed portions
            mask_3channel = cv2.cvtColor(reveal_mask, cv2.COLOR_GRAY2BGR) / 255.0
            frame = (frame * (1 - mask_3channel) + image_bgr * mask_3channel).astype(np.uint8)
            
            # Add hand at current position
            if path_idx > 0 and path_idx <= len(path):
                hand_pos = path[min(path_idx - 1, len(path) - 1)]
                hand_x = hand_pos[0] - pen_offset_x
                hand_y = hand_pos[1] - pen_offset_y
                frame = overlay_hand(frame, hand_img, hand_alpha, (hand_x, hand_y))
            
            video_writer.write(frame)
            frame_count += 1
    
    # Draw body parts
    print("[INFO] Drawing body parts...")
    for i, part in enumerate(body_parts):
        print(f"  Drawing body part {i+1}/{len(body_parts)}")
        
        part_mask = part['mask']
        center = part['center']
        
        # Create path for this body part
        path = create_spiral_path(part_mask, center)
        
        if not path:
            continue
        
        # Draw this part over multiple frames
        points_per_frame = max(1, len(path) // frames_per_component)
        path_idx = 0
        
        for _ in range(frames_per_component):
            if frame_count >= total_frames:
                break
            
            frame = canvas.copy()
            
            # Reveal points along the path
            for _ in range(points_per_frame):
                if path_idx < len(path):
                    px, py = path[path_idx]
                    cv2.circle(reveal_mask, (px, py), 12, 255, -1)
                    path_idx += 1
            
            # Apply revealed portions
            mask_3channel = cv2.cvtColor(reveal_mask, cv2.COLOR_GRAY2BGR) / 255.0
            frame = (frame * (1 - mask_3channel) + image_bgr * mask_3channel).astype(np.uint8)
            
            # Add hand at current position
            if path_idx > 0 and path_idx <= len(path):
                hand_pos = path[min(path_idx - 1, len(path) - 1)]
                hand_x = hand_pos[0] - pen_offset_x
                hand_y = hand_pos[1] - pen_offset_y
                frame = overlay_hand(frame, hand_img, hand_alpha, (hand_x, hand_y))
            
            video_writer.write(frame)
            frame_count += 1
    
    # Add final frames showing complete image
    print("[INFO] Adding final frames...")
    final_frame = image_bgr.copy()
    for _ in range(30):  # 1 second of final image
        video_writer.write(final_frame)
    
    # Release video writer
    video_writer.release()
    
    print(f"[SUCCESS] Video saved to: {output_video}")
    print(f"[INFO] Total frames written: {frame_count + 30}")
    
    # Save final mask for verification
    cv2.imwrite("../cartoon-test/final_reveal_mask.png", reveal_mask)
    print("[INFO] Saved final reveal mask")

if __name__ == "__main__":
    create_drawing_animation_with_head_priority()