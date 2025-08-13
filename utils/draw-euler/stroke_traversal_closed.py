#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced human-aware drawing with SAM2-based head detection:

This version replaces YOLO head detection with our improved SAM2-based algorithm
that creates solid masks without holes. The rest of the Euler path traversal
logic remains the same for smooth, continuous drawing.

Changes from original:
1. Replace YOLO with SAM2-based head detection
2. Use our solid mask generation (no holes)
3. Keep the sophisticated Euler path traversal for smooth drawing
"""

import sys
import os
import numpy as np
import cv2
from skimage.morphology import thin, skeletonize
import networkx as nx
import subprocess
import tempfile

# Import SAM2 head detector module
try:
    from sam2_head_detector import detect_head_with_sam2, remove_noise_and_fill_holes
    SAM2_DETECTOR_AVAILABLE = True
    print("[INFO] SAM2 head detector module loaded")
except Exception as e:
    print(f"[WARNING] SAM2 head detector not available: {e}")
    SAM2_DETECTOR_AVAILABLE = False

# =========================
# DEBUG FLAGS
# =========================
DEBUG = True
DEBUG_DUMP_COMPONENTS = True
DEBUG_VERBOSE = True  # Extra verbose logging
DEBUG_DIR = "test_output"

def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_mask(path, mask_uint8):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, mask_uint8)

def overlay_mask_on_image(base_rgb, mask_bool, color_bgr, alpha=0.6):
    base = base_rgb.copy()
    overlay = np.zeros_like(base)
    overlay[mask_bool] = color_bgr
    return cv2.addWeighted(base, 1.0, overlay, alpha, 0)

def component_bbox(points):
    ys = points[:, 0]
    xs = points[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())  # x1,y1,x2,y2

def bbox_center(x1,y1,x2,y2):
    return ((x1+x2)/2.0, (y1+y2)/2.0)

# =========================
# SAM2-based Head Detection (NEW)
# =========================
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

# If SAM2 detector is not available, create fallback functions
if not SAM2_DETECTOR_AVAILABLE:
    def detect_head_with_sam2(image_bgr: np.ndarray, debug_name: str = "image") -> tuple:
        """Fallback when SAM2 is not available"""
        return None, None, None
    
    def remove_noise_and_fill_holes(mask: np.ndarray) -> np.ndarray:
        """Advanced hole filling function"""
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
        
        # Fill contours completely
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

def detect_head_color_based(image_bgr: np.ndarray) -> tuple:
    """Color-based head detection with solid mask generation"""
    H, W = image_bgr.shape[:2]
    
    # Find head region using color
    head_box = find_head_region_color(image_bgr)
    if head_box is None:
        return None, None, None
    
    x0, y0, x1, y1 = head_box
    
    # Create initial mask based on detected region
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Use elliptical shape for head
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    width = x1 - x0
    height = y1 - y0
    
    # Draw ellipse for head shape
    cv2.ellipse(mask, (int(cx), int(cy)), 
                (int(width/2), int(height/2)), 
                0, 0, 360, 255, -1)
    
    # Make it solid (fill any gaps)
    mask = remove_noise_and_fill_holes(mask)
    
    # Create outline
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(mask, kernel, iterations=1)
    outline = cv2.subtract(mask, eroded)
    
    return mask, outline, head_box

def get_head_mask(image_bgr: np.ndarray, head_box: np.ndarray) -> tuple:
    """
    Get head mask using SAM2
    Returns: (filled_mask, outline_mask, head_box_refined)
    """
    # First try to use the standalone cartoon_face_mask_v5 script if available
    try:
        cartoon_test_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cartoon-test')
        if os.path.exists(cartoon_test_path):
            sys.path.insert(0, cartoon_test_path)
            from cartoon_face_mask_v5 import detect_and_save_head_mask
            
            # Use the v5 function which creates solid masks
            dprint("[Head] Using cartoon_face_mask_v5 for solid head detection")
            
            # Create temp file for the image
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cv2.imwrite(tmp.name, image_bgr)
                temp_path = tmp.name
            
            try:
                # Run detection
                mask, _, box = detect_and_save_head_mask(temp_path, None)
                
                if mask is not None:
                    # Create outline from the solid mask
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    eroded = cv2.erode(mask, kernel, iterations=1)
                    outline = cv2.subtract(mask, eroded)
                    
                    os.unlink(temp_path)  # Clean up temp file
                    dprint("[Head] Successfully detected head with cartoon_face_mask_v5")
                    return mask, outline, box if box is not None else head_box
            except Exception as e:
                dprint(f"[Head] cartoon_face_mask_v5 failed: {e}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    except Exception as e:
        dprint(f"[Head] Could not import cartoon_face_mask_v5: {e}")
    
    if not SAM2_AVAILABLE:
        return None, None, head_box
    
    try:
        device = "cpu"
        # Try different paths for the checkpoint
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'cartoon-head-detection', 'checkpoints', 'sam2.1_hiera_small.pt'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'cartoon-test', 'checkpoints', 'sam2.1_hiera_small.pt'),
            'checkpoints/sam2.1_hiera_small.pt'
        ]
        
        ckpt_path = None
        for path in possible_paths:
            if os.path.exists(path):
                ckpt_path = path
                break
        
        if not ckpt_path:
            dprint("[WARNING] SAM2 checkpoint not found")
            return None, None, head_box
        
        # Try to find config
        config_paths = [
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            os.path.join(os.path.dirname(__file__), '..', '..', 'cartoon-head-detection', 'sam2', 'sam2', 'configs', 'sam2.1', 'sam2.1_hiera_s.yaml')
        ]
        
        config_path = "configs/sam2.1/sam2.1_hiera_s.yaml"  # Use relative path for hydra
        
        sam2_model = build_sam2(config_path, ckpt_path, device=device)
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
        
        # Add negative points
        H, W = image_bgr.shape[:2]
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
        
        # Fill holes to make solid mask
        mask = fill_mask_holes(mask)
        
        # Create outline from filled mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = cv2.erode(mask, kernel, iterations=1)
        outline = cv2.subtract(mask, eroded)
        
        return mask, outline, head_box
        
    except Exception as e:
        dprint(f"[WARNING] SAM2 failed: {e}")
        return None, None, head_box

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
    
    # Progressive closing
    for kernel_size in [15, 25, 35]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Fill contour
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

# REMOVED: YOLO fallback is not needed and not working well

# =========================
# Image preprocessing (from original)
# =========================
def extract_lines(image_path, output_dir=None):
    """Extract lines using Anime2Sketch API, then skeletonize."""
    import requests
    import tempfile
    
    # Load original for background removal
    img_original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_original is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    dprint(f"[Extract] Loaded image: {image_path}, shape: {img_original.shape}")

    # Background removal if RGBA
    if len(img_original.shape) == 3 and img_original.shape[2] == 4:
        white_bg = np.ones((img_original.shape[0], img_original.shape[1], 3), dtype=np.uint8) * 255
        alpha = img_original[:, :, 3] / 255.0
        for c in range(3):
            white_bg[:, :, c] = (1 - alpha) * 255 + alpha * img_original[:, :, c]
        img = white_bg.astype(np.uint8)
        dprint(f"[Extract] Converted RGBA to RGB with white background")
    else:
        img = img_original[:, :, :3]
    
    # Call Anime2Sketch API to get sketch
    dprint("[Extract] Calling Anime2Sketch API...")
    url = "https://anime2sketch-968385204614.europe-west4.run.app/infer"
    
    try:
        # Create temp file for input
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_in:
            cv2.imwrite(tmp_in.name, img)
            tmp_in.flush()
            
            # Send to API
            with open(tmp_in.name, 'rb') as f:
                files = {'file': f}
                data = {'load_size': 512}
                response = requests.post(url, files=files, data=data, timeout=30)
            
            os.unlink(tmp_in.name)
        
        if response.status_code == 200:
            dprint(f"[Extract] Anime2Sketch API successful")
            
            # Save sketch and load as grayscale
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_sketch:
                tmp_sketch.write(response.content)
                tmp_sketch.flush()
                
                gray = cv2.imread(tmp_sketch.name, cv2.IMREAD_GRAYSCALE)
                
                # Save to debug folder if specified
                if output_dir:
                    cv2.imwrite(os.path.join(output_dir, "01_anime2sketch.jpg"), gray)
                
                os.unlink(tmp_sketch.name)
        else:
            dprint(f"[Extract] Anime2Sketch API failed with status {response.status_code}, using fallback")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    except Exception as e:
        dprint(f"[Extract] Anime2Sketch API error: {e}, using fallback")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    dprint(f"[Extract] Gray image stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.2f}")

    # Anime2Sketch already gives us thin lines, so let's preserve them
    # Use adaptive thresholding for better edge preservation
    black_threshold = 180  # Capture most lines including anti-aliased edges
    lines_binary = (gray < black_threshold).astype(np.uint8) * 255
    dprint(f"[Extract] Line pixels (gray<{black_threshold}): {np.count_nonzero(lines_binary)}")
    
    # Remove small noise components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lines_binary, connectivity=8)
    min_component_size = 15  # Lower threshold to keep fine details
    lines_cleaned = np.zeros_like(lines_binary, dtype=np.uint8)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_size:
            lines_cleaned[labels == i] = 255
    
    dprint(f"[Extract] After removing small components (<{min_component_size} px): {np.count_nonzero(lines_cleaned)}")
    
    # Only thin the thickest lines, preserve already-thin lines
    # Use distance transform to identify line thickness
    dist_transform = cv2.distanceTransform(lines_cleaned, cv2.DIST_L2, 5)
    
    # Separate thin and thick regions
    thin_lines = (dist_transform > 0) & (dist_transform <= 3)  # Already thin
    thick_lines = dist_transform > 3  # Need thinning
    
    dprint(f"[Extract] Thin lines (<=3px): {np.count_nonzero(thin_lines)}")
    dprint(f"[Extract] Thick lines (>3px): {np.count_nonzero(thick_lines)}")
    
    # Only skeletonize thick regions
    if np.any(thick_lines):
        thick_skeleton = skeletonize(thick_lines)
        # Combine: keep thin lines as-is, use skeleton for thick lines
        lines_bool = thin_lines | thick_skeleton
    else:
        lines_bool = thin_lines
    
    dprint(f"[Extract] Combined lines: {np.count_nonzero(lines_bool)}")
    
    # Convert to uint8
    lines = (lines_bool * 255).astype(np.uint8)
    
    # For debug visualization
    before_thin = np.count_nonzero(lines_cleaned)
    after_thin = np.count_nonzero(lines_bool)
    dprint(f"[Extract] Skeletonization: before={before_thin}, after={after_thin}")

    if DEBUG:
        save_dir = output_dir if output_dir else DEBUG_DIR
        ensure_dir(save_dir)
        cv2.imwrite(f"{save_dir}/00_original.png", img)
        cv2.imwrite(f"{save_dir}/01_sketch.png", gray)
        cv2.imwrite(f"{save_dir}/02_lines_binary.png", lines_binary)
        cv2.imwrite(f"{save_dir}/03_lines_cleaned.png", lines_cleaned)
        cv2.imwrite(f"{save_dir}/04_final_lines.png", lines)
        dprint(f"[Extract] Saved debug images to {save_dir}")
    
    return lines, img

# =========================
# Connected components & paths (from original)
# =========================
def get_all_components(lines):
    num_labels, labels = cv2.connectedComponents(lines, connectivity=8)
    components = []
    discarded = []
    dprint(f"[CC] num_labels={num_labels-1} (excluding background)")
    dprint(f"[CC] Input lines shape: {lines.shape}, non-zero pixels: {np.count_nonzero(lines)}")
    
    for label_id in range(1, num_labels):
        mask = (labels == label_id)
        size = int(mask.sum())
        points = np.argwhere(mask)
        
        if size >= 20:
            x1,y1,x2,y2 = component_bbox(points)
            cx, cy = bbox_center(x1,y1,x2,y2)
            components.append({
                'id': label_id,
                'points': points,
                'mask': mask,
                'size': size,
                'bbox': (x1,y1,x2,y2),
                'center': np.array([cy, cx])  # (y, x)
            })
            dprint(f"  [Comp {label_id}] KEPT size={size} bbox=({x1},{y1})-({x2},{y2}) center=({cx:.1f},{cy:.1f})")
        else:
            if size > 0:
                x1,y1,x2,y2 = component_bbox(points) if len(points) > 0 else (0,0,0,0)
                discarded.append({'id': label_id, 'size': size, 'bbox': (x1,y1,x2,y2)})
                if DEBUG_VERBOSE:
                    dprint(f"  [Comp {label_id}] DISCARDED size={size} bbox=({x1},{y1})-({x2},{y2}) (too small)")
    
    dprint(f"[CC] kept={len(components)} components (size>=20), discarded={len(discarded)} small components")
    
    if DEBUG_VERBOSE and discarded:
        dprint(f"[CC] Discarded component sizes: {[d['size'] for d in discarded[:10]]}...")
    
    return components

def create_path_graph(points):
    if len(points) < 2:
        return None, None

    point_dict = {tuple(p): i for i, p in enumerate(points)}
    G = nx.Graph()

    for i, p in enumerate(points):
        G.add_node(i, pos=tuple(p))
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbor = (p[0] + dy, p[1] + dx)
                if neighbor in point_dict:
                    j = point_dict[neighbor]
                    G.add_edge(i, j)

    return G, points

def find_paths_all_components(G):
    """Find paths for all connected components, not just the largest."""
    if not G or G.number_of_nodes() == 0:
        return []
    
    all_paths = []
    
    if nx.is_connected(G):
        # Single connected component
        path = find_single_path(G)
        if path:
            all_paths.append(path)
    else:
        # Multiple connected components
        ccs = list(nx.connected_components(G))
        ccs_sorted = sorted(ccs, key=len, reverse=True)
        
        dprint(f"[Path] Graph has {len(ccs)} connected components:")
        for i, cc in enumerate(ccs_sorted[:5]):  # Show top 5
            dprint(f"  CC{i+1}: {len(cc)} nodes")
        
        if len(ccs_sorted) > 5:
            remaining = sum(len(cc) for cc in ccs_sorted[5:])
            dprint(f"  ... and {len(ccs_sorted)-5} more CCs with {remaining} total nodes")
        
        # Process each connected component
        for i, cc in enumerate(ccs_sorted):
            if len(cc) < 2:  # Skip single nodes
                continue
            
            subgraph = G.subgraph(cc).copy()
            path = find_single_path(subgraph)
            if path:
                all_paths.append(path)
                if i < 3:  # Log first 3 components
                    dprint(f"  CC{i+1}: Found path with {len(path)} nodes")
    
    return all_paths

def find_single_path(G):
    """Find a single path through a connected graph."""
    if not G or G.number_of_nodes() == 0:
        return []
    
    odd_vertices = [v for v in G.nodes() if G.degree(v) % 2 == 1]

    if len(odd_vertices) == 0:
        try:
            path = list(nx.eulerian_circuit(G, source=list(G.nodes())[0]))
            return [edge[0] for edge in path] + [path[-1][1]]
        except Exception as e:
            pass
    elif len(odd_vertices) == 2:
        try:
            path = list(nx.eulerian_path(G, source=odd_vertices[0]))
            return [edge[0] for edge in path] + [path[-1][1]]
        except Exception as e:
            pass

    # Greedy DFS fallback
    visited = set()
    path = []
    stack = [list(G.nodes())[0]]
    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v)
            path.append(v)
            neighbors = list(G.neighbors(v))
            neighbors.sort(key=lambda n: len(set(G.neighbors(n)) - visited))
            stack.extend(neighbors[::-1])

    return path

def find_path(G):
    """Legacy function for compatibility - returns single largest path."""
    paths = find_paths_all_components(G)
    return paths[0] if paths else []

def draw_path(points, path_indices, image, color, thickness=2):
    if len(path_indices) < 2:
        return image
    for i in range(len(path_indices) - 1):
        pt1 = points[path_indices[i]]
        pt2 = points[path_indices[i + 1]]
        if abs(pt1[0] - pt2[0]) <= 1 and abs(pt1[1] - pt2[1]) <= 1:
            cv2.line(image, (pt1[1], pt1[0]), (pt2[1], pt2[0]), color, thickness)
    return image

# =========================
# Component splitting and classification (from original)
# =========================
def split_component_by_mask(comp, region_bool, id_suffix):
    """Split component into inside/outside parts based on mask"""
    H, W = region_bool.shape
    pts = comp['points']
    inside_flags = [region_bool[p[0], p[1]] for p in pts]
    inside_points = pts[np.where(inside_flags)[0]] if len(pts) else np.empty((0,2), dtype=np.int64)
    outside_points = pts[np.where(np.logical_not(inside_flags))[0]] if len(pts) else np.empty((0,2), dtype=np.int64)

    new = []

    def make_comp(points, tag):
        if len(points) < 10:
            return
        x1,y1,x2,y2 = component_bbox(points)
        cx, cy = bbox_center(x1,y1,x2,y2)
        mask = np.zeros((H, W), dtype=bool)
        mask[points[:,0], points[:,1]] = True
        new.append({
            'id': f"{comp['id']}_{id_suffix}_{tag}",
            'points': points,
            'mask': mask,
            'size': int(len(points)),
            'bbox': (x1,y1,x2,y2),
            'center': np.array([cy, cx])
        })

    make_comp(inside_points, "IN")
    make_comp(outside_points, "OUT")
    return new

def _dilate(mask_uint8, ksize=3, iters=1):
    if mask_uint8 is None:
        return None
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(mask_uint8, kernel, iterations=iters)

def classify_split_components(components, head_region_bool, head_outline_mask,
                              inside_thresh=0.75, outline_ratio_thresh=0.10, min_outline_px=15):
    """Classify split components into face_outline, face_interior, or body"""
    outline_dil = _dilate(head_outline_mask, 3, 1) if head_outline_mask is not None else None
    outline_bool = (outline_dil > 0) if outline_dil is not None else np.zeros_like(head_region_bool)

    face_outline_parts = []
    face_interior_parts = []
    body_parts = []

    for comp in components:
        # Check if original component overlaps with head outline
        comp_mask = comp['mask']
        comp_size = comp['size']
        outline_overlap_orig = int((comp_mask & outline_bool).sum()) if comp_size > 0 else 0
        outline_ratio_orig = outline_overlap_orig / float(comp_size) if comp_size > 0 else 0.0
        
        # Split component
        parts = split_component_by_mask(comp, head_region_bool, id_suffix="SPLIT")
        
        for part in parts:
            m = part['mask']; size = part['size']
            inside_ratio = (m & head_region_bool).sum() / float(size) if size > 0 else 0.0
            outline_overlap = int((m & outline_bool).sum())
            outline_ratio = outline_overlap / float(size) if size > 0 else 0.0

            # Check if this is part of the head outline
            # A component is face_outline if it significantly overlaps the outline mask
            # OR if the original component was on the outline (handles split components)
            if outline_ratio >= outline_ratio_thresh or outline_overlap >= min_outline_px or \
               (outline_ratio_orig >= outline_ratio_thresh and "_OUT" in part['id']):
                face_outline_parts.append(part)
                dprint(f"[Class] {part['id']} -> FACE_OUTLINE")
            elif inside_ratio >= inside_thresh:
                # Inside head region and not outline
                face_interior_parts.append(part)
                dprint(f"[Class] {part['id']} -> FACE_INTERIOR")
            else:
                body_parts.append(part)
                dprint(f"[Class] {part['id']} -> BODY")

    return face_outline_parts, face_interior_parts, body_parts

# =========================
# Main animation function
# =========================
def create_drawing_animation(lines, original_img, image_path, output_prefix="test_output/drawing", output_dir=None):
    H, W = lines.shape[:2]
    
    # Detect head using best available method
    dprint("[Head] Detecting head region...")
    
    # Get the original image for detection
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        img_bgr = original_img
    
    # Get image name for debug
    import os.path as osp
    img_name = osp.splitext(osp.basename(image_path))[0]
    
    # First check if we have pre-generated SAM2 masks (highest quality)
    mask_paths = [
        f"cartoon-test/{img_name}_head_mask_solid.png",
        f"cartoon-test/{img_name}_head_mask.png",
    ]
    
    head_filled_mask = None
    head_outline_mask = None
    head_box_refined = None
    
    for mask_path in mask_paths:
        if os.path.exists(mask_path):
            dprint(f"[Head] Using high-quality SAM2 mask: {mask_path}")
            head_filled_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if head_filled_mask is not None:
                # Ensure it's the right size
                if head_filled_mask.shape != (H, W):
                    head_filled_mask = cv2.resize(head_filled_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                
                # Create outline
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                eroded = cv2.erode(head_filled_mask, kernel, iterations=1)
                head_outline_mask = cv2.subtract(head_filled_mask, eroded)
                
                # Find bounding box
                coords = np.column_stack(np.where(head_filled_mask > 0))
                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    head_box_refined = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                break
    
    # If no pre-generated mask, try live detection
    if head_filled_mask is None:
        dprint("[Head] No pre-generated mask found, attempting live detection")
        head_filled_mask, head_outline_mask, head_box_refined = detect_head_with_sam2(img_bgr, img_name)
    
    # If still no mask, use color-based detection with solid mask generation
    if head_filled_mask is None:
        dprint("[Head] Using color-based head detection with solid mask generation")
        head_filled_mask, head_outline_mask, head_box_refined = detect_head_color_based(img_bgr)
    
    if head_filled_mask is None:
        dprint("[Head] No head detected, treating entire image as body")
        # Create empty masks to proceed without head detection
        head_filled_mask = np.zeros((H, W), dtype=np.uint8)
        head_outline_mask = np.zeros((H, W), dtype=np.uint8)
        head_region = np.zeros((H, W), dtype=bool)
    
    dprint(f"[Head] Detection successful, box: {head_box_refined}")
    
    # Build robust head region only if we have masks
    if head_filled_mask is not None and np.any(head_filled_mask > 0):
        outline_dil = _dilate(head_outline_mask, ksize=3, iters=1) if head_outline_mask is not None else np.zeros_like(head_filled_mask)
        head_region = ((head_filled_mask > 0) | (outline_dil > 0))
    else:
        # No head region if no mask
        head_region = np.zeros((H, W), dtype=bool)
    
    if DEBUG:
        save_dir = output_dir if output_dir else DEBUG_DIR
        ensure_dir(f"{save_dir}/debug_head")
        save_mask(f"{save_dir}/debug_head/head_filled_mask.png", head_filled_mask)
        save_mask(f"{save_dir}/debug_head/head_outline_mask.png", head_outline_mask if head_outline_mask is not None else np.zeros_like(head_filled_mask))
        save_mask(f"{save_dir}/debug_head/head_region_combined.png", (head_region.astype(np.uint8)*255))
        
        # Save overlays on original image for visualization
        overlay_filled = original_img.copy()
        overlay_filled[head_filled_mask > 0] = (overlay_filled[head_filled_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        cv2.imwrite(f"{save_dir}/debug_head/head_filled_overlay.png", overlay_filled)
        
        # Save the actual face+hair mask (what we use for classification)
        face_hair_mask = head_filled_mask.copy()
        cv2.imwrite(f"{save_dir}/debug_head/face_hair_mask_final.png", face_hair_mask)
        
        dprint(f"[Head] Saved head masks to {save_dir}/debug_head/")
        dprint(f"[Head] Face+hair mask shape: {face_hair_mask.shape}, pixels: {np.count_nonzero(face_hair_mask)}")
    
    # Get all components
    all_components = get_all_components(lines)
    
    # Save component visualization
    if DEBUG:
        save_dir = output_dir if output_dir else DEBUG_DIR
        comp_viz = np.zeros((H, W, 3), dtype=np.uint8)
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
        for i, comp in enumerate(all_components):
            color = colors[i % len(colors)]
            comp_viz[comp['mask']] = color
        cv2.imwrite(f"{save_dir}/06_all_components.png", comp_viz)
        dprint(f"[Components] Saved visualization with {len(all_components)} components")
    
    # Split and classify components
    dprint(f"[Split] Starting component splitting with {len(all_components)} components")
    face_outline_parts, face_interior_parts, body_parts = classify_split_components(
        all_components, head_region, head_outline_mask,
        inside_thresh=0.75, outline_ratio_thresh=0.10, min_outline_px=15
    )
    
    dprint(f"[Split] Results: face_outline={len(face_outline_parts)}, face_interior={len(face_interior_parts)}, body={len(body_parts)}")
    
    # Visualize classified components
    if DEBUG:
        save_dir = output_dir if output_dir else DEBUG_DIR
        class_viz = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Face outline in red
        for part in face_outline_parts:
            class_viz[part['mask']] = (0, 0, 255)
        
        # Face interior in green
        for part in face_interior_parts:
            class_viz[part['mask']] = (0, 255, 0)
        
        # Body in blue
        for part in body_parts:
            class_viz[part['mask']] = (255, 0, 0)
        
        cv2.imwrite(f"{save_dir}/07_classified_components.png", class_viz)
        dprint(f"[Split] Saved classified visualization (red=outline, green=interior, blue=body)")
    
    # Order: outline (largest first) -> interior (top→bottom) -> body (top→bottom)
    face_outline_parts.sort(key=lambda c: c['size'], reverse=True)
    face_interior_parts.sort(key=lambda c: c['center'][0])
    
    # Sort body parts by distance from head region
    # Find the lowest point of head components (neck connection point)
    head_bottom_y = 0
    head_center_x = W // 2
    
    # Get bottom of head from all face components
    all_face_parts = face_outline_parts + face_interior_parts
    if all_face_parts:
        # Find the lowest y-coordinate (bottom) of all face parts
        for comp in all_face_parts:
            x1, y1, x2, y2 = comp['bbox']
            if y2 > head_bottom_y:
                head_bottom_y = y2
                # Update center X based on this component
                head_center_x = (x1 + x2) // 2
    
    # Sort body parts by distance from the bottom of head (neck area)
    def distance_from_head_bottom(comp):
        cx, cy = comp['center']
        # Calculate distance from the bottom-center of head region
        # Prefer components that are below and close to head
        dx = abs(cx - head_center_x)
        dy = max(0, cy - head_bottom_y)  # Only positive if below head
        # Weight vertical distance less to prefer neck/collar area
        return dx + dy * 0.5
    
    body_parts.sort(key=distance_from_head_bottom)
    
    ordered = [('face_outline', c) for c in face_outline_parts] \
            + [('face_interior', c) for c in face_interior_parts] \
            + [('body', c) for c in body_parts]
    
    dprint("[Order] Final draw order:")
    for i, (tag, comp) in enumerate(ordered):
        x1,y1,x2,y2 = comp['bbox']
        dprint(f"  {i+1:03d}. {tag} id={comp['id']} size={comp['size']}")
    
    # Draw with Euler paths
    frames = []
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    
    # Old draw_component removed - using draw_component_from_nearest below
    def draw_component_old(tag, comp, canvas, frames, comp_index, total_comps, output_dir=None):
        G, points = create_path_graph(comp['points'])
        if G is None:
            dprint(f"[Draw] WARNING: No graph created for {tag} {comp['id']}")
            return False
        
        if G.number_of_nodes() == 0:
            dprint(f"[Draw] WARNING: Empty graph for {tag} {comp['id']}")
            return False
        
        # Check connectivity before finding paths
        is_connected = nx.is_connected(G)
        if not is_connected:
            ccs = list(nx.connected_components(G))
            dprint(f"[Draw] Component {tag} {comp['id']} has {len(ccs)} disconnected parts")
        
        # Get paths for ALL connected components
        all_paths = find_paths_all_components(G)
        if not all_paths:
            dprint(f"[Draw] WARNING: No paths found for {tag} {comp['id']}")
            return False
        
        size = comp['size']
        if size < 100:
            num_steps = 5
        elif size < 500:
            num_steps = 15
        else:
            num_steps = 30
        
        if tag in ('face_outline', 'face_interior'):
            num_steps = max(5, num_steps - 5)
        
        color = (0, 0, 0)
        
        # Calculate total path length
        total_path_len = sum(len(p) for p in all_paths)
        dprint(f"[Draw] [{comp_index}/{total_comps}] {tag} {comp['id']} size={size} points={len(points)} total_path_len={total_path_len} paths={len(all_paths)} steps={num_steps}")
        
        # Draw each path
        for path_idx, path in enumerate(all_paths):
            if len(path) < 2:
                continue
            
            # Adjust steps based on path size
            path_steps = max(1, (num_steps * len(path)) // total_path_len) if total_path_len > 0 else 1
            step = max(1, len(path) // path_steps)
            
            if path_idx < 3 or (path_idx == len(all_paths) - 1 and len(all_paths) > 3):
                dprint(f"  Path {path_idx+1}/{len(all_paths)}: {len(path)} nodes, {path_steps} steps")
            
            # Animate this path
            for i in range(0, len(path), step):
                sub = path[:i + step]
                frame = canvas.copy()
                
                # Draw all previous paths
                for prev_path in all_paths[:path_idx]:
                    draw_path(points, prev_path, frame, color, thickness=2)
                
                # Draw current path progress
                draw_path(points, sub, frame, color, thickness=2)
                frames.append(frame)
            
            # Draw final path on canvas
            draw_path(points, path, canvas, color, thickness=2)
        
        dprint(f"[Draw] Completed {tag} {comp['id']}: drew {total_path_len}/{len(points)} points across {len(all_paths)} paths")
        return True
    
    # Track drawing progress and last position
    total_points_to_draw = sum(comp['size'] for _, comp in ordered)
    total_points_drawn = 0
    components_drawn = 0
    components_failed = 0
    last_draw_position = None  # Track where we finished drawing
    
    def draw_component_from_nearest(tag, comp, canvas, frames, comp_index, total_comps, last_pos, output_dir=None):
        """Modified draw function that starts from nearest point to last position"""
        G, points = create_path_graph(comp['points'])
        if G is None or G.number_of_nodes() == 0:
            return False, last_pos
        
        # Check connectivity
        is_connected = nx.is_connected(G)
        if not is_connected:
            ccs = list(nx.connected_components(G))
            dprint(f"[Draw] Component {tag} {comp['id']} has {len(ccs)} disconnected parts")
        
        # Get paths for ALL connected components
        all_paths = find_paths_all_components(G)
        if not all_paths:
            return False, last_pos
        
        # If we have a last position and this is a body part, reorder paths to start from nearest
        if last_pos is not None and tag == 'body' and len(all_paths) > 0:
            # Find the path that starts closest to last position
            def path_start_distance(path):
                if len(path) == 0:
                    return float('inf')
                start_point = points[path[0]]
                return np.linalg.norm(np.array(start_point) - np.array(last_pos))
            
            # Sort paths by distance from last position
            all_paths.sort(key=path_start_distance)
            dprint(f"[Draw] Reordered paths to start from nearest to last position")
        
        size = comp['size']
        if size < 100:
            num_steps = 5
        elif size < 500:
            num_steps = 15
        else:
            num_steps = 30
        
        if tag in ('face_outline', 'face_interior'):
            num_steps = max(5, num_steps - 5)
        
        color = (0, 0, 0)
        
        # Calculate total path length
        total_path_len = sum(len(p) for p in all_paths)
        dprint(f"[Draw] [{comp_index}/{total_comps}] {tag} {comp['id']} size={size} points={len(points)} total_path_len={total_path_len} paths={len(all_paths)} steps={num_steps}")
        
        # Draw each path and track last position
        final_position = last_pos
        for path_idx, path in enumerate(all_paths):
            if len(path) < 2:
                continue
            
            # Update final position to end of this path
            if len(path) > 0:
                final_position = points[path[-1]]
            
            # Adjust steps based on path size
            path_steps = max(1, (num_steps * len(path)) // total_path_len) if total_path_len > 0 else 1
            step = max(1, len(path) // path_steps)
            
            if path_idx < 3 or (path_idx == len(all_paths) - 1 and len(all_paths) > 3):
                dprint(f"  Path {path_idx+1}/{len(all_paths)}: {len(path)} nodes, {path_steps} steps")
            
            # Animate this path
            for i in range(0, len(path), step):
                sub = path[:i + step]
                frame = canvas.copy()
                
                # Draw all previous paths
                for prev_path in all_paths[:path_idx]:
                    draw_path(points, prev_path, frame, color, thickness=2)
                
                # Draw current path progress
                draw_path(points, sub, frame, color, thickness=2)
                frames.append(frame)
            
            # Draw final path on canvas
            draw_path(points, path, canvas, color, thickness=2)
        
        dprint(f"[Draw] Completed {tag} {comp['id']}: drew {total_path_len}/{len(points)} points across {len(all_paths)} paths")
        return True, final_position
    
    for idx, (tag, comp) in enumerate(ordered, start=1):
        dprint(f"\n[Draw] Component {idx}/{len(ordered)} -> {tag} {comp['id']}")
        success, last_draw_position = draw_component_from_nearest(tag, comp, canvas, frames, idx, len(ordered), last_draw_position, output_dir)
        
        if success:
            components_drawn += 1
            total_points_drawn += comp['size']
        else:
            components_failed += 1
            dprint(f"[Draw] FAILED to draw {tag} {comp['id']} with {comp['size']} points")
    
    dprint(f"\n[Draw Summary] Components: {components_drawn}/{len(ordered)} drawn, {components_failed} failed")
    dprint(f"[Draw Summary] Points: ~{total_points_drawn}/{total_points_to_draw} drawn ({100*total_points_drawn/total_points_to_draw:.1f}%)")
    
    # Save intermediate canvases
    if DEBUG:
        save_dir = output_dir if output_dir else DEBUG_DIR
        cv2.imwrite(f"{save_dir}/08_final_drawing.png", canvas)
        
        # Compare with original lines
        comparison = np.zeros((H, W, 3), dtype=np.uint8)
        comparison[:,:,0] = lines  # Original in red channel
        comparison[:,:,1] = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)  # Drawn in green channel
        cv2.imwrite(f"{save_dir}/09_comparison_original_vs_drawn.png", comparison)
        dprint(f"[Draw] Saved comparison (red=original, green=drawn, yellow=overlap)")
    
    # Save results
    dprint(f"[Save] Total frames: {len(frames)}")
    ensure_dir(os.path.dirname(output_prefix) or ".")
    
    if frames:
        # First save as temporary file with OpenCV
        temp_video = f"{output_prefix}_temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for temp
        out = cv2.VideoWriter(temp_video, fourcc, 60.0, (W, H))  # 60 FPS for 6x faster playback
        for frame in frames:
            # Write each frame only once (was 2x before, now 1x at double FPS = same duration but 6x faster drawing)
            out.write(frame)
        for _ in range(60):  # 1 second pause at end (60 frames at 60fps)
            out.write(frames[-1] if frames else canvas)
        for _ in range(60):  # Show original for 1 second
            out.write(original_img)
        out.release()
        
        # Convert to H.264 using ffmpeg for compatibility
        final_video = f"{output_prefix}_animation.mp4"
        try:
            cmd = [
                'ffmpeg', '-y', '-i', temp_video,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', '-crf', '23',
                final_video
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            os.remove(temp_video)  # Clean up temp file
            dprint(f"[Save] Video: {final_video} (H.264 encoded)")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If ffmpeg fails, rename temp to final
            os.rename(temp_video, final_video)
            dprint(f"[Save] Video: {final_video} (mp4v codec)")
    
    cv2.imwrite(f"{output_prefix}_final.png", canvas)
    
    # Debug visualization
    debug_viz = original_img.copy()
    # Overlay head region in semi-transparent red
    debug_viz[head_region] = (debug_viz[head_region] * 0.6 + np.array([0, 0, 255]) * 0.4).astype(np.uint8)
    
    if head_box_refined is not None:
        x0, y0, x1, y1 = head_box_refined.astype(int)
        cv2.rectangle(debug_viz, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(debug_viz, "HEAD (SAM2)", (x0, max(0, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite(f"{output_prefix}_debug.png", debug_viz)
    dprint(f"[Save] Debug viz: {output_prefix}_debug.png")
    
    return frames

# =========================
# CLI
# =========================
def main():
    if len(sys.argv) < 2:
        print("Usage: python stroke_traversal_closed.py image.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Create output directory based on image name
    import os.path as osp
    img_name = osp.splitext(osp.basename(image_path))[0]
    output_dir = f"{DEBUG_DIR}/{img_name}_debug"
    ensure_dir(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    lines, original = extract_lines(image_path, output_dir=output_dir)
    
    frames = create_drawing_animation(lines, original, image_path, 
                                    output_prefix=f"{output_dir}/drawing",
                                    output_dir=output_dir)
    
    print(f"\n{'='*60}")
    print(f"Done! Generated {len(frames)} frames")
    print(f"Output: {output_dir}/drawing_animation.mp4")
    print(f"Debug images saved in: {output_dir}/")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()