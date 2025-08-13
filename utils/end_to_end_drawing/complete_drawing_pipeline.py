#!/usr/bin/env python3
"""
Complete end-to-end drawing pipeline that properly uses Euler path traversal
from stroke_traversal_closed.py for human-like continuous drawing.

Pipeline:
1. Call Anime2Sketch API for clean sketch
2. Skeletonize to get thin lines
3. Extract and classify components (face_outline, face_interior, body)
4. For each component, use Euler path traversal for continuous drawing
5. Create video with natural drawing motion
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
import requests
import os
import sys
import networkx as nx
from scipy.spatial.distance import cdist
from PIL import Image
import io

# Add path to import stroke_traversal module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'draw-euler'))

# Try to import SAM2 API client first
try:
    from sam2_api_client import detect_head_with_sam2_api
    SAM2_API_AVAILABLE = True
    print("SAM2 API client available")
except ImportError:
    SAM2_API_AVAILABLE = False
    print("SAM2 API client not available")

# Try to import local SAM2 head detector as fallback
if not SAM2_API_AVAILABLE:
    try:
        from sam2_head_detector_simple import detect_head_with_sam2_simple, initialize_sam2_simple
        SAM2_AVAILABLE = True
        SAM2_SIMPLE = True
        print("SAM2 simplified head detector available")
    except ImportError:
        try:
            from sam2_head_detector import detect_head_with_sam2, initialize_sam2
            SAM2_AVAILABLE = True
            SAM2_SIMPLE = False
            print("SAM2 head detector available")
        except ImportError:
            SAM2_AVAILABLE = False
            print("SAM2 head detector not available, will use color-based fallback")
else:
    SAM2_AVAILABLE = False

def remove_background(input_path, output_path):
    """Remove background using rembg package"""
    try:
        from rembg import remove
        print("Removing background...")
        
        # Read input image
        with open(input_path, 'rb') as input_file:
            input_data = input_file.read()
        
        # Remove background
        output_data = remove(input_data)
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(output_data))
        
        # Create white background
        if img.mode == 'RGBA':
            # Create white background image
            white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            # Paste image on white background using alpha channel
            white_bg.paste(img, mask=img.split()[3])
            # Convert to RGB
            white_bg = white_bg.convert('RGB')
        else:
            white_bg = img
        
        # Save with white background
        white_bg.save(output_path, 'PNG')
        print(f"Background removed and saved to {output_path}")
        return True
        
    except ImportError:
        print("Warning: rembg not installed. Install with: pip install rembg")
        print("Continuing without background removal...")
        # Just copy the file
        img = cv2.imread(input_path)
        cv2.imwrite(output_path, img)
        return False

def call_anime2sketch_api(input_path, output_path):
    """Call Anime2Sketch API to get sketch"""
    print("Calling Anime2Sketch API...")
    
    url = "https://anime2sketch-968385204614.europe-west4.run.app/infer"
    
    with open(input_path, 'rb') as f:
        files = {'file': f}
        data = {'load_size': 512}
        
        response = requests.post(url, files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as out:
                out.write(response.content)
            print(f"Sketch saved to {output_path}")
            return True
        else:
            print(f"API error: {response.status_code}")
            return False

def skeletonize_sketch(sketch_path, skeleton_path):
    """Convert sketch to thin lines"""
    print("Skeletonizing sketch...")
    
    # Read sketch
    img = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    
    # Threshold to get black lines
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    
    # Invert so lines are white
    binary = 255 - binary
    
    # Clean up noise
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Skeletonize
    skeleton = skeletonize(binary > 0)
    
    # Convert back to image (black lines on white)
    result = (1 - skeleton) * 255
    result = result.astype(np.uint8)
    
    cv2.imwrite(skeleton_path, result)
    print(f"Skeleton saved to {skeleton_path}")
    return result

def build_graph_from_skeleton(skeleton_mask):
    """
    Build a graph from skeleton pixels for Euler path traversal
    Based on stroke_traversal_closed.py logic
    """
    # Get all skeleton pixels (numpy returns y,x format)
    skeleton_points = np.argwhere(skeleton_mask)
    
    if len(skeleton_points) == 0:
        return None, []
    
    # Build graph where each pixel is a node
    G = nx.Graph()
    
    # Create a dictionary for fast lookup (convert to (y,x) tuples for consistency)
    point_set = set(map(tuple, skeleton_points))
    
    # Add nodes and edges
    for point in skeleton_points:
        y, x = point
        G.add_node((x, y))  # Store as (x,y) for OpenCV drawing
        
        # Check 8-connected neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                    
                neighbor_yx = (y + dy, x + dx)
                if neighbor_yx in point_set:
                    # Add edge using (x,y) format
                    G.add_edge((x, y), (x + dx, y + dy))
    
    return G, skeleton_points

def find_euler_path(G):
    """
    Find Eulerian path/circuit in the graph for continuous drawing
    Returns ordered list of points to draw
    """
    if G is None or len(G.nodes()) == 0:
        return []
    
    # Check for Eulerian path/circuit
    odd_degree_nodes = [n for n in G.nodes() if G.degree(n) % 2 == 1]
    
    try:
        if len(odd_degree_nodes) == 0:
            # Eulerian circuit exists - can start anywhere
            path = list(nx.eulerian_circuit(G, source=list(G.nodes())[0]))
            # Convert edges to continuous point list
            if path:
                points = [path[0][0]]
                for edge in path:
                    points.append(edge[1])
                return points
                
        elif len(odd_degree_nodes) == 2:
            # Eulerian path exists - must start at odd degree node
            path = list(nx.eulerian_path(G, source=odd_degree_nodes[0]))
            if path:
                points = [path[0][0]]
                for edge in path:
                    points.append(edge[1])
                return points
    except:
        pass
    
    # Fallback: greedy traversal if no Eulerian path
    visited = set()
    path = []
    
    # Start from a node with odd degree if available, else any node
    start = odd_degree_nodes[0] if odd_degree_nodes else list(G.nodes())[0]
    stack = [start]
    
    while stack:
        node = stack[-1]
        if node not in visited:
            visited.add(node)
            path.append(node)
        
        # Find unvisited neighbor
        unvisited_neighbors = [n for n in G.neighbors(node) if n not in visited]
        if unvisited_neighbors:
            stack.append(unvisited_neighbors[0])
        else:
            stack.pop()
    
    return path

def detect_head_from_original(original_img):
    """
    Detect head region using SAM2 if available, otherwise color-based segmentation
    Returns both bounding box and actual head mask
    """
    H, W = original_img.shape[:2]
    
    # Try SAM2 API first if available
    if SAM2_API_AVAILABLE:
        print("Using SAM2 API for head detection...")
        solid_mask, head_box = detect_head_with_sam2_api(original_img, expand_ratio=1.2)
        
        if solid_mask is not None and head_box is not None:
            # SAM2 API succeeded
            x0, y0, x1, y1 = head_box
            # Add padding to bbox for visualization
            padding = 20
            x0 = max(0, x0 - padding)
            y0 = max(0, y0 - padding)
            x1 = min(W, x1 + padding)
            y1 = min(H, y1 + padding)
            
            print(f"SAM2 API detected head successfully")
            return (int(x0), int(y0), int(x1), int(y1)), solid_mask
        else:
            print("SAM2 API failed, falling back to local SAM2")
    
    # Try local SAM2 if available
    if SAM2_AVAILABLE:
        print("Using local SAM2 for head detection...")
        if SAM2_SIMPLE:
            solid_mask, head_box = detect_head_with_sam2_simple(original_img, debug_name="image")
            outline_mask = None  # Simplified version doesn't return outline
        else:
            solid_mask, outline_mask, head_box = detect_head_with_sam2(original_img, debug_name="image")
        
        if solid_mask is not None and head_box is not None:
            # SAM2 succeeded
            x0, y0, x1, y1 = head_box
            # Add padding to bbox for visualization
            padding = 20
            x0 = max(0, x0 - padding)
            y0 = max(0, y0 - padding)
            x1 = min(W, x1 + padding)
            y1 = min(H, y1 + padding)
            
            print(f"Local SAM2 detected head successfully")
            return (int(x0), int(y0), int(x1), int(y1)), solid_mask
        else:
            print("Local SAM2 failed, falling back to color-based detection")
    
    # Fallback to color-based detection
    print("Using color-based head detection...")
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    
    # Define multiple color ranges for different skin tones and head colors
    masks = []
    
    # Pink/Red (for robot-like characters)
    lower_pink = np.array([160, 50, 50])
    upper_pink = np.array([180, 255, 255])
    masks.append(cv2.inRange(hsv, lower_pink, upper_pink))
    
    lower_pink2 = np.array([0, 50, 50])
    upper_pink2 = np.array([20, 255, 255])
    masks.append(cv2.inRange(hsv, lower_pink2, upper_pink2))
    
    # Skin tones - beige/peach/orange range
    # These cover most human skin colors
    lower_skin1 = np.array([0, 20, 70])
    upper_skin1 = np.array([20, 150, 255])
    masks.append(cv2.inRange(hsv, lower_skin1, upper_skin1))
    
    # More skin tones - yellowish/brownish
    lower_skin2 = np.array([15, 30, 50])
    upper_skin2 = np.array([30, 150, 255])
    masks.append(cv2.inRange(hsv, lower_skin2, upper_skin2))
    
    # Brown hair/skin
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([20, 255, 200])
    masks.append(cv2.inRange(hsv, lower_brown, upper_brown))
    
    # Combine all masks
    color_mask = masks[0]
    for mask in masks[1:]:
        color_mask = cv2.bitwise_or(color_mask, mask)
    
    # Clean up with morphology
    kernel = np.ones((5,5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(color_mask, connectivity=8)
    
    # Find best head candidate
    best_head = None
    best_score = 0
    
    for label in range(1, num_labels):
        mask = (labels == label)
        area = np.sum(mask)
        
        if area < 1000:
            continue
        
        points = np.argwhere(mask)
        y_min, x_min = points.min(axis=0)
        y_max, x_max = points.max(axis=0)
        
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        # Score based on position, size, and shape
        upper_score = max(0, 1 - (center_y / H))
        size_score = min(1, area / (H * W * 0.1))
        aspect = width / height if height > 0 else 0
        aspect_score = 1 - abs(1 - aspect) if aspect > 0 else 0
        
        total_score = upper_score * 2 + size_score + aspect_score
        
        if total_score > best_score:
            best_score = total_score
            best_head = {'bbox': (x_min, y_min, x_max, y_max), 'mask': mask}
    
    if best_head:
        # Create full-size mask
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[best_head['mask']] = 255
        
        # More conservative dilation - just enough to include facial features
        # Use smaller kernel and fewer iterations
        kernel = np.ones((7, 7), np.uint8)
        dilated_mask = cv2.dilate(full_mask, kernel, iterations=1)
        
        # Fill only interior holes (not extending outward)
        # Use flood fill from outside to identify exterior, then invert
        h, w = dilated_mask.shape
        flood_mask = np.zeros((h+2, w+2), np.uint8)
        filled = dilated_mask.copy()
        cv2.floodFill(filled, flood_mask, (0, 0), 255)
        filled_inv = cv2.bitwise_not(filled)
        final_mask = dilated_mask | filled_inv
        
        # Add padding to bbox for visualization
        x_min, y_min, x_max, y_max = best_head['bbox']
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(W, x_max + padding)
        y_max = min(H, y_max + padding)
        return (x_min, y_min, x_max, y_max), final_mask
    
    # Fallback to simple upper region
    return (int(W * 0.2), 0, int(W * 0.8), int(H * 0.4)), None

def extract_and_classify_components(skeleton_img, original_img, output_dir=None):
    """
    Extract components and classify them as face_outline, face_interior, or body
    Uses original image for better head detection
    """
    print("Extracting and classifying components...")
    
    # Detect head region from original image (returns bbox and mask)
    head_box, head_mask = detect_head_from_original(original_img)
    x0, y0, x1, y1 = head_box
    print(f"Detected head box: ({x0},{y0})-({x1},{y1})")
    
    # Create and save head visualization if output_dir provided
    if output_dir:
        # Create visualization showing detected head pixels
        viz = original_img.copy()
        
        # Draw head box in green
        cv2.rectangle(viz, (x0, y0), (x1, y1), (0, 255, 0), 3)
        
        # Highlight actual head mask (not just box)
        if head_mask is not None:
            overlay = viz.copy()
            head_pixels = np.where(head_mask > 0)
            overlay[head_pixels] = [0, 0, 255]
            viz = cv2.addWeighted(viz, 0.5, overlay, 0.5, 0)
        else:
            # Fallback to box highlighting
            overlay = viz.copy()
            overlay[y0:y1, x0:x1] = cv2.addWeighted(
                overlay[y0:y1, x0:x1], 0.3,
                np.full_like(overlay[y0:y1, x0:x1], [0, 0, 255]), 0.7,
                0
            )
            viz = cv2.addWeighted(viz, 0.5, overlay, 0.5, 0)
        
        # Add text
        cv2.putText(viz, "Detected Head Region", (x0, y0 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Also create a view showing skeleton with head mask
        skeleton_viz = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(skeleton_viz, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
        # Highlight only skeleton pixels that overlap with actual head mask
        lines_inv = 255 - skeleton_img
        if head_mask is not None:
            # Use actual head mask to filter skeleton pixels
            head_skeleton_mask = (lines_inv > 0) & (head_mask > 0)
            head_pixels = np.where(head_skeleton_mask)
        else:
            # Fallback to box
            head_region_mask = np.zeros_like(lines_inv)
            head_region_mask[y0:y1, x0:x1] = lines_inv[y0:y1, x0:x1]
            head_pixels = np.where(head_region_mask > 0)
        
        # Color head components red in skeleton viz
        skeleton_viz[head_pixels] = [0, 0, 255]
        
        # Create side-by-side comparison
        H, W = viz.shape[:2]
        comparison = np.hstack([viz, skeleton_viz])
        
        # Save visualizations
        head_viz_path = os.path.join(output_dir, "head_detection_viz.png")
        cv2.imwrite(head_viz_path, comparison)
        print(f"Saved head detection visualization: {head_viz_path}")
        
        # Open the visualization
        os.system(f'open "{head_viz_path}"')
    
    # Invert so lines are white
    lines = 255 - skeleton_img
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(lines, connectivity=8)
    print(f"Found {num_labels - 1} connected components")
    
    H, W = skeleton_img.shape[:2]
    
    face_outline_parts = []
    face_interior_parts = []
    body_parts = []
    
    for label in range(1, num_labels):
        # Get component mask
        component_mask = (labels == label).astype(np.uint8) * 255
        points = np.argwhere(labels == label)
        
        if len(points) < 10:  # Skip tiny components
            continue
        
        # Build graph for this component
        G, _ = build_graph_from_skeleton(component_mask > 0)
        
        # Find Euler path for continuous drawing
        euler_path = find_euler_path(G)
        
        if not euler_path:
            continue
        
        # Calculate component properties
        ys, xs = points[:, 0], points[:, 1]
        comp_center_x = (xs.min() + xs.max()) / 2
        comp_center_y = (ys.min() + ys.max()) / 2
        comp_size = len(points)
        
        # Calculate overlap with head MASK (not just box)
        if head_mask is not None:
            # Use actual head mask for precise overlap
            head_pixels = []
            for y, x in points:
                if head_mask[y, x] > 0:
                    head_pixels.append([y, x])
            head_pixels = np.array(head_pixels) if head_pixels else np.array([])
            overlap_ratio = len(head_pixels) / len(points) if len(points) > 0 else 0
        else:
            # Fallback to box-based overlap
            head_pixels = points[(points[:, 0] >= y0) & (points[:, 0] < y1) & 
                                (points[:, 1] >= x0) & (points[:, 1] < x1)]
            overlap_ratio = len(head_pixels) / len(points) if len(points) > 0 else 0
        
        component = {
            'id': label,
            'mask': component_mask,
            'euler_path': euler_path,
            'center': (comp_center_x, comp_center_y),
            'size': comp_size
        }
        
        # Classify based on position
        if overlap_ratio > 0.7:  # In head region
            width = xs.max() - xs.min()
            height = ys.max() - ys.min()
            
            # Large encompassing component is outline
            if comp_size > 500 and width > (x1 - x0) * 0.5 and height > (y1 - y0) * 0.5:
                face_outline_parts.append(component)
                print(f"  Component {label}: FACE_OUTLINE ({len(euler_path)} path points)")
            else:
                face_interior_parts.append(component)
                print(f"  Component {label}: FACE_INTERIOR ({len(euler_path)} path points)")
        else:
            body_parts.append(component)
            print(f"  Component {label}: BODY ({len(euler_path)} path points)")
    
    # Sort components
    face_outline_parts.sort(key=lambda c: c['size'], reverse=True)
    face_interior_parts.sort(key=lambda c: (c['center'][1], c['center'][0]))
    
    # Sort body parts to start with those connected/close to head
    # Calculate distance to head center for each body part
    if head_mask is not None and body_parts:
        # Find head center from mask
        head_pixels = np.argwhere(head_mask > 0)
        if len(head_pixels) > 0:
            head_center_y = head_pixels[:, 0].mean()
            head_center_x = head_pixels[:, 1].mean()
            
            # Sort body parts by distance to head center
            for part in body_parts:
                cx, cy = part['center']
                distance = np.sqrt((cx - head_center_x)**2 + (cy - head_center_y)**2)
                part['distance_to_head'] = distance
            
            body_parts.sort(key=lambda c: c['distance_to_head'])
        else:
            body_parts.sort(key=lambda c: c['center'][1])
    else:
        body_parts.sort(key=lambda c: c['center'][1])
    
    print(f"Classification: outline={len(face_outline_parts)}, interior={len(face_interior_parts)}, body={len(body_parts)}")
    
    return face_outline_parts, face_interior_parts, body_parts

def create_natural_drawing_video(face_outline, face_interior, body, img_shape, output_video, fps=30):
    """
    Create video with natural continuous drawing using Euler paths
    """
    print("Creating natural drawing animation...")
    
    height, width = img_shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Failed to open video writer")
        return False
    
    # Create canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Drawing parameters
    line_color = (0, 0, 0)  # Black
    current_color = (255, 0, 0)  # Red for current position
    jump_color = (0, 255, 0)  # Green for pen lift indicator
    
    # Combine all components in order
    ordered_components = []
    
    # 1. Face outline first
    for comp in face_outline:
        ordered_components.append(('face_outline', comp))
    
    # 2. Face interior
    for comp in face_interior:
        ordered_components.append(('face_interior', comp))
    
    # 3. Body last
    for comp in body:
        ordered_components.append(('body', comp))
    
    frame_count = 0
    total_components = len(ordered_components)
    
    for idx, (category, component) in enumerate(ordered_components):
        euler_path = component['euler_path']
        
        if len(euler_path) < 2:
            continue
        
        print(f"Drawing {idx+1}/{total_components}: {category} (id={component['id']}, {len(euler_path)} points)")
        
        # Adjust speed based on category
        if category == 'face_outline':
            points_per_frame = 15
        elif category == 'face_interior':
            points_per_frame = 8
        else:  # body
            points_per_frame = 20
        
        # Draw path continuously following Euler path
        for i in range(0, len(euler_path) - 1, points_per_frame):
            frame = canvas.copy()
            
            # Draw line segments
            end_idx = min(i + points_per_frame, len(euler_path) - 1)
            
            for j in range(i, end_idx):
                pt1 = euler_path[j]
                pt2 = euler_path[j + 1]
                
                # Check if points are connected (8-connected neighbors)
                dx = abs(pt2[0] - pt1[0])
                dy = abs(pt2[1] - pt1[1])
                
                if dx <= 1 and dy <= 1:
                    # Points are connected - draw line
                    cv2.line(canvas, pt1, pt2, line_color, 1, cv2.LINE_AA)
                    cv2.line(frame, pt1, pt2, current_color, 2, cv2.LINE_AA)
                else:
                    # Points are disconnected - pen lift, just show jump
                    # Show dotted line to indicate pen lift
                    num_dots = 5
                    for k in range(num_dots):
                        t = k / float(num_dots - 1) if num_dots > 1 else 0
                        dot_x = int(pt1[0] + t * (pt2[0] - pt1[0]))
                        dot_y = int(pt1[1] + t * (pt2[1] - pt1[1]))
                        cv2.circle(frame, (dot_x, dot_y), 1, jump_color, -1)
            
            # Show current position
            if end_idx < len(euler_path):
                cv2.circle(frame, euler_path[end_idx], 3, current_color, -1)
            
            # Add status text
            progress = ((idx + (i / len(euler_path))) / total_components) * 100
            cv2.putText(frame, f"Drawing: {category}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(frame, f"Progress: {progress:.1f}%", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            
            video_writer.write(frame)
            frame_count += 1
    
    # Add final frames
    for _ in range(fps):  # 1 second of final image
        final_frame = canvas.copy()
        cv2.putText(final_frame, "Complete!", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        video_writer.write(final_frame)
        frame_count += 1
    
    video_writer.release()
    
    print(f"Video saved: {output_video}")
    print(f"Duration: {frame_count/fps:.1f} seconds")
    
    return True

def complete_pipeline(input_image, output_dir):
    """Run complete pipeline with Euler path traversal"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    nobg_path = os.path.join(output_dir, f"{base_name}_nobg.png")
    sketch_path = os.path.join(output_dir, f"{base_name}_sketch.jpg")
    skeleton_path = os.path.join(output_dir, f"{base_name}_skeleton.png")
    video_path = os.path.join(output_dir, f"{base_name}_natural_drawing.mp4")
    
    print(f"\n{'='*50}")
    print(f"Processing: {input_image}")
    print(f"Output directory: {output_dir}")
    print(f"Using Euler path traversal for natural drawing")
    print(f"Using color-based head detection from original image")
    print(f"{'='*50}\n")
    
    # Step 0: Remove background
    remove_background(input_image, nobg_path)
    
    # Use the no-background version for processing
    processing_image = nobg_path if os.path.exists(nobg_path) else input_image
    
    # Read image for head detection (use cleaned version)
    original_img = cv2.imread(processing_image)
    if original_img is None:
        print(f"Failed to read image: {processing_image}")
        return False
    
    # Step 1: Get sketch from API (use cleaned version)
    if not call_anime2sketch_api(processing_image, sketch_path):
        print("Failed to get sketch from API")
        return False
    
    # Step 2: Skeletonize
    skeleton_img = skeletonize_sketch(sketch_path, skeleton_path)
    
    # Step 3: Extract and classify components with Euler paths
    # Pass original image for better head detection and output_dir for visualization
    face_outline, face_interior, body = extract_and_classify_components(skeleton_img, original_img, output_dir)
    
    # Step 4: Create video with natural drawing motion
    create_natural_drawing_video(
        face_outline, face_interior, body,
        skeleton_img.shape, video_path
    )
    
    # Convert to H.264
    h264_path = video_path.replace('.mp4', '_h264.mp4')
    os.system(f'ffmpeg -i "{video_path}" -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p -movflags +faststart "{h264_path}" -y > /dev/null 2>&1')
    
    print(f"\n{'='*50}")
    print("Pipeline complete!")
    print(f"Sketch: {sketch_path}")
    print(f"Skeleton: {skeleton_path}")
    print(f"Video (original): {video_path}")
    print(f"Video (H.264): {h264_path}")
    print(f"{'='*50}\n")
    
    return h264_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python complete_drawing_pipeline.py input_image [output_dir]")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "natural_drawing_output"
    
    h264_video = complete_pipeline(input_image, output_dir)
    
    # Open the video if successful
    if h264_video and os.path.exists(h264_video):
        print(f"Opening video: {h264_video}")
        os.system(f'open "{h264_video}"')