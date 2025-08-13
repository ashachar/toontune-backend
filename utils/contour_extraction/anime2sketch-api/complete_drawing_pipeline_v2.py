#!/usr/bin/env python3
"""
Fixed pipeline that includes ALL paths (internal and external)
and creates faster video (6x speed)
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
import requests
import os
import sys
from scipy import ndimage

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

def extract_all_paths_from_skeleton(skeleton_img):
    """Extract ALL paths (internal and external) from skeleton image"""
    print("Extracting ALL paths from skeleton...")
    
    # Invert so lines are white
    lines = 255 - skeleton_img
    
    # Find ALL contours including internal ones using RETR_TREE
    contours, hierarchy = cv2.findContours(lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    print(f"Found {len(contours)} total contours (internal and external)")
    
    # Also find connected components for better path detection
    num_labels, labels = cv2.connectedComponents(lines, connectivity=8)
    print(f"Found {num_labels - 1} connected components")
    
    # Extract paths from connected components
    paths = []
    for label in range(1, num_labels):  # Skip background (0)
        # Get all pixels for this component
        component_mask = (labels == label)
        points = np.argwhere(component_mask)
        
        if len(points) > 10:  # Skip very small components
            # Convert to (x, y) format
            path = [(int(p[1]), int(p[0])) for p in points]
            paths.append(path)
    
    # Alternative: use contours if connected components didn't work well
    if len(paths) < 5:  # If we got too few paths, use contours
        print("Using contours method for better detail extraction")
        paths = []
        for contour in contours:
            # Reshape contour to list of points
            path = contour.reshape(-1, 2).tolist()
            if len(path) > 10:  # Skip very small contours
                paths.append(path)
    
    print(f"Extracted {len(paths)} valid paths")
    return paths

def order_paths_for_drawing(paths, img_shape):
    """Order paths for natural drawing (head first, then by size and position)"""
    print("Ordering paths for natural drawing...")
    
    if not paths:
        return []
    
    # Calculate properties for each path
    path_info = []
    for i, path in enumerate(paths):
        if path:
            points = np.array(path)
            # Get bounding box
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            # Calculate properties
            top_y = y_min
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            size = len(path)
            area = (x_max - x_min) * (y_max - y_min)
            
            # Check if it's in the head region (top 40% of image)
            is_head = center_y < img_shape[0] * 0.4
            
            # Check if it's central (likely face features)
            is_central = abs(center_x - img_shape[1]/2) < img_shape[1] * 0.3
            
            path_info.append({
                'index': i,
                'top_y': top_y,
                'center_x': center_x,
                'center_y': center_y,
                'size': size,
                'area': area,
                'is_head': is_head,
                'is_central': is_central
            })
    
    # Sort: 
    # 1. Head components first
    # 2. Central components (face features) before peripheral
    # 3. Then by vertical position (top to bottom)
    # 4. Then by size (larger first)
    path_info.sort(key=lambda x: (
        not x['is_head'],  # Head first
        not x['is_central'],  # Central features next
        x['top_y'],  # Top to bottom
        -x['area']  # Larger components first
    ))
    
    # Reorder paths
    ordered_paths = [paths[info['index']] for info in path_info]
    
    print(f"Ordered {len(ordered_paths)} paths (head/central components first)")
    
    return ordered_paths

def create_fast_drawing_video(paths, img_shape, output_video, fps=30, draw_speed=30):
    """Create FAST video showing progressive drawing (6x faster than before)"""
    print(f"Creating FAST drawing animation video (speed={draw_speed} points/frame)...")
    
    height, width = img_shape[:2]
    
    # Create video writer with H.264 codec
    # Try H.264 first, fall back to mp4v if not available
    try:
        fourcc = cv2.VideoWriter_fourcc(*'h264')
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            raise Exception("H264 not available")
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Failed to open video writer")
        return False
    
    # Create canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Colors for drawing
    line_color = (0, 0, 0)  # Black
    current_color = (0, 0, 255)  # Red for current drawing
    
    frame_count = 0
    total_points = sum(len(p) for p in paths)
    points_drawn = 0
    
    # Draw each path
    for path_idx, path in enumerate(paths):
        if not path or len(path) < 2:
            continue
            
        print(f"Drawing path {path_idx + 1}/{len(paths)} ({len(path)} points)")
        
        # Convert path to numpy array
        points = np.array(path, dtype=np.int32)
        
        # Draw path progressively with FAST speed
        for i in range(0, len(points), draw_speed):
            # Create frame
            frame = canvas.copy()
            
            # Draw segments up to current point
            end_idx = min(i + draw_speed, len(points))
            
            # Draw on canvas (permanent) - connect points as lines
            for j in range(max(1, i), end_idx):
                # Find nearest previous point to connect to
                if j > 0:
                    pt2 = tuple(points[j])
                    # Find closest previous point
                    min_dist = float('inf')
                    best_pt1 = tuple(points[j-1])
                    for k in range(max(0, j-5), j):
                        pt1 = tuple(points[k])
                        dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
                        if dist < min_dist and dist < 10:  # Only connect if close enough
                            min_dist = dist
                            best_pt1 = pt1
                    
                    if min_dist < 10:  # Only draw line if points are close
                        cv2.line(canvas, best_pt1, pt2, line_color, 1, cv2.LINE_AA)
            
            # Highlight current drawing area with circle
            if end_idx < len(points):
                current_pt = tuple(points[end_idx-1])
                cv2.circle(frame, current_pt, 5, current_color, 2)
            
            # Add progress text
            points_drawn += min(draw_speed, end_idx - i)
            progress = points_drawn / total_points * 100
            cv2.putText(frame, f"Drawing: {progress:.1f}%", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            video_writer.write(frame)
            frame_count += 1
    
    # Add final frames
    for _ in range(fps):  # 1 second of final image
        final_frame = canvas.copy()
        cv2.putText(final_frame, "Complete!", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
        video_writer.write(final_frame)
        frame_count += 1
    
    video_writer.release()
    
    print(f"Video saved to {output_video}")
    print(f"Total frames: {frame_count} (duration: {frame_count/fps:.1f} seconds)")
    
    return True

def complete_pipeline(input_image, output_dir):
    """Run complete pipeline from image to drawing video"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    sketch_path = os.path.join(output_dir, f"{base_name}_sketch.jpg")
    skeleton_path = os.path.join(output_dir, f"{base_name}_skeleton.png")
    video_path = os.path.join(output_dir, f"{base_name}_drawing_fast.mp4")
    
    print(f"\n{'='*50}")
    print(f"Processing: {input_image}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}\n")
    
    # Step 1: Get sketch from API
    if not call_anime2sketch_api(input_image, sketch_path):
        print("Failed to get sketch from API")
        return False
    
    # Step 2: Skeletonize
    skeleton_img = skeletonize_sketch(sketch_path, skeleton_path)
    
    # Step 3: Extract ALL paths (internal and external)
    paths = extract_all_paths_from_skeleton(skeleton_img)
    
    if not paths:
        print("No paths found in skeleton")
        return False
    
    # Step 4: Order paths (head first, then by position and size)
    ordered_paths = order_paths_for_drawing(paths, skeleton_img.shape)
    
    # Step 5: Create FAST video (6x speed)
    create_fast_drawing_video(ordered_paths, skeleton_img.shape, video_path)
    
    print(f"\n{'='*50}")
    print("Pipeline complete!")
    print(f"Sketch: {sketch_path}")
    print(f"Skeleton: {skeleton_path}")
    print(f"Video: {video_path}")
    print(f"{'='*50}\n")
    
    # Create debug image showing ALL paths in different colors
    debug_img = np.ones((skeleton_img.shape[0], skeleton_img.shape[1], 3), dtype=np.uint8) * 255
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
    ]
    
    # Draw each path in different color
    for i, path in enumerate(ordered_paths[:20]):  # Show first 20 paths
        color = colors[i % len(colors)]
        points = np.array(path, dtype=np.int32)
        
        # Draw path as dots
        for pt in points[::5]:  # Sample every 5th point for visibility
            cv2.circle(debug_img, tuple(pt), 2, color, -1)
        
        # Draw path number at the start
        if len(points) > 0:
            cv2.putText(debug_img, str(i+1), tuple(points[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    debug_path = os.path.join(output_dir, f"{base_name}_all_paths.png")
    cv2.imwrite(debug_path, debug_img)
    print(f"All paths debug image: {debug_path}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python complete_drawing_pipeline_v2.py input_image [output_dir]")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "drawing_output"
    
    complete_pipeline(input_image, output_dir)