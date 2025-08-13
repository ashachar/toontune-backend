#!/usr/bin/env python3
"""
Simplified complete pipeline without networkx dependency
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
import requests
import os
import sys

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

def extract_contours_from_skeleton(skeleton_img):
    """Extract contours as paths from skeleton image"""
    print("Extracting paths from skeleton...")
    
    # Invert so lines are white
    lines = 255 - skeleton_img
    
    # Find contours
    contours, hierarchy = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    print(f"Found {len(contours)} contours")
    
    # Convert contours to paths
    paths = []
    for contour in contours:
        # Reshape contour to list of points
        path = contour.reshape(-1, 2).tolist()
        if len(path) > 5:  # Skip very small contours
            paths.append(path)
    
    return paths

def order_paths_for_drawing(paths, img_shape):
    """Order paths for natural drawing (top to bottom, large to small)"""
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
            center_y = (y_min + y_max) / 2
            size = len(path)
            area = (x_max - x_min) * (y_max - y_min)
            
            # Check if it's in the head region (top 40% of image)
            is_head = center_y < img_shape[0] * 0.4
            
            path_info.append({
                'index': i,
                'top_y': top_y,
                'center_y': center_y,
                'size': size,
                'area': area,
                'is_head': is_head
            })
    
    # Sort: head components first, then by vertical position, then by size
    path_info.sort(key=lambda x: (not x['is_head'], x['top_y'], -x['area']))
    
    # Reorder paths
    ordered_paths = [paths[info['index']] for info in path_info]
    
    print(f"Ordered {len(ordered_paths)} paths (head components first)")
    
    return ordered_paths

def create_drawing_video(paths, img_shape, output_video, fps=30, draw_speed=5):
    """Create video showing progressive drawing"""
    print("Creating drawing animation video...")
    
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
    
    # Draw each path
    for path_idx, path in enumerate(paths):
        if not path or len(path) < 2:
            continue
            
        print(f"Drawing path {path_idx + 1}/{len(paths)} ({len(path)} points)")
        
        # Convert path to numpy array
        points = np.array(path, dtype=np.int32)
        
        # Draw path progressively
        for i in range(0, len(points), draw_speed):
            # Create frame
            frame = canvas.copy()
            
            # Draw segments up to current point
            end_idx = min(i + draw_speed, len(points))
            
            # Draw on canvas (permanent)
            for j in range(max(1, i), end_idx):
                pt1 = tuple(points[j-1])
                pt2 = tuple(points[j])
                cv2.line(canvas, pt1, pt2, line_color, 1, cv2.LINE_AA)
            
            # Highlight current drawing segment
            if i > 0:
                segment_start = max(0, i - draw_speed * 2)
                for j in range(segment_start + 1, end_idx):
                    pt1 = tuple(points[j-1])
                    pt2 = tuple(points[j])
                    cv2.line(frame, pt1, pt2, current_color, 2, cv2.LINE_AA)
            
            # Add progress text
            progress = frame_count / total_points * 100
            cv2.putText(frame, f"Drawing: {progress:.1f}%", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            video_writer.write(frame)
            frame_count += min(draw_speed, end_idx - i)
    
    # Add final frames
    for _ in range(fps * 2):  # 2 seconds of final image
        cv2.putText(canvas, "Complete!", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
        video_writer.write(canvas)
    
    video_writer.release()
    
    print(f"Video saved to {output_video}")
    print(f"Total frames: {frame_count + fps * 2}")
    
    return True

def complete_pipeline(input_image, output_dir):
    """Run complete pipeline from image to drawing video"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    sketch_path = os.path.join(output_dir, f"{base_name}_sketch.jpg")
    skeleton_path = os.path.join(output_dir, f"{base_name}_skeleton.png")
    video_path = os.path.join(output_dir, f"{base_name}_drawing.mp4")
    
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
    
    # Step 3: Extract paths using contours
    paths = extract_contours_from_skeleton(skeleton_img)
    
    if not paths:
        print("No paths found in skeleton")
        return False
    
    # Step 4: Order paths (head first, then top to bottom)
    ordered_paths = order_paths_for_drawing(paths, skeleton_img.shape)
    
    # Step 5: Create video
    create_drawing_video(ordered_paths, skeleton_img.shape, video_path)
    
    print(f"\n{'='*50}")
    print("Pipeline complete!")
    print(f"Sketch: {sketch_path}")
    print(f"Skeleton: {skeleton_path}")
    print(f"Video: {video_path}")
    print(f"{'='*50}\n")
    
    # Also save a debug image showing path order
    debug_img = np.ones_like(skeleton_img) * 255
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, path in enumerate(ordered_paths[:6]):  # Show first 6 paths in different colors
        color = colors[i % len(colors)]
        points = np.array(path, dtype=np.int32)
        for j in range(1, len(points)):
            pt1 = tuple(points[j-1])
            pt2 = tuple(points[j])
            cv2.line(debug_img, pt1, pt2, color, 2)
    
    debug_path = os.path.join(output_dir, f"{base_name}_path_order.png")
    cv2.imwrite(debug_path, debug_img)
    print(f"Path order debug image: {debug_path}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_drawing_pipeline.py input_image [output_dir]")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "drawing_output"
    
    complete_pipeline(input_image, output_dir)