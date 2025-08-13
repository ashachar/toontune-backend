#!/usr/bin/env python3
"""
Complete pipeline: 
1. Call Anime2Sketch API
2. Skeletonize to get thin lines
3. Extract paths and order them for natural drawing
4. Create drawing animation video
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
import requests
import os
import sys
from collections import defaultdict
import networkx as nx
from scipy.spatial.distance import cdist
import imageio

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

def extract_paths_from_skeleton(skeleton_img):
    """Extract drawing paths from skeleton image"""
    print("Extracting paths from skeleton...")
    
    # Invert so lines are white
    lines = 255 - skeleton_img
    
    # Find all white pixels (line pixels)
    line_pixels = np.argwhere(lines > 0)
    
    if len(line_pixels) == 0:
        return []
    
    # Build adjacency graph
    G = nx.Graph()
    
    # Add all line pixels as nodes
    for y, x in line_pixels:
        G.add_node((x, y))
    
    # Connect adjacent pixels
    for y, x in line_pixels:
        # Check 8-neighborhood
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (nx, ny) in G.nodes():
                    G.add_edge((x, y), (nx, ny))
    
    # Find connected components
    components = list(nx.connected_components(G))
    print(f"Found {len(components)} connected components")
    
    paths = []
    for comp in components:
        # Convert component to subgraph
        subgraph = G.subgraph(comp)
        
        # Try to find Eulerian path
        if nx.is_connected(subgraph):
            # Find nodes with odd degree
            odd_degree_nodes = [n for n in subgraph.nodes() if subgraph.degree(n) % 2 == 1]
            
            if len(odd_degree_nodes) == 0:
                # Eulerian circuit exists
                try:
                    path = list(nx.eulerian_circuit(subgraph))
                    # Convert edges to continuous path
                    continuous_path = [path[0][0]]
                    for edge in path:
                        continuous_path.append(edge[1])
                    paths.append(continuous_path)
                except:
                    # Fallback to simple path
                    paths.append(list(comp))
            elif len(odd_degree_nodes) == 2:
                # Eulerian path exists
                try:
                    path = list(nx.eulerian_path(subgraph, source=odd_degree_nodes[0]))
                    continuous_path = [path[0][0]]
                    for edge in path:
                        continuous_path.append(edge[1])
                    paths.append(continuous_path)
                except:
                    paths.append(list(comp))
            else:
                # No Eulerian path, just use points
                paths.append(list(comp))
    
    return paths

def order_paths_for_drawing(paths, img_shape):
    """Order paths for natural drawing (head first if detected)"""
    print("Ordering paths for natural drawing...")
    
    if not paths:
        return []
    
    # Simple heuristic: order by vertical position (top to bottom)
    # and by size (larger components first)
    path_info = []
    for i, path in enumerate(paths):
        if path:
            points = np.array(path)
            centroid = points.mean(axis=0)
            top_y = points[:, 1].min()
            size = len(path)
            path_info.append((i, centroid, top_y, size))
    
    # Sort by top_y position primarily, then by size
    path_info.sort(key=lambda x: (x[2], -x[3]))
    
    # Reorder paths
    ordered_paths = [paths[info[0]] for info in path_info]
    
    return ordered_paths

def create_drawing_video(paths, img_shape, output_video, fps=30, draw_speed=10):
    """Create video showing progressive drawing"""
    print("Creating drawing animation video...")
    
    height, width = img_shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Create canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Colors for drawing
    line_color = (0, 0, 0)  # Black
    current_color = (255, 0, 0)  # Red for current drawing
    
    frames = []
    
    # Draw each path
    for path_idx, path in enumerate(paths):
        if not path:
            continue
            
        print(f"Drawing path {path_idx + 1}/{len(paths)} ({len(path)} points)")
        
        # Draw points progressively
        for i in range(1, len(path), draw_speed):
            # Create frame
            frame = canvas.copy()
            
            # Draw current segment
            end_idx = min(i + draw_speed, len(path))
            for j in range(i, end_idx):
                if j > 0:
                    pt1 = tuple(map(int, path[j-1]))
                    pt2 = tuple(map(int, path[j]))
                    cv2.line(canvas, pt1, pt2, line_color, 1, cv2.LINE_AA)
                    cv2.line(frame, pt1, pt2, current_color, 2, cv2.LINE_AA)
            
            frames.append(frame)
            video_writer.write(frame)
    
    # Add final frame
    for _ in range(fps):  # 1 second of final image
        video_writer.write(canvas)
        frames.append(canvas)
    
    video_writer.release()
    
    # Also save as GIF
    gif_path = output_video.replace('.mp4', '.gif')
    imageio.mimsave(gif_path, frames[::3], duration=50)  # Sample every 3rd frame for smaller GIF
    
    print(f"Video saved to {output_video}")
    print(f"GIF saved to {gif_path}")
    
    return output_video

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
    
    # Step 3: Extract paths
    paths = extract_paths_from_skeleton(skeleton_img)
    
    if not paths:
        print("No paths found in skeleton")
        return False
    
    # Step 4: Order paths
    ordered_paths = order_paths_for_drawing(paths, skeleton_img.shape)
    
    # Step 5: Create video
    create_drawing_video(ordered_paths, skeleton_img.shape, video_path)
    
    print(f"\n{'='*50}")
    print("Pipeline complete!")
    print(f"Sketch: {sketch_path}")
    print(f"Skeleton: {skeleton_path}")
    print(f"Video: {video_path}")
    print(f"GIF: {video_path.replace('.mp4', '.gif')}")
    print(f"{'='*50}\n")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python complete_drawing_pipeline.py input_image [output_dir]")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "drawing_output"
    
    complete_pipeline(input_image, output_dir)