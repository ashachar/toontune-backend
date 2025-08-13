#!/usr/bin/env python3
"""
Generate animated SVG from the stroke traversal paths
"""

import cv2
import numpy as np
import os
import sys
import networkx as nx
import json

# Import the actual functions from stroke_traversal_closed
from stroke_traversal_closed import (
    extract_lines, detect_head_color_based, get_all_components,
    split_component_by_mask, classify_split_components,
    find_paths_all_components, create_path_graph,
    ensure_dir, dprint, _dilate
)

def paths_to_svg_path(paths, scale=1.0):
    """Convert a list of paths (node coordinates) to SVG path data"""
    svg_paths = []
    
    for path in paths:
        if len(path) < 2:
            continue
            
        # Start with Move command
        path_str = f"M {path[0][1]*scale},{path[0][0]*scale}"
        
        # Add Line commands for rest of path
        for i in range(1, len(path)):
            y, x = path[i]
            path_str += f" L {x*scale},{y*scale}"
        
        svg_paths.append(path_str)
    
    return svg_paths

def create_animated_svg(image_path, output_dir=None):
    """Create animated SVG from image"""
    
    # Setup paths
    base_name = os.path.basename(image_path).rsplit('.', 1)[0]
    if output_dir is None:
        output_dir = f"test_output/{base_name}_svg"
    ensure_dir(output_dir)
    
    print(f"\nProcessing {image_path} -> SVG animation")
    
    # Extract lines
    lines, original = extract_lines(image_path, output_dir=None)
    H, W = lines.shape
    
    # Get head region
    head_mask, head_outline, head_box = detect_head_color_based(original)
    
    # Get all components
    components = get_all_components(lines)
    print(f"Found {len(components)} components")
    
    # Split components by head mask and classify
    split_components = []
    if head_mask is not None:
        head_region_bool = head_mask > 0
        head_outline_mask = head_outline if head_outline is not None else None
        
        for comp in components:
            new_comps = split_component_by_mask(comp, head_region_bool, 'SPLIT')
            split_components.extend(new_comps)
        
        # Classify components
        face_outline, face_interior, body = classify_split_components(split_components, head_region_bool, head_outline_mask)
        
        # Add type field to each component
        for c in face_outline:
            c['type'] = 'face_outline'
        for c in face_interior:
            c['type'] = 'face_interior'
        for c in body:
            c['type'] = 'body'
        
        # Sort by vertical position
        face_interior.sort(key=lambda c: c['center'][1])
        body.sort(key=lambda c: c['center'][1])
        
        ordered = face_outline + face_interior + body
    else:
        # No head detection - treat all as body
        for comp in components:
            comp['type'] = 'body'
        ordered = components
    
    # Generate SVG
    svg_width = W
    svg_height = H
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <defs>
    <style>
      .drawing-path {{
        fill: none;
        stroke: black;
        stroke-width: 1;
        stroke-linecap: round;
        stroke-linejoin: round;
      }}
      .face-outline {{ stroke: #ff0000; }}
      .face-interior {{ stroke: #00ff00; }}
      .body {{ stroke: #0000ff; }}
    </style>
  </defs>
  
  <g id="drawing">
'''
    
    # Track timing for animations
    total_duration = 0
    animation_speed = 0.002  # seconds per point (fast animation)
    
    all_paths_data = []
    
    # Process each component
    for comp_idx, comp in enumerate(ordered):
        comp_id = comp.get('id', f'comp_{comp_idx}')
        comp_type = comp.get('type', 'body')
        points = comp['points']
        
        if len(points) < 2:
            continue
        
        # Build graph
        G, edge_lookup = create_path_graph(points)
        
        # Find paths
        paths = find_paths_all_components(G)
        
        if not paths:
            continue
        
        # Convert to SVG paths
        svg_paths = paths_to_svg_path(paths)
        
        # Add each path to SVG with animation
        for path_idx, (svg_path, path_nodes) in enumerate(zip(svg_paths, paths)):
            path_id = f"{comp_id}_path_{path_idx}"
            path_length = len(path_nodes)
            duration = path_length * animation_speed
            
            # Estimate visual path length for stroke-dasharray
            visual_length = 0
            for i in range(1, len(path_nodes)):
                dy = abs(path_nodes[i][0] - path_nodes[i-1][0])
                dx = abs(path_nodes[i][1] - path_nodes[i-1][1])
                visual_length += np.sqrt(dy*dy + dx*dx)
            
            # Store path data for later use
            all_paths_data.append({
                'id': path_id,
                'type': comp_type,
                'path': svg_path,
                'duration': duration,
                'start': total_duration,
                'length': path_length,
                'visual_length': visual_length
            })
            
            svg_content += f'''    <path id="{path_id}" class="drawing-path {comp_type}" d="{svg_path}" 
          stroke-dasharray="{visual_length}" 
          stroke-dashoffset="{visual_length}">
      <animate attributeName="stroke-dashoffset" 
               from="{visual_length}" 
               to="0" 
               dur="{duration}s" 
               begin="{total_duration}s" 
               fill="freeze"/>
    </path>
'''
            total_duration += duration
    
    svg_content += '''  </g>
</svg>'''
    
    # Save SVG file
    svg_path = os.path.join(output_dir, f"{base_name}_animated.svg")
    with open(svg_path, 'w') as f:
        f.write(svg_content)
    
    print(f"Saved animated SVG: {svg_path}")
    print(f"Total animation duration: {total_duration:.2f} seconds")
    
    # Save path data as JSON for reference
    json_path = os.path.join(output_dir, f"{base_name}_paths.json")
    with open(json_path, 'w') as f:
        json.dump({
            'width': svg_width,
            'height': svg_height,
            'total_duration': total_duration,
            'paths': all_paths_data
        }, f, indent=2)
    
    print(f"Saved path data: {json_path}")
    
    # Create HTML preview with overlay capability
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>{base_name} - Animated Drawing</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: #f0f0f0;
        }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            margin-top: 0;
        }}
        .display-container {{
            position: relative;
            display: inline-block;
            border: 1px solid #ddd;
        }}
        .svg-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            z-index: 2;
        }}
        video {{
            display: block;
            width: {svg_width}px;
            height: {svg_height}px;
        }}
        .controls {{
            text-align: center;
            margin-top: 20px;
        }}
        button {{
            padding: 10px 20px;
            margin: 0 5px;
            font-size: 16px;
            cursor: pointer;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
        }}
        button:hover {{
            background: #0056b3;
        }}
        .toggle-btn {{
            background: #28a745;
        }}
        .toggle-btn:hover {{
            background: #218838;
        }}
        .info {{
            text-align: center;
            color: #666;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{base_name} - Animated Drawing</h1>
        <div class="display-container">
            <!-- Video background (optional) -->
            <video id="bgVideo" width="{svg_width}" height="{svg_height}" style="display:none;">
                <source src="../{base_name}_debug/drawing_animation.mp4" type="video/mp4">
            </video>
            
            <!-- SVG overlay -->
            <object class="svg-overlay" id="svgObject" data="{base_name}_animated.svg" 
                    type="image/svg+xml" width="{svg_width}" height="{svg_height}"></object>
        </div>
        <div class="controls">
            <button onclick="restartAnimation()">Restart Animation</button>
            <button class="toggle-btn" onclick="toggleVideo()">Toggle Video Background</button>
        </div>
        <div class="info">
            Animation duration: {total_duration:.2f} seconds<br>
            Total paths: {len(all_paths_data)}<br>
            <small>Colors: <span style="color:#ff0000">■</span> Face Outline | 
                   <span style="color:#00ff00">■</span> Face Interior | 
                   <span style="color:#0000ff">■</span> Body</small>
        </div>
    </div>
    
    <script>
        let videoVisible = false;
        
        function restartAnimation() {{
            // Reload the SVG
            const svg = document.getElementById('svgObject');
            const svgSrc = svg.data;
            svg.data = '';
            setTimeout(() => {{
                svg.data = svgSrc;
                if (videoVisible) {{
                    const video = document.getElementById('bgVideo');
                    video.currentTime = 0;
                    video.play();
                }}
            }}, 100);
        }}
        
        function toggleVideo() {{
            const video = document.getElementById('bgVideo');
            videoVisible = !videoVisible;
            if (videoVisible) {{
                video.style.display = 'block';
                video.play();
            }} else {{
                video.style.display = 'none';
                video.pause();
            }}
        }}
    </script>
</body>
</html>'''
    
    html_path = os.path.join(output_dir, f"{base_name}_preview.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Saved HTML preview: {html_path}")
    
    return svg_path, html_path

def main():
    if len(sys.argv) < 2:
        # Process all test images
        test_images = [
            "cartoon-test/man.png",
            "cartoon-test/woman.png",
            "cartoon-test/baby.png",
            "cartoon-test/robot.png",
            "cartoon-test/spring.png"
        ]
        
        for image_path in test_images:
            if os.path.exists(image_path):
                try:
                    create_animated_svg(image_path)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    else:
        image_path = sys.argv[1]
        create_animated_svg(image_path)

if __name__ == "__main__":
    main()