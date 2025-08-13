#!/usr/bin/env python3
"""
Final pipeline integrating the correct path ordering algorithm from stroke_traversal_closed.py
1. Call Anime2Sketch API
2. Skeletonize to get thin lines  
3. Extract and classify components (face_outline, face_interior, body)
4. Order properly: face_outline → face_interior → body
5. Create fast drawing video
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

def detect_head_region_simple(skeleton_img, original_img=None):
    """Simple head detection - assume head is in top 40% of image"""
    H, W = skeleton_img.shape[:2]
    
    # Create head region mask (top 40% of image)
    head_mask = np.zeros((H, W), dtype=np.uint8)
    head_mask[:int(H * 0.4), :] = 255
    
    # Head box
    head_box = [W * 0.2, 0, W * 0.8, H * 0.4]  # x0, y0, x1, y1
    
    return head_mask, head_box

def extract_and_classify_components(skeleton_img, head_mask, head_box):
    """
    Extract components and classify them as face_outline, face_interior, or body
    Based on the algorithm from stroke_traversal_closed.py
    """
    print("Extracting and classifying components...")
    
    # Invert so lines are white
    lines = 255 - skeleton_img
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(lines, connectivity=8)
    print(f"Found {num_labels - 1} connected components")
    
    # Head region boundaries
    x0, y0, x1, y1 = head_box
    head_center_x = (x0 + x1) / 2
    head_center_y = (y0 + y1) / 2
    
    # Create components with classification
    face_outline_parts = []
    face_interior_parts = []
    body_parts = []
    
    for label in range(1, num_labels):  # Skip background
        # Get component mask
        component_mask = (labels == label)
        points = np.argwhere(component_mask)
        
        if len(points) < 10:  # Skip tiny components
            continue
        
        # Calculate component properties
        ys, xs = points[:, 0], points[:, 1]
        comp_x_min, comp_x_max = xs.min(), xs.max()
        comp_y_min, comp_y_max = ys.min(), ys.max()
        comp_center_x = (comp_x_min + comp_x_max) / 2
        comp_center_y = (comp_y_min + comp_y_max) / 2
        comp_size = len(points)
        
        # Calculate overlap with head region
        head_pixels = points[(points[:, 0] < y1) & (points[:, 1] > x0) & (points[:, 1] < x1)]
        overlap_ratio = len(head_pixels) / len(points) if len(points) > 0 else 0
        
        # Create component dict
        component = {
            'id': label,
            'mask': component_mask,
            'points': [(int(p[1]), int(p[0])) for p in points],  # Convert to (x,y) format
            'center': (comp_center_x, comp_center_y),
            'size': comp_size,
            'bbox': (comp_x_min, comp_y_min, comp_x_max, comp_y_max)
        }
        
        # Classify based on position and overlap
        if overlap_ratio > 0.7:  # Mostly in head region
            # Check if it's outline (large and encompasses head) or interior
            width = comp_x_max - comp_x_min
            height = comp_y_max - comp_y_min
            aspect_ratio = width / (height + 0.001)
            
            # Face outline: large component that encompasses significant portion of head
            if comp_size > 500 and width > (x1 - x0) * 0.5 and height > (y1 - y0) * 0.5:
                face_outline_parts.append(component)
                print(f"  Component {label}: FACE_OUTLINE (size={comp_size})")
            else:
                # Face interior: smaller components inside head (eyes, mouth, etc)
                face_interior_parts.append(component)
                print(f"  Component {label}: FACE_INTERIOR (size={comp_size})")
        else:
            # Body: components outside head region
            body_parts.append(component)
            print(f"  Component {label}: BODY (size={comp_size})")
    
    # Sort components within each category
    # Face outline: largest first
    face_outline_parts.sort(key=lambda c: c['size'], reverse=True)
    
    # Face interior: top to bottom, left to right
    face_interior_parts.sort(key=lambda c: (c['center'][1], c['center'][0]))
    
    # Body: top to bottom
    body_parts.sort(key=lambda c: c['center'][1])
    
    print(f"Classification: outline={len(face_outline_parts)}, interior={len(face_interior_parts)}, body={len(body_parts)}")
    
    return face_outline_parts, face_interior_parts, body_parts

def create_ordered_drawing_video(face_outline, face_interior, body, img_shape, output_video, fps=30):
    """Create video with proper drawing order: face_outline → face_interior → body"""
    print("Creating drawing animation with correct order...")
    
    height, width = img_shape[:2]
    
    # Try H.264 codec
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
    
    # Combine all components in order
    ordered_components = []
    
    # 1. Face outline first (red)
    for comp in face_outline:
        ordered_components.append(('face_outline', comp, (0, 0, 255)))
    
    # 2. Face interior (green)
    for comp in face_interior:
        ordered_components.append(('face_interior', comp, (0, 255, 0)))
    
    # 3. Body last (blue)
    for comp in body:
        ordered_components.append(('body', comp, (255, 0, 0)))
    
    frame_count = 0
    total_components = len(ordered_components)
    
    # Draw speed based on component type
    for idx, (category, component, debug_color) in enumerate(ordered_components):
        points = component['points']
        
        if len(points) < 2:
            continue
        
        print(f"Drawing {idx+1}/{total_components}: {category} (id={component['id']}, {len(points)} points)")
        
        # Adjust speed based on category
        if category == 'face_outline':
            draw_speed = 40  # Faster for outline
        elif category == 'face_interior':
            draw_speed = 20  # Medium for details
        else:  # body
            draw_speed = 50  # Fast for body
        
        # Draw component progressively
        points_array = np.array(points, dtype=np.int32)
        
        for i in range(0, len(points), draw_speed):
            frame = canvas.copy()
            
            # Draw points in this batch
            end_idx = min(i + draw_speed, len(points))
            
            for j in range(i, end_idx):
                pt = tuple(points_array[j])
                cv2.circle(canvas, pt, 1, (0, 0, 0), -1)
                
                # Highlight current drawing
                cv2.circle(frame, pt, 3, (0, 0, 255), 2)
            
            # Add category label
            cv2.putText(frame, f"Drawing: {category}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, debug_color, 2)
            
            # Add progress
            progress = (idx / total_components) * 100
            cv2.putText(frame, f"Progress: {progress:.1f}%", (10, 60),
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
    
    print(f"Video saved: {output_video}")
    print(f"Duration: {frame_count/fps:.1f} seconds")
    
    return True

def complete_pipeline(input_image, output_dir):
    """Run complete pipeline with correct path ordering"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    sketch_path = os.path.join(output_dir, f"{base_name}_sketch.jpg")
    skeleton_path = os.path.join(output_dir, f"{base_name}_skeleton.png")
    video_path = os.path.join(output_dir, f"{base_name}_ordered_drawing.mp4")
    
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
    
    # Step 3: Detect head region (simple version)
    head_mask, head_box = detect_head_region_simple(skeleton_img)
    
    # Step 4: Extract and classify components
    face_outline, face_interior, body = extract_and_classify_components(
        skeleton_img, head_mask, head_box
    )
    
    # Step 5: Create video with correct order
    create_ordered_drawing_video(
        face_outline, face_interior, body,
        skeleton_img.shape, video_path
    )
    
    # Create debug visualization
    debug_img = np.ones((skeleton_img.shape[0], skeleton_img.shape[1], 3), dtype=np.uint8) * 255
    
    # Draw components in their categories with different colors
    for comp in face_outline:
        for pt in comp['points'][::5]:  # Sample for visibility
            cv2.circle(debug_img, pt, 2, (0, 0, 255), -1)  # Red for outline
    
    for comp in face_interior:
        for pt in comp['points'][::5]:
            cv2.circle(debug_img, pt, 2, (0, 255, 0), -1)  # Green for interior
    
    for comp in body:
        for pt in comp['points'][::5]:
            cv2.circle(debug_img, pt, 2, (255, 0, 0), -1)  # Blue for body
    
    # Add legend
    cv2.putText(debug_img, "Red: Face Outline", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(debug_img, "Green: Face Interior", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(debug_img, "Blue: Body", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    debug_path = os.path.join(output_dir, f"{base_name}_classified_components.png")
    cv2.imwrite(debug_path, debug_img)
    
    print(f"\n{'='*50}")
    print("Pipeline complete!")
    print(f"Sketch: {sketch_path}")
    print(f"Skeleton: {skeleton_path}")
    print(f"Video: {video_path}")
    print(f"Classification: {debug_path}")
    print(f"{'='*50}\n")
    
    # Convert to H.264
    h264_path = video_path.replace('.mp4', '_h264.mp4')
    os.system(f'ffmpeg -i "{video_path}" -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p -movflags +faststart "{h264_path}" -y > /dev/null 2>&1')
    print(f"H.264 video: {h264_path}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python final_drawing_pipeline.py input_image [output_dir]")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "drawing_output"
    
    complete_pipeline(input_image, output_dir)