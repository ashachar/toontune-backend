#!/usr/bin/env python3
"""
Create a comprehensive summary image showing what the motion tracking tracks.
"""

import cv2
import numpy as np
from pathlib import Path


def create_tracking_summary():
    """Create a single image that explains the tracking."""
    
    # Load the visualizations we created
    vis_dir = Path("tracking_visualization")
    
    # Load images
    grid_img = cv2.imread(str(vis_dir / "tracking_grid_overlay.png"))
    pixels_img = cv2.imread(str(vis_dir / "tracking_pixels.png"))
    detail_img = cv2.imread(str(vis_dir / "tracked_region_detail.png"))
    
    # Get dimensions
    h1, w1 = grid_img.shape[:2] if grid_img is not None else (534, 1166)
    
    # Create canvas for summary
    canvas_height = h1 + 400  # Extra space for text
    canvas_width = w1
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Add title
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "MOTION TRACKING ANALYSIS - What Gets Tracked?"
    title_size = cv2.getTextSize(title, font, 1.5, 3)[0]
    title_x = (canvas_width - title_size[0]) // 2
    cv2.putText(canvas, title, (title_x, 50), font, 1.5, (0, 0, 0), 3)
    
    # Add grid overlay image
    if grid_img is not None:
        canvas[80:80+h1, 0:w1] = grid_img
    
    # Add explanation text
    y_offset = 80 + h1 + 30
    explanations = [
        "How Motion Tracking Works:",
        "1. INITIAL POINT: Starts at specified coordinates (default: center)",
        "2. BOUNDING BOX: Creates 60x60 pixel box around initial point",
        "3. TRACKING: Uses OpenCV CSRT tracker to follow features in box",
        "4. WHAT IT TRACKS: Texture patterns, edges, color gradients",
        "",
        "In DO_RE_MI.MOV:",
        "- Red boxes show the 60x60 pixel region being tracked",
        "- The tracker follows whatever visual features are in that box",
        "- As the camera/subject moves, the box follows the pattern",
        "- Text is positioned relative to the tracked box center"
    ]
    
    for i, text in enumerate(explanations):
        if text.startswith("How") or text.startswith("In DO"):
            # Headers in bold
            cv2.putText(canvas, text, (50, y_offset + i * 30), 
                       font, 0.8, (0, 0, 200), 2, cv2.LINE_AA)
        elif text.startswith("-"):
            # Bullet points
            cv2.putText(canvas, text, (70, y_offset + i * 30), 
                       font, 0.6, (50, 50, 50), 1, cv2.LINE_AA)
        else:
            # Regular text
            cv2.putText(canvas, text, (50, y_offset + i * 30), 
                       font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Save summary
    output_path = vis_dir / "tracking_summary.png"
    cv2.imwrite(str(output_path), canvas)
    print(f"Created summary: {output_path}")
    
    # Also create a simple annotated frame
    create_annotated_frame()
    
    return output_path


def create_annotated_frame():
    """Create a single annotated frame showing what's tracked."""
    
    # Get a frame from the video
    video_path = Path("do_re_mi_quick_test/short_clip.mp4")
    if not video_path.exists():
        return
    
    cap = cv2.VideoCapture(str(video_path))
    
    # Go to middle of video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    # Get dimensions
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Draw the tracking box
    box_size = 60
    x = center_x - box_size // 2
    y = center_y - box_size // 2
    
    # Extract the tracked region
    tracked_region = frame[y:y+box_size, x:x+box_size].copy()
    
    # Highlight edges in the tracked region
    edges = cv2.Canny(tracked_region, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[:, :, 2] = edges  # Red channel for edges
    
    # Draw on main frame
    overlay = frame.copy()
    
    # Draw tracking box with thick red border
    cv2.rectangle(overlay, (x, y), (x + box_size, y + box_size), (0, 0, 255), 3)
    
    # Fill tracked area with semi-transparent red
    roi = overlay[y:y+box_size, x:x+box_size]
    red_overlay = np.ones_like(roi, dtype=np.uint8) * np.array([0, 0, 255], dtype=np.uint8)
    blended = cv2.addWeighted(roi, 0.7, red_overlay, 0.3, 0)
    overlay[y:y+box_size, x:x+box_size] = blended
    
    # Add arrows pointing to tracked features
    # Arrow to center
    cv2.arrowedLine(overlay, (x - 50, y - 50), (x + 10, y + 10), 
                    (0, 255, 255), 2, tipLength=0.3)
    cv2.putText(overlay, "TRACKED PIXELS", (x - 150, y - 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Show what features are tracked
    cv2.putText(overlay, "60x60 pixel box", (x + box_size + 10, y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, "tracks textures", (x + box_size + 10, y + 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, "and patterns", (x + box_size + 10, y + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Create side-by-side comparison
    comparison = np.hstack([frame, overlay])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(comparison, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.rectangle(comparison, (w, 0), (w * 2, 40), (0, 0, 0), -1)
    cv2.putText(comparison, "ORIGINAL FRAME", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "TRACKED REGION (RED)", (w + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Save
    output_path = Path("tracking_visualization") / "annotated_tracking.png"
    cv2.imwrite(str(output_path), comparison)
    print(f"Created annotated frame: {output_path}")
    
    # Also create zoomed detail
    detail = np.zeros((300, 600, 3), dtype=np.uint8)
    detail[:, :] = (240, 240, 240)
    
    # Place zoomed tracked region
    tracked_large = cv2.resize(tracked_region, (240, 240), interpolation=cv2.INTER_LINEAR)
    edges_large = cv2.resize(edges_colored, (240, 240), interpolation=cv2.INTER_LINEAR)
    
    detail[30:270, 30:270] = tracked_large
    detail[30:270, 330:570] = edges_large
    
    # Add labels
    cv2.putText(detail, "Tracked Region (4x zoom)", (35, 20), font, 0.6, (0, 0, 0), 1)
    cv2.putText(detail, "Edge Features", (335, 20), font, 0.6, (0, 0, 0), 1)
    
    # Add boxes
    cv2.rectangle(detail, (30, 30), (270, 270), (0, 0, 255), 2)
    cv2.rectangle(detail, (330, 30), (570, 270), (0, 0, 255), 2)
    
    # Save detail
    detail_path = Path("tracking_visualization") / "tracking_detail_zoom.png"
    cv2.imwrite(str(detail_path), detail)
    print(f"Created detail zoom: {detail_path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CREATING TRACKING EXPLANATION SUMMARY")
    print("="*60)
    
    create_tracking_summary()
    
    print("\n" + "="*60)
    print("SUMMARY COMPLETE")
    print("="*60)
    print("\nCreated files:")
    print("1. tracking_summary.png - Full explanation")
    print("2. annotated_tracking.png - Frame with annotations")
    print("3. tracking_detail_zoom.png - Zoomed view of tracked pixels")